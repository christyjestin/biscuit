import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

COT_MAX_LENGTH = 6

class Biscuit:
    def __init__(self, base_model_name = "Qwen/Qwen2-0.5B"):
        self.base_model_name = base_model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name, padding_side='left')
        self.token_trunk = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=torch.bfloat16,
                                                                device_map="auto", attn_implementation="sdpa")
        self.latent_trunk = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=torch.bfloat16, 
                                                                 device_map="auto", attn_implementation="sdpa")
        self.token_trunk.config.pad_token_id = self.token_trunk.config.eos_token_id
        self.token_trunk.config.output_hidden_states = True
        self.latent_trunk.config.output_hidden_states = True

        # beginning and end of thought special tokens
        self.bot, self.eot = "<bot>", "<eot>"
        self.bot_embedding = torch.empty(1, self.token_trunk.model.embed_tokens.embedding_dim, device=self.device)
        self.eot_embedding = torch.empty(1, self.token_trunk.model.embed_tokens.embedding_dim, device=self.device)
        torch.nn.init.normal_(self.bot_embedding)
        torch.nn.init.normal_(self.eot_embedding)
        self.token_trunk.train()
        self.latent_trunk.train()

    # same process but for token_batches, the loss is just predicting the beginning of thought token
    # while non token_batches are about using latents to upweight the probability of the discrete CoT
    def compute_batch(self, prompt, segments, keep_indices_lst, token_batch):
        # Step 0: just process the first segment without decoding the next token
        first_segment = [prompt + segment for segment in segments[0]]
        inputs = self.tokenizer(first_segment, return_tensors="pt", padding=True).to(self.device)
        outputs = self.token_trunk(**inputs)
        kv_cache = outputs.past_key_values
        attn_mask = inputs.attention_mask

        CE_loss = torch.nn.CrossEntropyLoss()

        loss = torch.tensor(0.).to(self.device)
        # continuous CoT loop: produce CoT -> use it to predict next segment -> repeat
        for segment, keep_indices in zip(segments[1:], keep_indices_lst):
            # Step 1: drop sequences that are done
            kv_cache.batch_select_indices(keep_indices)
            attn_mask = attn_mask[keep_indices]
            batch_size = keep_indices.shape[0]
            attn_ones = torch.ones(batch_size, 1, dtype=int).to(self.device)

            # learn to output bot token
            # if token_batch:
            #     bot_token = self.tokenizer([self.bot] * batch_size, return_tensors="pt").to(self.device)
            #     loss += CE_loss(outputs.logits[keep_indices, -1], bot_token.input_ids[:, 0])

            # Step 2: then autoregressively predict a continuous chain of thought sequence
            last_hidden_state = None
            k = np.random.randint(1, COT_MAX_LENGTH + 1) # the CoT sequence has a random length
            for i in range(k + 2):
                attn_mask = torch.cat((attn_mask, attn_ones), dim=1)
                if i == 0 or i == k + 1: # process beginning of thought or end of thought token
                    inp = self.bot_embedding if i == 0 else self.eot_embedding
                    outputs = self.token_trunk(inputs_embeds=inp.repeat(batch_size, 1, 1), attention_mask=attn_mask, 
                                               past_key_values=kv_cache)
                else: # process new continuous thought token
                    outputs = self.latent_trunk(inputs_embeds=last_hidden_state, attention_mask=attn_mask, 
                                                past_key_values=kv_cache)
                last_hidden_state = outputs.hidden_states[-1][:, -1:]
                kv_cache = outputs.past_key_values

            # Step 3: finally, predict the next segment and compute the loss
            # pad on the right side so that the CoT and the new input are contiguous
            inputs = self.tokenizer(segment, return_tensors="pt", padding=True, 
                                    padding_side='right').to(self.device)
            # ignore masked tokens when computing loss
            labels = torch.where(inputs.attention_mask.bool(), inputs.input_ids, -100)
            attn_mask = torch.cat((attn_mask, inputs.attention_mask), dim=1)
            outputs = self.token_trunk(input_ids=inputs.input_ids, attention_mask=attn_mask, 
                                 labels=labels, past_key_values=kv_cache)
            kv_cache = outputs.past_key_values
            if not token_batch:
                loss += outputs.loss

        return loss
