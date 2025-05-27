import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

COT_MAX_LENGTH = 6

class Biscuit:
    def __init__(self, base_model = "Qwen/Qwen2-0.5B"):
        self.base_model = base_model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(base_model, padding_side='left')
        self.model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.bfloat16,
                                                          device_map="auto", 
                                                          attn_implementation="sdpa")
        self.model.config.pad_token_id = self.model.config.eos_token_id
        self.model.config.output_hidden_states = True

        # beginning and end of thought special tokens
        self.bot, self.eot = "<bot>", "<eot>"
        self.tokenizer.add_tokens([self.bot, self.eot])
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.train()


    def compute_batch(self, prompt, segments, keep_indices_lst):
        # Step 0: just process the first segment without decoding the next token
        first_segment = [prompt + segment for segment in segments[0]]
        inputs = self.tokenizer(first_segment, return_tensors="pt", padding=True).to(self.device)
        outputs = self.model(**inputs)
        kv_cache = outputs.past_key_values
        attn_mask = inputs.attention_mask

        loss = torch.tensor(0.).to(self.device)
        # continuous CoT loop: produce CoT -> use it to predict next segment -> repeat
        for segment, keep_indices in zip(segments[1:], keep_indices_lst):
            # Step 1: drop sequences that are done
            kv_cache.batch_select_indices(keep_indices)
            attn_mask = attn_mask[keep_indices]
            batch_size = keep_indices.shape[0]
            attn_ones = torch.ones(batch_size, 1, dtype=int).to(self.device)


            # Step 2: then autoregressively predict a continuous chain of thought sequence
            last_hidden_state = None
            k = np.random.randint(1, COT_MAX_LENGTH + 1) # the CoT sequence has a random length
            for i in range(k + 2):
                attn_mask = torch.cat((attn_mask, attn_ones), dim=1)
                if i == 0 or i == k + 1: # process beginning of thought or end of thought token
                    seq = [self.bot if i == 0 else self.eot] * batch_size
                    inputs = self.tokenizer(seq, return_tensors="pt").to(self.device)
                    args = {'input_ids': inputs.input_ids}
                else: # process new continuous thought token
                    args = {'inputs_embeds': last_hidden_state}

                outputs = self.model(**args, attention_mask=attn_mask, past_key_values=kv_cache)
                last_hidden_state = outputs.hidden_states[-1][:, -1:]
                kv_cache = outputs.past_key_values


            # Step 3: finally, predict the next segment and compute the loss
            # pad on the right side so that the CoT and the new input are contiguous
            inputs = self.tokenizer(segment, return_tensors="pt", padding=True, 
                                    padding_side='right').to(self.device)
            # ignore masked tokens when computing loss
            labels = torch.where(inputs.attention_mask.bool(), inputs.input_ids, -100)
            attn_mask = torch.cat((attn_mask, inputs.attention_mask), dim=1)
            outputs = self.model(input_ids=inputs.input_ids, attention_mask=attn_mask, 
                                 labels=labels, past_key_values=kv_cache)
            kv_cache = outputs.past_key_values
            loss += outputs.loss

        return loss
