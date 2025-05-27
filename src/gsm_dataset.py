import re, json

import numpy as np
import torch
from torch.utils.data import Dataset

GSM_DATA_PATH = "../gsm/grade_school_math/data/train.jsonl"

GSM_PROMPT_PREFIX = '''You are a helpful and intelligent assistant. You will be asked a \
math problem. Please think step by step, and indicate your answer with four \
hashtags. Here is an example exchange:'''

GSM_PROMPT_SUFFIX = 'Respond to the following question using the same format as above.'

def format_example(example):
    question, answer_split = example
    return format_qa(question, ''.join(answer_split))

def gsm_prompt(examples):
    formatted_examples = "\n\n".join(map(format_example, examples))
    return f"{GSM_PROMPT_PREFIX}\n\n{formatted_examples}\n\n{GSM_PROMPT_SUFFIX}\n\n"

def format_qa(question, answer):
    return f"Question: {question}\n\nAnswer: {answer}"

# split by calculator statements
def split_answer(answer):
    return re.split(r'<<.*>>', answer)

class GSMDataset(Dataset):
    def __init__(self, ):
        with open(GSM_DATA_PATH, mode='r') as f:
            self.data = [json.loads(line) for line in f]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pair = self.data[idx]
        return pair['question'], split_answer(pair['answer'])
    
# usage:
#   batch = batch[keep_indices]
#   batch = [a + b for a, b in zip(batch, segment)]
def gsm_collate(pairs):
    questions, answers = zip(*pairs)
    # each answer is made up of multiple segment; we want to form batches s.t. the ith segment
    # of each answer is in the same batch; keep_indices tracks when a particular answer is
    # dropped from the batch because it has no segments left
    segments, keep_indices_lst = [], []
    segments.append([format_qa(question, answer[0]) for question, answer in zip(questions, answers)])

    while True:
        filtered = [(i, answer[1:]) for i, answer in enumerate(answers) if len(answer) > 1]
        if not filtered:
            break
        keep_indices, answers = zip(*filtered)
        keep_indices_lst.append(torch.tensor(keep_indices))
        segments.append([answer[0] for answer in answers])

    return segments, keep_indices_lst

def sample(dataset, num_samples):
    return [dataset[idx] for idx in np.random.randint(len(dataset), size=num_samples)]