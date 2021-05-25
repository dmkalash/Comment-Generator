#!/usr/bin/env python3

import torch
from transformers import get_linear_schedule_with_warmup
from transformers import GPT2Config,AdamW
from transformers import AutoTokenizer, AutoModel
import pickle5 as pickle
from tqdm.notebook import tqdm
import json
from tensorboardX import SummaryWriter
import torch.nn.functional as F
import os
from datetime import datetime
import random
from random import randint
from transformers import AutoModelForCausalLM
import gc
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.nn import CrossEntropyLoss
from tqdm.notebook import tnrange, tqdm
import numpy as np
from random import shuffle
import argparse


POST_CNT = 5880
TEST_CNT = 20
VAL_CNT = 10
IN_TOKENS = 768
device = 'cuda:0'
SEQ_LEN = 100
TEMP = 1.5
CP_PATH = 'state_dict.pt'


tokenizer = AutoTokenizer.from_pretrained("sberbank-ai/rugpt3small_based_on_gpt2")
special_tokens = {'pad_token':'<|pad|>','sep_token':'<|sep|>'}
num_add_toks = tokenizer.add_special_tokens(special_tokens)


ignore_idx = tokenizer.pad_token_id
#model = AutoModelForCausalLM.from_pretrained("Grossmend/rudialogpt3_medium_based_on_gpt2")
model = AutoModelForCausalLM.from_pretrained("sberbank-ai/rugpt3small_based_on_gpt2")
#model = AutoModelForCausalLM.from_pretrained("sberbank-ai/rugpt3medium_based_on_gpt2")
# sberbank-ai/rugpt3medium_based_on_gpt2
#model = torch.load('drive/MyDrive/Colab Notebooks/rus01.pth')

print(len(tokenizer))
model.resize_token_embeddings(len(tokenizer))

model.to(torch.device(device))


state = torch.load(CP_PATH)

optimizer.load_state_dict(state['optimizer_state_dict'])
model.load_state_dict(state['model_state_dict'])


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits

def sample_seq(model, context, length, device, temperature=TEMP, top_k=0, top_p=0.0):
    """ Generates a sequence of tokens 
        Args:
            model: gpt/gpt2 model
            context: tokenized text using gpt/gpt2 tokenizer
            length: length of generated sequence.
            device: torch.device object.
            temperature >0: used to control the randomness of predictions by scaling the logits before applying softmax.
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
    """
    
    context = torch.tensor(context, dtype=torch.long, device=device)
    context = context.unsqueeze(0)
    generated = context
    with torch.no_grad():  
        for _ in tnrange(length):
            inputs = {'input_ids': generated}
            outputs = model(**inputs)
            next_token_logits = outputs[0][0, -1, :] / temperature
            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
            next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
            generated = torch.cat((generated, next_token.unsqueeze(0)), dim=1)
    return generated


def generate_sample(data, tokenizer, model, eval_step=False, length=SEQ_LEN, temperature=1, top_k=10, top_p=0.5, device=torch.device(device)):
    """ Generate summaries for articles.
        Args:
            data = GPT21024Dataset object
            tokenizer = gpt/gpt2 tokenizer
            model = gpt/gpt2 model
            eval_step = can be True/False, checks generating during evaluation or not
    """

    for i in range(len(data)):
        sample = data[i]
        idx = sample['post_len']
        context = sample['post_com'][:idx]
        summary = sample['post_com'][idx+1:][:100]
        generated_text = sample_seq(model, context, length, device, temperature, top_k, top_p)
        generated_text = generated_text[0, len(context):]
        text = tokenizer.convert_ids_to_tokens(generated_text,skip_special_tokens=True)
        text = tokenizer.convert_tokens_to_string(text)
        text = re.sub(r'\s+', ' ', text)
        if eval_step==False:
            print('\n\n')
            print('new_post:')
            print(f'<<<{tokenizer.decode(context)}>>>', end='\n\n')
            print("generated_comment:")
            print(f'<<<{text}>>>', end='\n\n')
            print('actual_comment:')
            print(f'<<<{tokenizer.decode(summary)}>>>', end='\n\n')
        else:
            print("generated_comment:")
            print(f'<<<{text}>>>', end='\n\n')
            print('\n\n')


def data_preparation(fin):
	for_test_ids = []
	ids_to_items = {}
	i = 0
	train_ready = {}

	for post in vk_comments:
  		post_tok = tokenizer.encode(post)
  		for_test_ids.append(i)
  		for cmt in cmts:
    		text = tokenizer.encode(tokenizer.pad_token) * IN_TOKENS

    cmt_tok = tokenizer.encode(cmt)
    content = post_tok + tokenizer.encode(tokenizer.sep_token) + cmt_tok
    if len(content) > IN_TOKENS or len(cmt_tok) == 0 or len(post_tok) == 0:
        print(f'{i};', end='')
      	#print(tokenizer.decode(content))
        continue
    text[:len(content)] = content
    ids_to_items[i] = {'post': post_tok, 'comment': cmt_tok}
    train_ready[i] = {'post_com': text, 'post_len': len(post_tok)}
    i += 1


if __name__ == __main__:
	parser = argparse.ArgumentParser()
	args = parser.parse_args()
	parser.add_argument("--file", default='resrc.txt')
	samples_file = parser.file
	with open(samples_file, 'r') as fin:
		
		generate_sample(valid_dataset_paired, tokenizer, model, device=torch.device(device))
	
	
	