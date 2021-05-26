#!/usr/bin/env python3

import torch
from transformers import get_linear_schedule_with_warmup
from transformers import GPT2Config, AdamW
from transformers import AutoTokenizer, AutoModel
import pickle5 as pickle

import torch.nn.functional as F
import os
from datetime import datetime
import random
from random import randint, shuffle
from transformers import AutoModelForCausalLM
import gc
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.nn import CrossEntropyLoss
from tqdm.notebook import tqdm
import numpy as np
import argparse
import sys
import re

POST_CNT = 5880
IN_TOKENS = 768
device = 'cpu'
SEQ_LEN = 100
TEMP = 1.5
CP_PATH = 'state_dict.pt'
TOP_K = 10
MODEL_FN = 'model.pth'

seps = set('.!?')
garb = set(' .!?,')


def cut(s, num=2):
    if num <= 0:
        num = 2

    start = 0
    for symb in s:
        if symb not in garb:
            break
        start += 1

    end = start
    while end < len(s):
        if num == 0:
            break

        while s[end] in seps and end + 1 < len(s) and s[end + 1] in seps:
            end += 1

        if s[end] in seps:
            num -= 1
        end += 1
    return s[start:end]


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


def sample_seq(model, context, length, device, top_k, top_p, temperature=TEMP):
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
        for _ in range(length):
            inputs = {'input_ids': generated}
            outputs = model(**inputs)
            next_token_logits = outputs[0][0, -1, :] / temperature
            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
            next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
            generated = torch.cat((generated, next_token.unsqueeze(0)), dim=1)
    return generated


def generate_sample(data, tokenizer, model, length=SEQ_LEN, temperature=TEMP, top_k=TOP_K, top_p=0.5,
                    device=torch.device(device)):
    """ Generate summaries for articles.
        Args:
            data = GPT21024Dataset object
            tokenizer = gpt/gpt2 tokenizer
            model = gpt/gpt2 model
    """

    for i in range(len(data)):
        post_len = data[i][1]
        context = data[i][0][:post_len]

        generated_text = sample_seq(model, context, length, device, top_k, top_p, temperature)
        generated_text = generated_text[0, len(context):]
        text = tokenizer.convert_ids_to_tokens(generated_text, skip_special_tokens=True)
        text = tokenizer.convert_tokens_to_string(text)
        text = cut(re.sub(r'\s+', ' ', text), randint(1, 2))

        print(f'> P: {tokenizer.decode(context)}')
        print(f'> C: {text}', end='\n\n\n')


def data_preparation(fin, tokenizer):
    test_ready = []
    # comm = 'раз два три четыре пять.'
    for post in fin:
        post_tok = tokenizer.encode(post)
        # cmt_tok = tokenizer.encode(comm)
        # content = post_tok + tokenizer.encode(tokenizer.sep_token) + cmt_tok
        content = post_tok
        if len(content) > IN_TOKENS or len(post_tok) == 0:
            print(f'wrong len: {len(content)}', end='', file=sys.stderr)
            continue

        text = tokenizer.encode(tokenizer.pad_token) * IN_TOKENS
        text[:len(content)] = content
        test_ready.append((text, len(post_tok)))

    return test_ready


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--file", default='resrc.txt', help='file with input', )
    parser.add_argument("--seq", default=100, type=int)
    parser.add_argument("--temp", default=1.5, type=float)
    parser.add_argument("--topk", default=10, type=int)
    parser.add_argument("--device", default='cpu')
    parser.add_argument("--cppath", default='state_dict.pt')

    args = parser.parse_args()

    samples_file = args.file
    SEQ_LEN = args.seq
    TEMP = args.temp
    TOP_K = args.topk
    device = args.device
    CP_PATH = args.cppath

    print('loading tokenizer...', file=sys.stderr)
    try:
        tokenizer = torch.load('tokenizer', map_location=torch.device(device))
    except:
        tokenizer = AutoTokenizer.from_pretrained("sberbank-ai/rugpt3small_based_on_gpt2")

    special_tokens = {'pad_token': '<|pad|>', 'sep_token': '<|sep|>'}
    num_add_toks = tokenizer.add_special_tokens(special_tokens)

    ignore_idx = tokenizer.pad_token_id

    print('loading model...', file=sys.stderr)
    # if MODEL_FN not in os.listdir():
    #     print('Model not found. Use actual repo state.', file=sys.stderr)
    try:
        model = torch.load(MODEL_FN, map_location=torch.device(device))
        model.to(torch.device(device))
    except:
        model = AutoModelForCausalLM.from_pretrained("sberbank-ai/rugpt3small_based_on_gpt2")
        model.resize_token_embeddings(len(tokenizer))

        state = torch.load(CP_PATH, map_location=torch.device(device))
        model.load_state_dict(state['model_state_dict'])
        model.to(torch.device(device))
        torch.save(model, MODEL_FN)

    model.eval()
    with torch.no_grad():
        print('Prediction:', file=sys.stderr)
        with open(samples_file, 'r') as fin:
            test_ready = data_preparation(fin, tokenizer)
            generate_sample(test_ready, tokenizer, model, device=torch.device(device))
