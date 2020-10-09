import torch
from torch.utils.data import (
    DataLoader,
    Dataset,
    RandomSampler,
    SequentialSampler,
    TensorDataset
)

import json
import os
import pickle
import logging
import time

logging.basicConfig(
    handlers=[
        logging.FileHandler(f"log/{time.strftime('%Y-%m-%d-%H:%M')}"),
        logging.StreamHandler()
    ],
    format='%(asctime)s -- %(levelname)s -- %(message)s', 
    datefmt='%Y-%m-%d %H:%M', 
    level=logging.DEBUG
)

LOG = logging.getLogger()

def get_logger():
    return logging.getLogger()


def load_refs_from_txt(fpath):
    refs = []
    with open(fpath) as f:
        for line in f:
            pair = line.strip().split()
            if pair: refs.append(pair)
                
    return refs


def convert_examples_to_triple_dataset(
    examples,
    tokenizer,
    max_length
):

    text_center_tensors = []
    attn_center_tensors = []
    text_pos_tensors = []
    attn_pos_tensors = []
    text_neg_tensors = []
    attn_neg_tensors = []

    tokenizer_config = {
        "max_length": max_length,
        "return_attention_mask": True,
        "return_token_type_ids": False,
        "truncation": True,
        "padding": "max_length"
    }

    for e in examples:
        tokenized_center = tokenizer(e["text_center"], **tokenizer_config)
        tokenized_pos = tokenizer(e["text_pos"], **tokenizer_config)
        tokenized_neg = tokenizer(e["text_neg"], **tokenizer_config)
        text_center_tensors.append(tokenized_center['input_ids'])
        attn_center_tensors.append(tokenized_center['attention_mask'])
        text_pos_tensors.append(tokenized_pos['input_ids'])
        attn_pos_tensors.append(tokenized_pos['attention_mask'])
        text_neg_tensors.append(tokenized_neg['input_ids'])
        attn_neg_tensors.append(tokenized_neg['attention_mask'])

    dataset = TensorDataset(
        torch.tensor(text_center_tensors, dtype=torch.long),
        torch.tensor(attn_center_tensors, dtype=torch.long),
        torch.tensor(text_pos_tensors, dtype=torch.long),
        torch.tensor(attn_pos_tensors, dtype=torch.long),
        torch.tensor(text_neg_tensors, dtype=torch.long),
        torch.tensor(attn_neg_tensors, dtype=torch.long)
    )

    return dataset


def convert_examples_to_desc_dataset(    
    examples,
    tokenizer,
    max_length
):
    id_tensors = []
    desc_tensors = []
    attn_tensors = []

    tokenizer_config = {
        "max_length": max_length,
        "return_attention_mask": True,
        "return_token_type_ids": False,
        "truncation": True,
        "padding": "max_length"
    }

    for e in examples:
        id_tensors.append(torch.Tensor([int(e["id"])]))
        tokenized_desc = tokenizer(e["desc"], **tokenizer_config)
        desc_tensors.append(tokenized_desc['input_ids'])
        attn_tensors.append(tokenized_desc['attention_mask'])

    dataset = TensorDataset(
        torch.tensor(id_tensors, dtype=torch.long),
        torch.tensor(desc_tensors, dtype=torch.long),
        torch.tensor(attn_tensors, dtype=torch.long)
    )

    return dataset
        
        
def load_jsonl(fpath):
    jsonl = []
    with open(fpath) as f:
        for line in f:
            example = json.loads(line.strip())
            if example:
                jsonl.append(example)
    return jsonl


def load_and_cache_dataset(fpath, tokenizer, mode, max_len=128, cache=None):
    if cache and os.path.exists(cache):
        LOG.info(f"Loading from cache {cache}.")
        with open(cache, "rb") as f:
            dataset = pickle.load(f)
            
    else:
        examples = load_jsonl(fpath)
        if mode == "train":
            convert_func = convert_examples_to_triple_dataset
        else:
            convert_func = convert_examples_to_desc_dataset
            
        dataset = convert_func(examples, tokenizer, max_len)
        if cache and not os.path.exists(cache):
            with open(cache, "wb") as f:
                pickle.dump(dataset, f)
        
    return dataset
    
    