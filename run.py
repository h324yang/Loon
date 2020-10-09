from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
import torch
from torch.utils.data import DataLoader
import os
import numpy as np
import scipy
import argparse

from util import (
    load_and_cache_dataset, 
    get_logger, 
    load_refs_from_txt, 
    load_jsonl
)

from model import (
    PairwiseBERT, 
    compute_loss, 
    compute_desc_reprs
)



from transformers import (
    AutoTokenizer, 
    AutoModel, 
    AdamW
)

LOG = get_logger()

def get_args():
    parser = argparse.ArgumentParser(description='Run PairwiseBert on WikidataLow.')
    
    parser.add_argument('--language', type=str, required=True, 
                        help='specify the source language')
    
    parser.add_argument('--data_dir', type=str, default='../massive-align/WikidataLow',
                        help='specify the directory of data folders')
    
    parser.add_argument('--model_name', type=str, default='bert-base-multilingual-cased',
                        help='specify the BERT version')
    
    parser.add_argument('--pretrained_ckpt', type=str, default='',
                        help='specify the pre-trained checkpoint')    
    
    parser.add_argument('--output_dir', type=str, default='./ckpt',
                        help='specify the output directory of model checkpoints')
    
    parser.add_argument('--max_seq_length', type=int, default=128,
                        help='specify the maximum sequence length after tokenization')
    
    parser.add_argument('--dim', type=int, default=300,
                        help='specify the dimensionality of description representations')    
    parser.add_argument('--learning_rate', type=float, default=6e-6,
                        help='specify the learning rate')
    
    parser.add_argument('--adam_epsilon', type=float, default=1e-8,
                        help='specify the value of epsilon for Adam optimizer')

    parser.add_argument('--cache_path', type=str, default='./dataset_cased.cached',
                        help='specify the path for datase caches')
    
    parser.add_argument('--batch_size', type=int, default=12,
                        help='specify the batch size')    

    parser.add_argument('--device', type=str, default='cuda:0',
                        help='specify the device')
    
    parser.add_argument('--epochs', type=int, default=5,
                        help='specify the number of training epochs')    
    
    parser.add_argument('--do_train', action='store_true', 
                        help='run the training process')
    
    parser.add_argument('--do_test', action='store_true',
                        help='run the test process')    
    
    parser.add_argument('--log_every', type=int, default=50,
                        help='specify the logging frequency')
    
    args = parser.parse_args()
    return args


def run_train(train_dataset, model, optimizer, epoch, args):
    model.zero_grad()
    model = model.train()
    dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    total = len(dataloader)
    accu_loss = 0.
    for idx, batch in enumerate(dataloader):
        loss = compute_loss(model, batch)

        loss.backward()
        optimizer.step()
        model.zero_grad()
        
        accu_loss += loss.item()
        if (idx + 1) % args.log_every == 0:
            avg_loss = accu_loss / args.log_every
            info = f"Epoch: {epoch} | Step: {idx+1}/{total} | Loss: {avg_loss:.4f}"
            accu_loss = 0.
            LOG.info(info)

            
def ranking_task(vecs, test_pairs, top_k=(1, 10, 50)):
    head_vecs = np.array([vecs[e1] for e1, e2 in test_pairs])
    tail_vecs = np.array([vecs[e2] for e1, e2 in test_pairs])
    sim = scipy.spatial.distance.cdist(head_vecs, tail_vecs, metric='cityblock')
    hit_k_linked = [0] * len(top_k)
    hit_k_nil = [0] * len(top_k)
    for i, pair in enumerate(test_pairs):
        ranked_tail_ids = [test_pairs[idx][1] for idx in sim[i, :].argsort()]
        target_rank = ranked_tail_ids.index(pair[1])
        for j in range(len(top_k)):
            if target_rank < top_k[j]:
                hit_k = hit_k_nil if pair[1] == "-1" else hit_k_linked
                hit_k[j] += 1
                
    hit_k_all = [hitl + hitn for hitl, hitn in zip(hit_k_linked, hit_k_nil)]
    
    results = []
    total_all = len(test_pairs)
    total_linked = total_all - ranked_tail_ids.count("-1")
    total_nil = total_all - total_linked
    for hit_k, total, category in [
        (hit_k_all, total_all, "All"), 
        (hit_k_linked, total_linked, "Linked"), 
        (hit_k_nil, total_nil, "NIL"), 
    ]:
        for i in range(len(hit_k)):
            res = hit_k[i]/total*100
            LOG.info(f"Result (#{category}: {total}) | Hits@{top_k[i]}: {res:.2f}%")
            results.append((category, total, top_k[i], res))
        
    return results

        
def run_eval(eval_dataset, labels, model, args):
    model = model.eval()
    dataloader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False)
    with torch.no_grad():
        reprs = {}
        total = len(dataloader)
        for bid, batch in enumerate(dataloader):
            batch_reprs = compute_desc_reprs(model, batch)
            reprs.update(batch_reprs)
            if (bid + 1) % (total // 4) == 0:
                LOG.info(f"Inferred {bid+1}/{total} batches.")
                
    return ranking_task(reprs, labels)
          
            
if __name__ == "__main__":
    args = get_args()
    LOG.info(vars(args))
    
    # MODEL INIT
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model_path = args.pretrained_ckpt if args.pretrained_ckpt else args.model_name
    model = PairwiseBERT.from_pretrained(model_path, dim=args.dim)
    model = model.to(args.device)
    optimizer = AdamW(
        model.parameters(), 
        lr=args.learning_rate, 
        eps=args.adam_epsilon
    )
    
    LOG.info(model)

    # LOAD DATASETS
    # training triples
    train_dataset = load_and_cache_dataset(
        os.path.join(
            args.data_dir, 
            f"examples/{args.language}_en.triples.text.train.jsonl"
        ), 
        tokenizer,
        mode="train", 
        max_len=args.max_seq_length, 
        cache=args.cache_path+f".{args.language}.train.pkl"
    )
    
    # for dev & test ranking
    splits = {}
    for split in ["dev", "test"]:
        pairs = load_refs_from_txt(
            os.path.join(
                args.data_dir, 
                f"data/{args.language}_en/{split}"
            )
        )
        splits[split] = pairs
    
    # load desc
    desc_dataset = load_and_cache_dataset(
        os.path.join(
            args.data_dir, 
            f"examples/{args.language}_en.text.ref.jsonl"
        ), 
        tokenizer,
        mode="eval", 
        max_len=args.max_seq_length, 
        cache=args.cache_path+f".{args.language}.eval.pkl"
    )

    if args.do_train:
        dev_results = []
        for epoch in range(args.epochs):
            run_train(train_dataset, model, optimizer, epoch, args)
            model.save_pretrained(
                f"{args.output_dir}/{args.language}.ckpt.{epoch+1}"
            )
            LOG.info(f"Model (epoch: {epoch+1}) saved.")
            
            dev_result = run_eval(desc_dataset, splits['dev'], model, args)
            dev_results.append(dev_result)
            
        else:
            for i, res in enumerate(dev_results):
                LOG.info(f"{(i+1)}: {res}")
                
    if args.do_test:
        run_eval(desc_dataset, splits['test'], model, args)
