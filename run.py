from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
import torch
from torch.utils.data import DataLoader
import os
import numpy as np
import scipy

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
    AutoConfig, 
    AutoTokenizer, 
    AutoModel, 
    AdamW
)

LOG = get_logger()

class Args:
    language = "zh_yue"
    overwrite_cache = False
    data_dir = "../massive-align/WikidataLow"
    model_name = "bert-base-multilingual-cased"
    pretrained_ckpt = None
    output_dir = "./ckpt"
    max_seq_length = 128
    learning_rate = 6e-6
    adam_epsilon = 1e-8
    cache_path = "./dataset_cased.cache.{lang}.{mode}.pkl"
    batch_size = 12
    dim = 300
    device = "cuda:0"
    epochs = 10
    do_train = True
    do_test = True
    save_every = 3000
    log_every = 50


def run_train(train_dataset, model, optimizer, epoch, args):
    model.zero_grad()
    model = model.train()
    dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    total = len(dataloader)
    for idx, batch in enumerate(dataloader):
        loss = compute_loss(model, batch)

        loss.backward()
        optimizer.step()
        model.zero_grad()
        
        if (idx + 1) % args.log_every == 0:
            info = f"Epoch: {epoch} | Step: {idx+1}/{total} | Loss: {loss.item():.4f}"
            LOG.info(info)

            
def ranking_task(vecs, test_pairs, top_k=(1, 10, 50)):
    head_vecs = np.array([vecs[e1] for e1, e2 in test_pairs])
    tail_vecs = np.array([vecs[e2] for e1, e2 in test_pairs])
    sim = scipy.spatial.distance.cdist(head_vecs, tail_vecs, metric='cityblock')
    hit_k = [0] * len(top_k)
    for i in range(head_vecs.shape[0]):
        rank = sim[i, :].argsort()
        rank_index = np.where(rank == i)[0][0]
        for j in range(len(top_k)):
            if rank_index < top_k[j]:
                hit_k[j] += 1
    
    results = []
    for i in range(len(hit_k)):
        res = hit_k[i]/len(test_pairs)*100
        LOG.info(f"Hits@{top_k[i]}: {res:.2f}%")
        results.append((top_k[i], res))
        
    return results

        
def run_eval(eval_dataset, refs, labels, model, args):
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
    LOG.info(vars(Args))
    args = Args()
    
    # MODEL INIT
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model_path = args.pretrained_ckpt if args.pretrained_ckpt else args.model_name
    config = AutoConfig.from_pretrained(model_path)
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
        cache=args.cache_path.format(mode="train", lang=args.language)
    )
    
    # for dev & test ranking
    splits = {}
    for split in ["ref_ent_ids", "dev", "test"]:
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
        cache=args.cache_path.format(mode="eval", lang=args.language)
    )

    if args.do_train:
        dev_results = []
        for epoch in range(args.epochs):
            run_train(train_dataset, model, optimizer, epoch, args)
            model.save_pretrained(
                f"{args.output_dir}/{args.language}.ckpt.{epoch+1}"
            )
            LOG.info(f"Model (epoch: {epoch+1}) saved.")
            
            dev_result = run_eval(
                desc_dataset, 
                splits['ref_ent_ids'], 
                splits['dev'], 
                model, 
                args
            )
            
            dev_results.append(dev_result)
            
        else:
            for i, res in enumerate(dev_results):
                LOG.info(f"{(i+1)}: {res}")
                
    if args.do_test:
        run_eval(desc_dataset, splits['ref_ent_ids'], splits['test'], model, args)
