import torch
from torch import nn
from model.tokenization import BertTokenizer
from model.module import PairwiseBERT
from loader.hdf5 import HDF5er
from trainer import Trainer
import numpy as np
from utils import never_split
from utils.metrics import reciprocal_rank, recall
from utils.logger import get_logger
import pickle
import os


logger = get_logger("debug_dataset/debug.log")
vocab_path = "vocabs/tacred-bert-base-cased-vocab.txt"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
TOKENIZER = BertTokenizer(vocab_file=vocab_path, never_split=never_split)


class TrainProcess:
    def __init__(self):
        self.distance = nn.PairwiseDistance(p=1)
        self.loss = nn.MarginRankingLoss(margin=3).cuda()

    def train(self, model, batch):
        q = torch.tensor(batch[0][:], dtype=torch.long).to(DEVICE)
        p = torch.tensor(batch[1][:], dtype=torch.long).to(DEVICE)
        n = torch.tensor(batch[2][:], dtype=torch.long).to(DEVICE)
        query_repr = model(q, True) # is_query=True
        pos_repr = model(p, False)
        neg_repr = model(n, False)
        positive_distance = self.distance(query_repr, pos_repr)
        negative_distance = self.distance(query_repr, neg_repr)
        target = torch.ones_like(query_repr[:, 0]).float().to(DEVICE)
        loss = self.loss(positive_distance, negative_distance, -1*target)
        return loss

class EvalProcess:
    def __init__(self, query_candidate_dict):
        self.query_repr_dict = {}
        self.passage_repr_dict = {}
        self.batch_results = []
        self.query_candidate_dict = query_candidate_dict

    def eval(self, model, batch):
        with torch.no_grad():
            q_tokens, q_ids, p_tokens, p_ids = [], [], [], []
            for stype, sid, tokens in zip(batch[0], batch[1], batch[2]):
                # split a batch into query batch and passage batch
                if int(stype) == 0:
                    q_ids.append(int(sid))
                    q_tokens.append(tokens)
                else:
                    p_ids.append(int(sid))
                    p_tokens.append(tokens)

            if len(q_ids) > 0:
                q_repr = model(torch.tensor(q_tokens, dtype=torch.long).to(DEVICE), True).data.cpu().numpy()
                for qid, qrerp in zip(q_ids, q_repr):
                    self.query_repr_dict[qid] = qrerp
            if len(p_ids) > 0:
                p_repr = model(torch.tensor(p_tokens, dtype=torch.long).to(DEVICE), False).data.cpu().numpy()
                for pid, prepr in zip(p_ids, p_repr):
                    self.passage_repr_dict[pid] = prepr
        return None

    def aggregate(self):
        RRs = []
        Recalls = []
        total_hit = 0.
        total_base = 0.
        for q, (candi, qrels) in self.query_candidate_dict.items():
            q_repr = np.array([self.query_repr_dict[int(q)]])
            c_repr = np.array([self.passage_repr_dict[int(c)] for c in candi])
            RRs.append(reciprocal_rank(q_repr, c_repr, qrels, k=10))
            hit, base = recall(q_repr, c_repr, qrels, k=10)
            total_hit += hit
            total_base += base
            if base>0:
                Recalls.append(hit/float(base))
            else:
                Recalls.append(None)
        result = (np.mean(RRs), total_hit/float(total_base))
        logger.info(RRs)
        logger.info("MRR:%.5f"%result[0])
        logger.info(Recalls)
        logger.info("MRecalls:%.5f"%result[1])
        return result[0]

if __name__ == "__main__":
    epoch = 1000
    valid_every = 200
    learning_rate = 1e-6
    repr_size = 768
    grad_accu_step = 1
    fields = ["query", "pos", "neg"]
    max_length = [64, 256, 256]
    pretrained_version = 'bert-base-cased'
    data_batch_hdf5 = "debug_dataset/triples.debug.batch.hdf5"
    ckpt_path = "ckpt/debugging.pt"
    query_candidate_dict_path = "debug_dataset/candidate_pool.pkl"
    to_load_ckpt = False

    pairwise_bert = PairwiseBERT(pretrained_version=pretrained_version, repr_size=repr_size).to(DEVICE)

    train_proc = TrainProcess()
    eval_proc = EvalProcess(pickle.load(open(query_candidate_dict_path, "rb")))
    trainer = Trainer(pairwise_bert, learning_rate, train_proc, eval_proc, logger)
    if to_load_ckpt:
        trainer.load_ckpt(ckpt_path)

    # Training
    train_data = HDF5er(data_path=None, fields=fields, max_length=max_length, tokenizer=None, preprocess=None)
    train_mbs = train_data.get_minibatches(data_batch_hdf5)

    # Testing
    fields = ["type", "id", "sent"]
    max_length = [1, 1, 256]
    skip = [True, True, False]
    dev_batch_hdf5 = "debug_dataset/sent.debug.batch.hdf5"
    dev_data = HDF5er(data_path=None, fields=fields, max_length=max_length, preprocess=None, tokenizer=None, skip=skip)
    dev_mbs = dev_data.get_minibatches(dev_batch_hdf5)

    trainer.test(dev_mbs)
    trainer.start(train_mbs, epoch, dev_mbs, valid_every, ckpt_path, grad_accu_step)

