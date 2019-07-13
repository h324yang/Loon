from sklearn.metrics.pairwise import pairwise_distances
import numpy as np
from loader import read_embd, read_qrels
import annoy
import pickle
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def recall(q_repr, c_repr, qrels, k=None):
    hit = 0.
    sim = pairwise_distances(q_repr, c_repr, metric="cityblock")
    gold = {idx:int(qrel) for idx, qrel in enumerate(qrels)}
    rank = np.argsort(sim)[0]
    for place, idx in enumerate(rank):
        if gold[idx] > 0:
            hit += 1.
        if k and place+1 >= k:
            break
    return hit, np.sum(qrels)


def reciprocal_rank(q_repr, c_repr, qrels, k=None):
    sim = pairwise_distances(q_repr, c_repr, metric="cityblock")
    gold = {idx:int(qrel) for idx, qrel in enumerate(qrels)}
    rank = np.argsort(sim)[0]
    for place, idx in enumerate(rank):
        if gold[idx] > 0:
            return 1/float(place+1)
        if k and place+1 >= k:
            break
    return 0.


class AnnoyEvaluator:
    def __init__(self, q_file, p_file, qrel_file, metric="manhattan"):
        self.q_file = q_file
        self.p_file = p_file
        self.qrel_file = qrel_file
        self.query = read_embd(self.q_file)
        self.psg = read_embd(self.p_file)
        self.qrels = read_qrels(self.qrel_file)
        self.gen_new_index()
        self.metric = metric

    def gen_new_index(self):
        new_to_ori = {}
        ori_to_new = {}
        for newid, oriid in enumerate(self.query):
            new_to_ori[newid] = oriid
            ori_to_new[oriid] = newid
        self.query_new_to_ori = new_to_ori
        self.query_ori_to_new = ori_to_new
        self.dim = self.query[oriid].shape[0]

        last = newid+1
        new_to_ori = {}
        ori_to_new = {}
        for newid, oriid in enumerate(self.psg):
            ori_to_new[oriid] = last+newid
            new_to_ori[last+newid] = oriid
        self.psg_new_to_ori = new_to_ori
        self.psg_ori_to_new = ori_to_new

    def gen_annoy(self, opath, ntree=10):
        t = annoy.AnnoyIndex(self.dim, metric=self.metric)
        for ori_qid, qvec in self.query.items():
            t.add_item(self.query_ori_to_new[ori_qid], qvec)
        for ori_pid, pvec in self.psg.items():
            t.add_item(self.psg_ori_to_new[ori_pid], pvec)
        t.build(ntree) # 10 trees
        t.save(opath)

        config = {}
        index_map = {"query_new_to_ori": self.query_new_to_ori,
             "query_ori_to_new": self.query_ori_to_new,
             "psg_new_to_ori": self.psg_new_to_ori,
             "psg_ori_to_new": self.psg_ori_to_new}
        config["index_map"] = index_map
        config["dim"] = self.dim

        with open(opath+".meta", "wb") as p:
            pickle.dump(config, p)

    def load_meta(self, opath):
        with open(opath+".meta", "rb") as p:
            config = pickle.load(p)
        self.query_new_to_ori = config["index_map"]["query_new_to_ori"]
        self.query_ori_to_new = config["index_map"]["query_ori_to_new"]
        self.psg_new_to_ori = config["index_map"]["psg_new_to_ori"]
        self.psg_ori_to_new = config["index_map"]["psg_ori_to_new"]
        self.dim = config["dim"]

    def eval(self, annoy_index, gold_dict=None, topk=3):
        if not gold_dict:
            print("reading qrels...")
            gold_dict = self.qrels
        BUFF = len(self.query)
        u = annoy.AnnoyIndex(self.dim, metric=self.metric)
        u.load(annoy_index)
        self.load_meta(annoy_index)
        RR = []
        Recall = []
        for oriid, newid in self.query_ori_to_new.items():
            RR_checked = 0.
            postive = 0.
            if oriid in gold_dict:
                # exclude nn which are queries
                knn = u.get_nns_by_item(newid, topk+1+BUFF)
                cnt = 0
                valid_knn = []
                for nn in knn:
                    if nn in self.psg_new_to_ori:
                        valid_knn.append(nn)
                        cnt += 1
                        if cnt >= topk:
                            break

                for rank, nn in enumerate(valid_knn):
                    if self.psg_new_to_ori[nn] in gold_dict[oriid]:
                        postive += 1
                        if not RR_checked:
                            RR_checked = 1./float(rank+1)

                RR.append(RR_checked)
                Recall.append(postive/float(len(gold_dict[oriid])))
            else:
                RR.append(0.)
                Recall.append(0.)
        return np.average(RR), RR, np.average(Recall), Recall


if __name__ == "__main__":
    # test
    rr = reciprocal_rank([[1,1,1]], [[2,2,3], [1,4,5], [1,1,1]], [0,0,0], 3)
    print(rr)
