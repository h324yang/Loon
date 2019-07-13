import numpy as np
from collections import defaultdict


def read_embd(path):
    embd_dict = {}
    with open(path) as f:
        for line in f:
            spt = line.strip().split("\t")
            idx = int(spt[0])
            array = np.array(spt[1:]).astype(np.float32)
            embd_dict[idx] = array
    return embd_dict


def read_qrels(path):
    gold_dict = defaultdict(list)
    with open(path) as f:
        for line in f:
            spt = line.strip().split("\t")
            qid = int(spt[0])
            pid = int(spt[2])
            gold_dict[qid].append(pid)
    return gold_dict


def read_top1000(path):
    q_dict = {}
    p_dict = {}
    top1000_dict = defaultdict(list)
    with open(path) as f:
        for line in f:
            spt = line.strip().split("\t")
            if len(spt) == 4:
                qid, pid, qsent, psent = spt
                q_dict[int(qid)] = qsent
                p_dict[int(pid)] = psent
                top1000_dict[int(qid)].append(int(pid))
    return q_dict, p_dict, top1000_dict

