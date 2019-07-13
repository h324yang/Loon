from collections import defaultdict
from loader import read_qrels, read_top1000
import pickle
from random import shuffle


top1000_path = "msmarco/top1000.dev.tsv"
qrels_path = "msmarco/qrels.dev.small.tsv"
size = 10

if __name__ == "__main__":
    q_dict, p_dict, top1000_dict = read_top1000(top1000_path)
    gold_dict = read_qrels(qrels_path)
    sorted_q = sorted(list(top1000_dict.keys()))
    pids = list(p_dict.keys())
    triple = []
    candidate_pool = {}
    cnt = 0
    for qid in sorted_q:
        if cnt >= size:
            break

        if not qid in q_dict or not qid in gold_dict:
            continue

        pos_ps = gold_dict[qid]

        all_pos_valid = True
        for pos_p in pos_ps:
            if not pos_p in p_dict:
                all_pos_valid = False
        if not all_pos_valid:
            continue

        candi = top1000_dict[qid]
        qrels = [1 if c in pos_ps else 0 for c in candi]
        if sum(qrels) < 2:
            continue

        neg_ps = [c for c in candi if not c in gold_dict[qid]]

        for i, neg_p in enumerate(neg_ps):
            if neg_p in p_dict:
                pos_p = pos_ps[i%len(pos_ps)]
                triple.append((q_dict[qid], p_dict[pos_p], p_dict[neg_p]))

        cnt += 1
        print("{}({} positive)...".format(cnt, sum(qrels)))
        candidate_pool[qid] = (candi, qrels)

    shuffle(triple)
    with open("debug_dataset/triples.debug.tsv", "w") as f:
        for t in triple:
            f.write("\t".join(t)+"\n")

    with open("debug_dataset/candidate_pool.pkl", "wb") as f:
        pickle.dump(candidate_pool, f)

    with open("debug_dataset/sent.debug.tsv", "w") as f:
        qid_pool = set()
        pid_pool = set()
        for qid, (pids, qrels) in candidate_pool.items():
            qid_pool.add(qid)
            for pid in pids:
                pid_pool.add(pid)
        for qid in qid_pool:
            f.write("{}\t{}\t{}\n".format(0, qid, q_dict[qid]))

        for pid in pid_pool:
            f.write("{}\t{}\t{}\n".format(1, pid, p_dict[pid]))


