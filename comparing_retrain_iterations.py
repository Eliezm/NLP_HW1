import os, pickle, re
import numpy as np
from collections import Counter
from inference import tag_all_test
import spacy
from spacy.tokenizer import Tokenizer

RAW     = "data/comp2.words"
SOTA    = "comp2_sota.wtag"
ITERS   = 5   # however many iter*.pkl you saved
NLINES  = sum(1 for _ in open(RAW))

# ensure SOTA exists
if not os.path.exists(SOTA):
    nlp = spacy.load("en_core_web_sm", disable=["parser","ner"])
    nlp.tokenizer = Tokenizer(nlp.vocab, token_match=re.compile(r"[^ ]+").match)
    with open(RAW) as fin, open(SOTA,"w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                fout.write("\n")
            else:
                doc = nlp(line)
                fout.write(" ".join(f"{t.text}_{t.tag_}" for t in doc) + "\n")
    print("[eval] Generated spaCy SOTA tags.")

results = []

for itr in range(1, ITERS+1):
    wfile = f"weights2_selfTrain_iter{itr}.pkl"
    pred_out = f"predictions_iter{itr}.wtag"

    if not os.path.exists(wfile):
        print(f"[eval] Missing {wfile}, skipping")
        continue

    # 1) tag with your model
    print(f"[eval] Iter {itr}: loading {wfile}")
    (opt_params, ft2id) = pickle.load(open(wfile,"rb"))
    w = opt_params[0]

    print(f"[eval] Iter {itr}: tagging → {pred_out}")
    tag_all_test(RAW, w, ft2id, pred_out)

    # 2) compare to SOTA
    conf, gc = Counter(), Counter()
    with open(pred_out) as fm, open(SOTA) as fs:
        for lm, ls in zip(fm, fs):
            tm_seq = [wt.rsplit("_",1)[1] for wt in lm.split()]
            ts_seq = [wt.rsplit("_",1)[1] for wt in ls.split()]
            if len(tm_seq)!=len(ts_seq):
                continue
            for tm, ts in zip(tm_seq, ts_seq):
                conf[(ts,tm)] += 1
                gc[ts] += 1

    total   = sum(gc.values())
    correct = sum(conf[(t,t)] for t in gc)
    overall = correct/total if total else 0

    # compute macro‐F1 average
    f1_scores = []
    for t in gc:
        tp = conf[(t,t)]
        fp = sum(conf[(other,t)] for other in gc if other!=t)
        fn = sum(conf[(t,other)] for other in gc if other!=t)
        p  = tp/(tp+fp) if tp+fp else 0
        r  = tp/(tp+fn) if tp+fn else 0
        f1 = 2*p*r/(p+r) if p+r else 0
        f1_scores.append(f1)
    macro_f1 = np.mean(f1_scores)

    results.append((itr, overall, macro_f1))
    print(f"[eval] Iter {itr}: agreement {correct}/{total} = {overall:.2%}, macro-F1 = {macro_f1:.2%}")

# 3) summary
print("\n=== Summary over iterations ===")
print("Iter  Agreement   Macro-F1")
for itr, agr, f1 in results:
    print(f"{itr:4d}   {agr:8.2%}   {f1:8.2%}")

