# rerun_selftrain.py

import pickle, os, re
import numpy as np
from collections import Counter
from preprocessing import read_test, represent_input_with_features
from optimization import get_optimal_vector
from inference import tag_all_test
import spacy
from spacy.tokenizer import Tokenizer

# —— load CV‐tuned model ——
print("[rerun] Loading CV‐tuned weights from weights2_cv.pkl")
(opt_params_cv, ft2id) = pickle.load(open("weights2_cv.pkl","rb"))
w = opt_params_cv[0]
stats = ft2id.feature_statistics

# —— self‐train with capped sampling + model saving ——
# inside rerun_selftrain.py, replace the self_train_schedule with this version

def self_train_schedule(
    stat, ft2id, w, comp_path, thresholds, lam,
    max_pseudo=5000
):
    sents     = read_test(comp_path, tagged=False)
    tags      = [t for t in ft2id.feature_statistics.tags if t!="~"]
    # start with a set of all histories seen so far
    existing = set(stat.histories)

    for itr, thresh in enumerate(thresholds, 1):
        print(f"\n[self_train] Iter {itr}, thresh = {thresh:.2f}")
        candidates = []       # list of (prob, hist)
        seen_new   = set()    # dedupe within this round

        # collect high-confidence tokens not already in existing
        for words,_ in sents:
            for i in range(2, len(words)-1):
                # compute local MEMM scores
                scores = []
                for t in tags:
                    hist = (words[i], t,
                            words[i-1], "*",
                            words[i-2], "*",
                            words[i+1])
                    feats = represent_input_with_features(
                        hist, ft2id.feature_to_idx, stat.common_words
                    )
                    scores.append(sum(w[idx] for idx in feats))
                exps = np.exp(scores - np.max(scores))
                probs = exps/exps.sum()
                best = int(np.argmax(probs))
                p    = probs[best]

                if p >= thresh:
                    hist = (words[i], tags[best],
                            words[i-1], "*",
                            words[i-2], "*",
                            words[i+1])
                    # skip anything we’ve already added (or seen previously)
                    if hist not in existing and hist not in seen_new:
                        candidates.append((p, hist))
                        seen_new.add(hist)

        if not candidates:
            print(f"[self_train] no new candidates ≥{thresh:.2f}, stopping.")
            break

        # keep only the top max_pseudo by confidence
        candidates.sort(reverse=True, key=lambda x: x[0])
        kept = [hist for _,hist in candidates[:max_pseudo]]
        print(f"[self_train] collected {len(candidates)} total new, keeping top {len(kept)}")

        # add them into stat.histories & existing
        stat.histories.extend(kept)
        existing.update(kept)
        print(f"[self_train] histories now = {len(stat.histories)}")

        # rebuild matrices and retrain
        ft2id.calc_represent_input_with_features()
        out_w = f"weights2_selfTrain_iter{itr}.pkl"
        print(f"[self_train] retraining → {out_w}")
        w = get_optimal_vector(
            stat, ft2id,
            lam=lam,
            weights_path=out_w,
            init_weights=w
        )[0]

    return w


# define thresholds from 0.96 down to 0.88 in 5 steps
thresholds = np.linspace(0.96, 0.88, 5).tolist()

w = self_train_schedule(
    stats, ft2id, w,
    comp_path="data/comp2.words",
    thresholds=thresholds,
    lam=0.001,
    max_pseudo=5000
)

# —— tag with the last iter’s model ——
print("\n[rerun] Tagging comp2.words with updated self-trained model…")
tag_all_test(
    test_path="data/comp2.words",
    pre_trained_weights=w,
    feature2id=ft2id,
    predictions_path="predictions2_updated.wtag"
)

# —— spaCy‐whitespace SOTA tagging (if needed) ——
SOTA="comp2_sota.wtag"
if not os.path.exists(SOTA):
    print("[rerun] Running spaCy-whitespace SOTA tagging…")
    nlp = spacy.load("en_core_web_sm", disable=["parser","ner"])
    token_match = re.compile(r"[^ ]+").match
    nlp.tokenizer = Tokenizer(nlp.vocab, token_match=token_match)
    with open("data/comp2.words") as fin, open(SOTA,"w") as fout:
        for L in fin:
            line = L.strip()
            if not line:
                fout.write("\n"); continue
            doc = nlp(line)
            fout.write(" ".join(f"{t.text}_{t.tag_}" for t in doc) + "\n")
    print(f"[rerun] SOTA written to {SOTA}")

# —— compare token‐level agreement & per‐tag P/R/F1 ——
print("\n[rerun] Comparing updated predictions to spaCy SOTA…")
conf, gc = Counter(), Counter()
with open("predictions2_updated.wtag") as fm, open(SOTA) as fs:
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
print(f"\n=== Updated agreement = {correct}/{total} = {correct/total:.2%} ===")
print(f"{'TAG':6s}  {'P':>6s}  {'R':>6s}  {'F1':>6s}  support")
for t in sorted(gc):
    tp = conf[(t,t)]
    fp = sum(conf[(o,t)] for o in gc if o!=t)
    fn = sum(conf[(t,o)] for o in gc if o!=t)
    P  = tp/(tp+fp) if tp+fp else 0
    R  = tp/(tp+fn) if tp+fn else 0
    F1 = 2*P*R/(P+R) if P+R else 0
    print(f"{t:6s}  {P:6.2%}  {R:6.2%}  {F1:6.2%}  {gc[t]}")
