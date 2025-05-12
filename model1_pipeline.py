#!/usr/bin/env python3
import os
import pickle
import numpy as np
from collections import OrderedDict

from preprocessing import preprocess_train, read_test
from optimization   import get_optimal_vector
from inference      import tag_all_test
from evaluate       import predict_all, word_accuracy

#
# ── CONFIG ────────────────────────────────────────────────────────────────────
#
TRAIN1    = "data/train1.wtag"
DEV1      = "data/test1.wtag"
COMP1     = "data/comp1.words"
OUT_PRED1 = "predictions1_noextctx.wtag"

THRESH    = 1        # keep features seen ≥1
LAM       = 2.0      # your chosen λ
PRUNE_TO  = 10000    # final budget of 10 000
#
# ──────────────────────────────────────────────────────────────────────────────
#

def prune_and_retrain(stats, f2i, w, K, lam, out_path):
    """Prune to top-K by |w|, rebuild matrices, warm-start retrain."""
    mask = np.zeros_like(w, bool)
    topk = np.argsort(np.abs(w))[-K:]
    mask[topk] = True

    # rebuild feature→idx
    new_map, new_i = {}, 0
    for fc, fmap in f2i.feature_to_idx.items():
        new_map[fc] = OrderedDict()
        for feat, old_i in fmap.items():
            if mask[old_i]:
                new_map[fc][feat] = new_i
                new_i += 1

    f2i.feature_to_idx   = new_map
    f2i.n_total_features = new_i
    f2i.calc_represent_input_with_features()

    # warm-start
    w0 = np.zeros(new_i)
    w0[:min(len(w), new_i)] = w[:new_i]

    w_pruned = get_optimal_vector(stats, f2i, lam, out_path, init_weights=w0)[0]
    return w_pruned

if __name__ == "__main__":
    # 1) build statistics, index features, build matrices
    print("[1] preprocess + build features")
    stats, f2i = preprocess_train(TRAIN1, threshold=THRESH)

    # 2) train full 100k-dim MEMM
    print(f"[2] training full model (λ={LAM})")
    w_full = get_optimal_vector(stats, f2i, LAM, "w1_full_noextctx.pkl")[0]

    # 3) prune + retrain
    print(f"[3] pruning to {PRUNE_TO} features + retrain")
    w_10k = prune_and_retrain(stats, f2i, w_full, PRUNE_TO, LAM, "w1_10k_noextctx.pkl")

    # 4) tag competition file (untagged!)
    print("[4] tagging comp file →", OUT_PRED1)
    tag_all_test(COMP1, w_10k, f2i, OUT_PRED1, tagged=False)

    # 5) evaluate on train set
    print("[5] evaluating on train set")
    _, train_gold, train_pred = predict_all(TRAIN1, f2i, w_10k)
    train_acc = word_accuracy(train_gold, train_pred)
    print(f"→ Train set accuracy = {train_acc*100:.2f}%")

    # 6) evaluate on dev set
    print("[6] evaluating on dev set")
    _, dev_gold, dev_pred = predict_all(DEV1, f2i, w_10k)
    dev_acc = word_accuracy(dev_gold, dev_pred)
    print(f"→ Dev   set accuracy = {dev_acc*100:.2f}%")
