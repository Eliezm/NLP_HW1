#!/usr/bin/env python3
import os
import pickle
import numpy as np
import re
from collections import Counter, OrderedDict
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, classification_report
import spacy
from spacy.tokenizer import Tokenizer

from preprocessing import (
    preprocess_train,
    read_test,
    represent_input_with_features
)
from optimization   import get_optimal_vector
from inference      import tag_all_test, memm_viterbi

from evaluate import predict_all, word_accuracy
from sklearn.metrics import confusion_matrix, classification_report

# ─── CONFIG ────────────────────────────────────────────────────────────────────

TRAIN_PATH      = "data/train2.wtag"
UNLABELED_PATH  = "data/comp2.words"
OUTPUT_PRED     = "predictions2_final.wtag"
CV_DIR          = "cv_tmp"
MODEL_5K        = "weights_5k.pkl"
MODEL_SELF      = "weights_5k_selftrain.pkl"
SOTA_OUT        = "comp2_sota_whitespace.wtag"

FEATURE_THRESHOLDS = [1,2]
LAM_VALUES         = [0.1,1.0,2.0]
CV_FOLDS           = 3
K_FEATURES         = 5000

# Drop‐in subset of feature‐classes for the small model:
CORE_FEATURE_CLASSES = [
    "f100","f101","f102","f103","f104","f105","f106","f107",
    "f108","f109","f110","f112","f113","f114","f115","f132",
    "f141","f142","f143","f300"
]

# self‐training params
SELF_ON         = True
SELF_MAX_ROUNDS = 2
SELF_BATCH      = 1000
SELF_THRESH     = [0.99, 0.98]

# toggles
DO_CV           = True
DO_TRAIN_PRUNE  = True
DO_SELFTRAIN    = True
DO_TAG          = True
DO_COMPARE      = True

# ──────────────────────────────────────────────────────────────────────────────

# …inside your model2_pipeline.py…

# drop‐in subset of feature‐classes for the small model:
CORE_FEATURE_CLASSES = [
    "f100","f101","f102","f103","f104","f105","f106","f107",
    "f108","f109","f110","f112","f113","f114","f115","f132",
    "f141","f142","f143","f300"
]

def prune_to_core(f2i):
    """
    Rebuild f2i.feature_to_idx so that:
      - every feature‐class key is still present (so represent_input never KeyErrors)
      - only CORE_FEATURE_CLASSES get non‐empty maps, and we re‐index them 0..N_core−1
    """
    # start with all empty
    new_map = {fc: OrderedDict() for fc in f2i.feature_to_idx}
    new_idx = 0

    # fill only the core classes
    for fc in CORE_FEATURE_CLASSES:
        for feat, old_i in f2i.feature_to_idx.get(fc, {}).items():
            new_map[fc][feat] = new_idx
            new_idx += 1

    f2i.feature_to_idx   = new_map
    f2i.n_total_features = new_idx


def cross_validate():
    os.makedirs(CV_DIR, exist_ok=True)
    best = {"lam":None,"th":None,"score":-1.0}
    sents = read_test(TRAIN_PATH, tagged=True)
    for th in FEATURE_THRESHOLDS:
        for lam in LAM_VALUES:
            scores = []
            kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=0)
            for f, (tr, dev) in enumerate(kf.split(sents),1):
                # write train‐fold to disk
                fold_file = f"{CV_DIR}/fold{f}.wtag"
                with open(fold_file,"w") as out:
                    for i in tr:
                        wds,tags = sents[i]
                        seq = [f"{w}_{t}" for w,t in zip(wds[2:-1],tags[2:-1])]
                        out.write(" ".join(seq)+"\n")

                stats, f2i = preprocess_train(fold_file, threshold=th)
                f2i.get_features_idx()
                prune_to_core(f2i)
                f2i.calc_represent_input_with_features()
                w_fold = get_optimal_vector(stats, f2i, lam, f"{CV_DIR}/w{f}.pkl")[0]

                # score
                corr = tot = 0
                for i in dev:
                    wds, tags = sents[i]
                    pred = memm_viterbi(wds, w_fold, f2i)[2:-1]
                    for p,g in zip(pred, tags[2:-1]):
                        corr += (p==g); tot += 1
                scores.append(corr/tot)

                os.remove(fold_file)
                os.remove(f"{CV_DIR}/w{f}.pkl")

            mean = np.mean(scores)
            if mean > best["score"]:
                best.update(lam=lam, th=th, score=mean)

    print(f"→ CV picks threshold={best['th']} λ={best['lam']}  (acc={best['score']:.4f})")
    return best["th"], best["lam"]

def train_and_prune(feat_thresh, lam):
    # initial feature build
    stats, f2i = preprocess_train(TRAIN_PATH, threshold=feat_thresh)
    f2i.get_features_idx()
    prune_to_core(f2i)
    f2i.calc_represent_input_with_features()

    # train full
    w_full = get_optimal_vector(stats, f2i, lam, "weights_full.pkl")[0]

    # prune to K_FEATURES by |w|
    absw, topk = np.abs(w_full), np.argsort(np.abs(w_full))[-K_FEATURES:]
    mask = np.zeros_like(absw, bool); mask[topk] = True

    new_map = {}
    new_idx = 0
    for fc, fmap in f2i.feature_to_idx.items():
        new_map[fc] = OrderedDict()
        for feat, old_i in fmap.items():
            if mask[old_i]:
                new_map[fc][feat] = new_idx
                new_idx += 1

    f2i.feature_to_idx   = new_map
    f2i.n_total_features = new_idx
    f2i.calc_represent_input_with_features()

    w5k = get_optimal_vector(stats, f2i, lam, MODEL_5K, init_weights=w_full)[0]
    print(f"[train/prune] small model has {len(w5k)} features")
    return stats, f2i, w5k

def self_train(stats, f2i, w, lam):
    if not SELF_ON:
        return w

    sents = read_test(UNLABELED_PATH, tagged=False)
    tags  = [t for t in f2i.feature_statistics.tags if t!="~"]
    existing = set(stats.histories)

    for rd, thr in enumerate(SELF_THRESH,1):
        if rd > SELF_MAX_ROUNDS:
            break
        candidates, seen = [], set()
        for words,_ in sents:
            for i in range(2, len(words)-1):
                # score each possible tag
                scores = []
                for t in tags:
                    hist = (words[i],t,words[i-1],"*",words[i-2],"*",words[i+1])
                    feats= represent_input_with_features(hist,
                           f2i.feature_to_idx,
                           f2i.feature_statistics.common_words)
                    scores.append(sum(w[j] for j in feats))
                exps  = np.exp(scores - max(scores))
                probs = exps/exps.sum()
                bi    = int(np.argmax(probs))
                conf  = probs[bi]
                if conf >= thr:
                    hist2 = (words[i], tags[bi],
                             words[i-1],"*",
                             words[i-2],"*",
                             words[i+1])
                    if hist2 not in existing and hist2 not in seen:
                        candidates.append((conf,hist2))
                        seen.add(hist2)

        if not candidates:
            break

        # pick top SELF_BATCH
        candidates.sort(key=lambda x: x[0], reverse=True)
        to_add = [h for _,h in candidates[:SELF_BATCH]]
        stats.histories.extend(to_add)
        existing.update(to_add)

        # retrain
        f2i.calc_represent_input_with_features()
        w = get_optimal_vector(stats, f2i, lam, MODEL_SELF, init_weights=w)[0]
        print(f"[self] round {rd}: +{len(to_add)} pseudo-histories")

    return w

def make_sota(raw, sota_out):
    if os.path.exists(sota_out):
        return
    nlp = spacy.load("en_core_web_sm", disable=["parser","ner"])
    tok_match = re.compile(r"[^ ]+").match
    nlp.tokenizer = Tokenizer(nlp.vocab, token_match=tok_match)
    with open(raw) as fin, open(sota_out,"w") as fout:
        for L in fin:
            line = L.strip()
            if not line:
                fout.write("\n"); continue
            doc = nlp(line)
            fout.write(" ".join(f"{tok.text}_{tok.tag_}" for tok in doc)+"\n")

def compare(pred, sota):
    conf, supp = Counter(), Counter()
    with open(pred) as fp, open(sota) as fs:
        for lm, ls in zip(fp, fs):
            p = lm.split(); s = ls.split()
            if len(p)!=len(s): continue
            for wm, ws in zip(p, s):
                pm = wm.rsplit("_",1)[1]
                sm = ws.rsplit("_",1)[1]
                conf[(sm,pm)] += 1
                supp[sm]     += 1

    labels = sorted(supp)
    tot   = sum(supp.values())
    corr  = sum(conf[(t,t)] for t in labels)
    print(f"\nAgreement: {corr}/{tot} = {corr/tot:.2%}\n")
    print("TAG     P      R     F1   support")
    for t in labels:
        tp = conf[(t,t)]
        fp = sum(conf[(o,t)] for o in labels if o!=t)
        fn = sum(conf[(t,o)] for o in labels if o!=t)
        P  = tp/(tp+fp) if tp+fp else 0
        R  = tp/(tp+fn) if tp+fn else 0
        F  = 2*P*R/(P+R) if P+R else 0
        print(f"{t:6s} {P:6.2%} {R:6.2%} {F:6.2%} {supp[t]}")

if __name__=="__main__":
    if DO_CV:
        feat_thresh, lam = cross_validate()
    else:
        feat_thresh, lam = FEATURE_THRESHOLDS[0], LAM_VALUES[0]

    if DO_TRAIN_PRUNE:
        stats, f2i, w = train_and_prune(feat_thresh, lam)

    if DO_SELFTRAIN:
        w = self_train(stats, f2i, w, lam)

    if DO_TAG:
        tag_all_test(UNLABELED_PATH, w, f2i, OUTPUT_PRED, tagged=False)
        print(f"[TAG] → {OUTPUT_PRED}")

    if DO_COMPARE:
        make_sota(UNLABELED_PATH, SOTA_OUT)
        compare(OUTPUT_PRED, SOTA_OUT)

    # ─── CONFIG ────────────────────────────────────────────────────────────────────
    TRAIN_PATH   = "data/train2.wtag"
    # choose whichever you want to evaluate:
    WEIGHTS_PKL  = "weights_5k_selftrain.pkl"
    # WEIGHTS_PKL = "weights_5k.pkl"
    # ───────────────────────────────────────────────────────────────────────────────

    # 1) load your model
    with open(WEIGHTS_PKL, "rb") as f:
        (opt_params, f2i) = pickle.load(f)
    w = opt_params[0]  # the weight vector

    # 2) run your tagger on the train set
    _, gold, pred = predict_all(TRAIN_PATH, f2i, w)

    # 3) overall word‐level accuracy
    acc = word_accuracy(gold, pred)
    print(f"Word‐level accuracy on TRAIN2 = {acc*100:.2f}%\n")

    # 4) (optional) confusion matrix & per‐tag classification report
    labels = sorted(set(gold))
    print("Confusion matrix (TRAIN2):")
    print(confusion_matrix(gold, pred, labels=labels))
    print("\nClassification report (TRAIN2):")
    print(classification_report(gold, pred, labels=labels, zero_division=0))
