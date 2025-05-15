import os
import pickle
import numpy as np
from collections import OrderedDict

from sklearn.model_selection import KFold
from sklearn.metrics import (confusion_matrix, classification_report)

from preprocessing import (preprocess_train, read_test, represent_input_with_features)
from optimization   import get_optimal_vector
from inference import memm_viterbi

def predict_all(test_path, feature2id, weights):
    """
    This function received a tagged test set and computes the predictions of a trained model on it.
    Returns:
        1. A list of all the words in the test set
        2. A list of all the true tags in the test set
        3. A list of all the predicted tags in the test set by the model
    """
    # 1. Read all sentences, extract in format of [(sentence_words, sentence_word_tags), ...]
    sentences = read_test(test_path, tagged=True)
    # 2. Prepare lists for storing words, true tags, and predicted tags.
    all_words, all_gold, all_pred = [], [], []

    # 3. Iterate over all sentences in the set, use Viterbi to tag their words,
    #    and extend our all_words, all_gold, all_pred
    for words, gold_tags in sentences:
        pred_tags = memm_viterbi(words, weights, feature2id)
        # Each sentence starts with two "*" signs and ends with a "~", so we cut it.
        pred = pred_tags[2:-1]
        real_words = words[2:-1]
        all_words .extend(real_words)
        all_gold  .extend(gold_tags[2:-1])
        all_pred  .extend(pred)

    return all_words, all_gold, all_pred

def word_accuracy(gold, pred):
    """
    A simple function that takes two lists, one list of true tags, and one list of predicted tags.
    And computes the accuracy of the predictions.
    """
    correct = sum(g==p for g,p in zip(gold,pred))
    return correct / len(gold)


###



"""
MODEL 1 CONSTANTS
"""
TRAIN1    = "data/train1.wtag"
TEST1      = "data/test1.wtag"
COMP1     = "data/comp1.words"
OUT_PRED1 = "comp_m1_207476763_322409376.wtag"

THRESH    = 1
LAM       = 2.0
PRUNE_TO  = 10000

"""
MODEL 2 CONSTANTS
"""
TRAIN_PATH      = "data/train2.wtag"
UNLABELED_PATH  = "data/comp2.words"
OUTPUT_PRED     = "comp_m2_207476763_322409376.wtag"
CV_DIR          = "cv_tmp"
MODEL_5K        = "weights_5k.pkl"
MODEL_SELF      = "weights_5k_selftrain.pkl"

FEATURE_THRESHOLDS = [1,2]
LAM_VALUES         = [0.1,1.0,2.0]
CV_FOLDS           = 3
K_FEATURES         = 5000

CORE_FEATURE_CLASSES = [
    "f100","f101","f102","f103","f104","f105","f106","f107",
    "f108","f109","f110","f112","f113","f114","f115","f132",
    "f141","f142","f143","f300"
]

SELF_ON         = True
SELF_MAX_ROUNDS = 2
SELF_BATCH      = 1000
SELF_THRESH     = [0.99, 0.98]


### MODEL 1 FUNCTIONS
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


### MODEL 2 FUNCTIONS

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


if __name__ == "__main__":

    ### MODEL 1 PIPELINE

    # build statistics, index features, build matrices
    print("[model 1] preprocess + build features")
    stats, f2i = preprocess_train(TRAIN1, threshold=THRESH)

    # train full 100k-feature MEMM
    print(f"[model 1] training full model (λ={LAM})")
    w_full = get_optimal_vector(stats, f2i, LAM, "w1_full.pkl")[0]

    # prune + retrain
    print(f"[model 1] pruning to {PRUNE_TO} features + retrain")
    w_10k = prune_and_retrain(stats, f2i, w_full, PRUNE_TO, LAM, "w1_10k.pkl")

    # evaluate on train set
    print("[model 1] evaluating on train set")
    _, train_gold, train_pred = predict_all(TRAIN1, f2i, w_10k)
    train_acc = word_accuracy(train_gold, train_pred)
    print(f"[model 1] Train set accuracy = {train_acc*100:.2f}%")

    # evaluate on test set
    print("[model 1] evaluating on test set")
    _, test_gold, test_pred = predict_all(TEST1, f2i, w_10k)
    test_acc = word_accuracy(test_gold, test_pred)
    print(f"[model 1] Test set accuracy = {test_acc*100:.2f}%")


    ### MODEL 2 PIPELINE

    ### Cross - Validation ###
    print("[model 2] preforming cross validation on threshold and lambda")
    feat_thresh, lam = cross_validate()
    feat_thresh, lam = FEATURE_THRESHOLDS[0], LAM_VALUES[0]

    ### Pruning to 5k features ###
    print("[model 2] training model on best parameters + pruning to 5k features")
    stats, f2i, w = train_and_prune(feat_thresh, lam)

    ### Self - Training Based On High Probability Predictions ###
    print("[model 2] performing self-training based on the models most probable predictions")
    w = self_train(stats, f2i, w, lam)

    TRAIN_PATH   = "data/train2.wtag"
    WEIGHTS_PKL  = "weights_5k_selftrain.pkl"


    print("[model 2] opening the 5k model")
    with open(WEIGHTS_PKL, "rb") as f:
        (opt_params, f2i) = pickle.load(f)
    w = opt_params[0]

    print("[model 2] tagging the train set via trained 5k model")
    # run your tagger on the train set
    _, gold, pred = predict_all(TRAIN_PATH, f2i, w)

    # overall word‐level accuracy
    acc = word_accuracy(gold, pred)
    print(f"[model 2] Word‐level accuracy on train2.wtag = {acc*100:.2f}%\n")

    labels = sorted(set(gold))
    print("[model 2] Confusion matrix (train2.wtag):")
    print(confusion_matrix(gold, pred, labels=labels))
    print("\n[model 2] Classification report (train2.wtag):")
    print(classification_report(gold, pred, labels=labels, zero_division=0))