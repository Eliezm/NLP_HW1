import os
import pickle
from collections import OrderedDict
from copy import deepcopy

from preprocessing import preprocess_train
from optimization import get_optimal_vector
from inference import tag_all_test
from evaluate import predict_all, word_accuracy

import pickle

def inspect_weights(path="weights.pkl"):
    with open(path, "rb") as f:
        (optimal_params, feature2id) = pickle.load(f)
    # optimal_params is the tuple returned by fmin_l_bfgs_b:
    #   optimal_params[0] is the weight‐vector
    w = optimal_params[0]
    print(f"Number of learned parameters (len(w)): {len(w)}")
    print(f"feature2id.n_total_features:        {feature2id.n_total_features}")


# def main():
#     # # # inspect_weights("weights.pkl")
#     #
#     # threshold        = 1
#     # lam              = 3
#     # train_path       = "data/train1.wtag"
#     # test_path        = "data/comp1.words"
#     # weights_path     = "weights.pkl"
#     # predictions_path = "predictions.wtag"
#     # pruned_weights_path = "weights_pruned.pkl"
#     # #
#     # # 1) Load or train full model
#     # if os.path.exists(pruned_weights_path):
#     #     print(f"[main] Loading pretrained model from {pruned_weights_path}")
#     #     with open(weights_path, "rb") as f:
#     #         (optimal_params, feature2id) = pickle.load(f)
#     #     # pull statistics back out of feature2id
#     #     statistics = feature2id.feature_statistics
#     #     w = optimal_params[0]
#     # #
#     # else:
#     #     print("[main] No existing weights found—training from scratch.")
#     #     statistics, feature2id = preprocess_train(train_path, threshold)
#     #
#     #     # initial training on full feature set
#     #     optimal_params = get_optimal_vector(
#     #         statistics=statistics,
#     #         feature2id=feature2id,
#     #         lam=lam,
#     #         weights_path=weights_path
#     #     )
#     #     w = optimal_params[0]
#     # #
#     # # # # ─── prune to top-K by |w|, then fine-tune ───
#     # # # K = 10000
#     # # # print(f"[main] Pruning to top {K} features by |w|…")
#     # # # feature2id.prune_top_k_by_weight(w, K)
#     # # # print(f"[main] {feature2id.n_total_features} features remain after pruning")
#     # # #
#     # # # # rebuild both small_matrix and big_matrix to match the pruned feature set
#     # # # feature2id.calc_represent_input_with_features()
#     # # #
#     # # # print("[main] Fine-tuning on pruned feature set…")
#     # # # # re-train (L-BFGS) starting from the pruned weights
#     # # # optimal_params = get_optimal_vector(
#     # # #     statistics=statistics,
#     # # #     feature2id=feature2id,
#     # # #     lam=lam,
#     # # #     weights_path=pruned_weights_path,
#     # # #     init_weights=w
#     # # # )
#     # # # w = optimal_params[0]
#     # #
#     # # # ─── Prune the 10k→5 000 & fine-tune ───
#     # # # K2 = 5_000
#     # # # print(f"[main] Pruning to top {K2} features by |w|…")
#     # # # feature2id.prune_top_k_by_weight(w, K2)
#     # # # print(f"[main] {feature2id.n_total_features} features remain after 5k-prune")
#     # # # feature2id.calc_represent_input_with_features()
#     # # # print("[main] Fine-tuning on 5k feature set…")
#     # # # optimal_params = get_optimal_vector(
#     # # #     statistics=statistics,
#     # # #     feature2id=feature2id,
#     # # #     lam=lam,
#     # # #     weights_path=pruned_weights_path,
#     # # #     init_weights=w
#     # # # )
#     # # # w = optimal_params[0]
#     # #
#     # # # 3) Tag the test set
#     # # print("[main] Tagging test set…")
#     # # tag_all_test(
#     # #     test_path=test_path,
#     # #     pre_trained_weights=w,
#     # #     feature2id=feature2id,
#     # #     predictions_path=predictions_path
#     # # )
#     # # print(f"[main] Predictions written to {predictions_path}")
#     # #
#     # # # 4) Evaluate (optional)
#     # # print("[main] Evaluating on held-out test1.wtag…")
#     # # words, gold, pred = predict_all("data/test1.wtag", feature2id, w)
#     # # acc = word_accuracy(gold, pred)
#     # # print(f"Word-level accuracy: {acc*100:.2f}%")
#     # from tag_spacy_tokens import tag_with_spacy_tokens
#     #
#     # # read your comp1.words
#     # with open("data/comp1.words", encoding="utf8") as f:
#     #     lines = f.readlines()
#     #
#     # # get MEMM predictions on spaCy tokens
#     # preds = tag_with_spacy_tokens(lines, w, feature2id)
#     #
#     # # write them out
#     # with open("comp1.memmtag.spacytok", "w", encoding="utf8") as out:
#     #     for sent in preds:
#     #         out.write(" ".join(f"{tok}_{tag}" for tok, tag in sent) + "\n")
#     #
#     # import spacy
#     #
#     # nlp = spacy.load("en_core_web_sm")  # same model!
#     #
#     # with open("data/comp1.words", encoding="utf8") as fin, \
#     #         open("comp1.silver.spacytok", "w", encoding="utf8") as fout:
#     #
#     #     for line in fin:
#     #         line = line.rstrip("\n")
#     #         if not line:
#     #             fout.write("\n")
#     #             continue
#     #
#     #         doc = nlp(line)
#     #         tagged = [f"{tok.text}_{tok.tag_}" for tok in doc]
#     #         fout.write(" ".join(tagged) + "\n")
#
#
#
#     threshold        = 1
#     lam              = 3
#     train_path       = "data/train2.wtag"    # <-- מודול 2 קטן
#     test_path        = "data/comp2.words"     # <-- קובץ ללא תגים
#     weights_path     = "weights2.pkl"
#     predictions_path = "predictions2.wtag"
#
#
#     # 1) Load or train full model
#     if os.path.exists(weights_path):
#         print(f"[main] Loading pretrained model from {weights_path}")
#         with open(weights_path, "rb") as f:
#             (optimal_params, feature2id) = pickle.load(f)
#         # pull statistics back out of feature2id
#         statistics = feature2id.feature_statistics
#         w = optimal_params[0]
#     #
#     else:
#         print("[main] No existing weights found—training from scratch.")
#         statistics, feature2id = preprocess_train(train_path, threshold)
#
#         # initial training on full feature set
#         optimal_params = get_optimal_vector(
#             statistics=statistics,
#             feature2id=feature2id,
#             lam=lam,
#             weights_path=weights_path
#         )
#         w = optimal_params[0]
#
#     # # 1) אימון
#     # statistics, feature2id = preprocess_train(train_path, threshold)
#     # optimal_params = get_optimal_vector(
#     #     statistics=statistics,
#     #     feature2id=feature2id,
#     #     lam=lam,
#     #     weights_path=weights_path
#     # )
#     # w = optimal_params[0]
#
#     # 2) תיוג
#     print("[small] Tagging comp2.words…")
#     tag_all_test(
#         test_path=test_path,
#         pre_trained_weights=w,
#         feature2id=feature2id,
#         predictions_path=predictions_path
#     )
#     print(f"[small] Predictions written to {predictions_path}")
#
#     # 4) Evaluate (optional)
#     print("[main] Evaluating on held-out test1.wtag…")
#     words, gold, pred = predict_all("data/test2.wtag", feature2id, w)
#     acc = word_accuracy(gold, pred)
#     print(f"Word-level accuracy: {acc*100:.2f}%")
#
# def score(memm_path, silver_path):
#     total = correct = 0
#     with open(memm_path) as f1, open(silver_path) as f2:
#         for l1, l2 in zip(f1, f2):
#             t1 = [wt.rsplit("_",1)[1] for wt in l1.split()]
#             t2 = [wt.rsplit("_",1)[1] for wt in l2.split()]
#             assert len(t1)==len(t2), "Token mismatch!"
#             for a,b in zip(t1,t2):
#                 total   += 1
#                 correct += (a==b)
#     print(f"Accuracy vs spaCy = {correct/total*100:.2f}%")


import numpy as np
from sklearn.model_selection import KFold
from scipy.optimize import fmin_l_bfgs_b



#################################


# import os
# import pickle
# import numpy as np
# from copy import deepcopy
# from collections import OrderedDict
# from sklearn.model_selection import KFold
#
# from preprocessing import preprocess_train, read_test, represent_input_with_features
# from optimization import get_optimal_vector
# from inference import tag_all_test
# from evaluate import predict_all, word_accuracy
# from scipy.sparse import vstack, csr_matrix
#
# # --- Helper functions for self-training and CV ---
# def prune_old_model(old_weights_path, K=5000):
#     with open(old_weights_path, "rb") as f:
#         (old_params, old_ft2id) = pickle.load(f)
#     old_w = old_params[0]
#     # build old feat->idx
#     old_map = {feat: idx for fc in old_ft2id.feature_to_idx
#                           for feat, idx in old_ft2id.feature_to_idx[fc].items()}
#     # pick top-K
#     abs_w = np.abs(old_w)
#     topk = np.argsort(abs_w)[-K:]
#     mask = np.zeros_like(abs_w, bool); mask[topk] = True
#     # build new feature2id with only those K
#     new_ft2id = deepcopy(old_ft2id)
#     new_map = OrderedDict()
#     new_i = 0
#     for fc, fmap in old_ft2id.feature_to_idx.items():
#         new_ft2id.feature_to_idx[fc] = OrderedDict()
#         for feat, old_i in fmap.items():
#             if mask[old_i]:
#                 new_ft2id.feature_to_idx[fc][feat] = new_i
#                 new_i += 1
#     new_ft2id.n_total_features = new_i
#     return old_w, old_map, new_ft2id
#
#
# from inference import memm_viterbi
# from preprocessing import read_test
# from preprocessing import preprocess_train
# from optimization import get_optimal_vector
# from scipy.sparse import csr_matrix
# from sklearn.model_selection import KFold
# import numpy as np
# import os
#
# def cross_validate(train_path, lam_values, th_values, n_splits=5, threshold=1):
#     """
#     5-fold CV on train2.wtag to pick best (lam, self-train threshold).
#     """
#     # 1) load all tagged sentences: each as (words_list, tags_list)
#     all_sents = read_test(train_path, tagged=True)
#
#     kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)
#     best_score = -1.0
#     best_lam = None
#     best_th  = None
#
#     for lam in lam_values:
#         for th in th_values:
#             fold_scores = []
#             for train_idx, dev_idx in kf.split(all_sents):
#                 # --- build a small temp file for this fold's training set ---
#                 tmp_train = "cv_fold_train.wtag"
#                 with open(tmp_train, "w", encoding="utf8") as out:
#                     for i in train_idx:
#                         words, tags = all_sents[i]
#                         # drop the padding [*,*] and [~]
#                         seq = [f"{w}_{t}" for w,t in zip(words[2:-1], tags[2:-1])]
#                         out.write(" ".join(seq) + "\n")
#
#                 # 2) preprocess & train on this fold
#                 stat_fold, ft2id_fold = preprocess_train(tmp_train, threshold)
#                 opt = get_optimal_vector(stat_fold, ft2id_fold, lam, "/tmp/cv_w.pkl")
#                 w_fold = opt[0]
#
#                 # 3) evaluate on dev fold
#                 correct = total = 0
#                 for j in dev_idx:
#                     words, tags = all_sents[j]
#                     # run Viterbi
#                     pred_hist = memm_viterbi(words, w_fold, ft2id_fold)
#                     pred_tags = pred_hist[2:-1]
#                     gold_tags = tags[2:-1]
#                     for p,g in zip(pred_tags, gold_tags):
#                         total   += 1
#                         correct += (p == g)
#                 # avoid divide-by-zero
#                 acc = correct/total if total>0 else 0.0
#                 fold_scores.append(acc)
#
#                 # cleanup
#                 os.remove(tmp_train)
#
#             mean_acc = float(np.mean(fold_scores))
#             if mean_acc > best_score:
#                 best_score, best_lam, best_th = mean_acc, lam, th
#
#     return best_lam, best_th
#
#
# from preprocessing import read_test, represent_input_with_features
# from optimization import get_optimal_vector
# import math
#
# def self_train(stat, ft2id, w, comp_words_path, conf_thresh=0.9, max_iter=3, lam=3, weights_path="weights2_st.pkl"):
#     """
#     Semi-supervised self-training loop.
#     - stat: FeatureStatistics (with histories from train2)
#     - ft2id: Feature2id (already built & matrices computed for train2)
#     - w:      current weight vector for ft2id
#     - comp_words_path: unlabeled text file (one sentence per line)
#     - conf_thresh: only accept sentences whose every token has max-tag-prob >= this
#     - max_iter: how many self-train rounds to do
#     - lam, weights_path: passed through to get_optimal_vector
#     Returns new weight vector.
#     """
#     all_sents = read_test(comp_words_path, tagged=False)
#     tags = [t for t in ft2id.feature_statistics.tags if t != "~"]
#     for iteration in range(max_iter):
#         new_histories = []
#         for words, _ in all_sents:
#             # pad is already part of read_test output: words = ['*','*', w1, w2,...,'~']
#             token_tags = []
#             token_confs = []
#             for i in range(2, len(words)-1):
#                 c_word = words[i]
#                 p_word, pp_word = words[i-1], words[i-2]
#                 p_tag, pp_tag = None, None  # we’ll use greedy on past predicted tags
#                 # but for simplicity here we assume previous tags are "*","*"
#                 # instead we do a local classifier (MEMM) ignoring dependencies
#                 scores = []
#                 for t in tags:
#                     # build history tuple
#                     hist = (
#                         c_word, t,
#                         words[i-1], "*",
#                         words[i-2], "*",
#                         words[i+1]
#                     )
#                     feats = represent_input_with_features(
#                         hist, ft2id.feature_to_idx, ft2id.feature_statistics.common_words
#                     )
#                     scores.append(sum(w[idx] for idx in feats))
#                 # softmax
#                 max_s = max(scores)
#                 exps = [math.exp(s-max_s) for s in scores]
#                 S = sum(exps)
#                 probs = [e/S for e in exps]
#                 best_i = max(range(len(tags)), key=lambda j: probs[j])
#                 token_tags.append(tags[best_i])
#                 token_confs.append(probs[best_i])
#
#             # accept only if **all** confidences ≥ threshold
#             if min(token_confs, default=0) >= conf_thresh:
#                 # build full histories for this sentence, with pseudo tags
#                 for i, pred_t in enumerate(token_tags, start=2):
#                     hist = (
#                         words[i], pred_t,
#                         words[i-1], token_tags[i-3] if i>2 else "*",
#                         words[i-2], "*",
#                         words[i+1]
#                     )
#                     new_histories.append(hist)
#
#         if not new_histories:
#             print(f"[self_train] no new sentences at iter {iteration} → stopping")
#             break
#
#         # append pseudo-histories
#         stat.histories.extend(new_histories)
#         # rebuild matrices
#         ft2id.calc_represent_input_with_features()
#
#         # re-train from current w
#         opt = get_optimal_vector(
#             stat, ft2id,
#             lam=lam,
#             weights_path=weights_path,
#             init_weights=w
#         )
#         w = opt[0]
#         print(f"[self_train] completed iteration {iteration+1}, total histories now {len(stat.histories)}")
#
#     return w
#
#
#
# def main():
#     # 1) Transfer / Fine-tuning
#     old_w, old_map, ft2id = prune_old_model("weights_pruned.pkl", K=5000)
#     # build train2 stats
#     stat2, _ = preprocess_train("data/train2.wtag", threshold=1)
#
#     ft2id.feature_statistics = stat2
#
#     # rebuild both small_matrix & big_matrix to match the pruned 5 000 features
#     ft2id.calc_represent_input_with_features()
#     # warm-start weights
#     w0 = np.zeros(ft2id.n_total_features)
#     for fc, fmap in ft2id.feature_to_idx.items():
#         for feat, new_i in fmap.items():
#             old_i = old_map.get(feat)
#             if old_i is not None:
#                 w0[new_i] = old_w[old_i]
#     opt = get_optimal_vector(stat2, ft2id, lam=3, weights_path="weights2.pkl", init_weights=w0)
#     w = opt[0]
#
#     # inspect_weights("weights2.pkl")
#
#     # # 2) Cross-Validation & Early Stopping (optional tuning)
#     lam_values = [0.001, 0.01, 0.1, 1]
#     th_values = [1, 2, 3]
#     best_lam, best_th = cross_validate(stat2.histories, ft2id.small_matrix, ft2id, lam_values, th_values)
#     print(f"CV tuned lam={best_lam}, thresh={best_th}")
#
#     # 3) (Re-)Train with best_λ
#     opt = get_optimal_vector(
#         stat2,
#         ft2id,
#         lam=best_lam,
#         weights_path="weights2_cv.pkl",
#         init_weights=w  # warm-start from your transfer-trained w
#     )
#     w = opt[0]
#     #
#     # # 3) Semi-supervised Self-Training
#     w = self_train(stat2, ft2id, w, "data/comp2.words",
#                    conf_thresh=0.9, max_iter=3,
#                    lam=best_lam, weights_path="weights2_selfTrain.pkl")


###########################

import os
import pickle
import numpy as np
from copy import deepcopy
from collections import OrderedDict
from sklearn.model_selection import KFold

from preprocessing import preprocess_train, read_test, represent_input_with_features
from optimization import get_optimal_vector
from inference import tag_all_test, memm_viterbi
from evaluate import predict_all, word_accuracy
from scipy.sparse import csr_matrix
import math

# --- Helper functions for self-training and CV ---
def prune_old_model(old_weights_path, K=5000):
    print(f"[prune_old_model] Loading old weights from {old_weights_path}")
    with open(old_weights_path, "rb") as f:
        (old_params, old_ft2id) = pickle.load(f)
    old_w = old_params[0]
    print(f"[prune_old_model] Original model has {old_w.shape[0]} features")
    # build old feat->idx
    old_map = {feat: idx for fc in old_ft2id.feature_to_idx
                          for feat, idx in old_ft2id.feature_to_idx[fc].items()}
    # pick top-K
    abs_w = np.abs(old_w)
    topk = np.argsort(abs_w)[-K:]
    mask = np.zeros_like(abs_w, bool); mask[topk] = True
    print(f"[prune_old_model] Pruning to top {K} features")
    # build new feature2id with only those K
    new_ft2id = deepcopy(old_ft2id)
    new_i = 0
    for fc, fmap in old_ft2id.feature_to_idx.items():
        new_ft2id.feature_to_idx[fc] = OrderedDict()
        for feat, old_i in fmap.items():
            if mask[old_i]:
                new_ft2id.feature_to_idx[fc][feat] = new_i
                new_i += 1
    new_ft2id.n_total_features = new_i
    print(f"[prune_old_model] New model will have {new_i} features")
    return old_w, old_map, new_ft2id


def cross_validate(train_path, lam_values, th_values, n_splits=5, threshold=1):
    print("[cross_validate] Starting cross-validation")
    all_sents = read_test(train_path, tagged=True)
    print(f"[cross_validate] Loaded {len(all_sents)} tagged sentences from {train_path}")
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)
    best_score = -1.0
    best_lam = None
    best_th  = None
    for lam in lam_values:
        for th in [1]:
            print(f"[cross_validate] Testing lam={lam}, thresh={th}")
            fold_scores = []
            for fold, (train_idx, dev_idx) in enumerate(kf.split(all_sents), 1):
                print(f"  [CV fold {fold}/{n_splits}] Building train fold")
                tmp_train = f"cv_fold_{fold}.wtag"
                with open(tmp_train, "w", encoding="utf8") as out:
                    for i in train_idx:
                        words, tags = all_sents[i]
                        seq = [f"{w}_{t}" for w,t in zip(words[2:-1], tags[2:-1])]
                        out.write(" ".join(seq) + "\n")
                stat_fold, ft2id_fold = preprocess_train(tmp_train, threshold)
                ft2id_fold.calc_represent_input_with_features()
                print(f"  [CV fold {fold}] Training on fold with lam={lam}")
                opt = get_optimal_vector(stat_fold, ft2id_fold, lam, "cv_w_fold.pkl")
                w_fold = opt[0]
                correct = total = 0
                for j in dev_idx:
                    words, tags = all_sents[j]
                    pred_hist = memm_viterbi(words, w_fold, ft2id_fold)
                    pred_tags = pred_hist[2:-1]
                    gold_tags = tags[2:-1]
                    for p,g in zip(pred_tags, gold_tags):
                        total   += 1
                        correct += (p == g)
                acc = correct/total if total>0 else 0.0
                print(f"  [CV fold {fold}] Accuracy = {acc:.4f}")
                fold_scores.append(acc)
                os.remove(tmp_train)
            mean_acc = float(np.mean(fold_scores))
            print(f"[cross_validate] Mean CV accuracy for lam={lam}, th={th}: {mean_acc:.4f}")
            if mean_acc > best_score:
                best_score, best_lam, best_th = mean_acc, lam, th
    print(f"[cross_validate] Best lam={best_lam}, thresh={best_th}, accuracy={best_score:.4f}")
    return best_lam, best_th


def self_train(stat, ft2id, w, comp_words_path, conf_thresh=0.8, max_iter=3, lam=3, weights_path="weights2_st.pkl"):
    print(f"[self_train] Starting self-training with conf_thresh={conf_thresh}, max_iter={max_iter}")
    all_sents = read_test(comp_words_path, tagged=False)
    tags = [t for t in ft2id.feature_statistics.tags if t != "~"]
    for iteration in range(1, max_iter+1):
        print(f"[self_train] Iteration {iteration}")
        new_histories = []
        for words, _ in all_sents:
            token_tags = []
            token_confs = []
            for i in range(2, len(words)-1):
                scores = []
                for t in tags:
                    hist = (words[i], t, words[i-1], "*", words[i-2], "*", words[i+1])
                    feats = represent_input_with_features(hist,
                                ft2id.feature_to_idx,
                                ft2id.feature_statistics.common_words)
                    scores.append(sum(w[idx] for idx in feats))
                max_s = max(scores)
                exps = [math.exp(s-max_s) for s in scores]
                S = sum(exps)
                probs = [e/S for e in exps]
                best_i = int(np.argmax(probs))
                token_tags.append(tags[best_i])
                token_confs.append(probs[best_i])
            if len(token_confs)>0 and min(token_confs) >= conf_thresh:
                new_histories.extend(
                    (words[i], token_tags[i-2], words[i-1],
                     token_tags[i-3] if i>2 else "*",
                     words[i-2], "*", words[i+1])
                    for i in range(2, len(words)-1)
                )
        if not new_histories:
            print(f"[self_train] No new pseudo-labels in iteration {iteration}, stopping.")
            break
        stat.histories.extend(new_histories)
        print(f"[self_train] Added {len(new_histories)} pseudo-histories, total histories: {len(stat.histories)}")
        ft2id.calc_represent_input_with_features()
        print(f"[self_train] Retraining model with lam={lam}")
        opt = get_optimal_vector(stat, ft2id, lam, weights_path, init_weights=w)
        w = opt[0]
        print(f"[self_train] Completed iteration {iteration}")
    return w


def main():
    print("=== STEP 1: Transfer/Fine-tuning ===")
    old_w, old_map, ft2id = prune_old_model("weights_pruned.pkl", K=5000)
    stat2, _ = preprocess_train("data/train2.wtag", threshold=1)
    ft2id.feature_statistics = stat2
    ft2id.calc_represent_input_with_features()
    print("[main] Warm-starting weights from pruned model")
    w0 = np.zeros(ft2id.n_total_features)
    for fc,fmap in ft2id.feature_to_idx.items():
        for feat,new_i in fmap.items():
            old_i = old_map.get(feat)
            if old_i is not None:
                w0[new_i] = old_w[old_i]
    opt = get_optimal_vector(stat2, ft2id, lam=3, weights_path="weights2.pkl", init_weights=w0)
    w = opt[0]
    print("=== STEP 2: Cross-Validation ===")
    lam_values = [0.001, 0.01, 0.1, 1]
    th_values  = [0.9,   0.85, 0.8]
    best_lam, best_th = cross_validate("data/train2.wtag", lam_values, 1)
    print(f"[main] CV results -> lam={best_lam}, thresh={best_th}")
    print("=== STEP 3: Re-train with best lambda ===")
    opt = get_optimal_vector(stat2, ft2id, lam=best_lam, weights_path="weights2_cv.pkl", init_weights=w)
    w = opt[0]
    print("=== STEP 4: Semi-supervised Self-Training ===")
    w = self_train(stat2, ft2id, w, "data/comp2.words", conf_thresh=best_th,
                   max_iter=3, lam=best_lam, weights_path="weights2_selfTrain.pkl")
    print("=== STEP 5: Tagging final predictions ===")
    tag_all_test(test_path="data/comp2.words",
                 pre_trained_weights=w,
                 feature2id=ft2id,
                 predictions_path="predictions2_final.wtag")
    if os.path.exists("data/test2.wtag"):
        print("[main] Evaluating on data/test2.wtag")
        words, gold, pred = predict_all("data/test2.wtag", ft2id, w)
        acc = word_accuracy(gold, pred)
        print(f"[main] Final word accuracy: {acc*100:.2f}%")

if __name__ == "__main__":
    main()


# if __name__=="__main__":
#     main()
#     # score("comp1.memmtag.spacytok", "comp1.silver.spacytok")


# if __name__ == '__main__':
#     main()


"""
How to lower the parameter count?:
1. Increase threshold
2. Remove features
3. Maybe drop features with some probability
4. Maybe drop features with respect to some confusion matrix behaviour

How to lower accuracy on test?:
1. Lower lam
2. Lower threshold


"""