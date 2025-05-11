# rerun_selftrain_full_pipeline.py

import os, pickle, re
import numpy as np
from collections import Counter
from preprocessing import preprocess_train, read_test, represent_input_with_features
from optimization import get_optimal_vector
from inference import tag_all_test, memm_viterbi
from spacy.tokenizer import Tokenizer
import spacy

# ── CONFIG ────────────────────────────────────────────────────────────────────
GOLD      = "data/train2.wtag"
UNLABELED = "data/comp2.words"
CV_MODEL  = "weights2_cv.pkl"
K_FEATURES = 5000   # final feature‐count cap
LAM        = 0.001  # regularization
THRESH_SCHEDULE = [0.96, 0.92, 0.88]  # your self‐train rounds
# ──────────────────────────────────────────────────────────────────────────────

def load_cv_model(path):
    with open(path,"rb") as f:
        (opt_params, ft2id_cv) = pickle.load(f)
    w_cv = opt_params[0]
    return w_cv, ft2id_cv

def write_combined_gold_silver(gold_path, silver_sentences, out_path="all_combined.wtag"):
    """Write gold train2 + list of silver sentences (List[List[(w,tag)]]) to a single .wtag file."""
    with open(out_path,"w",encoding="utf8") as out:
        # gold
        for line in open(gold_path, encoding="utf8"):
            out.write(line)
        # silver
        for sent in silver_sentences:
            out.write(" ".join(f"{w}_{t}" for w,t in sent)+"\n")
    return out_path

def collect_whole_sent_silver(unlabeled_path, w, ft2id, thresh):
    """
    Only keep sentences whose *every* token has MEMM-prob ≥ thresh.
    Returns List[List[(word,tag)]].
    """
    silver = []
    # read_test returns words padded: ['*','*', w1,w2,...,'~'], _
    for words,_ in read_test(unlabeled_path, tagged=False):
        all_tags, all_probs = [], []
        for i in range(2,len(words)-1):
            hist = (words[i], None, words[i-1], "*", words[i-2], "*", words[i+1])
            # we’ll call memm_viterbi just for this token:
            # build score vector:
            scores = []
            for t in [tg for tg in ft2id.feature_statistics.tags if tg!="~"]:
                hist_t = (words[i], t, words[i-1], "*", words[i-2], "*", words[i+1])
                feats = represent_input_with_features(hist_t,
                               ft2id.feature_to_idx,
                               ft2id.feature_statistics.common_words)
                scores.append(sum(w[idx] for idx in feats))
            exps = np.exp(scores - np.max(scores))
            probs = exps / exps.sum()
            best_i = int(np.argmax(probs))
            all_tags.append(ft2id.feature_statistics.tags - {"~"} and list(ft2id.feature_statistics.tags - {"~"})[best_i])
            all_probs.append(probs[best_i])

        if len(all_probs)>0 and min(all_probs) >= thresh:
            # keep this sentence
            silver.append(list(zip(words[2:-1],all_tags)))
    print(f"[silver] threshold {thresh:.2f} → kept {len(silver)} silver sentences")
    return silver

def prune_to_top_k(ft2id, w, K):
    """Prune ft2id to the top-K features by |w|."""
    print(f"[prune] pruning to top {K} features…")
    ft2id.prune_top_k_by_weight(w, K)
    ft2id.calc_represent_input_with_features()

def self_train_full():
    # 1) load CV‐tuned
    (opt_cv, ft2id_curr) = pickle.load(open("weights2_cv.pkl","rb"))
    w_curr = opt_cv[0]

    for itr, thresh in enumerate([0.96, 0.94, 0.92, 0.90, 0.88], 1):
        print(f"\n=== SELF-TRAIN ITER {itr}: thresh={thresh:.2f} ===")

        # collect silver histories...
        silvers = collect_whole_sent_silver("data/comp2.words", w_curr, ft2id_curr, thresh)
        write_combined_gold_silver("data/train2.wtag", silvers, "all_combined.wtag")

        # rebuild on *all* features
        stats_new, ft2id_new = preprocess_train("all_combined.wtag", threshold=1)

        # train full
        print("[train-full] training on full feature set…")
        w_full = get_optimal_vector(
            stats_new, ft2id_new,
            lam=0.001,
            weights_path=f"weights2_full_iter{itr}.pkl"
        )[0]

        # prune to top-K
        print("[prune] pruning to top 5000 features…")
        ft2id_new.prune_top_k_by_weight(w_full, 5000)
        # rebuild *both* big_matrix and small_matrix
        ft2id_new.calc_represent_input_with_features()
        # warm-start
        print("[warmstart] extracting top-K subvector…")
        old_map = {}
        for fmap in ft2id_curr.feature_to_idx.values():
            for feat, idx in fmap.items():
                old_map[feat] = idx

        w0 = np.zeros(ft2id_new.n_total_features)
        for fmap in ft2id_new.feature_to_idx.values():
            for feat, new_i in fmap.items():
                old_i = old_map.get(feat)
                if old_i is not None and old_i < len(w_full):
                    w0[new_i] = w_full[old_i]

        # retrain on pruned
        out_pkl = f"weights2_selfTrain_pruned_iter{itr}.pkl"
        print(f"[retrain-pruned] → {out_pkl}")
        w_pruned = get_optimal_vector(
            stats_new, ft2id_new,
            lam=0.001,
            weights_path=out_pkl,
            init_weights=w0
        )[0]

        # rotate
        w_curr, ft2id_curr = w_pruned, ft2id_new

    return w_curr, ft2id_curr



if __name__=="__main__":
    # run self-train
    w_final, ft2id_final = self_train_full()

    # final tagging
    print("\n=== TAGGING final comp2.words ===")
    tag_all_test(UNLABELED, w_final, ft2id_final, "predictions2_final.wtag")

    # spaCy‐whitespace SOTA tagging
    SOTA = "comp2_sota_whitespace.wtag"
    if not os.path.exists(SOTA):
        print("\n=== SPAcy SOTA tokenization + tagging ===")
        nlp = spacy.load("en_core_web_sm", disable=["parser","ner"])
        token_match = re.compile(r"[^ ]+").match
        nlp.tokenizer = Tokenizer(nlp.vocab, token_match=token_match)
        with open(UNLABELED, encoding="utf8") as fin, open(SOTA,"w",encoding="utf8") as fout:
            for line in fin:
                line = line.strip()
                if not line:
                    fout.write("\n"); continue
                doc = nlp(line)
                fout.write(" ".join(f"{tok.text}_{tok.tag_}" for tok in doc) + "\n")
        print(f"[SOTA] written to {SOTA}")

    # compare
    print("\n=== FINAL compare to SOTA ===")
    conf, gold_counts = Counter(), Counter()
    with open("predictions2_final.wtag",encoding="utf8") as fm, open(SOTA,encoding="utf8") as fs:
        for lm, ls in zip(fm,fs):
            tm = [wt.rsplit("_",1)[1] for wt in lm.split()]
            ts = [wt.rsplit("_",1)[1] for wt in ls.split()]
            if len(tm)!=len(ts):
                continue
            for p,g in zip(tm,ts):
                conf[(g,p)] += 1
                gold_counts[g]  += 1
    total = sum(gold_counts.values())
    correct = sum(conf[(t,t)] for t in gold_counts)
    print(f" overall agreement = {correct}/{total} = {correct/total:.2%}\n")
    print(f"{'TAG':6s}  {'P':>6s}  {'R':>6s}  {'F1':>6s}  support")
    for t in sorted(gold_counts):
        tp = conf[(t,t)]
        fp = sum(conf[(o,t)] for o in gold_counts if o!=t)
        fn = sum(conf[(t,o)] for o in gold_counts if o!=t)
        P  = tp/(tp+fp) if tp+fp else 0.0
        R  = tp/(tp+fn) if tp+fn else 0.0
        F1 = 2*P*R/(P+R)  if P+R   else 0.0
        print(f"{t:6s}  {P:6.2%}  {R:6.2%}  {F1:6.2%}  {gold_counts[t]}")
