import os
import pickle
import numpy as np
from copy import deepcopy
from collections import OrderedDict, Counter

from preprocessing import preprocess_train, read_test, represent_input_with_features
from optimization   import get_optimal_vector
from inference      import tag_all_test, memm_viterbi
from evaluate       import predict_all, word_accuracy
from sklearn.model_selection import KFold

import spacy
from spacy.tokenizer import Tokenizer
import re

# ── Stage 1: Train & Prune base model ────────────────────────────────────────
def stage1_prune(train1="data/train1.wtag", K=5000):
    print("=== STAGE 1: Train full on train1.wtag then prune to top K features ===")
    stats1, ft2id1 = preprocess_train(train1, threshold=1)
    opt1 = get_optimal_vector(stats1, ft2id1, lam=3.0, weights_path="weights1.pkl")
    w1 = opt1[0]
    # prune to top-K
    abs_w = np.abs(w1)
    topk = np.argsort(abs_w)[-K:]
    mask = np.zeros_like(abs_w, bool); mask[topk]=True

    pruned = deepcopy(ft2id1)
    new_i = 0
    for fc, fmap in ft2id1.feature_to_idx.items():
        pruned.feature_to_idx[fc] = OrderedDict()
        for feat, idx in fmap.items():
            if mask[idx]:
                pruned.feature_to_idx[fc][feat] = new_i
                new_i += 1
    pruned.n_total_features = new_i

    # save
    with open("weights_pruned.pkl", "wb") as f:
        pickle.dump(((w1, ), pruned), f)
    print(f"[stage1] Pruned model saved with {new_i} features")

# ── Stage 2: Transfer & Fine‐tune on train2 ──────────────────────────────────
def stage2_transfer(train2="data/train2.wtag"):
    print("=== STAGE 2: Transfer/fine-tune on train2.wtag ===")
    # load pruned
    (old_w,), ft2id = pickle.load(open("weights_pruned.pkl","rb"))
    old_map = {feat:idx for fc in ft2id.feature_to_idx for feat,idx in ft2id.feature_to_idx[fc].items()}

    # build stats2
    stats2, _ = preprocess_train(train2, threshold=1)
    ft2id.feature_statistics = stats2
    ft2id.calc_represent_input_with_features()

    # warm-start
    w0 = np.zeros(ft2id.n_total_features)
    for fc,fmap in ft2id.feature_to_idx.items():
        for feat,new_i in fmap.items():
            old_i = old_map.get(feat)
            if old_i is not None and old_i < len(old_w):
                w0[new_i] = old_w[old_i]

    opt2 = get_optimal_vector(stats2, ft2id, lam=3.0, weights_path="weights2_init.pkl", init_weights=w0)
    print("[stage2] Saved transfer-tuned model to weights2_init.pkl")

# ── Stage 3: Cross-validate λ and threshold ──────────────────────────────────
def stage3_cv(train2="data/train2.wtag"):
    print("=== STAGE 3: CV over λ and threshold on train2 ===")
    # load stats2 & init ft2id from stage2
    stats2, ft2id = preprocess_train(train2, threshold=1)
    ft2id.calc_represent_input_with_features()
    all_sents = read_test(train2, tagged=True)

    lam_values = [0.001,0.01,0.1,1.0]
    th_values  = [0.9,0.85,0.8]
    best = {"score":-1}

    kf = KFold(n_splits=5,shuffle=True,random_state=0)
    for lam in lam_values:
        for th in th_values:
            scores=[]
            for train_idx,dev_idx in kf.split(all_sents):
                # write fold train file
                tmp="cv.wtag"
                with open(tmp,"w",encoding="utf8") as out:
                    for i in train_idx:
                        wds, tags = all_sents[i]
                        seq = [f"{w}_{t}" for w,t in zip(wds[2:-1],tags[2:-1])]
                        out.write(" ".join(seq)+"\n")
                # train
                s2, f2 = preprocess_train(tmp,threshold=1)
                f2.calc_represent_input_with_features()
                w_fold = get_optimal_vector(s2,f2,lam,"cv_w.pkl")[0]
                # eval
                corr=tot=0
                for i in dev_idx:
                    wds,tags = all_sents[i]
                    pred = memm_viterbi(wds,w_fold,f2)[2:-1]
                    for p,g in zip(pred,tags[2:-1]):
                        corr += (p==g); tot +=1
                scores.append(corr/tot)
                os.remove(tmp)
            m = np.mean(scores)
            print(f"  λ={lam}, th={th} → {m:.4f}")
            if m>best["score"]:
                best.update({"score":m,"lam":lam,"th":th})
    print(f"[stage3] Best: λ={best['lam']} th={best['th']} ({best['score']:.4f})")
    return best["lam"],best["th"]

# ── Stage 4: Self-training ───────────────────────────────────────────────────
def stage4_selftrain(comp="data/comp2.words", lam=0.01, init_thresh=0.9):
    print("=== STAGE 4: Semi-supervised self-training ===")
    # load cv-tuned model
    (opt_params, ft2id) = pickle.load(open("weights2_cv.pkl","rb"))
    w = opt_params[0]
    stats = ft2id.feature_statistics

    # simple per-token threshold decay
    min_t=0.7; max_iter=3
    decay=(init_thresh-min_t)/(max_iter-1)
    sents = read_test(comp, tagged=False)
    tags  = [t for t in ft2id.feature_statistics.tags if t!="~"]

    for itr in range(1,max_iter+1):
        t_i = init_thresh - decay*(itr-1)
        new=[]
        for wds,_ in sents:
            for i in range(2,len(wds)-1):
                # local MEMM log-probs
                scores=[]
                for t in tags:
                    hist=(wds[i],t,wds[i-1],"*",wds[i-2],"*",wds[i+1])
                    feats=represent_input_with_features(hist,ft2id.feature_to_idx,stats.common_words)
                    scores.append(sum(w[idx] for idx in feats))
                exps=np.exp(scores-np.max(scores))
                probs=exps/exps.sum()
                bi=np.argmax(probs)
                if probs[bi]>=t_i:
                    new.append((wds[i],tags[bi],wds[i-1],"*",wds[i-2],"*",wds[i+1]))
        if not new:
            print(f"[selftrain] no new ≥{t_i:.2f}, stopping.")
            break
        stats.histories.extend(new)
        print(f"[selftrain] +{len(new)} pseudo-histories")
        ft2id.calc_represent_input_with_features()
        w = get_optimal_vector(stats,ft2id,lam,"weights2_selfTrain.pkl",init_weights=w)[0]
    print("[stage4] Self-training done.")

# ── Stage 5: Final tagging ──────────────────────────────────────────────────
def stage5_tag(comp="data/comp2.words"):
    print("=== STAGE 5: Tag comp2.words with final model ===")
    (opt_params, ft2id) = pickle.load(open("weights2_selfTrain.pkl","rb"))
    w = opt_params[0]
    tag_all_test(comp, w, ft2id, "predictions2_final.wtag")
    if os.path.exists("data/test2.wtag"):
        words,gold,pred = predict_all("data/test2.wtag",ft2id,w)
        print("[stage5] dev acc:",word_accuracy(gold,pred))

# ── Stage 6: spaCy SOTA + compare ───────────────────────────────────────────
def stage6_compare(raw="data/comp2.words", mine="predictions2_final.wtag", sota="comp2_sota.wtag"):
    print("=== STAGE 6: spaCy‐whitespace SOTA + compare ===")
    if not os.path.exists(sota):
        nlp=spacy.load("en_core_web_sm",disable=["parser","ner"])
        token_match=re.compile(r"[^ ]+").match
        nlp.tokenizer=Tokenizer(nlp.vocab,token_match=token_match)
        with open(raw) as fin, open(sota,"w") as fout:
            for L in fin:
                line=L.strip()
                if not line: fout.write("\n"); continue
                doc=nlp(line)
                fout.write(" ".join(f"{t.text}_{t.tag_}" for t in doc)+"\n")
        print("[stage6] SOTA written to",sota)

    # compare
    conf=Counter(); gc=Counter()
    with open(mine) as fm, open(sota) as fs:
        for lm,ls in zip(fm,fs):
            for pm,ps in zip(lm.split(), ls.split()):
                tm=pm.rsplit("_",1)[1]; ts=ps.rsplit("_",1)[1]
                conf[(ts,tm)]+=1; gc[ts]+=1
    tot=sum(gc.values()); cor=sum(conf[(t,t)] for t in gc)
    print(f"[stage6] overall agreement {cor}/{tot} = {cor/tot:.2%}")
    print("TAG   P     R    F1   support")
    for t in sorted(gc):
        tp=conf[(t,t)]
        fp=sum(conf[(o,t)] for o in gc if o!=t)
        fn=sum(conf[(t,o)] for o in gc if o!=t)
        P=tp/(tp+fp) if tp+fp else 0
        R=tp/(tp+fn) if tp+fn else 0
        F=2*P*R/(P+R) if P+R else 0
        print(f"{t:4s} {P:5.2%} {R:5.2%} {F:5.2%} {gc[t]}")

if __name__=="__main__":
    # just run stages in order:
    # stage1_prune()
    # stage2_transfer()
    # lam,th = stage3_cv()
    # # save CV‐tuned
    # # (this re‐trains with best lam and zero‐init; you can skip if you want warm‐start)
    # stats2,ft2id=preprocess_train("data/train2.wtag",1); ft2id.calc_represent_input_with_features()
    # w_cv = get_optimal_vector(stats2,ft2id,lam,"weights2_cv.pkl")[0]
    # open("weights2_cv.pkl","wb").write(pickle.dumps(((w_cv,),ft2id)))
    # stage4_selftrain(lam=lam, init_thresh=th)
    stage5_tag()
    stage6_compare()
