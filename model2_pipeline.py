#!/usr/bin/env python3
import os, pickle, numpy as np, re
from collections import Counter, OrderedDict
from sklearn.model_selection import KFold
import spacy
from spacy.tokenizer import Tokenizer

from preprocessing import preprocess_train, read_test, represent_input_with_features
from optimization   import get_optimal_vector
from inference      import tag_all_test, memm_viterbi

# ─── CONFIGURE HERE ────────────────────────────────────────────────────────────

# Paths
TRAIN_PATH     = "data/train2.wtag"
UNLABELED_PATH = "data/comp2.words"
OUTPUT_PRED    = "predictions2_final.wtag"
CV_DIR         = "cv_tmp"
MODEL_5K       = "weights_5k.pkl"
MODEL_SELF     = "weights_5k_selftrain.pkl"
SOTA_OUT       = "comp2_sota_whitespace.wtag"

# Hyperparameters
FEATURE_THRESHOLDS = [1,2,5]
LAM_VALUES         = [0.001,0.01,0.1]
CV_FOLDS           = 5
K_FEATURES         = 5000
SELF_THRESH        = [0.96,0.92,0.88]

# Stages ON/OFF
DO_CV         = True   # run cross‐validation to pick feat_thresh & λ
DO_TRAIN_PRUNE= True   # train full → prune to 5k
DO_SELFTRAIN  = True   # run token‐level self‐training
DO_TAG        = True   # tag comp2.words with final model
DO_COMPARE    = True   # compute agreement vs spaCy SOTA

# ────────────────────────────────────────────────────────────────────────────────

def cross_validate():
    os.makedirs(CV_DIR, exist_ok=True)
    best = {"lam":None,"th":None,"score":-1.0}
    all_sents = read_test(TRAIN_PATH, tagged=True)
    for th in FEATURE_THRESHOLDS:
        for lam in LAM_VALUES:
            scores=[]
            kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=0)
            for f,(tr,dev) in enumerate(kf.split(all_sents),1):
                fold_file = f"{CV_DIR}/fold{f}.wtag"
                with open(fold_file,"w") as out:
                    for i in tr:
                        wds,tags=all_sents[i]
                        seq=[f"{w}_{t}" for w,t in zip(wds[2:-1],tags[2:-1])]
                        out.write(" ".join(seq)+"\n")
                stats,f2i = preprocess_train(fold_file,threshold=th)
                f2i.calc_represent_input_with_features()
                w_fold = get_optimal_vector(stats,f2i,lam,f"{CV_DIR}/w{f}.pkl")[0]
                corr=tot=0
                for i in dev:
                    wds,tags=all_sents[i]
                    pred=memm_viterbi(wds,w_fold,f2i)[2:-1]
                    for p,g in zip(pred,tags[2:-1]):
                        corr+=p==g; tot+=1
                scores.append(corr/tot)
                os.remove(fold_file); os.remove(f"{CV_DIR}/w{f}.pkl")
            mean=np.mean(scores)
            if mean>best["score"]:
                best.update(lam=lam,th=th,score=mean)
    print(f"→ CV picks threshold={best['th']} λ={best['lam']} (accuracy={best['score']:.4f})")
    return best["th"],best["lam"]

def train_and_prune(feat_thresh, lam):
    stats,f2i = preprocess_train(TRAIN_PATH,threshold=feat_thresh)
    f2i.calc_represent_input_with_features()
    w_full = get_optimal_vector(stats,f2i,lam,"weights_full.pkl")[0]
    # prune
    absw,topk=np.abs(w_full),np.argsort(np.abs(w_full))[-K_FEATURES:]
    mask=np.zeros_like(absw,bool); mask[topk]=True
    new_map,new_i={},0
    for fc,fmap in f2i.feature_to_idx.items():
        new_map[fc]=OrderedDict()
        for feat,idx in fmap.items():
            if mask[idx]:
                new_map[fc][feat]=new_i; new_i+=1
    f2i.feature_to_idx, f2i.n_total_features = new_map, new_i
    f2i.calc_represent_input_with_features()
    w5k = get_optimal_vector(stats,f2i,lam,MODEL_5K,init_weights=w_full)[0]
    print(f"[train/prune] pruned → {len(w5k)} features")
    return stats,f2i,w5k

def self_train(stats,f2i,w):
    existing = set(stats.histories)
    sents = read_test(UNLABELED_PATH,tagged=False)
    tags=[t for t in f2i.feature_statistics.tags if t!="~"]
    for itr,thr in enumerate(SELF_THRESH,1):
        cands,seen=[],set()
        for words,_ in sents:
            for i in range(2,len(words)-1):
                scores=[]
                for t in tags:
                    hist=(words[i],t,words[i-1],"*",words[i-2],"*",words[i+1])
                    feats=represent_input_with_features(hist,f2i.feature_to_idx,f2i.feature_statistics.common_words)
                    scores.append(sum(w[idx] for idx in feats))
                exps=np.exp(scores-max(scores));probs=exps/exps.sum()
                bi=int(np.argmax(probs));conf=probs[bi]
                if conf>=thr:
                    hist=(words[i],tags[bi],words[i-1],"*",words[i-2],"*",words[i+1])
                    if hist not in existing and hist not in seen:
                        cands.append((conf,hist)); seen.add(hist)
        if not cands: break
        ps=np.array([c for c,_ in cands]); ps/=ps.sum()
        pick=np.random.choice(len(cands),min(len(cands),K_FEATURES),p=ps,replace=False)
        new_h=[cands[i][1] for i in pick]
        stats.histories.extend(new_h); existing.update(new_h)
        f2i.calc_represent_input_with_features()
        w = get_optimal_vector(stats,f2i,lam,MODEL_SELF,init_weights=w)[0]
        print(f"[self] round{itr}: +{len(new_h)} pseudo-histories")
    return w

def make_sota(raw,sota_out):
    if os.path.exists(sota_out): return
    print(f"[SOTA] → {sota_out}")
    nlp=spacy.load("en_core_web_sm",disable=["parser","ner"])
    token_match=re.compile(r"[^ ]+").match
    nlp.tokenizer=Tokenizer(nlp.vocab,token_match=token_match)
    with open(raw) as fin, open(sota_out,"w") as fout:
        for L in fin:
            line=L.strip()
            if not line: fout.write("\n"); continue
            doc=nlp(line)
            fout.write(" ".join(f"{tok.text}_{tok.tag_}" for tok in doc)+"\n")

def compare(pred,sota):
    conf, supp = Counter(), Counter()
    with open(pred) as fp, open(sota) as fs:
        for lm,ls in zip(fp,fs):
            p=lm.split();s=ls.split()
            if len(p)!=len(s): continue
            for wm,ws in zip(p,s):
                pm,sm=wm.rsplit("_",1)[1],ws.rsplit("_",1)[1]
                conf[(sm,pm)]+=1; supp[sm]+=1
    tot=sum(supp.values()); corr=sum(conf[(t,t)] for t in supp)
    print(f"\nAgreement: {corr}/{tot} = {corr/tot:.2%}\n")
    print(f"{'TAG':6s}  {'P':>6s}  {'R':>6s}  {'F1':>6s}  support")
    for t in sorted(supp):
        tp=conf[(t,t)]; fp=sum(conf[(o,t)] for o in supp if o!=t)
        fn=sum(conf[(t,o)] for o in supp if o!=t)
        P=tp/(tp+fp) if tp+fp else 0; R=tp/(tp+fn) if tp+fn else 0
        F=2*P*R/(P+R) if P+R else 0
        print(f"{t:6s}  {P:6.2%}  {R:6.2%}  {F:6.2%}  {supp[t]}")

if __name__=="__main__":
    if DO_CV:
        feat_thresh, lam = cross_validate()
    if DO_TRAIN_PRUNE:
        stats,f2i,w = train_and_prune(feat_thresh, lam)
    if DO_SELFTRAIN:
        w = self_train(stats,f2i,w)
    if DO_TAG:
        tag_all_test(UNLABELED_PATH, w, f2i, OUTPUT_PRED)
        print(f"[TAG] → {OUTPUT_PRED}")
    if DO_COMPARE:
        make_sota(UNLABELED_PATH, SOTA_OUT)
        compare(OUTPUT_PRED, SOTA_OUT)

    # ─── NEW: VALIDATE SELF-TRAINED MODELS ON TRAIN2 VIA 5-FOLD ─────────────────
    from sklearn.model_selection import KFold
    from inference import memm_viterbi
    import glob

    if True:  # or guard on a new flag, e.g. DO_VALIDATE_SELF
        print("\n[VALIDATION] 5-fold accuracy on train2.wtag for each self-trained model\n")
        # 1) load all the model files you want to validate
        #    adjust the glob to match your iteration filenames
        model_paths = sorted(glob.glob("weights_selftrain_iter*.pkl")) \
                    + ["weights_5k_selftrain.pkl"]

        # 2) read in all train2 sentences once
        all_sents = read_test(TRAIN_PATH, tagged=True)
        kf = KFold(n_splits=5, shuffle=True, random_state=0)

        for mpath in model_paths:
            print(f"→ Evaluating `{mpath}` …")
            # load (w_tuple, f2i) from pickle
            with open(mpath, "rb") as f:
                opt_params, f2i_model = pickle.load(f)
            w_model = opt_params[0]

            fold_scores = []
            # 3) for each fold, tag and score
            for fold_idx, (train_idx, dev_idx) in enumerate(kf.split(all_sents), 1):
                corr = tot = 0
                for i in dev_idx:
                    wds, gold_tags = all_sents[i]
                    # Viterbi tag with your MEMM
                    pred_tags = memm_viterbi(wds, w_model, f2i_model)[2:-1]
                    # compare
                    for p, g in zip(pred_tags, gold_tags[2:-1]):
                        corr += (p == g)
                        tot  += 1
                acc = corr / tot
                fold_scores.append(acc)
                print(f"  fold {fold_idx}: {acc*100:5.2f}%")

            mean_acc = np.mean(fold_scores)
            std_acc  = np.std(fold_scores)
            print(f"  → mean = {mean_acc*100:5.2f}%  (± {std_acc*100:.2f}%)\n")

