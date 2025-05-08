import time, pickle
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from scipy.special import logsumexp
from collections import Counter
from itertools import chain


def data_preprocessing(path, mode):
    """
    Reads input file into:
      V        = vocabulary list
      T        = tag list (no padding symbols)
      data     = list of word lists
      data_tag = list of tag lists
    mode: 'train','test','comp'
    """
    data, data_tag = [], []
    with open(path) as f:
        for line in f:
            toks = line.strip().split()
            if mode == 'comp':
                data.append(toks)
            else:
                words, tags = [], []
                for tok in toks:
                    w,t = tok.split('_')
                    words.append(w);
                    tags.append(t)
                data.append(words)
                data_tag.append(tags)

    if mode == 'train':
        # pad each sentence for 2-start and 1-end
        data_p, data_tag_p = [], []
        for words,tags in zip(data,data_tag):
            w2 = ['*','*'] + words + ['~']
            t2 = ['*','*'] + tags  + ['~']
            data_p.append(w2)
            data_tag_p.append(t2)
        data, data_tag = data_p, data_tag_p

    T = []
    V = []
    if mode != 'comp':
        T = sorted(set(chain(*data_tag)) - {'*','~'})
    V = sorted(set(chain(*data)))
    return V, T, data, data_tag


class POS_MEMM:
    def __init__(self, suffix_max=4, prefix_max=4, reg=1.0):
        self.smax = suffix_max
        self.pmax = prefix_max
        self.reg  = reg

    def build_index(self, V, T):
        """Index vocab and tags (with padding)"""
        # T_with_pad includes '*' and '~'
        self.T = ['*','*'] + T + ['~']
        self.V = V
        self.V2i = {w:i for i,w in enumerate(V)}
        self.T2i = {t:i for i,t in enumerate(self.T)}
        self.V_size = len(V)
        self.T_size = len(self.T)

    def get_feature_size(self):
        # f100: V*T
        F100 = self.V_size * self.T_size
        # f101/102: (sum_{l=1..smax} V? but dynamic) approximate smax*V*T
        F101 = self.smax * len(self.suffixes) * self.T_size
        F102 = self.pmax * len(self.prefixes) * self.T_size
        # f103: T^3, f104: T^2, f105: T, f106/107: V*T
        F103 = self.T_size**3
        F104 = self.T_size**2
        F105 = self.T_size
        F106 = self.V_size * self.T_size
        F107 = self.V_size * self.T_size
        # f108..f112: 5 binary features * T
        Fbin = 5 * self.T_size
        return sum([F100, F101, F102, F103, F104, F105, F106, F107, Fbin])

    def init_affixes(self, data):
        # collect all suffixes/prefixes up to length
        suffixes, prefixes = set(), set()
        for sent in data:
            for w in sent:
                L = len(w)
                for l in range(1, min(self.smax, L)+1): suffixes.add(w[-l:])
                for l in range(1, min(self.pmax, L)+1): prefixes.add(w[:l])
        self.suffixes = sorted(suffixes)
        self.prefixes = sorted(prefixes)
        self.s2i = {s:i for i,s in enumerate(self.suffixes)}
        self.p2i = {p:i for i,p in enumerate(self.prefixes)}

    def build_features(self):
        """Compute empirical and expected features across training set"""
        # Precompute feature indices for every training history
        self.histories = []      # list of (ctx, pos i)
        self.emp_counts = []     # empirical counts vector accumulators
        # Build expected features array for loss & grads
        self.exp_F = []  # list of lists of feature‐lists for each tag

        for words, tags in zip(self.data, self.data_tag):
            n = len(words)
            for i in range(2, n-1):
                ctx = (words[i-2], words[i-1], words[i], words[i+1])
                self.histories.append((ctx, tags[i]))
                # features for each candidate tag
                feats_all = []
                for t in self.T:
                    feats_all.append(self.get_features(ctx, t, is_first=(i==2)))
                self.exp_F.append(feats_all)
                # find empirical
                true_feats = feats_all[self.T.index(tags[i])]
                self.emp_counts.extend(true_feats)
        # convert emp_counts list to count vector
        self.emp_counts = Counter(self.emp_counts)

    def get_features(self, ctx, tag, is_first=False):
        """Return a list of feature indices for context+tag"""
        w2, w1, w, w3 = ctx
        t_i = self.T2i[tag]
        feats = []
        # base offset
        offset = 0
        # f100
        try:
            feats.append(self.V2i[w] * self.T_size + t_i)
        except: pass
        offset += self.V_size * self.T_size
        # f101: suffixes
        for l in range(1, min(self.smax,len(w))+1):
            s = w[-l:]
            if s in self.s2i:
                feats.append(offset + self.s2i[s]*self.T_size + t_i)
        offset += len(self.suffixes)*self.T_size
        # f102: prefixes
        for l in range(1, min(self.pmax,len(w))+1):
            p = w[:l]
            if p in self.p2i:
                feats.append(offset + self.p2i[p]*self.T_size + t_i)
        offset += len(self.prefixes)*self.T_size
        # f103: tag trigram
        feats.append(offset + (self.T2i[ctx[2]]*(self.T_size**2) + self.T2i[ctx[1]]*self.T_size + self.T2i[tag]))
        offset += self.T_size**3
        # f104: bigram
        feats.append(offset + (self.T2i[ctx[1]]*self.T_size + self.T2i[tag]))
        offset += self.T_size**2
        # f105: unigram tag
        feats.append(offset + t_i)
        offset += self.T_size
        # f106: prev-word * tag
        try:
            feats.append(offset + self.V2i[w1]*self.T_size + t_i)
        except: pass
        offset += self.V_size * self.T_size
        # f107: next-word * tag
        try:
            feats.append(offset + self.V2i[w3]*self.T_size + t_i)
        except: pass
        offset += self.V_size * self.T_size
        # f108: has digit
        if any(ch.isdigit() for ch in w): feats.append(offset + t_i)
        offset += self.T_size
        # f109: is numeric
        if w.isdigit(): feats.append(offset + t_i)
        offset += self.T_size
        # f110: has uppercase
        if any(ch.isupper() for ch in w): feats.append(offset + t_i)
        offset += self.T_size
        # f111: init cap
        if len(w)>0 and w[0].isupper(): feats.append(offset + t_i)
        offset += self.T_size
        # f112: all caps
        if w.isupper(): feats.append(offset + t_i)
        # done
        return feats

    def loss_and_grads(self, w):
        # build score matrix
        S = np.array([[np.sum(w[f]) for f in feats] for feats in self.exp_F])
        logZ = logsumexp(S, axis=1)
        # loss = w⋅emp - Σ logZ - .5⋅reg⋅||w||^2
        emp_sum = sum(w[i]*c for i,c in self.emp_counts.items())
        loss = emp_sum - logZ.sum() - 0.5*self.reg*(w@w)
        # grad
        P = np.exp(S - logZ[:,None])
        grad = np.zeros_like(w)
        for i, feats_all in enumerate(self.exp_F):
            for t_idx, fidxs in enumerate(feats_all):
                for f in fidxs:
                    grad[f] -= P[i,t_idx]
        for f,c in self.emp_counts.items(): grad[f] += c
        grad -= self.reg*w
        return -loss, -grad

    def train(self, path):
        V,T,data,data_tag = data_preprocessing(path,'train')
        self.build_index(V,T)
        self.init_affixes(data)
        self.data, self.data_tag = data, data_tag
        self.feature_size = self.get_feature_size()
        self.build_features()
        w0 = np.zeros(self.feature_size)
        w_opt,_,_ = fmin_l_bfgs_b(self.loss_and_grads, x0=w0)
        self.w = w_opt
        pickle.dump(self, open('model.pkl','wb'))

    def viterbi(self, sent, beam=10):
        # pad and decode with beam search using self.get_features
        from heapq import nlargest
        padded = ['*','*'] + sent + ['~']
        N = len(padded)
        Tset = self.T
        # beam: list of (score, hist)
        beam_list = [(0.0,['*','*'])]
        for i in range(2,N-1):
            new_cands=[]
            for sc,h in beam_list:
                for t in Tset:
                    feats = self.get_features((padded[i-2],padded[i-1],padded[i],padded[i+1]),t,i==2)
                    score = sc + sum(self.w[f] for f in feats) - logsumexp([sum(self.w[f] for f in self.get_features((padded[i-2],padded[i-1],padded[i],padded[i+1]),t2,i==2)) for t2 in Tset])
                    new_cands.append((score,h+[t]))
            beam_list = nlargest(beam,new_cands,key=lambda x:x[0])
        best = max(beam_list,key=lambda x:x[0])[1]+['~']
        return best

    def predict(self, path, out='pred.txt'):
        V,T,data,_ = data_preprocessing(path,'test')
        with open(out,'w') as f:
            for sent in data:
                tags = self.viterbi(sent)
                f.write(' '.join(f"{w}_{t}" for w,t in zip(sent,tags[2:-1]))+"\n")

    def analyze(self, gold_path):
        from sklearn.metrics import confusion_matrix, classification_report
        pred_tags = []
        gold_tags = []
        _,_,data,gt = data_preprocessing(gold_path,'test')
        for sent,goldt in zip(data,gt):
            pr = self.viterbi(sent)[2:-1]
            pred_tags.extend(pr)
            gold_tags.extend(goldt)
        print("Accuracy:", np.mean([p==g for p,g in zip(pred_tags,gold_tags)]))
        print(classification_report(gold_tags,pred_tags))


if __name__=='__main__':
    m = POS_MEMM(suffix_max=4,prefix_max=4,reg=0.05)
    m.train('data/train1.wtag')
    m.predict('data/test1.wtag','predictions.txt')
    m.analyze('data/test1.wtag')
