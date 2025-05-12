from scipy import sparse
from collections import OrderedDict, defaultdict
import numpy as np
from typing import List, Dict, Tuple


WORD = 0
TAG = 1

def word_shape(w: str) -> str:
    # X=upper, x=lower, 0=digit, else itself; collapse runs
    s = []
    for ch in w:
        if   ch.isupper(): s.append("X")
        elif ch.islower(): s.append("x")
        elif ch.isdigit(): s.append("0")
        else:              s.append(ch)
    out = [s[0]] if s else []
    for c in s[1:]:
        if c != out[-1]:
            out.append(c)
    return "".join(out)


class FeatureStatistics:
    def __init__(self):
        self.n_total_features = 0  # Total number of features accumulated

        # Init all features dictionaries
        feature_dict_list = ["f100", "f101", "f102", "f103", "f104",
                             "f105", "f106", "f107","f108","f109","f110",
                             "f112",
                             "f113",
                                "f114",  # capital‐start
                                "f115",  # init‐cap & first‐word
                               # ——— new features ———
                                     "f132",  # word‐shape
                                     "f_char3",  # char 3‐grams
                                     "f_char4",  # char 4‐grams
                                     "f_wordbigram_prev",
                                     "f_wordbigram_next",
                                     "f_tagshape",  # cross of tag‐bigram × shape
                             "f141",  # sentence-position bucket
                             "f142",  # common-word flag
                             # # common‐verb suffix
                             "f300",
                             # # prev/next word affixes
                             "f304", "f305", "f306", "f307",
                             # sentence‐position flags
                             "f309", "f310", "f311",
                             "f143", # word length indicator
                             ]  # the feature classes used in the code
        self.feature_rep_dict = {fd: OrderedDict() for fd in feature_dict_list}
        '''
        A dictionary containing the counts of each data regarding a feature class. For example in f100, would contain
        the number of times each (word, tag) pair appeared in the text.
        '''
        self.tags = set()  # a set of all the seen tags
        self.tags.add("~")
        self.tags_counts = defaultdict(int)  # a dictionary with the number of times each tag appeared in the text
        self.words_count = defaultdict(int)  # a dictionary with the number of times each word appeared in the text
        self.histories = []  # a list of all the histories seen at the test
        self.words_count = defaultdict(int)
        self.word_count_threshold = 1


        self.global_word_counts = defaultdict(int)
        self.common_words = set()

    def increase_instances_count(self, feat_class: str, key):
        self.feature_rep_dict[feat_class][key] = self.feature_rep_dict[feat_class].get(key, 0) + 1

    def increment_val_in_feature_dict(self, feat_class: str, key):
        # alias for symmetry with your snippet
        self.increase_instances_count(feat_class, key)


    # --- helper methods for counting each feature group ---
    def _count_word_tag_basics(self, w, t, pp_word, p_word, n_word):
        # f100: (word, tag)
        self.feature_rep_dict["f100"][(w, t)] = self.feature_rep_dict["f100"].get((w, t), 0) + 1
        # f101: suffixes
        for L in range(1, min(4, len(w)) + 1):
            suf = w[-L:]
            self.feature_rep_dict["f101"][(suf, t)] = self.feature_rep_dict["f101"].get((suf, t), 0) + 1
        # f102: prefixes
        for L in range(1, min(4, len(w)) + 1):
            pre = w[:L]
            self.feature_rep_dict["f102"][(pre, t)] = self.feature_rep_dict["f102"].get((pre, t), 0) + 1
        # f103: tag-trigram
        if pp_word[0] is not None and p_word[0] is not None:
            tri = (pp_word[1], p_word[1], t)
            self.feature_rep_dict["f103"][tri] = self.feature_rep_dict["f103"].get(tri, 0) + 1
        # f104: tag-bigram
        if p_word[0] is not None:
            bi = (p_word[1], t)
            self.feature_rep_dict["f104"][bi] = self.feature_rep_dict["f104"].get(bi, 0) + 1
        # f105: tag-unigram
        self.feature_rep_dict["f105"][t] = self.feature_rep_dict["f105"].get(t, 0) + 1
        # f106: prev-word+tag
        if p_word[0] is not None:
            self.feature_rep_dict["f106"][(p_word[0], t)] = self.feature_rep_dict["f106"].get((p_word[0], t), 0) + 1
        # f107: next-word+tag
        if n_word[0] is not None:
            self.feature_rep_dict["f107"][(n_word[0], t)] = self.feature_rep_dict["f107"].get((n_word[0], t), 0) + 1

    def _count_orthographic(self, w, t):
        # f108: has digit
        if any(ch.isdigit() for ch in w):
            self.feature_rep_dict["f108"][('has_digit', t)] = self.feature_rep_dict["f108"].get(('has_digit', t), 0) + 1
        # f109: is numeric
        if w.isdigit():
            self.feature_rep_dict["f109"][('is_numeric', t)] = self.feature_rep_dict["f109"].get(('is_numeric', t), 0) + 1
        # f110: has uppercase
        if any(ch.isupper() for ch in w):
            self.feature_rep_dict["f110"][('has_upper', t)] = self.feature_rep_dict["f110"].get(('has_upper', t), 0) + 1
        # f112: all caps
        if w.isupper():
            self.feature_rep_dict["f112"][('all_caps', t)] = self.feature_rep_dict["f112"].get(('all_caps', t), 0) + 1
        # f113: number-noun
        head, sep, tail = w.partition("-")
        if head.isdigit() and sep == "-" and not tail.isdigit():
            self.feature_rep_dict["f113"][('num_noun', t)] = self.feature_rep_dict["f113"].get(('num_noun', t), 0) + 1

    def _count_capitalization(self, w, t, position):
        # f114: capital-start
        if w and w[0].isupper() and w[0].isalpha():
            self.feature_rep_dict["f114"][('capital_start', t)] = self.feature_rep_dict["f114"].get(('capital_start', t), 0) + 1
        # f115: initcap_first
        if w and w[0].isupper() and position == 0:
            self.feature_rep_dict["f115"][('initcap_first', t)] = self.feature_rep_dict["f115"].get(('initcap_first', t), 0) + 1

    def _count_word_shape(self, w, t):
        # f132: word shape
        shp = word_shape(w)
        self.feature_rep_dict["f132"][(shp, t)] = self.feature_rep_dict["f132"].get((shp, t), 0) + 1

    def _count_char_ngrams(self, w, t):
        # f_char3 & f_char4
        for n, key in ((3, 'f_char3'), (4, 'f_char4')):
            for j in range(len(w) - n + 1):
                gram = w[j:j+n]
                self.feature_rep_dict[key][(gram, t)] = self.feature_rep_dict[key].get((gram, t), 0) + 1

    def _count_word_bigrams(self, w, t, pp_word, p_word, n_word):
        # f_wordbigram_prev
        if p_word:
            self.feature_rep_dict['f_wordbigram_prev'][((p_word[0], w), t)] = self.feature_rep_dict['f_wordbigram_prev'].get(((p_word[0], w), t), 0) + 1
        # f_wordbigram_next
        if n_word:
            self.feature_rep_dict['f_wordbigram_next'][((w, n_word[0]), t)] = self.feature_rep_dict['f_wordbigram_next'].get(((w, n_word[0]), t), 0) + 1

    def _count_tagshape(self, p_tag, shp, t):
        # f_tagshape
        self.feature_rep_dict['f_tagshape'][((p_tag, t), shp)] = self.feature_rep_dict['f_tagshape'].get(((p_tag, t), shp), 0) + 1

    def _count_position_bucket(self, t, pos_idx, sentence_len):
        # f141
        frac = pos_idx / float(sentence_len - 1) if sentence_len > 1 else 0.0
        bucket = 'start' if frac < 0.25 else ('end' if frac > 0.75 else 'mid')
        self.feature_rep_dict['f141'][(bucket, t)] = self.feature_rep_dict['f141'].get((bucket, t), 0) + 1

    def _count_common_word_flag(self, w, t):
        # f142: only if w is in the set we just built
        if w in self.common_words:
            self.feature_rep_dict['f142'][('common_word', t)] = \
                self.feature_rep_dict['f142'].get(('common_word', t), 0) + 1

    def _count_verb_suffix(self, w, t):
        # f300
        for suf in ('ing','ed','en','s','es','ies'):
            if w.lower().endswith(suf):
                self.feature_rep_dict['f300'][(suf,t)] = self.feature_rep_dict['f300'].get((suf,t),0) + 1
                break

    def _count_neighbor_affixes(self, w, t, pp_word, p_word, n_word):
        # f304-307 suffixes/prefixes of neighbors
        for key, neigh in [('f304', pp_word), ('f305', pp_word[:1]),
                           ('f306', n_word[-1:]), ('f307', n_word[:1])]:
            if neigh is not None:
                self.feature_rep_dict[key][(neigh, t)] = self.feature_rep_dict[key].get((neigh, t),0) + 1

    def _count_position_flags(self, pp_word, p_word, n_word, t):
        # f309: second
        self.feature_rep_dict['f309'][((pp_word is None and p_word is not None), t)] = \
            self.feature_rep_dict['f309'].get(((pp_word is None and p_word is not None), t),0) + 1
        # f310: last
        self.feature_rep_dict['f310'][((n_word is None), t)] = self.feature_rep_dict['f310'].get(((n_word is None), t),0) + 1
        # f311: middle
        cond = (p_word is not None and pp_word is not None and n_word is not None)
        self.feature_rep_dict['f311'][(cond, t)] = self.feature_rep_dict['f311'].get((cond, t),0) + 1

    def _count_word_length(self, w, t):
        # f143
        l = len(w)
        bucket = 'short' if l<4 else ('med' if l<=7 else 'long')
        self.feature_rep_dict['f143'][(bucket,t)] = self.feature_rep_dict['f143'].get((bucket,t),0) + 1

    def get_word_tag_pair_count(self, file_path) -> None:
        """
        Count raw feature events f100–f113 and G2–G8 (f114–f131) in two passes.
        """

        # padding templates
        PAD_LEFT = [("*", "*"), ("*", "*")]
        PAD_RIGHT = [("~", "~")]

        # ——— PASS 1: build global counts ———
        with open(file_path, encoding='utf8') as f:
            for line in f:
                line = line.rstrip("\n")
                for wt in line.split():
                    w, _ = wt.split("_")
                    self.global_word_counts[w] += 1

        # derive common_words once
        common_threshold = 100
        self.common_words = {w for w, c in self.global_word_counts.items() if c > common_threshold}

        # ——— PASS 2: count features & collect histories ———
        with open(file_path, encoding='utf8') as f:
            for line in f:
                line = line.rstrip("\n")
                words_tags = [wt.split("_") for wt in line.split()]
                sentence_len = len(words_tags)
                padded = PAD_LEFT + words_tags + PAD_RIGHT

                # count each token’s features
                for i, (w, t) in enumerate(words_tags):
                    # fetch context tuples
                    pp = padded[i]  # (word_{i-2}, tag_{i-2})
                    p = padded[i + 1]  # (word_{i-1}, tag_{i-1})
                    n = padded[i + 2]  # (word_{i+1}, tag_{i+1})

                    self._count_word_tag_basics(w, t, pp, p, n)
                    self._count_orthographic(w, t)
                    self._count_capitalization(w, t, i)
                    self._count_word_shape(w, t)
                    self._count_char_ngrams(w, t)
                    self._count_word_bigrams(w, t, pp, p, n)
                    self._count_tagshape(p[1], word_shape(w), t)
                    self._count_position_bucket(t, i, sentence_len)
                    self._count_common_word_flag(w, t)
                    self._count_verb_suffix(w, t)
                    self._count_neighbor_affixes(w, t, pp, p, n)
                    self._count_position_flags(pp, p, n, t)
                    self._count_word_length(w, t)

                # build histories for MEMM training
                padded_full = PAD_LEFT + words_tags + PAD_RIGHT
                for prev2, prev1, curr, nxt in zip(padded_full,
                                                   padded_full[1:],
                                                   padded_full[2:],
                                                   padded_full[3:]):
                    self.histories.append((
                        curr[0], curr[1],
                        prev1[0], prev1[1],
                        prev2[0], prev2[1],
                        nxt[0]
                    ))


class Feature2id:
    def __init__(self, feature_statistics: FeatureStatistics, threshold: int):
        """
        @param feature_statistics: the feature statistics object
        @param threshold: the minimal number of appearances a feature should have to be taken
        """
        self.feature_statistics = feature_statistics  # statistics class, for each feature gives empirical counts
        self.threshold = threshold  # feature count threshold - empirical count must be higher than this

        self.n_total_features = 0  # Total number of features accumulated

        # Init all features dictionaries
        self.feature_to_idx = {
            feat_class: OrderedDict()
            for feat_class in feature_statistics.feature_rep_dict
        }

        self.represent_input_with_features = OrderedDict()
        self.histories_matrix = OrderedDict()
        self.histories_features = OrderedDict()
        self.small_matrix = sparse.csr_matrix
        self.big_matrix = sparse.csr_matrix



    def prune_top_k_by_weight(self, weights: np.ndarray, K: int) -> None:
        """
        Prune top k features by their |w| parameter value
        And small_matrix with the pruned features
        """
        # 1) rank features by |w|
        abs_w = np.abs(weights)
        # get the indices of the top-K largest absolute weights
        topk = np.argsort(abs_w)[-K:]

        # 2) build a boolean mask of kept feature‐indices
        mask = np.zeros(len(abs_w), dtype=bool)
        mask[topk] = True

        # 3) rebuild feature_to_idx mapping
        new_map = {fc: OrderedDict() for fc in self.feature_to_idx}
        new_idx = 0
        for fc, fmap in self.feature_to_idx.items():
            for feat, idx in fmap.items():
                if mask[idx]:
                    new_map[fc][feat] = new_idx
                    new_idx += 1
        self.feature_to_idx = new_map
        self.n_total_features = new_idx

        # 4) rebuild small_matrix (for training)
        rows, cols = [], []
        for i, hist in enumerate(self.feature_statistics.histories):
            for c in represent_input_with_features(hist, self.feature_to_idx):
                rows.append(i)
                cols.append(c)
        self.small_matrix = sparse.csr_matrix(
            (np.ones(len(rows)), (np.array(rows), np.array(cols))),
            shape=(len(self.feature_statistics.histories), self.n_total_features),
            dtype=bool
        )


    def get_features_idx(self) -> None:
        """
        Assigns each feature that appeared enough time in the train files an idx.
        Saves those indices to self.feature_to_idx
        """
        for feat_class in self.feature_statistics.feature_rep_dict:
            if feat_class not in self.feature_to_idx:
                continue
            for feat, count in self.feature_statistics.feature_rep_dict[feat_class].items():
                if count >= self.threshold:
                    self.feature_to_idx[feat_class][feat] = self.n_total_features
                    self.n_total_features += 1
        print(f"you have {self.n_total_features} features!")

    def calc_represent_input_with_features(self) -> None:
        """
        initializes the matrices used in the optimization process - self.big_matrix and self.small_matrix
        """
        big_r = 0
        big_rows = []
        big_cols = []
        small_rows = []
        small_cols = []
        for small_r, hist in enumerate(self.feature_statistics.histories):
            for c in represent_input_with_features(hist, self.feature_to_idx):
                small_rows.append(small_r)
                small_cols.append(c)
            for r, y_tag in enumerate(self.feature_statistics.tags):
                demi_hist = (hist[0], y_tag, hist[2], hist[3], hist[4], hist[5], hist[6])
                self.histories_features[demi_hist] = []
                for c in represent_input_with_features(demi_hist, self.feature_to_idx):
                    big_rows.append(big_r)
                    big_cols.append(c)
                    self.histories_features[demi_hist].append(c)
                big_r += 1
        self.big_matrix = sparse.csr_matrix((np.ones(len(big_rows)), (np.array(big_rows), np.array(big_cols))),
                                            shape=(len(self.feature_statistics.tags) * len(
                                                self.feature_statistics.histories), self.n_total_features),
                                            dtype=bool)
        self.small_matrix = sparse.csr_matrix(
            (np.ones(len(small_rows)), (np.array(small_rows), np.array(small_cols))),
            shape=(len(
                self.feature_statistics.histories), self.n_total_features), dtype=bool)


# ——————————————————————————————————————————————————————
# Helpers for represent_input_with_features
# ——————————————————————————————————————————————————————
def _fe_word_tag_basics(history: Tuple, dicts: Dict[str, Dict]):
    """
    f100–f107: word/tag, suffixes, prefixes, tag-trigram, tag-bigram,
    tag-unigram, prev-word+tag, next-word+tag
    """
    c_word, c_tag, p_word, p_tag, pp_word, pp_tag, n_word = history
    feats: List[int] = []
    # f100
    if (c_word, c_tag) in dicts["f100"]:
        feats.append(dicts["f100"][(c_word, c_tag)])
    # f101 suffixes
    for L in range(1, min(4, len(c_word)) + 1):
        suf = c_word[-L:]
        idx = dicts["f101"].get((suf, c_tag))
        if idx is not None: feats.append(idx)
    # f102 prefixes
    for L in range(1, min(4, len(c_word)) + 1):
        pre = c_word[:L]
        idx = dicts["f102"].get((pre, c_tag))
        if idx is not None: feats.append(idx)
    # f103 tag-trigram
    if pp_word is not None and p_word is not None:
        tri = (pp_tag, p_tag, c_tag)
        idx = dicts["f103"].get(tri)
        if idx is not None: feats.append(idx)
    # f104 tag-bigram
    if p_word is not None:
        bi = (p_tag, c_tag)
        idx = dicts["f104"].get(bi)
        if idx is not None: feats.append(idx)
    # f105 tag-unigram
    idx = dicts["f105"].get(c_tag)
    if idx is not None: feats.append(idx)
    # f106 prev-word + tag
    if p_word is not None:
        idx = dicts["f106"].get((p_word, c_tag))
        if idx is not None: feats.append(idx)
    # f107 next-word + tag
    if n_word is not None:
        idx = dicts["f107"].get((n_word, c_tag))
        if idx is not None: feats.append(idx)
    return feats


def _fe_orthographic(history: Tuple, dicts: Dict[str, Dict]):
    """
    f108–f113: has_digit, is_numeric, has_upper, all_caps, num_noun
    """
    c_word, c_tag, *_ = history
    feats: List[int] = []
    # f108
    if any(ch.isdigit() for ch in c_word):
        idx = dicts["f108"].get(("has_digit", c_tag))
        if idx is not None: feats.append(idx)
    # f109
    if c_word.isdigit():
        idx = dicts["f109"].get(("is_numeric", c_tag))
        if idx is not None: feats.append(idx)
    # f110
    if any(ch.isupper() for ch in c_word):
        idx = dicts["f110"].get(("has_upper", c_tag))
        if idx is not None: feats.append(idx)
    # f112
    if c_word.isupper():
        idx = dicts["f112"].get(("all_caps", c_tag))
        if idx is not None: feats.append(idx)
    # f113
    head, sep, tail = c_word.partition("-")
    if head.isdigit() and sep == "-" and not tail.isdigit():
        idx = dicts["f113"].get(("num_noun", c_tag))
        if idx is not None: feats.append(idx)
    return feats


def _fe_capitalization(history: Tuple, dicts: Dict[str, Dict], position: int):
    """
    f114: capital_start; f115: initcap_first
    """
    c_word, c_tag, *_ = history
    feats: List[int] = []
    if c_word and c_word[0].isupper() and c_word[0].isalpha():
        idx = dicts["f114"].get(("capital_start", c_tag))
        if idx is not None: feats.append(idx)
        if position == 0:
            idx2 = dicts["f115"].get(("initcap_first", c_tag))
            if idx2 is not None: feats.append(idx2)
    return feats


def _fe_word_shape(history: Tuple, dicts: Dict[str, Dict]):
    """f132: word_shape"""
    c_word, c_tag, *_ = history
    feats: List[int] = []
    shp = word_shape(c_word)
    idx = dicts["f132"].get((shp, c_tag))
    if idx is not None: feats.append(idx)
    return feats


def _fe_char_ngrams(history: Tuple, dicts: Dict[str, Dict]):
    """f_char3, f_char4: character n-grams"""
    c_word, c_tag, *_ = history
    feats: List[int] = []
    for n, key in ((3, 'f_char3'), (4, 'f_char4')):
        for j in range(len(c_word) - n + 1):
            gram = c_word[j:j+n]
            idx = dicts[key].get((gram, c_tag))
            if idx is not None: feats.append(idx)
    return feats


def _fe_word_bigrams(history: Tuple, dicts: Dict[str, Dict]):
    """f_wordbigram_prev, f_wordbigram_next"""
    c_word, c_tag, p_word, _p_tag, pp_word, _pp_tag, n_word = history
    feats: List[int] = []
    if p_word is not None:
        idx = dicts['f_wordbigram_prev'].get(((p_word, c_word), c_tag))
        if idx is not None: feats.append(idx)
    if n_word is not None:
        idx = dicts['f_wordbigram_next'].get(((c_word, n_word), c_tag))
        if idx is not None: feats.append(idx)
    return feats


def _fe_tagshape(history: Tuple, dicts: Dict[str, Dict]):
    """f_tagshape: ((prev_tag, curr_tag), shape)"""
    c_word, c_tag, p_word, p_tag, *_ = history
    feats: List[int] = []
    shp = word_shape(c_word)
    idx = dicts['f_tagshape'].get(((p_tag, c_tag), shp))
    if idx is not None: feats.append(idx)
    return feats


def _fe_position_bucket(history: Tuple, dicts: Dict[str, Dict], position: int, sentence_len: int):
    """f141: sentence-position bucket"""
    c_word, c_tag, *_ = history
    feats: List[int] = []
    frac = position / float(sentence_len - 1) if sentence_len > 1 else 0.0
    bucket = 'start' if frac < 0.25 else ('end' if frac > 0.75 else 'mid')
    idx = dicts['f141'].get((bucket, c_tag))
    if idx is not None: feats.append(idx)
    return feats


def _fe_common_word_flag(history: Tuple, dicts: Dict[str, Dict], common_words: set):
    """f142: common-word flag"""
    c_word, c_tag, *_ = history
    feats: List[int] = []
    if c_word in common_words:
        idx = dicts['f142'].get(("common_word", c_tag))
        if idx is not None: feats.append(idx)
    return feats


def _fe_verb_suffix(history: Tuple, dicts: Dict[str, Dict]):
    """f300: common-verb suffix"""
    c_word, c_tag, *_ = history
    feats: List[int] = []
    for suf in ("ing","ed","en","s","es","ies"):
        if c_word.lower().endswith(suf):
            idx = dicts['f300'].get((suf, c_tag))
            if idx is not None: feats.append(idx)
            break
    return feats


def _fe_neighbor_affixes(history: Tuple, dicts: Dict[str, Dict]):
    """f304–f307: affixes of neighboring words"""
    c_word, c_tag, p_word, _p_tag, pp_word, _pp_tag, n_word = history
    feats: List[int] = []
    for key, neigh in [('f304', pp_word), ('f305', pp_word[:1]),
                       ('f306', n_word[-1:]), ('f307', n_word[:1])]:
        if neigh is not None:
            idx = dicts[key].get((neigh, c_tag))
            if idx is not None: feats.append(idx)
    return feats


def _fe_position_flags(history: Tuple, dicts: Dict[str, Dict]):
    """f309–f311: boolean position flags"""
    c_word, c_tag, p_word, p_tag, pp_word, pp_tag, n_word = history
    feats: List[int] = []
    # second?
    cond2 = (pp_word is None and p_word is not None)
    idx = dicts['f309'].get((cond2, c_tag))
    if idx is not None: feats.append(idx)
    # last?
    cond_last = (n_word is None)
    idx = dicts['f310'].get((cond_last, c_tag))
    if idx is not None: feats.append(idx)
    # middle?
    cond_mid = (p_word is not None and pp_word is not None and n_word is not None)
    idx = dicts['f311'].get((cond_mid, c_tag))
    if idx is not None: feats.append(idx)
    return feats


def _fe_word_length(history: Tuple, dicts: Dict[str, Dict]):
    """f143: word-length bucket"""
    c_word, c_tag, *_ = history
    feats: List[int] = []
    l = len(c_word)
    bucket = 'short' if l<4 else ('med' if l<=7 else 'long')
    idx = dicts['f143'].get((bucket, c_tag))
    if idx is not None: feats.append(idx)
    return feats

# main dispatcher:
def represent_input_with_features(history: Tuple, dicts: Dict[str, Dict], common_words: set, position: int=None, sentence_len: int=None) -> List[int]:
    features: List[int] = []
    features += _fe_word_tag_basics(history, dicts)
    features += _fe_orthographic(history, dicts)
    # position and sentence_len must be passed in by caller for below:
    if position is not None:
        features += _fe_capitalization(history, dicts, position)
    features += _fe_word_shape(history, dicts)
    features += _fe_char_ngrams(history, dicts)
    features += _fe_word_bigrams(history, dicts)
    features += _fe_tagshape(history, dicts)
    if position is not None and sentence_len is not None:
        features += _fe_position_bucket(history, dicts, position, sentence_len)
        features += _fe_common_word_flag(history, dicts, common_words)
    features += _fe_verb_suffix(history, dicts)
    features += _fe_neighbor_affixes(history, dicts)
    features += _fe_position_flags(history, dicts)
    features += _fe_word_length(history, dicts)
    return features


def preprocess_train(train_path, threshold):
    # Statistics
    statistics = FeatureStatistics()
    statistics.get_word_tag_pair_count(train_path)

    # build the global common_words set
    # e.g. threshold = 100 occurrences
    common_threshold = 100
    # you counted them in statistics.global_word_counts
    global common_words
    common_words = {w for w, cnt in statistics.global_word_counts.items() if cnt > common_threshold}
    print(f"Keeping {len(common_words)} words as ‘common’ (>{common_threshold} occurrences).")

    # feature2id
    feature2id = Feature2id(statistics, threshold)
    feature2id.get_features_idx()
    feature2id.calc_represent_input_with_features()
    print(feature2id.n_total_features)

    for dict_key in feature2id.feature_to_idx:
        print(dict_key, len(feature2id.feature_to_idx[dict_key]))
    return statistics, feature2id


def read_test(file_path, tagged=True) -> List[Tuple[List[str], List[str]]]:
    """
    reads a test file
    @param file_path: the path to the file
    @param tagged: whether the file is tagged (validation set) or not (test set)
    @return: a list of all the sentences, each sentence represented as tuple of list of the words and a list of tags
    """
    list_of_sentences = []
    with open(file_path) as f:
        for line in f:
            if line[-1:] == "\n":
                line = line[:-1]
            sentence = (["*", "*"], ["*", "*"])
            split_words = line.split(' ')
            for word_idx in range(len(split_words)):
                if tagged:
                    cur_word, cur_tag = split_words[word_idx].split('_')
                else:
                    cur_word, cur_tag = split_words[word_idx], ""
                sentence[WORD].append(cur_word)
                sentence[TAG].append(cur_tag)
            sentence[WORD].append("~")
            sentence[TAG].append("~")
            list_of_sentences.append(sentence)
    return list_of_sentences
