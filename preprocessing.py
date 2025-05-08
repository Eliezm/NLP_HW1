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
                             "f105", "f106", "f107","f108","f109","f110", "f111", "f112",
                             "f113",
                                "f114",  # capital‐start
                                "f115",  # init‐cap & first‐word
                                "f116",  # init‐cap & not‐first
                                "f117",  # ALL‐CAPS
                               # ——— new features ———
                                     "f132",  # word‐shape
                                     "f_char3",  # char 3‐grams
                                     "f_char4",  # char 4‐grams
                                     "f_wordbigram_prev",
                                     "f_wordbigram_next",
                                     "f_tagshape",  # cross of tag‐bigram × shape
                             "f138",  # word-shape
                             "f141",  # sentence-position bucket
                             "f142",  # common-word flag
                             # new boolean and count‐features
                             # new Boolean & positional features:
                             "f200", "f201", "f202", "f203", "f204",
                             # # common‐verb suffix
                             "f300",
                             # # punctuation / hyphen / dot
                             # "f301", "f302", "f303",
                             # # prev/next word affixes
                             "f304", "f305", "f306", "f307", "f308",
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

    def get_word_tag_pair_count(self, file_path) -> None:
        """
        Count raw feature events f100–f113 and G2–G8 (f114–f131).
        """

        def is_digit_char(ch: str) -> bool:
            return ch.isdigit()

        def is_numeric_word(word: str) -> bool:
            # all digits, no separators
            return word.isdigit()

        with open(file_path, encoding="utf8") as file:
            for line in file:
                line = line.rstrip("\n")
                words_tags = [wt.split("_") for wt in line.split()]
                PAD_LEFT = [("*", "*"), ("*", "*")]
                PAD_RIGHT = [("~", "~")]
                padded = PAD_LEFT + words_tags + PAD_RIGHT



                for i, (w, t) in enumerate(words_tags):
                    # ── track raw word counts ─────────────────────────────────────
                    self.global_word_counts[w] += 1

                    self.tags.add(t)

                    # safe–lookup helper
                    def safe(tags, idx, default):
                        return tags[idx] if 0 <= idx < len(tags) else default

                    # now pull out previous‐previous, previous, next
                    pp_word, pp_tag = safe(words_tags, i - 2, ("*", "*"))
                    p_word, p_tag = safe(words_tags, i - 1, ("*", "*"))
                    n_word, n_tag = safe(words_tags, i + 1, ("~", "~"))

                    # ── f100–f107 ─────────────────────────────────────────────────
                    # f100
                    self.feature_rep_dict["f100"][(w, t)] = self.feature_rep_dict["f100"].get((w, t), 0) + 1
                    # f101 suffixes
                    for L in range(1, min(4, len(w)) + 1):
                        key = (w[-L:], t)
                        self.feature_rep_dict["f101"][key] = self.feature_rep_dict["f101"].get(key, 0) + 1
                    # f102 prefixes
                    for L in range(1, min(4, len(w)) + 1):
                        key = (w[:L], t)
                        self.feature_rep_dict["f102"][key] = self.feature_rep_dict["f102"].get(key, 0) + 1
                    # f103 tag-trigram
                    if i >= 2:
                        tri = (words_tags[i - 2][1], words_tags[i - 1][1], t)
                        self.feature_rep_dict["f103"][tri] = self.feature_rep_dict["f103"].get(tri, 0) + 1
                    # f104 tag-bigram
                    if i >= 1:
                        bi = (words_tags[i - 1][1], t)
                        self.feature_rep_dict["f104"][bi] = self.feature_rep_dict["f104"].get(bi, 0) + 1
                    # f105 tag-unigram
                    self.feature_rep_dict["f105"][t] = self.feature_rep_dict["f105"].get(t, 0) + 1
                    # f106 prev-word + tag
                    if i >= 1:
                        prevw = words_tags[i - 1][0]
                        key = (prevw, t)
                        self.feature_rep_dict["f106"][key] = self.feature_rep_dict["f106"].get(key, 0) + 1
                    # f107 next-word + tag
                    if i < len(words_tags) - 1:
                        nextw = words_tags[i + 1][0]
                        key = (nextw, t)
                        self.feature_rep_dict["f107"][key] = self.feature_rep_dict["f107"].get(key, 0) + 1

                    # ── f108–f113 ─────────────────────────────────────────────────
                    # f108 has any digit
                    if any(is_digit_char(ch) for ch in w):
                        key = ("has_digit", t)
                        self.feature_rep_dict["f108"][key] = self.feature_rep_dict["f108"].get(key, 0) + 1
                    # f109 is all digits
                    if is_numeric_word(w):
                        key = ("is_numeric", t)
                        self.feature_rep_dict["f109"][key] = self.feature_rep_dict["f109"].get(key, 0) + 1
                    # f110 has any uppercase
                    if any(ch.isupper() for ch in w):
                        key = ("has_upper", t)
                        self.feature_rep_dict["f110"][key] = self.feature_rep_dict["f110"].get(key, 0) + 1
                    # f111 initial capital
                    if w and w[0].isupper():
                        key = ("init_cap", t)
                        self.feature_rep_dict["f111"][key] = self.feature_rep_dict["f111"].get(key, 0) + 1
                    # f112 all caps
                    if w.isupper():
                        key = ("all_caps", t)
                        self.feature_rep_dict["f112"][key] = self.feature_rep_dict["f112"].get(key, 0) + 1
                    # f113 number-noun
                    head, sep, tail = w.partition("-")
                    if head.isdigit() and sep == "-" and not tail.isdigit():
                        key = ("num_noun", t)
                        self.feature_rep_dict["f113"][key] = self.feature_rep_dict["f113"].get(key, 0) + 1

                    # ── G2–G5 (f114–f117) ──────────────────────────────────────────
                    # f114: capital-start
                    if w and w[0].isupper() and w[0].isalpha():
                        key = ("capital_start", t)
                        self.feature_rep_dict["f114"][key] = self.feature_rep_dict["f114"].get(key, 0) + 1
                    # f115: init-cap & first position
                    if w and w[0].isupper() and i == 0:
                        key = ("initcap_first", t)
                        self.feature_rep_dict["f115"][key] = self.feature_rep_dict["f115"].get(key, 0) + 1
                    # f116: init-cap & not first
                    if w and w[0].isupper() and i != 0:
                        key = ("initcap_notfirst", t)
                        self.feature_rep_dict["f116"][key] = self.feature_rep_dict["f116"].get(key, 0) + 1
                    # f117: all-caps word
                    if w.isupper():
                        key = ("all_caps_word", t)
                        self.feature_rep_dict["f117"][key] = self.feature_rep_dict["f117"].get(key, 0) + 1

                    # ——— f132: word‐shape + tag ———
                    shape = word_shape(w)
                    key = (shape, t)
                    self.feature_rep_dict["f132"][key] = self.feature_rep_dict["f132"].get(key, 0) + 1

                    # ——— f_char3 / f_char4: character n‐grams ———
                    for n, feat_name in ((3, "f_char3"), (4, "f_char4")):
                        for j in range(len(w) - n + 1):
                            gram = w[j:j + n]
                            key = (gram, t)
                            self.feature_rep_dict[feat_name][key] = self.feature_rep_dict[feat_name].get(key, 0) + 1

                    # ——— f_wordbigram_prev / f_wordbigram_next ———
                    if i >= 1:
                        prev_word = words_tags[i - 1][0]
                        key = ((prev_word, w), t)
                        self.feature_rep_dict["f_wordbigram_prev"][key] = \
                            self.feature_rep_dict["f_wordbigram_prev"].get(key, 0) + 1
                    if i < len(words_tags) - 1:
                        next_word = words_tags[i + 1][0]
                        key = ((w, next_word), t)
                        self.feature_rep_dict["f_wordbigram_next"][key] = \
                            self.feature_rep_dict["f_wordbigram_next"].get(key, 0) + 1

                    # ——— f_tagshape: cross previous‐tag + current shape ———
                    prev_tag = words_tags[i - 1][1] if i >= 1 else "*"
                    bigram = (prev_tag, t)
                    key = (bigram, shape)
                    self.feature_rep_dict["f_tagshape"][key] = \
                        self.feature_rep_dict["f_tagshape"].get(key, 0) + 1

                    ### More Features - But Not Trained For The 95% ###

                    # ── f138: word‐shape ─────────────────────────────────────────
                    shape = word_shape(w)
                    key = (shape, t)
                    self.feature_rep_dict["f138"][key] = self.feature_rep_dict["f138"].get(key, 0) + 1

                    # ── f141: sentence‐position bucket ─────────────────────────
                    n = len(words_tags)
                    pos = i / float(n - 1) if n > 1 else 0.0
                    bucket = "start" if pos < 0.25 else ("end" if pos > 0.75 else "mid")
                    self.feature_rep_dict["f141"][(bucket, t)] = \
                        self.feature_rep_dict["f141"].get((bucket, t), 0) + 1

                    # ── f142: common‐word flag ────────────────────────────────────
                    # you’ll want to pick a threshold; e.g. any word count > 100
                    if self.global_word_counts[w] > 100:
                        key = ("common_word", t)
                        self.feature_rep_dict["f142"][key] = \
                            self.feature_rep_dict["f142"].get(key, 0) + 1

                    # f143 - word length
                    length = len(w)
                    if length < 4:
                        bucket = "short"
                    elif length <= 7:
                        bucket = "med"
                    else:
                        bucket = "long"
                    self.feature_rep_dict["f143"][(bucket, t)] = \
                        self.feature_rep_dict["f143"].get((bucket, t), 0) + 1

                    # ── f300: common‐verb suffixes ────────────────────────────────────────────
                    for suffix in ("ing", "ed", "en", "s", "es", "ies"):
                        if w.lower().endswith(suffix):
                            self.feature_rep_dict["f300"][(suffix, t)] = self.feature_rep_dict["f300"].get((suffix, t),
                                                                                                           0) + 1
                            break

                    # ── f309–f311: sentence‐position flags ───────────────────────────────────
                    is_second = (pp_word == "*") and (p_word != "*")
                    self.feature_rep_dict["f309"][(is_second, t)] = self.feature_rep_dict["f309"].get((is_second, t), 0) + 1

                    is_last = (n_word == "~")
                    self.feature_rep_dict["f310"][(is_last, t)] = self.feature_rep_dict["f310"].get((is_last, t), 0) + 1

                    is_middle = (p_word != "*") and (pp_word != "*") and (n_word != "~")
                    self.feature_rep_dict["f311"][(is_middle, t)] = self.feature_rep_dict["f311"].get((is_middle, t), 0) + 1

                    # ── f304–f307: prev/next word suffixes & prefixes ───────────────────────
                    for L in range(1, min(4, len(p_word)) + 1):
                        self.feature_rep_dict["f304"][(p_word[-L:], t)] = self.feature_rep_dict["f304"].get(
                            (p_word[-L:], t), 0) + 1
                        self.feature_rep_dict["f305"][(p_word[:L], t)] = self.feature_rep_dict["f305"].get((p_word[:L], t),
                                                                                                           0) + 1
                    for L in range(1, min(4, len(n_word)) + 1):
                        self.feature_rep_dict["f306"][(n_word[-L:], t)] = self.feature_rep_dict["f306"].get(
                            (n_word[-L:], t), 0) + 1
                        self.feature_rep_dict["f307"][(n_word[:L], t)] = self.feature_rep_dict["f307"].get((n_word[:L], t),
                                                                                                           0) + 1


                # ── histories (same as before) ───────────────────────────────────
                for prev2, prev1, curr, next1 in zip(padded, padded[1:], padded[2:], padded[3:]):
                    w_m2, t_m2 = prev2
                    w_m1, t_m1 = prev1
                    w_i, t_i = curr
                    w_p1, _ = next1
                    history = (w_i, t_i, w_m1, t_m1, w_m2, t_m2, w_p1)
                    self.histories.append(history)

                for prev2, prev1, curr, next1 in zip(
                        padded,  # [x0, x1, ... xn]
                        padded[1:],  # [x1, x2, ... ,xn]
                        padded[2:], # [x2, x3, ..., xn]
                        padded[3:] # [x3, x4, ..., xn]
                ):
                    # unpack tuples
                    w_m2, t_m2 = prev2
                    w_m1, t_m1 = prev1
                    w_i, t_i = curr
                    w_p1, _ = next1

                    # f‐style “history” tuple:
                    history = (w_i, t_i,
                               w_m1, t_m1,
                               w_m2, t_m2,
                               w_p1)

                    self.histories.append(history)





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

def represent_input_with_features(history: Tuple, dict_of_dicts: Dict[str, Dict[Tuple[str, str], int]])\
        -> List[int]:
    """
        Extract feature vector in per a given history
        @param history: tuple{c_word, c_tag, p_word, p_tag, pp_word, pp_tag, n_word}
        @param dict_of_dicts: a dictionary of each feature and the index it was given
        @return a list with all features that are relevant to the given history
    """
    c_word = history[0]
    c_tag = history[1]
    c_word, c_tag, p_word, p_tag, pp_word, pp_tag, n_word = history

    features = []

    # f100
    if (c_word, c_tag) in dict_of_dicts["f100"]:
        features.append(dict_of_dicts["f100"][(c_word, c_tag)])


    # f114: capital‐start
    if ("capital_start", c_tag) in dict_of_dicts["f114"]:
        features.append(dict_of_dicts["f114"][("capital_start", c_tag)])

    # f115: initcap_first
    if ("initcap_first", c_tag) in dict_of_dicts["f115"]:
        features.append(dict_of_dicts["f115"][("initcap_first", c_tag)])

    # f116: initcap_notfirst
    if ("initcap_notfirst", c_tag) in dict_of_dicts["f116"]:
        features.append(dict_of_dicts["f116"][("initcap_notfirst", c_tag)])

    # f117: all_caps_word
    if ("all_caps_word", c_tag) in dict_of_dicts["f117"]:
        features.append(dict_of_dicts["f117"][("all_caps_word", c_tag)])

    # ——— f132: word‐shape
    shape = word_shape(c_word)
    idx = dict_of_dicts["f132"].get((shape, c_tag))
    if idx is not None:
        features.append(idx)

    # ——— f_char3 / f_char4
    for n, feat_name in ((3, "f_char3"), (4, "f_char4")):
        for j in range(len(c_word) - n + 1):
            gram = c_word[j:j + n]
            idx = dict_of_dicts[feat_name].get((gram, c_tag))
            if idx is not None:
                features.append(idx)

    # ——— word‐bigrams
    if p_word != "*":
        idx = dict_of_dicts["f_wordbigram_prev"].get(((p_word, c_word), c_tag))
        if idx is not None: features.append(idx)
    if n_word != "~":
        idx = dict_of_dicts["f_wordbigram_next"].get(((c_word, n_word), c_tag))
        if idx is not None: features.append(idx)



    # ——— tag×shape
    idx = dict_of_dicts["f_tagshape"].get(((p_tag, c_tag), shape))
    if idx is not None:
        features.append(idx)

    ### More Features That Hasn't Been Used For Training 95% ###

    # f138: word‐shape
    shape = word_shape(c_word)
    idx = dict_of_dicts["f138"].get((shape, c_tag))
    if idx is not None:
        features.append(idx)

    # f141: sentence‐position bucket
    # (we need sentence length & position: easiest is to include `bucket` in the history tuple,
    #  but if you can’t, you can approximate using the two PAD tags:
    is_first = (pp_tag == "*" and p_tag == "*")
    is_last = (n_word == "~")
    if is_first:
        bucket = "start"
    elif is_last:
        bucket = "end"
    else:
        bucket = "mid"
    idx = dict_of_dicts["f141"].get((bucket, c_tag))
    if idx is not None:
        features.append(idx)

    # f142: common‐word flag
    # you’ll need access to the same `global_word_counts` threshold; simplest is to pass
    # a set `common_words` into your tagger or store it on feature2id.
    # If you have `common_words`, then:
    if c_word in common_words:
        idx = dict_of_dicts["f142"].get(("common_word", c_tag))
        if idx is not None:
            features.append(idx)


    # f143
    length = len(c_word)
    if length < 4:
        bucket = "short"
    elif length <= 7:
        bucket = "med"
    else:
        bucket = "long"
    idx = dict_of_dicts["f143"].get((bucket, c_tag))
    if idx is not None: features.append(idx)

    # f304–f307
    for fc, neigh in [("f304", pp_word), ("f305", pp_word[:1]), ("f306", n_word[-1:]), ("f307", n_word[:1])]:
        idx = dict_of_dicts[fc].get((neigh, c_tag))
        if idx is not None: features.append(idx)

    # f309–311
    for fc, cond in [
        ("f309", (pp_word == "*" and p_word != "*")),
        ("f310", (n_word == "~")),
        ("f311", (p_word != "*" and pp_word != "*" and n_word != "~"))
    ]:
        idx = dict_of_dicts[fc].get((cond, c_tag))
        if idx is not None: features.append(idx)

    # # --- f118 — is_number
    # idx = dict_of_dicts["f118"].get(("is_number", c_tag))
    # if idx is not None:
    #     features.append(idx)

    # # --- f119, f120, f121 — unknown-word features
    # unk_keys = [
    #     (c_word, c_tag),
    #     (p_tag, c_tag),
    #     (pp_tag, p_tag, c_tag)
    # ]
    # for i, key in enumerate(unk_keys, start=119):
    #     idx = dict_of_dicts[f"f{i}"].get(key)
    #     if idx is not None:
    #         features.append(idx)
    #
    # # replace your whole G8 lookup with this
    # for offset, label in enumerate(
    #         ["g8_1", "g8_2", "g8_3", "g8_4", "g8_5",
    #          "g8_6", "g8_7", "g8_8", "g8_9", "g8_10"],
    #         start=122
    # ):
    #     feat_class = f"f{offset}"  # 122..131
    #     idx = dict_of_dicts[feat_class].get((label, c_tag))
    #     if idx is not None:
    #         features.append(idx)

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

