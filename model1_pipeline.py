# #!/usr/bin/env python3
# import os
# import pickle
# import numpy as np
# from collections import OrderedDict
#
# from preprocessing import preprocess_train, read_test
# from optimization   import get_optimal_vector
# from inference      import tag_all_test
# from evaluate       import predict_all, word_accuracy
#
# #
# # ── CONFIG ────────────────────────────────────────────────────────────────────
# #
# TRAIN1       = "data/train1.wtag"
# DEV1         = "data/test1.wtag"
# COMP1        = "data/comp1.words"
# OUT_PRED1    = "predictions1_noextctx.wtag"
# TRAIN1         = "data/train1.wtag"
#
#
# THRESH       = 1        # keep features seen ≥1
# LAM          = 2.0      # your chosen λ
# PRUNE_TO     = 10000    # final budget of 10 000
#
# #
# # ──────────────────────────────────────────────────────────────────────────────
# #
#
#
#
# def prune_and_retrain(stats, f2i, w, K, lam, out_path):
#     """Prune to top-K by |w|, rebuild matrices, retrain (warm-start)."""
#     abs_w = np.abs(w)
#     topk  = np.argsort(abs_w)[-K:]
#     mask  = np.zeros_like(w, bool); mask[topk] = True
#
#     # rebuild feature→idx
#     new_map, new_i = {}, 0
#     for fc, fmap in f2i.feature_to_idx.items():
#         new_map[fc] = OrderedDict()
#         for feat, old_i in fmap.items():
#             if mask[old_i]:
#                 new_map[fc][feat] = new_i
#                 new_i += 1
#
#     f2i.feature_to_idx   = new_map
#     f2i.n_total_features = new_i
#     f2i.calc_represent_input_with_features()
#
#     # warm-start vector
#     w0 = np.zeros(new_i)
#     w0[:min(len(w), new_i)] = w[:new_i]
#
#     w_pruned = get_optimal_vector(stats, f2i, lam, out_path, init_weights=w0)[0]
#     return w_pruned
#
# if __name__=="__main__":
#     # 1) build all features & drop the extended‐context ones
#     print("[1] preprocess + drop ext‐ctx classes")
#     stats, f2i = preprocess_train(TRAIN1, threshold=THRESH)
#     f2i.calc_represent_input_with_features()
#
#     # 2) train full 100 000-dim MEMM
#     print("[2] training full 100k-dim model (λ=2.0)")
#     w_full = get_optimal_vector(stats, f2i, LAM, "w1_full_noextctx.pkl")[0]
#
#     # 3) prune to 10 000 and retrain
#     print(f"[3] pruning to {PRUNE_TO} + retrain")
#     w_10k = prune_and_retrain(stats, f2i, w_full, PRUNE_TO, LAM, "w1_10k_noextctx.pkl")
#
#     # 4) tag comp1.words
#     print("[4] tagging competition file →", OUT_PRED1)
#     tag_all_test(COMP1, w_10k, f2i, OUT_PRED1)
#
#     # 5) evaluate on test1.wtag
#     print("[5] evaluating on", DEV1)
#     _, gold, pred = predict_all(DEV1, f2i, w_10k)
#     acc = word_accuracy(gold, pred)
#     print(f"→ Word‐level accuracy without ext‐ctx = {acc*100:.2f}%")

#!/usr/bin/env python3
import pickle
import numpy as np

from preprocessing import preprocess_train
from evaluate      import predict_all, word_accuracy

TRAIN1   = "data/train1.wtag"
PKL_FILE = "w1_10k_noextctx.pkl"
THRESH   = 1
PRUNE_TO = 10000

# 1) load what's in the pickle
with open(PKL_FILE, "rb") as f:
    data = pickle.load(f)

# 2) unpack to get the actual numpy‐vector
#    common patterns:
if isinstance(data, tuple) and isinstance(data[0], (list, tuple, np.ndarray)):
    # sometimes it's ((w,), f2i) or (w_vector,)
    candidate = data[0]
    if isinstance(candidate, (list, tuple)) and len(candidate)==1:
        w_10k = np.asarray(candidate[0])
    else:
        w_10k = np.asarray(candidate)
else:
    # assume it's already the raw array
    w_10k = np.asarray(data)

# 3) rebuild exactly the same feature‐to‐idx map
stats, f2i = preprocess_train(TRAIN1, threshold=THRESH)
f2i.calc_represent_input_with_features()

# 4) prune the mapping down so indices line up
f2i.prune_top_k_by_weight(w_10k, PRUNE_TO)

# 5) evaluate on train
_, gold_tr, pred_tr = predict_all(TRAIN1, f2i, w_10k)
acc_tr = word_accuracy(gold_tr, pred_tr)
print(f"→ Word‐level train accuracy = {acc_tr*100:.2f}%")
