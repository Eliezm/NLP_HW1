import os
import pickle
from preprocessing import preprocess_train
from optimization import get_optimal_vector
from inference import tag_all_test
from evaluate import predict_all, word_accuracy


# def main():
#     threshold = 2
#     lam = 1
#
#     train_path = "data/train1.wtag"
#     test_path = "data/comp1.words"
#
#     weights_path = 'weights.pkl'
#     predictions_path = 'predictions.wtag'
#
#     statistics, feature2id = preprocess_train(train_path, threshold)
#     # statistics, feature2id = preprocess_train(train_path, threshold, top_k=10000)
#     # statistics, feature2id = preprocess_train(
#     #     train_path,
#     #     threshold= 2,  # your original raw‐count threshold
#     #     count_thresh=5,  # drop features seen fewer than 5×
#     #     top_k=10000  # keep only the 10 000 highest‐χ² features
#     # )
#
#     get_optimal_vector(statistics=statistics, feature2id=feature2id, weights_path=weights_path, lam=lam)
#
#     # get_optimal_vector(statistics=statistics,
#     #                    feature2id=feature2id,
#     #                    weights_path=weights_path,
#     #                    lam=lam)
#
#     with open(weights_path, 'rb') as f:
#         optimal_params, feature2id = pickle.load(f)
#     pre_trained_weights = optimal_params[0]
#
#     print(pre_trained_weights)
#     tag_all_test(test_path, pre_trained_weights, feature2id, predictions_path)
#
#     print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
#     print()
#     words, gold, pred = predict_all("data/test1.wtag", feature2id, pre_trained_weights)
#     acc = word_accuracy(gold, pred)
#     print(f"Word-level accuracy: {acc * 100:.2f}%")
#
#     # cm_evaluate(gold, pred)

import pickle

def inspect_weights(path="weights.pkl"):
    with open(path, "rb") as f:
        (optimal_params, feature2id) = pickle.load(f)
    # optimal_params is the tuple returned by fmin_l_bfgs_b:
    #   optimal_params[0] is the weight‐vector
    w = optimal_params[0]
    print(f"Number of learned parameters (len(w)): {len(w)}")
    print(f"feature2id.n_total_features:        {feature2id.n_total_features}")


def main():
    # inspect_weights("weights.pkl")

    threshold        = 1
    lam              = 3
    train_path       = "data/train1.wtag"
    test_path        = "data/comp1.words"
    weights_path     = "weights.pkl"
    predictions_path = "predictions.wtag"
    pruned_weights_path = "weights_pruned.pkl"

    # 1) Load or train full model
    # if os.path.exists(weights_path):
    #     print(f"[main] Loading pretrained model from {weights_path}")
    #     with open(weights_path, "rb") as f:
    #         (optimal_params, feature2id) = pickle.load(f)
    #     # pull statistics back out of feature2id
    #     statistics = feature2id.feature_statistics
    #     w = optimal_params[0]
    #
    # else:
    print("[main] No existing weights found—training from scratch.")
    statistics, feature2id = preprocess_train(train_path, threshold)

    # initial training on full feature set
    optimal_params = get_optimal_vector(
        statistics=statistics,
        feature2id=feature2id,
        lam=lam,
        weights_path=weights_path
    )
    w = optimal_params[0]

    # ─── prune to top-K by |w|, then fine-tune ───
    # K = 10000
    # print(f"[main] Pruning to top {K} features by |w|…")
    # feature2id.prune_top_k_by_weight(w, K)
    # print(f"[main] {feature2id.n_total_features} features remain after pruning")
    #
    # # rebuild both small_matrix and big_matrix to match the pruned feature set
    # feature2id.calc_represent_input_with_features()
    #
    # print("[main] Fine-tuning on pruned feature set…")
    # # re-train (L-BFGS) starting from the pruned weights
    # optimal_params = get_optimal_vector(
    #     statistics=statistics,
    #     feature2id=feature2id,
    #     lam=lam,
    #     weights_path=pruned_weights_path,
    #     init_weights=w
    # )
    # w = optimal_params[0]

    # ─── Prune the 10k→5 000 & fine-tune ───
    # K2 = 5_000
    # print(f"[main] Pruning to top {K2} features by |w|…")
    # feature2id.prune_top_k_by_weight(w, K2)
    # print(f"[main] {feature2id.n_total_features} features remain after 5k-prune")
    # feature2id.calc_represent_input_with_features()
    # print("[main] Fine-tuning on 5k feature set…")
    # optimal_params = get_optimal_vector(
    #     statistics=statistics,
    #     feature2id=feature2id,
    #     lam=lam,
    #     weights_path=pruned_weights_path,
    #     init_weights=w
    # )
    # w = optimal_params[0]

    # 3) Tag the test set
    print("[main] Tagging test set…")
    tag_all_test(
        test_path=test_path,
        pre_trained_weights=w,
        feature2id=feature2id,
        predictions_path=predictions_path
    )
    print(f"[main] Predictions written to {predictions_path}")

    # 4) Evaluate (optional)
    print("[main] Evaluating on held-out test1.wtag…")
    words, gold, pred = predict_all("data/test1.wtag", feature2id, w)
    acc = word_accuracy(gold, pred)
    print(f"Word-level accuracy: {acc*100:.2f}%")


if __name__ == '__main__':
    main()


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

# import pickle
# import numpy as np
#
# def prune_by_weight_magnitude(feature2id, w, K=10000):
#     abs_w   = np.abs(w)
#     topk_idx = np.argsort(abs_w)[-K:]            # <-- these are the indices you keep
#     mask     = np.zeros_like(abs_w, bool)
#     mask[topk_idx] = True
#
#     # rebuild feature_to_idx as before...
#     new_map = {fc: OrderedDict() for fc in feature2id.feature_to_idx}
#     new_idx = 0
#     for fc, fmap in feature2id.feature_to_idx.items():
#         for feat_key, old_idx in fmap.items():
#             if mask[old_idx]:
#                 new_map[fc][feat_key] = new_idx
#                 new_idx += 1
#
#     feature2id.feature_to_idx   = new_map
#     feature2id.n_total_features = new_idx
#     feature2id.calc_represent_input_with_features()
#
#     return feature2id, topk_idx
#
#
# def main():
#     threshold = 1
#     lam = 3
#     train_path = "data/train1.wtag"
#     test_path  = "data/comp1.words"
#     weights_path     = "weights.pkl"
#     pruned_weights_p = "weights_pruned.pkl"
#     predictions_path = "predictions.wtag"
#
#     # 1) Preprocess & train full model
#     statistics, feature2id = preprocess_train(train_path, threshold)
#     get_optimal_vector(statistics, feature2id, weights_path, lam)
#
#     # 2) Load back the full model
#     with open(weights_path, "rb") as f:
#         (optimal_params, feature2id) = pickle.load(f)
#     w_full = optimal_params[0]
#
#     print(f"Full model: {feature2id.n_total_features} features")
#
#     # 3) Prune to top 10k by |w|
#     feature2id, topk_idx = prune_by_weight_magnitude(feature2id, w_full, K=10000)
#     print(f"Pruned model: {feature2id.n_total_features} features")
#
#     # 4a) (Optionally) re-train on the reduced set:
#     # get_optimal_vector(statistics, feature2id, pruned_weights_p, lam)
#     # with open(pruned_weights_p, "rb") as f:
#     #     (optimal_params, feature2id) = pickle.load(f)
#     # w = optimal_params[0]
#
#     # 4b) Or just keep using w_full but masked to top-K:
#     w = w_full[topk_idx]  # make sure to re-index w to match the new ordering above
#
#     # 5) Tag test with the pruned model
#     tag_all_test(test_path, w, feature2id, predictions_path)
#
# if __name__ == "__main__":
#     from preprocessing import preprocess_train
#     from optimization  import get_optimal_vector
#     from inference     import tag_all_test
#     from collections   import OrderedDict
#
#     main()