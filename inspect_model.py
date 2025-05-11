import pickle

# change this to the model you care about:
MODEL_PKL = "weights2_selfTrain_iter1.pkl"

with open(MODEL_PKL, "rb") as f:
    (optimal_params, feature2id) = pickle.load(f)

print(f"Loaded {MODEL_PKL}")
print(f"  • λ / regularization used:   {getattr(feature2id, 'lam', 'unknown')}")
print(f"  • feature‐count threshold:    {feature2id.threshold}")
print(f"  • total features in model:    {feature2id.n_total_features}\n")

print("Features per family:")
for feat_class, fmap in feature2id.feature_to_idx.items():
    print(f"  {feat_class:12s}: {len(fmap)}")

# if you want to see the actual feature keys for one family, uncomment:
# e.g. list the first 20 suffix‐features (f101):
# sample = list(feature2id.feature_to_idx["f101"].keys())[:20]
# print("\nSample f101 suffixes:", sample)
