import pandas as pd

# file paths
spacy_path = "comp1.silver.spacy"
pred_path = "predictions.wtag"

# read spaCy output: one line per sentence
with open(spacy_path, encoding='utf8') as f:
    spacy_sents = [line.strip() for line in f if line.strip()]

# read your predictions: reconstruct sentences by empty lines
pred_sents = []
current = []
with open(pred_path, encoding='utf8') as f:
    for line in f:
        line = line.strip()
        if not line:
            if current:
                pred_sents.append(" ".join(current))
                current = []
        else:
            current.append(line)
    if current:
        pred_sents.append(" ".join(current))

assert len(spacy_sents) == len(pred_sents), \
    f"Still a mismatch: {len(spacy_sents)} vs {len(pred_sents)} sentences"

total = correct = 0
mismatches = []

for i, (s_line, p_line) in enumerate(zip(spacy_sents, pred_sents)):
    s_tokens = s_line.split()
    p_tokens = p_line.split()
    assert len(s_tokens) == len(p_tokens), f"Token count mismatch in sentence {i}"
    for s_tok, p_tok in zip(s_tokens, p_tokens):
        sw, st = s_tok.rsplit("_",1)
        pw, pt = p_tok.rsplit("_",1)
        assert sw == pw, f"Word mismatch {sw} vs {pw}"
        total += 1
        if st == pt:
            correct += 1
        else:
            mismatches.append((i, sw, st, pt))

accuracy = correct/total*100
print(f"Accuracy vs. spaCy silver: {accuracy:.2f}%")
# Optionally view a few mismatches
print("Sample mismatches:", mismatches[:10])
