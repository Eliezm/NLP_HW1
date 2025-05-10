#!/usr/bin/env python
from tqdm import tqdm

def load_sentences(path):
    """
    Reads a file with one sentence per line, tokens WORD_TAG separated by spaces.
    Returns a list of lists of (word, tag) pairs.
    """
    sents = []
    with open(path, encoding="utf8") as f:
        for line in f:
            line = line.strip()
            if not line:
                # skip empty
                continue
            toks = []
            for wt in line.split():
                try:
                    w, t = wt.rsplit("_", 1)
                except ValueError:
                    raise ValueError(f"Bad token format (no underscore?): '{wt}' in line: {line}")
                toks.append((w, t))
            sents.append(toks)
    return sents

def main():
    yours_path = "predictions.wtag"
    spacy_path = "comp1.silver.spacy"

    yours = load_sentences(yours_path)
    spacy = load_sentences(spacy_path)

    assert len(yours) == len(spacy), \
        f"Sentence count mismatch: you={len(yours)} vs spacy={len(spacy)}"

    total = correct = 0
    mismatches = []

    for i, (ys, ss) in enumerate(zip(yours, spacy)):
        assert len(ys) == len(ss), \
            f"Token count mismatch sentence {i}: you={len(ys)} vs spacy={len(ss)}"
        for (wy, ty), (ws, ts) in zip(ys, ss):
            assert wy == ws, f"Word mismatch in sent {i}: you='{wy}' vs spacy='{ws}'"
            total += 1
            if ty == ts:
                correct += 1
            else:
                # record a few examples
                if len(mismatches) < 20:
                    mismatches.append((i, wy, ty, ts))

    acc = correct / total * 100
    print(f"Token-level accuracy vs. spaCy silver: {acc:.2f}%")
    print("Example mismatches (sent idx, word, your_tag, spacy_tag):")
    for m in mismatches:
        print(m)

if __name__ == "__main__":
    main()
