import spacy
from inference import memm_viterbi
from collections import namedtuple

# 1) load spaCy once (we only need its tokenizer)
_nlp = _nlp = spacy.load("en_core_web_trf", disable=["parser","ner","lemmatizer"])

def tag_with_spacy_tokens(lines, weights, feature2id):
    """
    lines: iterable of raw text lines (no tags)
    returns: list of lists of (token, predicted_tag)
    """
    all_preds = []
    for line in lines:
        line = line.rstrip("\n")
        if not line:
            all_preds.append([])  # blank sentence
            continue

        # 2) get spaCy tokens
        doc  = _nlp(line)
        toks = [tok.text for tok in doc]

        # 3) prepare padded sentence for your Viterbi
        sent = ["*", "*"] + toks + ["~"]

        # 4) run your memm
        hist = memm_viterbi(sent, weights, feature2id)

        # 5) drop the two '*' and the final '~'
        pred_tags = hist[2:-1]

        # 6) zip back
        all_preds.append(list(zip(toks, pred_tags)))

    return all_preds
