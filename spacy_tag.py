# import re
# import spacy
# from spacy.tokenizer import Tokenizer
#
# # 1) load the transformer model
# nlp = spacy.load("en_core_web_trf")
#
# # 2) override the tokenizer so it just splits on whitespace
# #    i.e. any sequence of non‐space chars is one token
# nlp.tokenizer = Tokenizer(
#     nlp.vocab,
#     token_match=re.compile(r"\S+").match
# )
#
# in_path  = "data/comp1.words"
# out_path = "comp1.silver.spacy"
#
# with open(in_path, encoding="utf8") as fin, open(out_path, "w", encoding="utf8") as fout:
#     for line in fin:
#         line = line.rstrip("\n")
#         if not line:
#             fout.write("\n")
#             continue
#         doc = nlp(line)
#         # now doc has exactly one token per original whitespace‐piece
#         tagged = [f"{tok.text}_{tok.tag_}" for tok in doc]
#         fout.write(" ".join(tagged) + "\n")


import spacy
from preprocessing import read_test, represent_input_with_features
from inference  import memm_viterbi
from tqdm       import tqdm

# 1) load spaCy just once
_nlp = spacy.load("en_core_web_sm", disable=["parser","ner","lemmatizer"])

def tag_all_test_spacy_tokenized(test_path, w, feature2id, predictions_path):
    out = open(predictions_path, "w", encoding="utf8")

    with open(test_path, encoding="utf8") as fin:
        for line in fin:
            line = line.rstrip("\n")
            if not line:
                out.write("\n")
                continue

            # 2) use spaCy to get the token sequence
            doc = _nlp(line)
            toks = [tok.text for tok in doc]

            # 3) build the padded sentence the way your Viterbi expects:
            sent = ["*", "*"] + toks + ["~"]

            # 4) run your Viterbi
            hist = memm_viterbi(sent, w, feature2id)

            # 5) drop the two '*' and the final '~'
            pred_tags = hist[2:-1]

            # 6) write out TOK_TAG
            out.write(" ".join(f"{tok}_{tag}" for tok, tag in zip(toks, pred_tags)))
            out.write("\n")

    out.close()
