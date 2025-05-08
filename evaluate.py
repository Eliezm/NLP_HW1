from inference import memm_viterbi
from preprocessing import read_test

def predict_all(test_path, feature2id, weights):
    """
    Returns two parallel lists:
      - words:   a flat list of all words in test.wtag (in sentence order)
      - gold:    the corresponding gold tags
      - pred:    the tags your model predicted
    """
    sentences = read_test(test_path, tagged=True)   # returns [ (words, tags), ... ]
    all_words, all_gold, all_pred = [], [], []

    for words, gold_tags in sentences:
        # words includes ["*","*", w1, w2, ..., "~"]
        pred_tags = memm_viterbi(words, weights, feature2id)
        # drop the two leading "*" and the final "~"
        pred = pred_tags[2:-1]
        real_words = words[2:-1]
        all_words .extend(real_words)
        all_gold  .extend(gold_tags[2:-1])
        all_pred  .extend(pred)

    return all_words, all_gold, all_pred

def word_accuracy(gold, pred):
    correct = sum(g==p for g,p in zip(gold,pred))
    return correct / len(gold)
