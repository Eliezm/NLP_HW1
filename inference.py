from preprocessing import read_test, represent_input_with_features
from tqdm import tqdm
import math

def logsumexp(full_term: list) -> float:
    m = max(full_term)
    return m + math.log(sum(math.exp(x - m) for x in full_term))

def memm_viterbi(sentence: list,
                 pre_trained_weights: list,
                 feature2id) -> list:
    """
    Write your MEMM Viterbi implementation below
    You can implement Beam Search to improve runtime
    Implement q efficiently (refer to conditional probability definition in MEMM slides)

    מימשנו את אלגוריתם ויטרבי על פי הפסאודוקוד שניתן לנו בהרצאה, מבדיקה קצרה זמני הריצה של האלגוריתם להיסק על 1000 משפטים היו,
    בכל הרצה של המודל שלקחה לנו מס' דקות ואף יותר מ-10 דקות, מימשנו את הגרסה של ויטרבי עם beam search. ראינו שמעבר ל-K=5 אין שיפורים משמעותיים.
    מימוש האלגוריתם באופן יעיל תרם לקיצור זמני התיוג מרבע שעה למס' דקות, ולכן יכולנו לבדוק יותר וריאציות של המודל שלנו בפחות זמן.

    """
    K = 5

    all_tags = [t for t in feature2id.feature_statistics.tags if t != "~"]

    # start with the two padded '*' tags
    beam = [(0.0, ["*", "*"])]

    # iterate positions 2 .. len(sentence)-2
    for i in range(2, len(sentence)-1):
        word = sentence[i]
        next_word = sentence[i+1]

        candidates = []
        for log_p, hist in beam:
            # last two tags
            pp_tag, p_tag = hist[-2], hist[-1]

            scores = []
            for t_prime in all_tags:
                hist_tp = (
                    word, t_prime,
                    sentence[i-1], p_tag,
                    sentence[i-2], pp_tag,
                    next_word
                )
                feats = represent_input_with_features(
                    hist_tp,
                    feature2id.feature_to_idx,
                    feature2id.feature_statistics.common_words
                )

                score_tp = sum(pre_trained_weights[idx] for idx in feats)
                scores.append(score_tp)

            lse = logsumexp(scores)

            for idx, t in enumerate(all_tags):
                score_t = scores[idx]
                log_q = score_t - lse
                candidates.append((log_p + log_q, hist + [t]))

        beam = sorted(candidates, key=lambda x: x[0], reverse=True)[:K]

    beam = [(lp, hist + ["~"]) for lp, hist in beam]

    return beam[0][1]

def tag_all_test(test_path, pre_trained_weights, feature2id, predictions_path):
    tagged = "test" in test_path
    test = read_test(test_path, tagged=tagged)

    output_file = open(predictions_path, "a+")

    for sen in tqdm(test, total=len(test)):
        words = sen[0]
        # run Viterbi: hist = ['*','*', tag1, tag2, …, tagN, '~']
        hist = memm_viterbi(words, pre_trained_weights, feature2id)

        # real words are at positions 2 .. len(words)-2
        real_words = words[2:-1]
        pred_tags = hist[2:-1]

        # write word_tag for each real word
        for w, t in zip(real_words, pred_tags):
            output_file.write(f"{w}_{t} ")
        output_file.write("\n")
    output_file.close()