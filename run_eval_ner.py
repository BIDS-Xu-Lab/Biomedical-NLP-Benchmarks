import argparse
import ast

def calculate_scores(gold_span, predict_span, exact=True):
    """
    Calculate precision, recall, and F1 score along with some count metrics for a single instance.

    Parameters:
        gold_span (list): List of (start, end) tuples representing the gold standard spans.
        predict_span (list): List of (start, end) tuples representing the predicted spans.
        exact (bool): If True, returns a condensed result string.

    Returns:
        str: Formatted string containing computed scores and counts.
    """
    right = 0
    right_gold = 0
    right_predict = 0

    # Count exact matches
    for s1, e1 in gold_span:
        for s2, e2 in predict_span:
            if s1 == s2 and e1 == e2:
                right += 1
                break

    # Count overlapping spans from the perspective of gold spans
    for s1, e1 in gold_span:
        for s2, e2 in predict_span:
            if s1 <= e2 and e1 >= s2:
                right_gold += 1
                break

    # Count overlapping spans from the perspective of predicted spans
    for s1, e1 in predict_span:
        for s2, e2 in gold_span:
            if s1 <= e2 and e1 >= s2:
                right_predict += 1
                break

    # Calculate precision, recall, and F1 for exact matches
    if predict_span:
        p = float(right) / len(predict_span)
    else:
        p = 0.0

    if gold_span:
        r = float(right) / len(gold_span)
    else:
        r = 0.0

    if p == 0.0 or r == 0.0:
        f = 0.0
    else:
        f = 2 * p * r / (p + r)

    # Calculate adjusted precision, recall, and F1 based on overlapping counts
    if predict_span:
        p2 = float(right_gold) / len(predict_span)
    else:
        p2 = 0.0

    if gold_span:
        r2 = float(right_predict) / len(gold_span)
    else:
        r2 = 0.0

    if p2 == 0.0 or r2 == 0.0:
        f2 = 0.0
    else:
        f2 = 2 * p2 * r2 / (p2 + r2)

    if not exact:
        return '%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%d\t%d\t%d\t%d\t%d' % (
            p, r, f, p2, r2, f2, right, right_gold, right_predict,
            len(predict_span), len(gold_span)
        )
    else:
        return '%.3f\t%.3f\t%.3f\t%d\t%d\t%d' % (p, r, f, right, len(predict_span), len(gold_span))

def process_text(file_path):
    """
    Reads a file containing a string representation of a Python object and converts it into the object.

    Parameters:
        file_path (str): Path to the text file.

    Returns:
        object: The evaluated Python object from the file.
    """
    with open(file_path, 'r') as file:
        content = file.read()
        data = ast.literal_eval(content)
    return data

def compute_score(gold_list, pred_list):
    """
    Compute aggregate precision, recall, and F1 score across all instances.

    Parameters:
        gold_list (list): List of gold spans for each instance.
        pred_list (list): List of predicted spans for each instance.

    Returns:
        tuple: Overall precision, recall, and F1 score.
    """
    TP = 0       # True positives (exact match counts)
    Pred_P = 0   # Total predicted positives (counts per instance)
    True_P = 0   # Total true positives (counts per instance)

    for i in range(len(gold_list)):
        gold_span = gold_list[i]
        predict_span = pred_list[i]

        scores = calculate_scores(gold_span, predict_span, exact=True)
        scores_list = scores.split('\t')

        TP_sent = float(scores_list[3])
        Pred_P_sent = float(scores_list[4])
        True_P_sent = float(scores_list[5])

        TP += TP_sent
        Pred_P += Pred_P_sent
        True_P += True_P_sent

    if Pred_P == 0:
        P = 0
    else:
        P = round(TP / Pred_P, 4)

    if True_P == 0:
        R = 0
    else:
        R = round(TP / True_P, 4)

    if P + R == 0:
        F1 = 0
    else:
        F1 = round(2 * P * R / (P + R), 4)

    return P, R, F1

def main():
    parser = argparse.ArgumentParser(description="Compute evaluation metrics from two txt files containing span lists.")
    parser.add_argument("--gold_file", type=str, required=True, help="Path to the gold span file (e.g., NCBI_Disease_gpt3.5_5s_gold_span.txt)")
    parser.add_argument("--pred_file", type=str, required=True, help="Path to the predicted span file (e.g., NCBI_Disease_gpt3.5_5s_pre_span.txt)")
    args = parser.parse_args()

    gold_list = process_text(args.gold_file)
    pred_list = process_text(args.pred_file)
    assert len(gold_list) == len(pred_list), "Mismatch in the number of instances between files."
    P, R, F1 = compute_score(gold_list, pred_list)
    print("Precision:", P)
    print("Recall:", R)
    print("F1 Score:", F1)

if __name__ == '__main__':
    main()