import json
import textstat
import evaluate
import numpy as np
import argparse
from sklearn.metrics import accuracy_score, f1_score
from BARTScore.bart_score import BARTScorer

rouge = evaluate.load("rouge")
bertscore = evaluate.load("evaluate-metric/bertscore")
bart_scorer = BARTScorer(device='cuda', checkpoint="facebook/bart-large-cnn")
bart_scorer.load(path="bart_score.pth")

def process_json(json_file, format_type):
    """
    Process a JSON file to extract predictions and true labels.
    
    Parameters:
    - json_file: The path to the JSON file.
    - format_type: The format of the JSON file ('llama' or 'gpt').
    
    Returns:
    - golds: A list of true labels.
    - preds: A list of predictions.
    """
    try:
        with open(json_file, 'r') as file:
            data = json.load(file)

        if format_type == 'llama':
            golds = [item['truth'] for item in data]
            preds = [item['logit_0'] for item in data]
        elif format_type == 'gpt':
            golds = data['truth']
            preds = data['preds']
        else:
            raise ValueError(f"Unsupported format type: {format_type}")

        return golds, preds
    except Exception as e:
        print(f"Error processing JSON file: {e}")
        return [], []
    
    
def compute_scores(predictions, true_labels, task):
    """
    Compute evaluation metrics for predictions, with optional adjustments for specific tasks.
    
    Parameters:
    - predictions: A list of prediction strings.
    - true_labels: A list of true label strings.
    - task: The type of task: summarization, simplification, QA, MLC, NER, RE.
    
    Returns:
    - A dictionary containing all computed scores.
    """

    scores = {}

    if task == "summarization" or task == "simplification":

        # Compute ROUGE scores
        rouge_results = rouge.compute(predictions=predictions, references=true_labels)
        rouge_scores = {
            'rouge1': round(rouge_results["rouge1"], 4), 
            'rouge2': round(rouge_results["rouge2"], 4), 
            'rougeL': round(rouge_results["rougeL"], 4)
        }
        
        # Compute FKG and DCR scores
        fkg = round(np.mean([textstat.flesch_kincaid_grade(pred) for pred in predictions]), 4)
        dcr = round(np.mean([textstat.dale_chall_readability_score(pred) for pred in predictions]), 4)
        
        # Compute BERT scores
        bert_results = bertscore.compute(predictions=predictions, references=true_labels, model_type="bert-base-multilingual-cased")
        bert_score_f1 = round(sum(bert_results["f1"]) / len(bert_results["f1"]), 4)
        
        # Compute BART scores
        bart_results = bart_scorer.score(srcs=list(predictions), tgts=list(true_labels), batch_size=8)
        bart_score = round(sum(bart_results) / len(bart_results), 4)
        
        # Combine all scores into a single dictionary
        scores = {
            'ROUGE': rouge_scores,
            'FKG': fkg,
            'DCR': dcr,
            'BERT F1': bert_score_f1,
            'BART': bart_score
        }
        
    elif task == "QA":

        accuracy = accuracy_score(true_labels, predictions)
        macro_f1 = f1_score(true_labels, predictions, average='macro')

        # Combine all scores into a single dictionary
        scores = {
            'Accuracy': round(accuracy, 4),
            'Macro F1': round(macro_f1, 4)
        }

    elif task == "MLC" or task == "RE":

        macro_f1 = f1_score(true_labels, predictions, average='macro')
        weighted_f1 = f1_score(true_labels, predictions, average='weighted')

        # Combine all scores into a single dictionary
        scores = {
            'Macro F1': round(macro_f1, 4),
            'Wighted F1': round(weighted_f1, 4)
        }

    else:
        print(f"Warning: Task '{task}' not recognized. No scores computed.")

    return scores


def main():

    parser = argparse.ArgumentParser(description="Process JSON data and compute evaluation scores.")
    parser.add_argument("json_file", type=str, help="Path to the JSON file.")
    parser.add_argument("format_type", type=str, help="Format of the JSON file ('llama' or 'gpt').")
    parser.add_argument("task", type=str, help="Type of task (e.g., summarization, QA).")
    
    args = parser.parse_args()
    
    golds, preds = process_json(args.json_file, args.format_type)
    scores = compute_scores(preds, golds, args.task)
    print(scores)

if __name__ == "__main__":
    main()
