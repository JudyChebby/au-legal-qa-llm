import os
import json
import pandas as pd
from rouge_score import rouge_scorer

def compute_token_f1(prediction, reference):
    """Computes token-level F1 score between prediction and reference text."""
    pred_tokens = prediction.lower().split()
    ref_tokens = reference.lower().split()
    
    if not pred_tokens or not ref_tokens:
        return 0.0
        
    common = set(pred_tokens) & set(ref_tokens)
    if not common:
        return 0.0
        
    num_same = sum(min(pred_tokens.count(t), ref_tokens.count(t)) for t in common)
    precision = num_same / len(pred_tokens)
    recall = num_same / len(ref_tokens)
    
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)

def evaluate_all_results(results_dir="results", output_report="results/evaluation_summary.json"):
    print("--- Running Unified Evaluation Across All Setups ---")
    
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    summary = {}
    
    prediction_files = {
        "Closed-Book Baseline": "baseline_predictions.jsonl",
        "RAG Pipeline": "rag_predictions.jsonl",
        "QLoRA Fine-Tuned": "qlora_predictions.jsonl"
    }
    
    for model_name, filename in prediction_files.items():
        path = os.path.join(results_dir, filename)
        if not os.path.exists(path):
            print(f"Warning: {path} not found. Skipping {model_name}.")
            continue
            
        preds = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                preds.append(json.loads(line))
                
        f1_scores = []
        rouge_scores = []
        latencies = []
        retrieval_hits = []
        
        for p in preds:
            ref = p["reference_answer"]
            pred = p["predicted_answer"]
            
            f1_scores.append(compute_token_f1(pred, ref))
            rouge_scores.append(scorer.score(ref, pred)['rougeL'].fmeasure)
            latencies.append(p.get("latency_sec", 0.0))
            if "retrieval_hit" in p:
                retrieval_hits.append(p["retrieval_hit"])
                
        metrics = {
            "num_samples": len(preds),
            "mean_token_f1": round(sum(f1_scores) / len(f1_scores), 4) if f1_scores else 0.0,
            "mean_rouge_l": round(sum(rouge_scores) / len(rouge_scores), 4) if rouge_scores else 0.0,
            "mean_latency_sec": round(sum(latencies) / len(latencies), 4) if latencies else 0.0,
            "retrieval_recall_at_k": round(sum(retrieval_hits) / len(retrieval_hits), 4) if retrieval_hits else None
        }
        summary[model_name] = metrics
        print(f"\nResults for {model_name}:")
        print(json.dumps(metrics, indent=2))
        
    os.makedirs(results_dir, exist_ok=True)
    with open(output_report, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved evaluation summary to {output_report}")

if __name__ == "__main__":
    evaluate_all_results()
