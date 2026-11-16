import os
import time
import json
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

def run_closed_book_baseline(test_path="data/processed/test.jsonl", output_path="results/baseline_predictions.jsonl", max_samples=None):
    print(f"--- Step 3: Running Closed-Book Baseline ({MODEL_ID}) ---")
    
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test split not found at {test_path}. Run src/data_prep.py first.")
        
    test_df = pd.read_json(test_path, lines=True)
    if max_samples:
        test_df = test_df.head(max_samples)
    print(f"Loaded {len(test_df)} test questions.")
    
    # Load tokenizer and model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None
    )
    if device == "cpu":
        model.to(device)
    model.eval()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    results = []
    
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Generating Baseline Answers"):
        question = row["question"]
        messages = [
            {"role": "system", "content": "You are an Australian legal research assistant. Answer the following question accurately and concisely. Note that this is general legal information, not legal advice."},
            {"role": "user", "content": question}
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        start_time = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        latency = time.perf_counter() - start_time
        
        generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        prediction = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        
        results.append({
            "question": question,
            "reference_answer": row["answer"],
            "predicted_answer": prediction,
            "latency_sec": round(latency, 4)
        })
        
    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
            
    avg_latency = sum(r["latency_sec"] for r in results) / len(results)
    print(f"\nSaved {len(results)} baseline predictions to {output_path}")
    print(f"Average latency per question: {avg_latency:.3f}s")

if __name__ == "__main__":
    run_closed_book_baseline()
