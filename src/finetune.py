import os
import time
import json
import torch
import pandas as pd
from tqdm import tqdm
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_DIR = "models/qwen25_au_legal_qlora"

def format_as_messages(df):
    """Formats question-answer pairs into conversational messages for SFTTrainer."""
    records = []
    for _, row in df.iterrows():
        records.append({
            "messages": [
                {
                    "role": "system",
                    "content": "You are an Australian legal research assistant. Answer the following question accurately and concisely. Note that this is general legal information, not legal advice."
                },
                {"role": "user", "content": str(row["question"])},
                {"role": "assistant", "content": str(row["answer"])}
            ]
        })
    return Dataset.from_list(records)

def train_qlora(train_path="data/processed/train.jsonl", val_path="data/processed/val.jsonl", output_dir=ADAPTER_DIR):
    print(f"--- Step 5a: Training QLoRA Adapter on {MODEL_ID} ---")
    if not os.path.exists(train_path) or not os.path.exists(val_path):
        raise FileNotFoundError("Train/val splits not found. Run src/data_prep.py first.")

    train_df = pd.read_json(train_path, lines=True)
    val_df = pd.read_json(val_path, lines=True)

    train_dataset = format_as_messages(train_df)
    val_dataset = format_as_messages(val_df)

    # 4-bit quantization configuration for T4 GPU efficiency
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto"
    )
    model = prepare_model_for_kbit_training(model)

    # LoRA adapter configuration
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )

    sft_config = SFTConfig(
        output_dir="checkpoints/qlora_runs",
        num_train_epochs=2,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        max_length=512,
        report_to="none"
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        peft_config=peft_config,
        processing_class=tokenizer
    )

    start_time = time.perf_counter()
    trainer.train()
    train_duration = time.perf_counter() - start_time

    os.makedirs(output_dir, exist_ok=True)
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save training metrics for report
    os.makedirs("results", exist_ok=True)
    metrics = {
        "training_time_sec": round(train_duration, 2),
        "peak_gpu_memory_mb": round(torch.cuda.max_memory_allocated() / (1024 ** 2), 2) if torch.cuda.is_available() else 0,
        "log_history": trainer.state.log_history
    }
    with open("results/qlora_train_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"QLoRA adapter saved to {output_dir} in {train_duration:.1f}s")

def evaluate_qlora(test_path="data/processed/test.jsonl", adapter_dir=ADAPTER_DIR, output_path="results/qlora_predictions.jsonl", max_samples=None):
    print(f"--- Step 5b: Generating Test Predictions with QLoRA Model ---")
    test_df = pd.read_json(test_path, lines=True)
    if max_samples:
        test_df = test_df.head(max_samples)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None
    )
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    if device == "cpu":
        model.to(device)
    model.eval()

    results = []
    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="QLoRA Inference"):
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
    print(f"Saved {len(results)} QLoRA predictions to {output_path} (Avg latency: {avg_latency:.3f}s)")

if __name__ == "__main__":
    train_qlora()
    evaluate_qlora()
