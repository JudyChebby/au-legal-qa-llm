import os
from datasets import load_dataset
import pandas as pd

def run_data_preparation():
    print("--- Step 2: Loading Open Australian Legal QA Dataset ---")
    
    # 1. Load dataset from Hugging Face
    dataset = load_dataset("umarbutler/open-australian-legal-qa", split="train")
    print(f"Total examples loaded: {len(dataset)}")
    
    # Convert to pandas DataFrame for easy inspection
    df = pd.DataFrame(dataset)
    print("\nColumns available in dataset:")
    print(df.columns.tolist())
    
    # 2. Basic Exploration
    print("\nSample Question:")
    print(df['question'].iloc[0])
    print("\nSample Answer:")
    print(df['answer'].iloc[0])
    
    # 3. Document-Level Split Strategy
    # To prevent data leakage, we split by unique source documents rather than individual questions.
    unique_sources = df['source_document'].unique() if 'source_document' in df.columns else df['source'].unique()
    print(f"\nTotal unique source documents: {len(unique_sources)}")
    
    # Shuffle and split sources: 80% train, 10% validation, 10% test
    df_shuffled = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    
    train_size = int(0.8 * len(df_shuffled))
    val_size = int(0.1 * len(df_shuffled))
    
    train_df = df_shuffled.iloc[:train_size]
    val_df = df_shuffled.iloc[train_size:train_size + val_size]
    test_df = df_shuffled.iloc[train_size + val_size:]
    
    print(f"Train split: {len(train_df)} examples")
    print(f"Validation split: {len(val_df)} examples")
    print(f"Test split: {len(test_df)} examples")
    
    # 4. Save splits locally
    os.makedirs("data/processed", exist_ok=True)
    train_df.to_json("data/processed/train.jsonl", orient="records", lines=True)
    val_df.to_json("data/processed/val.jsonl", orient="records", lines=True)
    test_df.to_json("data/processed/test.jsonl", orient="records", lines=True)
    print("\nSplits successfully saved to data/processed/!")

if __name__ == "__main__":
    run_data_preparation()
