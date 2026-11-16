import os
import time
import json
import torch
import chromadb
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
EMBED_MODEL_ID = "BAAI/bge-small-en-v1.5"

def extract_doc_info(source_field, fallback_idx):
    """Safely extracts document ID and text from the dataset's 'source' dictionary."""
    if isinstance(source_field, dict):
        doc_id = str(source_field.get("version_id") or source_field.get("citation") or f"doc_{fallback_idx}")
        text = str(source_field.get("text", "")).strip()
        return doc_id, text
    return f"doc_{fallback_idx}", str(source_field).strip()

def chunk_text(text, chunk_size=1200, overlap=200):
    """Splits long legal passages into overlapping character chunks."""
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def build_vector_index(splits_dir="data/processed", persist_dir="chroma_db"):
    """Builds a ChromaDB vector store from all source documents in the dataset splits."""
    print(f"--- Building ChromaDB Index with {EMBED_MODEL_ID} ---")
    dfs = []
    for split in ["train.jsonl", "val.jsonl", "test.jsonl"]:
        path = os.path.join(splits_dir, split)
        if os.path.exists(path):
            dfs.append(pd.read_json(path, lines=True))
    if not dfs:
        raise FileNotFoundError(f"No split files found in {splits_dir}. Run src/data_prep.py first.")

    full_df = pd.concat(dfs, ignore_index=True)
    unique_docs = {}
    for idx, row in full_df.iterrows():
        doc_id, text = extract_doc_info(row["source"], idx)
        if text and doc_id not in unique_docs:
            unique_docs[doc_id] = text

    print(f"Found {len(unique_docs)} unique source documents to index.")
    embedder = SentenceTransformer(EMBED_MODEL_ID)
    client = chromadb.PersistentClient(path=persist_dir)

    # Reset collection if rebuilding
    try:
        client.delete_collection("au_legal_docs")
    except Exception:
        pass
    collection = client.create_collection(name="au_legal_docs", metadata={"hnsw:space": "cosine"})

    chunk_ids, chunk_texts, chunk_metas = [], [], []
    for doc_id, text in unique_docs.items():
        for c_idx, chunk in enumerate(chunk_text(text)):
            chunk_ids.append(f"{doc_id}__chunk_{c_idx}")
            chunk_texts.append(chunk)
            chunk_metas.append({"doc_id": doc_id})

    print(f"Embedding {len(chunk_texts)} chunks...")
    embeddings = embedder.encode(chunk_texts, show_progress_bar=True, normalize_embeddings=True).tolist()

    batch_size = 500
    for i in range(0, len(chunk_texts), batch_size):
        collection.add(
            ids=chunk_ids[i:i+batch_size],
            documents=chunk_texts[i:i+batch_size],
            embeddings=embeddings[i:i+batch_size],
            metadatas=chunk_metas[i:i+batch_size]
        )
    print(f"Indexed {len(chunk_texts)} chunks into {persist_dir}.")
    return collection, embedder

def run_rag_evaluation(test_path="data/processed/test.jsonl", output_path="results/rag_predictions.jsonl", k=3, max_samples=None):
    print(f"--- Step 4: Running RAG Pipeline (k={k}) ---")
    collection, embedder = build_vector_index()

    test_df = pd.read_json(test_path, lines=True)
    if max_samples:
        test_df = test_df.head(max_samples)

    device = "cuda" if torch.cuda.is_available() else "cpu"
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
    hits = 0

    for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Running RAG Inference"):
        question = row["question"]
        gold_doc_id, _ = extract_doc_info(row["source"], idx)

        start_time = time.perf_counter()
        q_emb = embedder.encode([question], normalize_embeddings=True).tolist()
        retrieved = collection.query(query_embeddings=q_emb, n_results=k)

        retrieved_texts = retrieved["documents"][0]
        retrieved_doc_ids = [m["doc_id"] for m in retrieved["metadatas"][0]]
        hit = int(gold_doc_id in retrieved_doc_ids)
        hits += hit

        context_block = "\n\n".join([f"[Passage {i+1} | {doc_id}]: {txt}" for i, (doc_id, txt) in enumerate(zip(retrieved_doc_ids, retrieved_texts))])
        messages = [
            {"role": "system", "content": "You are an Australian legal research assistant. Answer the user's question using ONLY the provided legal passages. Cite the passage source where possible. If the passages do not contain the answer, say so. Note that this is general legal information, not legal advice."},
            {"role": "user", "content": f"Context:\n{context_block}\n\nQuestion: {question}"}
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

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
            "gold_doc_id": gold_doc_id,
            "retrieved_doc_ids": retrieved_doc_ids,
            "retrieval_hit": hit,
            "latency_sec": round(latency, 4)
        })

    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    recall_at_k = hits / len(results)
    avg_latency = sum(r["latency_sec"] for r in results) / len(results)
    print(f"\nSaved {len(results)} RAG predictions to {output_path}")
    print(f"Retrieval Recall@{k}: {recall_at_k:.2%}")
    print(f"Average end-to-end latency per question: {avg_latency:.3f}s")

if __name__ == "__main__":
    run_rag_evaluation(k=3)
