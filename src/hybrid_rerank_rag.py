import os
import time
import json
import torch
import chromadb
import pandas as pd
from tqdm import tqdm
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoModelForCausalLM, AutoTokenizer
from rag_pipeline import extract_doc_info, chunk_text

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
EMBED_MODEL_ID = "BAAI/bge-small-en-v1.5"
RERANKER_MODEL_ID = "cross-encoder/ms-marco-MiniLM-L-6-v2"

def run_hybrid_rerank_rag(splits_dir="data/processed", output_path="results/hybrid_rerank_predictions.jsonl", top_k_candidates=10, final_k=3, max_samples=None):
    print(f"--- Running Hybrid (BM25 + Dense) + Cross-Encoder Reranking RAG ---")
    
    dfs = [pd.read_json(os.path.join(splits_dir, s), lines=True) for s in ["train.jsonl", "val.jsonl", "test.jsonl"] if os.path.exists(os.path.join(splits_dir, s))]
    full_df = pd.concat(dfs, ignore_index=True)
    
    unique_docs = {}
    for idx, row in full_df.iterrows():
        doc_id, text = extract_doc_info(row["source"], idx)
        if text and doc_id not in unique_docs:
            unique_docs[doc_id] = text

    chunk_ids, chunk_texts, chunk_metas = [], [], []
    for doc_id, text in unique_docs.items():
        for c_idx, chunk in enumerate(chunk_text(text)):
            chunk_ids.append(f"{doc_id}__chunk_{c_idx}")
            chunk_texts.append(chunk)
            chunk_metas.append({"doc_id": doc_id})

    # 1. Sparse BM25 Index
    tokenized_corpus = [doc.lower().split() for doc in chunk_texts]
    bm25 = BM25Okapi(tokenized_corpus)

    # 2. Dense Index via ChromaDB
    embedder = SentenceTransformer(EMBED_MODEL_ID)
    client = chromadb.PersistentClient(path="chroma_db")
    collection = client.get_or_create_collection(name="au_legal_docs", metadata={"hnsw:space": "cosine"})
    if collection.count() == 0:
        embeddings = embedder.encode(chunk_texts, normalize_embeddings=True).tolist()
        for i in range(0, len(chunk_texts), 500):
            collection.add(ids=chunk_ids[i:i+500], documents=chunk_texts[i:i+500], embeddings=embeddings[i:i+500], metadatas=chunk_metas[i:i+500])

    # 3. Cross-Encoder Reranker
    reranker = CrossEncoder(RERANKER_MODEL_ID)

    test_df = pd.read_json(os.path.join(splits_dir, "test.jsonl"), lines=True)
    if max_samples:
        test_df = test_df.head(max_samples)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16 if device == "cuda" else torch.float32, device_map="auto" if device == "cuda" else None)
    if device == "cpu":
        model.to(device)
    model.eval()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    results, hits = [], 0

    for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Hybrid + Rerank Inference"):
        question = row["question"]
        gold_doc_id, _ = extract_doc_info(row["source"], idx)
        start_time = time.perf_counter()

        # Dense top candidates
        q_emb = embedder.encode([question], normalize_embeddings=True).tolist()
        dense_res = collection.query(query_embeddings=q_emb, n_results=top_k_candidates)
        dense_pairs = list(zip(dense_res["documents"][0], [m["doc_id"] for m in dense_res["metadatas"][0]]))

        # Sparse BM25 top candidates
        bm25_scores = bm25.get_scores(question.lower().split())
        top_bm25_idx = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:top_k_candidates]
        bm25_pairs = [(chunk_texts[i], chunk_metas[i]["doc_id"]) for i in top_bm25_idx]

        # Deduplicate merged candidates
        seen, merged = set(), []
        for txt, d_id in dense_pairs + bm25_pairs:
            if txt not in seen:
                seen.add(txt)
                merged.append((txt, d_id))

        # Rerank with Cross-Encoder
        ce_scores = reranker.predict([(question, txt) for txt, _ in merged])
        reranked = sorted(zip(merged, ce_scores), key=lambda x: x[1], reverse=True)[:final_k]

        top_texts = [item[0][0] for item in reranked]
        top_doc_ids = [item[0][1] for item in reranked]
        hit = int(gold_doc_id in top_doc_ids)
        hits += hit

        context_block = "\n\n".join([f"[Passage {i+1} | {d_id}]: {txt}" for i, (d_id, txt) in enumerate(zip(top_doc_ids, top_texts))])
        messages = [
            {"role": "system", "content": "You are an Australian legal research assistant. Answer the user's question using ONLY the provided legal passages. Cite the passage source where possible. Note that this is general legal information, not legal advice."},
            {"role": "user", "content": f"Context:\n{context_block}\n\nQuestion: {question}"}
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=256, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        latency = time.perf_counter() - start_time

        prediction = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        results.append({
            "question": question,
            "reference_answer": row["answer"],
            "predicted_answer": prediction,
            "gold_doc_id": gold_doc_id,
            "retrieved_doc_ids": top_doc_ids,
            "retrieval_hit": hit,
            "latency_sec": round(latency, 4)
        })

    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"Recall@{final_k}: {hits / len(results):.2%} | Saved to {output_path}")

if __name__ == "__main__":
    run_hybrid_rerank_rag()
