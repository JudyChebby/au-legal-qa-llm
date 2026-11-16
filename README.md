# Australian Legal Question Answering with Open-Weights LLMs

CP5403 Assessment 2 (LLM Project), James Cook University.

This project compares four approaches to answering Australian legal questions using a small open-weights LLM (`Qwen/Qwen2.5-1.5B-Instruct`):

1. **Closed-book baseline (`src/baseline.py`):** The base model answers directly from parametric knowledge without external context.
2. **Dense RAG (`src/rag_pipeline.py`):** Chunks source legal documents, embeds them with `BAAI/bge-small-en-v1.5` in ChromaDB, retrieves the top-$k$ passages, and generates grounded answers.
3. **LoRA/QLoRA fine-tuning (`src/finetune.py`):** Parameter-efficient 4-bit QLoRA fine-tuning on training question-answer pairs.
4. **Hybrid + Cross-Encoder Reranking RAG (`src/hybrid_rerank_rag.py`):** Literature-based enhancement combining sparse BM25 keyword retrieval, dense vector search, and `ms-marco-MiniLM-L-6-v2` cross-encoder reranking.

Dataset: [Open Australian Legal QA](https://huggingface.co/datasets/umarbutler/open-australian-legal-qa) (Butler, 2023).

> **Disclaimer:** This system is an academic research prototype and does not constitute legal advice.

## Repository Structure

- `src/data_prep.py` — Loads the dataset and creates reproducible train/validation/test splits.
- `src/baseline.py` — Runs closed-book zero-shot inference and logs latency.
- `src/rag_pipeline.py` — Builds the ChromaDB vector index and runs dense RAG inference + Recall@k tracking.
- `src/finetune.py` — Fine-tunes `Qwen2.5-1.5B-Instruct` with 4-bit QLoRA and evaluates on the test split.
- `src/hybrid_rerank_rag.py` — Implements hybrid BM25 + dense retrieval with cross-encoder reranking.
- `src/evaluate.py` — Computes unified Token F1, ROUGE-L, Recall@k, and latency metrics across all setups.
- `docs/prompt_history/` — Running log of GenAI prompts used during development.
- `requirements.txt` — Python dependencies.

## Setup & Execution Order

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
