# Australian Legal Question Answering with Open-Weights LLMs

CP5403 Assessment 2 (LLM Project), James Cook University.

This project compares three ways of answering Australian legal questions with a small open-weights LLM:

1. **Closed-book:** the base model answers from its own knowledge.
2. **RAG:** retrieve relevant passages from Australian legal documents, then answer.
3. **LoRA/QLoRA fine-tuning:** fine-tune the model on legal question-answer pairs.

Dataset: [Open Australian Legal QA](https://huggingface.co/datasets/umarbutler/open-australian-legal-qa) (Butler, 2023).

> This system is a university research project. It does not provide legal advice.

## Repository structure

- `notebooks/` - Colab notebooks, in order of use
- `docs/prompt_history/` - log of every GenAI prompt used (required for submission)
- `requirements.txt` - Python libraries needed

## Setup (Google Colab)

1. Open a new notebook at colab.research.google.com and set the runtime to a T4 GPU.
2. Clone this repo and run `pip install -r requirements.txt`.

## Status

- [x] Step 1: repo and environment
- [ ] Step 2: data loading, exploration and document-level split
- [ ] Step 3: closed-book baseline
- [ ] Step 4: RAG
- [ ] Step 5: LoRA fine-tuning
- [ ] Step 6: literature-based improvement
- [ ] Step 7: evaluation
- [ ] Step 8: report
- [ ] Step 9: presentation

## Generative AI use

GenAI tools were used to assist this project. Prompts are logged in `docs/prompt_history/`.
