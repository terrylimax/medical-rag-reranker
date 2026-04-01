# VKR Demo Guide

This document describes the shortest reproducible flow to demonstrate the thesis-relevant functionality already implemented in this repository.

## What This Demo Covers

The current project can already demonstrate:

- preparation of retrieval artifacts from source medical QA data
- building a baseline retriever index
- retrieval evaluation with `P@k`, `R@k`, `NDCG@k`
- answer generation on top of retrieved context
- reference-free evaluation of generated answers
- optional retrieval reranking comparison if a trained reranker checkpoint exists

The demo does not require Airflow, Kubernetes, or a separate MLflow service.
Those are infrastructure extensions, not blockers for the baseline RAG evaluation loop.

## Preconditions

1. Install dependencies:

```bash
poetry install --with dev
```

2. If you want generation without network calls to Hugging Face, ensure the model is already cached locally and use:

```bash
generation.local_files_only=true
```

3. If you want reranked retrieval comparison, you need a trained checkpoint path for:

```text
/absolute/path/to/reranker.ckpt
```

## Demo Flow

### 1. Prepare artifacts

```bash
poetry run python -m medical_rag_reranker.commands prep_data --overrides 'data.use_dvc=false'
```

Expected outputs:

- `data/processed/qa.jsonl`
- `data/processed/corpus.jsonl`
- `data/processed/splits.json`
- `data/processed/eval_queries.jsonl`
- `data/processed/qrels.tsv`

### 2. Build retrieval index

Baseline BM25 example:

```bash
poetry run python -m medical_rag_reranker.commands index --overrides 'data.use_dvc=false,retrieval=bm25'
```

You can also switch to:

- `retrieval=dense`
- `retrieval=hybrid`

### 3. Evaluate retrieval quality

```bash
poetry run python -m medical_rag_reranker.commands eval_retrieval --overrides 'data.use_dvc=false,retrieval=bm25'
```

This computes and logs:

- `P@5`, `P@10`
- `R@5`, `R@10`
- `NDCG@5`, `NDCG@10`

Expected artifacts:

- `runs/bm25.trec`
- `runs/bm25.trec.metrics.json`
- MLflow run under experiment `retrieval_eval`

### 4. Generate answers end-to-end

Single-question example:

```bash
poetry run python -m medical_rag_reranker.commands generate \
  --question "What is metformin used for?" \
  --overrides 'data.use_dvc=false,retrieval=bm25,generation.local_files_only=true'
```

What this does:

- retrieves context documents
- builds a prompt with document identifiers
- generates an answer with a local `transformers` model
- checks whether answer citations refer to retrieved `doc_id`s

### 5. Evaluate generated answers

```bash
poetry run python -m medical_rag_reranker.commands eval_generation \
  --overrides 'data.use_dvc=false,retrieval=bm25,generation.local_files_only=true'
```

Current answer evaluation is reference-free and heuristic.
It reports:

- `avg_context_relevance`
- `avg_groundedness`
- `avg_answer_relevance`
- citation support rates
- retrieval / generation / end-to-end latency averages

Expected artifacts:

- `reports/eval_generation_examples.md`
- `reports/eval_generation.jsonl`
- `reports/eval_generation.summary.json`
- `reports/eval_generation.md`
- MLflow run under experiment `generation_eval`

### 6. Optional: compare retrieval before and after reranking

Only run this if you already have a trained reranker checkpoint.

```bash
poetry run python -m medical_rag_reranker.commands eval_reranked_retrieval \
  --overrides "data.use_dvc=false,retrieval=bm25,retrieval_run.top_k=20,run.eval_reranked_retrieval.reranker_checkpoint_path=/absolute/path/to/reranker.ckpt"
```

Expected artifacts:

- `runs/bm25_reranked.trec`
- `reports/bm25_reranked_comparison.md`
- `reports/bm25_reranked_comparison.jsonl`
- MLflow run under experiment `reranked_retrieval_eval`

## How To Present This In The Thesis

A compact and defensible narrative is:

1. Show reproducible data preparation for retrieval evaluation.
2. Show baseline retrieval metrics.
3. Show end-to-end answer generation with citations.
4. Show reference-free evaluation of generated answers.
5. Show reranked retrieval comparison once a trained checkpoint is available.

This is enough to demonstrate that the baseline experimental framework already exists.
Airflow, GraphRAG, and additional infrastructure should be presented as next steps, not as prerequisites for validating the current baseline.

## Recommended Next Steps

If the goal is to strengthen the thesis demonstration rather than add infrastructure noise, the next priorities are:

1. Train and validate the reranker checkpoint.
2. Compare `baseline` vs `reranked` runs on the same eval set.
3. Add a stronger judge backend for answer evaluation.
4. Expand tests around reranker-enabled generation.
