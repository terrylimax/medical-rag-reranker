# Docker Run Guide (for assignment check)

Этот файл отдельный от основного README и нужен только для проверки запуска проекта в Docker.

## 1) Build image

```bash
docker build -t medical-rag-reranker:latest .
```

## 2) Prepare host folders for mounted outputs

```bash
mkdir -p data artifacts runs reports .hf_cache
```

## 3) Baseline pipeline inside container (BM25)

### 3.1 Prepare processed files

```bash
docker run --rm -it \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/artifacts:/app/artifacts" \
  -v "$(pwd)/runs:/app/runs" \
  -v "$(pwd)/reports:/app/reports" \
  -v "$(pwd)/.hf_cache:/app/.cache/huggingface" \
  medical-rag-reranker:latest \
  python -m medical_rag_reranker.commands prep_data --overrides 'data.use_dvc=false'
```

### 3.2 Build BM25 index

```bash
docker run --rm -it \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/artifacts:/app/artifacts" \
  -v "$(pwd)/runs:/app/runs" \
  -v "$(pwd)/reports:/app/reports" \
  -v "$(pwd)/.hf_cache:/app/.cache/huggingface" \
  medical-rag-reranker:latest \
  python -m medical_rag_reranker.commands index --overrides 'data.use_dvc=false,retrieval=bm25'
```

### 3.3 Generate answer (retriever + generation, no reranker)

```bash
docker run --rm -it \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/artifacts:/app/artifacts" \
  -v "$(pwd)/runs:/app/runs" \
  -v "$(pwd)/reports:/app/reports" \
  -v "$(pwd)/.hf_cache:/app/.cache/huggingface" \
  medical-rag-reranker:latest \
  python -m medical_rag_reranker.commands generate --question "What is metformin used for?" --overrides 'data.use_dvc=false,retrieval=bm25'
```

## 4) Optional: retrieval evaluation

### 4.1 Build run.trec

```bash
docker run --rm -it \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/artifacts:/app/artifacts" \
  -v "$(pwd)/runs:/app/runs" \
  -v "$(pwd)/reports:/app/reports" \
  -v "$(pwd)/.hf_cache:/app/.cache/huggingface" \
  medical-rag-reranker:latest \
  python -m medical_rag_reranker.commands retrieval_run --overrides 'data.use_dvc=false,retrieval=bm25'
```

### 4.2 Compute P@k / R@k / NDCG@k

```bash
docker run --rm -it \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/artifacts:/app/artifacts" \
  -v "$(pwd)/runs:/app/runs" \
  -v "$(pwd)/reports:/app/reports" \
  -v "$(pwd)/.hf_cache:/app/.cache/huggingface" \
  medical-rag-reranker:latest \
  python -m medical_rag_reranker.commands eval_retrieval --overrides 'data.use_dvc=false,retrieval=bm25'
```

## Notes

- Первый вызов `generate` скачает модель `google/flan-t5-small` в `.hf_cache`; это может занять время.
- Если интернет ограничен, укажи локальную модель через override:

```bash
--overrides 'data.use_dvc=false,retrieval=bm25,generation.llm_model_name=/app/models/your_local_model'
```

и добавь volume с моделью:

```bash
-v "$(pwd)/models:/app/models"
```

## Optional: run with PostgreSQL using docker compose

From repository root:

```bash
cp .env.example .env
docker compose up -d --build
docker compose exec app python -m medical_rag_reranker.commands init_jobs_schema
```

Submit a job and inspect it:

```bash
docker compose exec app python -m medical_rag_reranker.commands submit_job --question "What is metformin used for?"
docker compose exec app python -m medical_rag_reranker.commands job_status --job_id "<job_id>"
docker compose exec app python -m medical_rag_reranker.commands job_result --job_id "<job_id>"
```
