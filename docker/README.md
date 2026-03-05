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

## Optional: run with PostgreSQL + RabbitMQ (producer/consumer) using docker compose

From repository root:

```bash
cp .env.example .env
docker compose up -d --build
docker compose ps
docker compose exec app python -m medical_rag_reranker.commands init_jobs_schema
```

`init_jobs_schema` prepares DB tables for jobs storage. It uses `JOBS_POSTGRES_DSN`,
applies `medical_rag_reranker/jobs/sql/postgres_schema.sql`, and creates
`inference_jobs` + `inference_results` if they do not exist.
It is idempotent, and it does not publish tasks to broker or start worker execution.

Quick check:

```bash
docker compose exec postgres psql -U medical_rag -d medical_rag -c "\dt"
```

For explicit broker verification, stop worker first:

```bash
docker compose stop worker
```

Submit a job to producer API and capture `job_id`:

```bash
RESP=$(curl -sS -X POST "http://127.0.0.1:${JOBS_API_PORT:-8080}/jobs" \
  -H "Content-Type: application/json" \
  -d '{"question":"What is metformin used for?"}')
echo "$RESP"
JOB_ID=$(echo "$RESP" | python -c 'import sys,json; print(json.load(sys.stdin)["job_id"])')
echo "$JOB_ID"
```

Check RabbitMQ queue state:

```bash
docker compose exec rabbitmq rabbitmqctl list_queues \
  name messages_ready messages_unacknowledged consumers
```

Expected for `inference_jobs`: `messages_ready >= 1`, `consumers = 0`.

Start worker and watch processing:

```bash
docker compose start worker
docker compose logs -f worker
```

Then check status/result and DB row for this job:

```bash
curl -sS "http://127.0.0.1:${JOBS_API_PORT:-8080}/jobs/$JOB_ID"
curl -sS "http://127.0.0.1:${JOBS_API_PORT:-8080}/jobs/$JOB_ID/result"

docker compose exec postgres psql -U medical_rag -d medical_rag -c \
"SELECT job_id,status,created_at,started_at,finished_at,error_message \
FROM inference_jobs WHERE job_id='$JOB_ID';"
```
