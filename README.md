# Training a Reranker for a Medical RAG System

This project implements training code for a neural reranker (Cross-Encoder) used in a medical Retrieval-Augmented Generation (RAG) system.
The reranker scores the relevance of a (query, document) pair and is used to reorder candidate documents retrieved by a classical retriever before answer generation.

## Project Overview

**Goal:**
Develop and train a neural reranker for a medical question-answering RAG system that can be used by both clinicians and patients.

**For patients:**

- Improve awareness and understanding of diseases, medications, and their indications.

**For clinicians:**

- Reduce information retrieval time.
- Decrease diagnostic errors.
- Assist in identifying rare conditions.

**Core idea:**
A Cross-Encoder model takes a `(Query, Document)` pair as input and outputs a relevance score in `[0, 1]`.
This score is used to rank retrieved documents inside the RAG pipeline.

---

## Quickstart for Reviewers (Checklist)

The project is set up so a reviewer can validate it end-to-end with the steps below.

1. Clone the repository
   ```bash
   git clone https://github.com/terrylimax/medical-rag-reranker.git
   cd medical-rag-reranker
   ```
2. Create a clean environment and install dependencies
   ```bash
   poetry install
   ```
3. Install git hooks
   ```bash
   poetry run pre-commit install
   ```
4. Run all checks
   ```bash
   poetry run pre-commit run -a
   ```
5. (Optional) Start MLflow UI locally
   ```bash
   poetry run mlflow ui --host 127.0.0.1 --port 8080
   ```
6. Run training
   ```bash
   poetry run python -m medical_rag_reranker.commands train
   ```

Expected result: the training run finishes successfully and `train/loss` decreases over time.

## Repository Structure

```text
medical-rag-reranker/
├── configs/                   # Hydra configs
│   ├── config.yaml
│   ├── data/
│   │   └── data.yaml
│   ├── model/
│   │   └── model.yaml
│   ├── logging/
│   │   └── mlflow.yaml
│   └── train/
│       └── train.yaml
├── medical_rag_reranker/
│   ├── commands.py            # CLI entry point
│   ├── data/
│   │   ├── download.py        # data downloading logic
│   │   └── datamodule.py      # PyTorch Lightning DataModule
│   ├── models/
│   │   └── reranker_module.py # LightningModule
│   ├── training/
│   │   └── train.py            # train_from_cfg(cfg)
│   └── utils/
│       ├── git.py
├── pyproject.toml
├── README.md
├── .pre-commit-config.yaml
└── .gitignore
```

---

## Setup

### Prerequisites

- Python 3.11–3.14 (project requires `>=3.11,<3.15`)
- Poetry
- Git

Note: `datasets` imports `lzma`. If you use `pyenv` and see `ModuleNotFoundError: No module named '_lzma'`, reinstall Python with `xz` support (or switch Poetry to a Python build that has `lzma`).

### Environment setup

```bash
git clone https://github.com/terrylimax/medical-rag-reranker.git
cd medical-rag-reranker

poetry install
poetry run pre-commit install
```

Verify that code quality checks pass:

```bash
poetry run pre-commit run -a
```

---

## Data Management (DVC)

This project uses **DVC** to manage datasets. The default remote is a **local** folder `dvc_storage/` (not committed to Git).

One-time setup (already configured in this repo):

- `dvc init`
- `dvc remote add -d local ./dvc_storage`

Runtime behavior:

- `train` and `infer` first try to fetch data via DVC (pull).
- If DVC pull fails (e.g., first run), the code falls back to downloading from open sources and then adds/pushes the data to the local DVC remote.

---

## Data Management

Data is not stored in Git. Datasets are downloaded from open sources and managed locally.

If data is not present, it will be automatically downloaded using the provided helper function.

Main datasets:

- Medical QA dataset (used for positive `(query, document)` pairs)
- Medical document corpora (used to generate negative examples)

Negatives are constructed as:

- Answers to other questions from the QA dataset
- Unrelated snippets from the medical document corpus

## Training

Training is implemented using Lightning and configured with Hydra. The entrypoint is Fire-based:
`python -m medical_rag_reranker.commands train`.

To start training:

```bash
poetry run python -m medical_rag_reranker.commands train
```

Override config values from CLI:

```bash
# one override
poetry run python -m medical_rag_reranker.commands train --overrides "train.max_epochs=2"

# multiple overrides (comma- or space-separated)
poetry run python -m medical_rag_reranker.commands train --overrides "train.max_epochs=2,train.batch_size=16"

# JSON list also supported
poetry run python -m medical_rag_reranker.commands train --overrides '["train.max_epochs=2","train.batch_size=16"]'

Supported override formats:
- single override string
- comma/space-separated overrides
- JSON list
```

What happens under the hood:

1. Data is downloaded if missing.
2. Train/validation/test splits are created at the question level.
3. The Cross-Encoder reranker is fine-tuned.
4. Training and validation loss are logged via Lightning.

Expected behavior:

- training loss decreases over epochs
- validation metrics (e.g. AUC/F1) improve

---

## Configuration

All hyperparameters and paths are controlled via Hydra configs located in `configs/`:

- `config.yaml` — root config (defaults)
- `data/data.yaml` — dataset paths
- `model/model.yaml` — model name, max sequence length, negatives per query
- `train/train.yaml` — learning rate, epochs, batch size, trainer settings

No magic constants are hard-coded in the training code.

---

## Logging and Experiment Tracking

Training is logged to **MLflow** via Lightning's `MLFlowLogger`.

- Tracking URI is configured in `configs/logging/mlflow.yaml` (default: `http://127.0.0.1:8080`).
- Logged hyperparameters include key config values plus `git.commit_id`.
- Logged metrics include at least: `train/loss`, `val/loss`, `val/auroc`, `val/f1` (>= 3 charts in MLflow).

Start the MLflow UI locally:

```bash
poetry run mlflow ui --host 127.0.0.1 --port 8080
```

---

## Notes

- The model is designed as a reusable component of a larger RAG system.
- Deployment (inference API, Docker, orchestration) is considered but out of scope for this training task.
- No trained models or datasets are committed to the repository.

---

## Inference

Minimal single-pair scoring example:

```bash
poetry run python -m medical_rag_reranker.commands infer \
	--query "What is metformin used for?" \
	--document "Metformin is used to treat type 2 diabetes..."
```

---

## Baseline Generation (No Reranker)

This project also supports a baseline `retriever + generation` flow (without cross-encoder reranking):

- retrieve top-`k` docs with BM25 / dense / hybrid
- build a context prompt
- generate an answer with a local `transformers` model
- require citations in `[doc_id]` format

### Scenario A: Single question

```bash
poetry run python -m medical_rag_reranker.commands generate \
  --question "What is metformin used for?"
```

Output:

- generated answer
- `citations_detected=[...]`

### Scenario B: Batch examples report

```bash
poetry run python -m medical_rag_reranker.commands generate \
  --overrides "generation.mode=batch,generation.examples_limit=20"
```

Default report path:

- `reports/baseline_examples.md`

You can override it:

```bash
poetry run python -m medical_rag_reranker.commands generate \
  --output_path reports/my_examples.md \
  --overrides "generation.mode=batch"
```

Notes:

- Batch input JSONL should have `query_id` and either `text` or `question`.
- Retrieval metrics (`Precision@k`, `Recall@k`, `NDCG@k`) remain the primary KPI.
- Generation here is a quality/demo layer via curated examples.

---

## Async-Ready Job Storage

The project now includes a job abstraction for inference:

- `submit_job` creates a job (`PENDING -> RUNNING -> SUCCEEDED/FAILED`)
- `job_status` returns current status and timestamps
- `job_result` returns persisted result payload

Default storage is file-based (`runs/jobs/`), but you can switch to PostgreSQL
without changing business logic (repositories are selected via config).

### File storage (default)

```bash
poetry run python -m medical_rag_reranker.commands submit_job \
  --question "What is metformin used for?"
```

### PostgreSQL storage

1. Install PostgreSQL driver for SQLAlchemy (once):

```bash
poetry add "psycopg[binary]"
```

2. Apply migrations (`jobs.storage=postgres` + DSN):

```bash
poetry run python -m medical_rag_reranker.commands migrate_jobs_schema \
  --overrides "jobs.storage=postgres,jobs.postgres.dsn=postgresql+psycopg://user:password@127.0.0.1:5432/medical_rag"
```

3. Submit a job using PostgreSQL-backed repositories:

```bash
poetry run python -m medical_rag_reranker.commands submit_job \
  --question "What is metformin used for?" \
  --overrides "jobs.storage=postgres,jobs.postgres.dsn=postgresql+psycopg://user:password@127.0.0.1:5432/medical_rag"
```

The PostgreSQL schema is managed via Alembic migrations in `alembic/versions/`.

---

## Docker Compose (Nginx + App + Static + Broker)

This is the recommended end-to-end check for reviewers.

1. Prepare env and host folders:

```bash
cp .env.example .env
mkdir -p data artifacts runs reports .hf_cache
```

2. Start services:

```bash
docker compose up -d --build
docker compose ps
```

Expected services:

- `nginx` public gateway (the only published web port)
- `app` internal FastAPI ASGI application
- `static` internal static site
- `worker` Celery consumer
- `rabbitmq` broker
- `postgres` storage

Reviewer shortcut for the nginx task:

```bash
docker compose up -d --build
curl -sS "http://127.0.0.1:${NGINX_PORT:-8080}/" | head
curl -sS "http://127.0.0.1:${NGINX_PORT:-8080}/api/health"
docker compose exec app python -m medical_rag_reranker.commands migrate_jobs_schema
curl -sS -X POST "http://127.0.0.1:${NGINX_PORT:-8080}/api/jobs" \
  -H "Content-Type: application/json" \
  -d '{"question":"What is metformin used for?"}'
docker compose exec rabbitmq rabbitmqctl list_queues \
  name messages_ready messages_unacknowledged consumers
docker compose exec postgres psql -U medical_rag -d medical_rag -c "\dt"
```

Quick web checks:

```bash
curl -sS "http://127.0.0.1:${NGINX_PORT:-8080}/" | head
curl -sS "http://127.0.0.1:${NGINX_PORT:-8080}/api/health"
```

3. Apply Postgres migrations for jobs:

```bash
docker compose exec app \
  python -m medical_rag_reranker.commands migrate_jobs_schema
```

What this command does:

- connects to PostgreSQL via `JOBS_POSTGRES_DSN`
- applies Alembic migrations up to `head`
- creates or upgrades `inference_jobs` and `inference_results` schema as needed
- is idempotent (safe to run multiple times)

Important: `migrate_jobs_schema` does not publish anything to broker and does not start worker execution.

Quick DB check:

```bash
docker compose exec postgres psql -U medical_rag -d medical_rag -c "\dt"
```

4. Stop consumer for an explicit broker check:

```bash
docker compose stop worker
```

5. Send a test request to producer API and capture `job_id`:

```bash
RESP=$(curl -sS -X POST "http://127.0.0.1:${NGINX_PORT:-8080}/api/jobs" \
  -H "Content-Type: application/json" \
  -d '{"question":"What is metformin used for?"}')
echo "$RESP"
JOB_ID=$(echo "$RESP" | python -c 'import sys,json; print(json.load(sys.stdin)["job_id"])')
echo "$JOB_ID"
```

6. Verify that message is waiting in RabbitMQ queue:

```bash
docker compose exec rabbitmq rabbitmqctl list_queues \
  name messages_ready messages_unacknowledged consumers
```

Expected for `inference_jobs`: `messages_ready >= 1` and `consumers = 0`.

7. Start consumer and watch processing:

```bash
docker compose start worker
docker compose logs -f worker
```

Expected worker logs: `Task ... received` and then `succeeded` (or `failed`).

8. Check status and result by `job_id`:

```bash
curl -sS "http://127.0.0.1:${NGINX_PORT:-8080}/api/jobs/$JOB_ID"
curl -sS "http://127.0.0.1:${NGINX_PORT:-8080}/api/jobs/$JOB_ID/result"
```

Expected: status eventually becomes `SUCCEEDED` and `/result` returns payload.

9. Verify persisted data in PostgreSQL:

```bash
docker compose exec postgres psql -U medical_rag -d medical_rag -c \
"SELECT job_id,status,created_at,started_at,finished_at,error_message \
FROM inference_jobs WHERE job_id='$JOB_ID';"
```

```bash
docker compose exec postgres psql -U medical_rag -d medical_rag -c \
"SELECT job_id, jsonb_pretty(result) FROM inference_results WHERE job_id='$JOB_ID';"
```

If you keep worker running during submission, queue may be drained immediately and step 6 will be less visible.

10. Stop services:

```bash
docker compose down
```

Use `docker compose down -v` if you also want to remove Postgres data volume.

---

## Docker Image (Single-Container CLI)

If you want to validate the baseline pipeline without `docker compose`, you can
run the project as a single Docker image and execute CLI commands inside it.

1. Build image:

```bash
docker build -t medical-rag-reranker:latest .
```

2. Prepare host folders for mounted outputs:

```bash
mkdir -p data artifacts runs reports .hf_cache
```

3. Prepare processed files:

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

4. Build BM25 index:

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

5. Generate an answer:

```bash
docker run --rm -it \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/artifacts:/app/artifacts" \
  -v "$(pwd)/runs:/app/runs" \
  -v "$(pwd)/reports:/app/reports" \
  -v "$(pwd)/.hf_cache:/app/.cache/huggingface" \
  medical-rag-reranker:latest \
  python -m medical_rag_reranker.commands generate \
  --question "What is metformin used for?" \
  --overrides 'data.use_dvc=false,retrieval=bm25'
```

6. Optional retrieval evaluation:

```bash
docker run --rm -it \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/artifacts:/app/artifacts" \
  -v "$(pwd)/runs:/app/runs" \
  -v "$(pwd)/reports:/app/reports" \
  -v "$(pwd)/.hf_cache:/app/.cache/huggingface" \
  medical-rag-reranker:latest \
  python -m medical_rag_reranker.commands retrieval_run --overrides 'data.use_dvc=false,retrieval=bm25'

docker run --rm -it \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/artifacts:/app/artifacts" \
  -v "$(pwd)/runs:/app/runs" \
  -v "$(pwd)/reports:/app/reports" \
  -v "$(pwd)/.hf_cache:/app/.cache/huggingface" \
  medical-rag-reranker:latest \
  python -m medical_rag_reranker.commands eval_retrieval --overrides 'data.use_dvc=false,retrieval=bm25'
```

Notes:

- The first `generate` call downloads `google/flan-t5-small` into `.hf_cache`,
  so it may take some time.
- If network access is limited, point generation to a local model:

```bash
--overrides 'data.use_dvc=false,retrieval=bm25,generation.llm_model_name=/app/models/your_local_model'
```

Mount the model directory as well:

```bash
-v "$(pwd)/models:/app/models"
```

---

## Offline Retrieval Evaluation

This project includes a small CLI to evaluate retrieval runs against qrels using `pytrec_eval`.
It computes mean Precision@k / Recall@k / NDCG@k and logs results to MLflow.

### Scenario A: You already have a run file (`.trec`)

```bash
poetry run python -m medical_rag_reranker.commands.eval_retrieval \
   --eval_queries data/eval_queries.jsonl \
   --qrels data/qrels.tsv \
   --run_path runs/bm25.run.trec \
   --experiment retrieval_eval \
   --retriever bm25 \
   --top_k 100
```

### Scenario B: Generate the run by calling an external retriever

Your retriever command must accept `--queries` and `--out` and write a TREC run file.
Pass a command template to `--retrieve_cmd` with placeholders `{queries}` and `{out_run}`.

```bash
poetry run python -m medical_rag_reranker.commands.eval_retrieval \
   --eval_queries data/eval_queries.jsonl \
   --qrels data/qrels.tsv \
   --retrieve_cmd "python -m medical_rag_reranker.commands.retrieval_run --retriever bm25 --index artifacts/bm25_index.json.gz --queries {queries} --out {out_run}" \
   --experiment retrieval_eval \
   --retriever bm25 \
   --top_k 100
```

Notes:

- Metrics are saved next to the run file as `*.metrics.json`.
- If you see an error about missing `pytrec_eval`, add it to your environment dependencies.

---

## Baseline Retrieval Eval Checklist

Use the unified Fire entrypoint to run baseline retrieval evaluation end-to-end:

1. Prepare baseline artifacts (`qa/corpus/splits/eval_queries/qrels`):

```bash
poetry run python -m medical_rag_reranker.commands prep_data
```

2. Build index for current retriever (`bm25` / `dense` / `hybrid`):

```bash
poetry run python -m medical_rag_reranker.commands index --overrides "retrieval=hybrid"
```

3. (Optional) Generate TREC run file directly:

```bash
poetry run python -m medical_rag_reranker.commands retrieval_run --overrides "retrieval=hybrid"
```

4. Evaluate retrieval metrics and log to MLflow:

```bash
poetry run python -m medical_rag_reranker.commands eval_retrieval --overrides "retrieval=hybrid"
```

5. Run an end-to-end RAG demo (1 question by default, up to 5):

```bash
poetry run python -m medical_rag_reranker.commands rag_demo
poetry run python -m medical_rag_reranker.commands rag_demo --overrides "run.rag_demo.num_questions=5"
```

Notes:

- `medical_rag_reranker/retrieval/` and `configs/retrieval/` are already present in this repository.
- A separate `medical_rag_reranker/evaluation/` package is not required for this baseline step.

---

## License

This project is provided for educational purposes.
