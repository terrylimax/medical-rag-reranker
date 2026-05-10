# Medical RAG Pipeline: Retrieval, Reranking, Generation, and Evaluation

This repository is no longer only training code for a Cross-Encoder reranker.
It now contains an end-to-end medical RAG baseline with:

- data preparation for retrieval experiments
- BM25 / dense / hybrid retrieval
- optional Cross-Encoder reranking
- answer generation with local `transformers` models
- retrieval evaluation with `P@k`, `R@k`, `NDCG@k`
- reference-free generation evaluation
- MLflow tracking for training and evaluation runs
- async job API and deployment skeletons via Docker Compose and Kubernetes

## Project Scope

This repository implements a medical RAG system with hybrid retrieval and graph-aware extensions.
The current codebase covers the reproducible baseline and most of the evaluation harness:

- reproducible retrieval corpus / queries / qrels preparation
- baseline `retriever -> generation`
- `retriever -> reranker -> generation` path, provided a trained reranker checkpoint exists
- retrieval-only comparison before and after reranking
- reference-free answer evaluation for end-to-end RAG runs

Not fully implemented yet:

- training and validating a final reranker checkpoint
- full GraphRAG-style community summaries
- richer judge backends beyond the current heuristic reference-free scorer
- production-grade monitoring and scheduling layers such as Prometheus/Grafana or Airflow

## Current Functional Status

Implemented now:

- `train`: Cross-Encoder reranker training
- `infer`: single `(query, document)` relevance scoring
- `prep_data`: build `qa.jsonl`, `corpus.jsonl`, `splits.json`, `eval_queries.jsonl`, `qrels.tsv`
- `index`: build retrieval indices for `bm25`, `dense`, `hybrid`, or graph-expanded variants
- `retrieval_run`: generate a TREC run file
- `eval_retrieval`: compute retrieval metrics and log them to MLflow
- `generate`: run baseline or reranked RAG generation with citations
- `graph_benchmark`: build multi-document graph benchmark queries/references
- `eval_generation`: compute heuristic or LLM-as-a-Judge answer metrics and log them to MLflow
- `eval_reranked_retrieval`: compare retrieval quality before and after reranking
- `rag_demo`: run a compact end-to-end demo and save markdown / JSONL reports
- `submit_job`, `job_status`, `job_result`, `serve_jobs_api`: async-ready inference API flow

## Minimal RAG Demo Path

The shortest reproducible end-to-end path is:

1. Prepare retrieval artifacts
2. Build an index
3. Evaluate retrieval quality
4. Generate answers on top of retrieval
5. Evaluate generated answers
6. Optionally compare baseline retrieval with reranked retrieval

Commands:

```bash
poetry run python -m medical_rag_reranker.commands prep_data --overrides 'data.use_dvc=false'
poetry run python -m medical_rag_reranker.commands index --overrides 'data.use_dvc=false,retrieval=bm25'
poetry run python -m medical_rag_reranker.commands eval_retrieval --overrides 'data.use_dvc=false,retrieval=bm25'
poetry run python -m medical_rag_reranker.commands generate --question "What is metformin used for?" --overrides 'data.use_dvc=false,retrieval=bm25,generation.local_files_only=true'
poetry run python -m medical_rag_reranker.commands eval_generation --overrides 'data.use_dvc=false,retrieval=bm25,generation.local_files_only=true'
```

If you already have a trained reranker checkpoint:

```bash
poetry run python -m medical_rag_reranker.commands eval_reranked_retrieval \
  --overrides "data.use_dvc=false,retrieval=bm25,retrieval_run.top_k=20,run.eval_reranked_retrieval.reranker_checkpoint_path=/absolute/path/to/reranker.ckpt"
```

A focused walkthrough is available in `docs/rag_demo.md`.

## Quickstart

1. Clone the repository
   ```bash
   git clone https://github.com/terrylimax/medical-rag-reranker.git
   cd medical-rag-reranker
   ```
2. Install dependencies
   ```bash
   poetry install
   ```
3. Install hooks
   ```bash
   poetry run pre-commit install
   ```
4. Run quality checks
   ```bash
   poetry run pre-commit run -a
   ```
5. Run tests
   ```bash
   poetry install --with dev
   poetry run pytest
   ```

## Reviewer Shortcut: Kubernetes Iteration

If you are validating the Kubernetes deployment path rather than the RAG pipeline,
the fastest path is:

```bash
minikube start
minikube addons enable ingress

minikube image build -t medical-rag-app:local -f Dockerfile .
minikube image build -t medical-rag-nginx:local -f docker/nginx/Dockerfile .
minikube image build -t medical-rag-static:local -f docker/static/Dockerfile .

kubectl apply -f k8s/00-namespace.yaml
kubectl apply -f k8s/

kubectl -n medical-rag delete job jobs-db-migrate --ignore-not-found

kubectl -n medical-rag rollout status statefulset/postgres-master
kubectl -n medical-rag rollout status statefulset/postgres-slave
kubectl -n medical-rag rollout status deployment/rabbitmq
kubectl -n medical-rag rollout status deployment/static
kubectl -n medical-rag rollout status deployment/nginx
kubectl -n medical-rag rollout status deployment/app
kubectl -n medical-rag rollout status deployment/worker

kubectl apply -f k8s/12-migrate-job.yaml
kubectl -n medical-rag wait --for=condition=complete job/jobs-db-migrate --timeout=180s
```

Quick HA verification:

```bash
kubectl -n medical-rag rollout status statefulset/postgres-master
kubectl -n medical-rag rollout status statefulset/postgres-slave

kubectl -n medical-rag exec pod/postgres-master-0 -- sh -lc \
'PGPASSWORD="$POSTGRESQL_POSTGRES_PASSWORD" /opt/bitnami/postgresql/bin/psql -U postgres -d medical_rag -c "SELECT application_name, state, sync_state FROM pg_stat_replication;"'

kubectl -n medical-rag exec pod/postgres-slave-0 -- sh -lc \
'PGPASSWORD="$POSTGRESQL_POSTGRES_PASSWORD" /opt/bitnami/postgresql/bin/psql -U postgres -d medical_rag -c "SELECT pg_is_in_recovery();"'
```

Terminal 1:

```bash
# keep this running
kubectl -n ingress-nginx port-forward service/ingress-nginx-controller 8081:80
```

Terminal 2:

```bash
export APP_URL=http://127.0.0.1:8081
export BASIC_AUTH=reviewer:reviewer
curl -s -o /dev/null -w "%{http_code}\n" "$APP_URL/"
curl -sS -u "$BASIC_AUTH" "$APP_URL/api/health"

JOB_ID=$(curl -sS -u "$BASIC_AUTH" -X POST "$APP_URL/api/jobs" \
  -H "Content-Type: application/json" \
  -d '{"question":"What is metformin used for?"}' | python -c 'import sys, json; print(json.load(sys.stdin)["job_id"])')

until curl -sS -u "$BASIC_AUTH" "$APP_URL/api/jobs/$JOB_ID" | grep -q '"status":"SUCCEEDED"'; do
  sleep 5
done

curl -sS -u "$BASIC_AUTH" "$APP_URL/api/jobs/$JOB_ID"
curl -sS -u "$BASIC_AUTH" "$APP_URL/api/jobs/$JOB_ID/result"
```

`APP_URL` will stay available in the current shell session, so you can reuse it
for the next `curl` commands without retyping the URL.
`BASIC_AUTH` must also be exported in the same shell where you run `curl -u`,
otherwise `curl` will prompt for a password for an empty user.
If ingress briefly returns `503` right after `kubectl apply`, wait until
`kubectl -n medical-rag get endpoints app nginx` shows non-empty endpoints and
retry the `curl` commands.

For the full verification flow with broker queue check, Postgres persistence, and
pod restart stability, see `Kubernetes (Minikube)` below.

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
│   ├── commands/             # Fire CLI package (`python -m medical_rag_reranker.commands`)
│   ├── data/                 # data download, DVC helpers, dataset preparation
│   ├── retrieval/            # bm25 / dense / hybrid retrievers
│   ├── inference/            # infer, generate, rerank
│   ├── evaluation/           # reference-free generation evaluation helpers
│   ├── jobs/                 # async job service, storage, HTTP API
│   ├── models/              # Lightning reranker module
│   ├── training/            # training entrypoints
│   └── utils/               # config / git / shared helpers
├── configs/                  # Hydra configs for train/run/retrieval/jobs
├── tests/                    # lightweight regression tests
├── docker/                   # nginx/static support images and configs
├── k8s/                      # Kubernetes manifests
├── pyproject.toml
├── README.md
└── docs/rag_demo.md
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

- CLI commands that depend on managed artifacts call `ensure_data()` first, including `train`, `infer`, `prep_data`, `index`, `retrieval_run`, `generate`, `eval_retrieval`, `eval_generation`, `eval_reranked_retrieval`, and `rag_demo`.
- If DVC pull fails or DVC is disabled, the code falls back to downloading or building the required local artifacts.

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
- The repository already includes inference, evaluation, async job handling, Docker Compose, and Kubernetes manifests.
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

If the model is already cached locally and you want to avoid any network access
to the Hugging Face Hub, run:

```bash
poetry run python -m medical_rag_reranker.commands generate \
  --question "What is metformin used for?" \
  --overrides "generation.local_files_only=true"
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

## RAG With Reranker

The generation pipeline can now optionally rerank retrieved candidates with the
trained Cross-Encoder before building the final prompt.

Example:

```bash
poetry run python -m medical_rag_reranker.commands generate \
  --question "What is metformin used for?" \
  --overrides "retrieval=bm25,generation.use_reranker=true,generation.reranker_checkpoint_path=/absolute/path/to/reranker.ckpt,generation.retrieve_top_k=20,generation.top_k=5,generation.local_files_only=true"
```

To compare retrieval quality before and after reranking on the same eval set:

```bash
poetry run python -m medical_rag_reranker.commands eval_reranked_retrieval \
  --overrides "retrieval=bm25,retrieval_run.top_k=20,run.eval_reranked_retrieval.reranker_checkpoint_path=/absolute/path/to/reranker.ckpt"
```

This command writes a reranked `run.trec`, logs baseline/reranked metrics to
MLflow, stores metric deltas such as `delta_NDCG@10`, and now also saves:

- a markdown comparison report with top docs before/after reranking
- a JSONL file with per-query comparison examples

Example output artifacts:

- `reports/bm25_reranked_comparison.md`
- `reports/bm25_reranked_comparison.jsonl`

The batch generation report now also includes:

- top docs before rerank
- top docs after rerank
- detected citations
- which citations were supported by retrieved docs vs invented by the model

To evaluate generated answers end-to-end with reference-free heuristics:

```bash
poetry run python -m medical_rag_reranker.commands eval_generation \
  --overrides "retrieval=bm25,generation.local_files_only=true"
```

This command writes:

- `reports/eval_generation_examples.md` with raw generated examples
- `reports/eval_generation.jsonl` with per-example scores
- `reports/eval_generation.summary.json` with aggregated metrics
- `reports/eval_generation.md` with a compact markdown summary

The first implemented judge mode is `heuristic`, which logs:

- `avg_context_relevance`
- `avg_groundedness`
- `avg_answer_relevance`
- citation support / unsupported citation rates
- retrieval / generation / end-to-end latency averages

You can also run LLM-as-a-Judge against a local OpenAI-compatible endpoint such
as vLLM. In this mode, vLLM serves the judge model only; the RAG answer itself
is still generated by the local `transformers` backend configured under
`generation.*`.

Example vLLM server:

```bash
vllm serve mistralai/Mistral-7B-Instruct-v0.3 \
  --dtype auto \
  --api-key token-abc123 \
  --generation-config vllm
```

Example eval run:

```bash
LLM_JUDGE_BASE_URL=http://localhost:8000/v1 \
LLM_JUDGE_MODEL=mistralai/Mistral-7B-Instruct-v0.3 \
LLM_JUDGE_API_KEY=token-abc123 \
poetry run python -m medical_rag_reranker.commands eval_generation \
  --overrides "retrieval=bm25,generation.local_files_only=true,run.eval_generation.judge_mode=llm"
```

The LLM judge writes `faithfulness`, `relevance`, `completeness`, `safety`,
`verdict`, and `rationale` into the same evaluation JSONL/report artifacts.

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

## Kubernetes (Minikube)

The repository also includes a minimal Kubernetes deployment for local cluster
validation in Minikube. This iteration uses `Deployment`, `Service`, and
`Ingress` resources. PostgreSQL now runs in a master/slave topology using two
separate `StatefulSet` resources based on the Bitnami master/slave replication
pattern. As of March 24, 2026, the live Docker manifest available during
validation was `docker.io/bitnamilegacy/postgresql:17.6.0-debian-12-r4`, so the
Kubernetes manifests pin to that image rather than the newer example tag.

Kubernetes manifests are located in `/Users/terrylimax/medical-rag-reranker/k8s/`.

Architecture in cluster:

- `ingress-nginx` public entrypoint with Basic Auth
- `nginx` internal ClusterIP gateway
- `app` internal FastAPI producer
- `static` internal static site
- `worker` Celery consumer
- `rabbitmq` internal broker
- `postgres-master` write database (`StatefulSet`)
- `postgres-slave` read replica (`StatefulSet`)

The namespace used by manifests is `medical-rag`.

Registry secret note:

- `/Users/terrylimax/medical-rag-reranker/k8s/01a-registry-secret.placeholder.yaml`
  creates a placeholder `regcred` secret for the local Minikube flow.
- `/Users/terrylimax/medical-rag-reranker/k8s/01b-workload-serviceaccount.yaml`
  attaches `imagePullSecrets: [regcred]` to custom workload pods
  (`app`, `worker`, `nginx`, `static`).
- This placeholder is enough when you use `minikube image build`.
- For a real private registry, replace `regcred` with actual credentials:

```bash
kubectl -n medical-rag delete secret regcred
kubectl -n medical-rag create secret docker-registry regcred \
  --docker-server=<registry> \
  --docker-username=<user> \
  --docker-password=<password> \
  --docker-email=<email>
```

1. Start Minikube:

```bash
minikube start
minikube addons enable ingress
```

2. Build local images directly into Minikube:

```bash
minikube image build -t medical-rag-app:local -f Dockerfile .
minikube image build -t medical-rag-nginx:local -f docker/nginx/Dockerfile .
minikube image build -t medical-rag-static:local -f docker/static/Dockerfile .
```

3. Apply manifests:

```bash
# if you are upgrading from the previous Kubernetes iteration:
kubectl -n medical-rag delete deployment postgres --ignore-not-found
kubectl -n medical-rag delete statefulset postgres --ignore-not-found
kubectl -n medical-rag delete service postgres-headless --ignore-not-found
kubectl -n medical-rag delete pvc postgres-data --ignore-not-found
kubectl -n medical-rag delete pvc postgres-data-postgres-0 --ignore-not-found

kubectl apply -f k8s/00-namespace.yaml
kubectl apply -f k8s/
kubectl -n medical-rag delete job jobs-db-migrate --ignore-not-found
```

4. Wait until core Deployments are ready:

```bash
kubectl -n medical-rag rollout status statefulset/postgres-master
kubectl -n medical-rag rollout status statefulset/postgres-slave
kubectl -n medical-rag rollout status deployment/rabbitmq
kubectl -n medical-rag rollout status deployment/static
kubectl -n medical-rag rollout status deployment/app
kubectl -n medical-rag rollout status deployment/worker
kubectl -n medical-rag rollout status deployment/nginx
kubectl -n medical-rag get pods
kubectl -n medical-rag get svc
```

5. Apply DB migrations for jobs storage with a Kubernetes `Job`:

```bash
kubectl apply -f k8s/12-migrate-job.yaml
kubectl -n medical-rag wait --for=condition=complete job/jobs-db-migrate --timeout=180s
kubectl -n medical-rag logs job/jobs-db-migrate
```

If you need to rerun the migration job, delete and recreate it:

```bash
kubectl -n medical-rag delete job jobs-db-migrate --ignore-not-found
kubectl apply -f k8s/12-migrate-job.yaml
```

Fallback for debugging only:

```bash
kubectl -n medical-rag exec deploy/app -- \
  python -m medical_rag_reranker.commands migrate_jobs_schema
```

Kubernetes smoke-test note:

The Kubernetes manifests are configured to use small bundled demo retrieval
assets already shipped inside the image:

- `/app/medical_rag_reranker/demo_assets/corpus.jsonl`
- `/app/medical_rag_reranker/demo_assets/bm25_index.json`

Because of that, `prep_data` and `index` are not required for the basic
Minikube validation flow after rebuilding the images.

If you still use older images or deliberately switch back to the default
`data/processed` + `artifacts/` paths and see
`FileNotFoundError: artifacts/bm25_index.json.gz`, the temporary workaround is:

```bash
kubectl -n medical-rag exec deploy/worker -- \
  python -m medical_rag_reranker.commands prep_data

kubectl -n medical-rag exec deploy/worker -- \
  python -m medical_rag_reranker.commands index
```

Optional quick check:

```bash
kubectl -n medical-rag exec deploy/worker -- sh -lc 'ls -lah artifacts && ls -lah data/processed'
```

That fallback is slower. A proper follow-up is to mount a shared persistent
volume for `data/` and `artifacts/` into both `app` and `worker`.

PostgreSQL replication checks:

```bash
kubectl -n medical-rag exec pod/postgres-master-0 -- sh -lc \
'PGPASSWORD="$POSTGRESQL_POSTGRES_PASSWORD" /opt/bitnami/postgresql/bin/psql -U postgres -d medical_rag -c "SELECT application_name, state, sync_state FROM pg_stat_replication;"'

kubectl -n medical-rag exec pod/postgres-slave-0 -- sh -lc \
'PGPASSWORD="$POSTGRESQL_POSTGRES_PASSWORD" /opt/bitnami/postgresql/bin/psql -U postgres -d medical_rag -c "SELECT pg_is_in_recovery();"'
```

Expected:

- master sees at least one replica in `pg_stat_replication`
- slave returns `t` for `pg_is_in_recovery()`

6. Expose the ingress controller locally and verify Basic Auth.

Recommended on macOS:

```bash
# keep this command running in a separate terminal
kubectl -n ingress-nginx port-forward service/ingress-nginx-controller 8081:80
```

Then in another terminal:

```bash
export APP_URL=http://127.0.0.1:8081
export BASIC_AUTH=reviewer:reviewer

# without credentials ingress should reject the request
curl -s -o /dev/null -w "%{http_code}\n" "$APP_URL/"

# with credentials the gateway should allow access
curl -sS -u "$BASIC_AUTH" "$APP_URL/" | head
curl -sS -u "$BASIC_AUTH" "$APP_URL/api/health"
```

If ingress briefly returns `503` here, wait until both `app` and `nginx` have
backend endpoints:

```bash
kubectl -n medical-rag get endpoints app nginx
```

7. Explicit broker check: stop consumer, submit a job, confirm message is queued:

```bash
kubectl -n medical-rag scale deployment/worker --replicas=0

RESP=$(curl -sS -X POST "$APP_URL/api/jobs" \
  -u "$BASIC_AUTH" \
  -H "Content-Type: application/json" \
  -d '{"question":"What is metformin used for?"}')
echo "$RESP"
JOB_ID=$(echo "$RESP" | python -c 'import sys,json; print(json.load(sys.stdin)["job_id"])')
echo "$JOB_ID"

kubectl -n medical-rag exec deploy/rabbitmq -- \
  rabbitmqctl list_queues name messages_ready messages_unacknowledged consumers
```

Expected for `inference_jobs`: `messages_ready >= 1` and `consumers = 0`.

8. Start consumer, watch logs, and verify result persistence:

```bash
kubectl -n medical-rag scale deployment/worker --replicas=1
kubectl -n medical-rag rollout status deployment/worker
kubectl -n medical-rag logs -f deployment/worker
```

In a separate shell:

```bash
until curl -sS -u "$BASIC_AUTH" "$APP_URL/api/jobs/$JOB_ID" | grep -q '"status":"SUCCEEDED"'; do
  sleep 5
done

curl -sS -u "$BASIC_AUTH" "$APP_URL/api/jobs/$JOB_ID"
curl -sS -u "$BASIC_AUTH" "$APP_URL/api/jobs/$JOB_ID/result"

kubectl -n medical-rag exec pod/postgres-master-0 -- sh -lc \
"PGPASSWORD=\$POSTGRESQL_PASSWORD /opt/bitnami/postgresql/bin/psql -U medical_rag -d medical_rag -c \"SELECT job_id,status,created_at,started_at,finished_at,error_message FROM inference_jobs WHERE job_id = '$JOB_ID';\""

kubectl -n medical-rag exec pod/postgres-master-0 -- sh -lc \
"PGPASSWORD=\$POSTGRESQL_PASSWORD /opt/bitnami/postgresql/bin/psql -U medical_rag -d medical_rag -c \"SELECT job_id, result->>'answer' AS answer FROM inference_results WHERE job_id = '$JOB_ID';\""
```

9. Stability check: delete the API pod and verify that Deployment restores it:

```bash
kubectl -n medical-rag delete pod -l app=app
kubectl -n medical-rag rollout status deployment/app
curl -sS -u "$BASIC_AUTH" "$APP_URL/api/health"
```

This demonstrates that the service is not tied to a single pod instance and is
restored by Kubernetes after failure simulation.

PostgreSQL HA note:

- write traffic goes to `service/postgres`, which points to `postgres-master`
- replica traffic can go to `service/postgres-slave`
- stable pod names are `postgres-master-0` and `postgres-slave-0`
- storage is created through `volumeClaimTemplates`
- application DSN stays unchanged because `service/postgres` is preserved for
  client connections

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
poetry run python -m medical_rag_reranker.commands eval_retrieval \
  --overrides "run.eval_retrieval.run_path=runs/bm25.run.trec,run.eval_retrieval.retriever=bm25,run.eval_retrieval.top_k=100"
```

### Scenario B: Generate the run by calling an external retriever

Your retriever command must accept `--queries` and `--out` and write a TREC run file.
Pass a command template to `--retrieve_cmd` with placeholders `{queries}` and `{out_run}`.

```bash
poetry run python -m medical_rag_reranker.commands eval_retrieval \
  --overrides "run.eval_retrieval.retrieve_cmd=python -m medical_rag_reranker.commands.retrieval_run --retriever bm25 --index artifacts/bm25_index.json.gz --queries {queries} --out {out_run},run.eval_retrieval.retriever=bm25,run.eval_retrieval.top_k=100"
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

5. Build and evaluate the graph-guided MedQuAD benchmark:

```bash
poetry run python -m medical_rag_reranker.commands graph_benchmark
poetry run python -m medical_rag_reranker.commands index --overrides "retrieval=graph_hybrid"
poetry run python -m medical_rag_reranker.commands retrieval_run \
  --overrides "retrieval=graph_hybrid,retrieval_run.queries=data/processed/graph_eval_queries.jsonl"
poetry run python -m medical_rag_reranker.commands eval_retrieval \
  --overrides "retrieval=graph_hybrid,run.eval_retrieval.eval_queries=data/processed/graph_eval_queries.jsonl,run.eval_retrieval.qrels=data/processed/graph_qrels.tsv"
```

6. Compare baseline retrieval vs reranked retrieval:

```bash
poetry run python -m medical_rag_reranker.commands eval_reranked_retrieval \
  --overrides "retrieval=hybrid,retrieval_run.top_k=20,run.eval_reranked_retrieval.reranker_checkpoint_path=/absolute/path/to/reranker.ckpt"
```

7. Run an end-to-end RAG demo (1 question by default, up to 5):

```bash
poetry run python -m medical_rag_reranker.commands rag_demo
poetry run python -m medical_rag_reranker.commands rag_demo --overrides "run.rag_demo.num_questions=5"
poetry run python -m medical_rag_reranker.commands rag_demo --overrides "retrieval=hybrid,generation.use_reranker=true,generation.reranker_checkpoint_path=/absolute/path/to/reranker.ckpt,run.rag_demo.num_questions=3"
```

Notes:

- `medical_rag_reranker/retrieval/` and `configs/retrieval/` are already present in this repository.
- A separate `medical_rag_reranker/evaluation/` package is not required for this baseline step.
- Graph retrieval is implemented as a wrapper over the existing seed retriever. Use `retrieval=graph_bm25`, `retrieval=graph_hybrid`, or `retrieval=graph_hybrid_medcpt`.
- `rag_demo` writes a detailed markdown report to `reports/rag_demo.md` and a JSONL dump to `reports/rag_demo.jsonl`.

### Tests

Install dev dependencies and run the lightweight regression suite:

```bash
poetry install --with dev
poetry run pytest
```

---

## License

This project is provided for educational purposes.
