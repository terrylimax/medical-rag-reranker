# Medical RAG Reranker

End-to-end medical Retrieval-Augmented Generation (RAG) research pipeline for
comparing retrieval methods, optional reranking, answer generation, and
generation-quality evaluation.

The project started as Cross-Encoder reranker training code and now includes a
complete experimental harness:

- MedQuAD-style medical QA data preparation.
- Sparse, dense, hybrid, RAG-fusion, MedCPT, and graph-expanded retrieval.
- Optional Cross-Encoder reranking.
- Local `transformers` generation or remote OpenAI-compatible generation with
  vLLM / RunPod.
- Retrieval evaluation with TREC-style run files.
- Reference-free generation evaluation with heuristic metrics.
- Sampled LLM-as-a-Judge evaluation over existing generated answers.
- Experiment matrix runner for reproducible sweeps and final comparison tables.
- MLflow tracking.
- DVC/S3-compatible artifact sync.
- Async job API plus Docker Compose and Kubernetes deployment skeletons.

Last project-level update: 2026-05-13. The repository includes the latest
experiment reports for `e2e_practical_20260512` under
`artifacts/experiments/e2e_practical_20260512/`.

<img width="1219" height="718" alt="Architecture medical-rag-reranker-Architecure (1)" src="https://github.com/user-attachments/assets/e6ff4e15-35c2-407e-9a50-0992b8e682c9" />


## Current Status

Implemented and actively used:

- `prep_data`: builds processed QA/corpus/query/qrels files.
- `index`: builds retrieval indexes and manifests.
- `retrieval_run`: writes TREC retrieval runs.
- `eval_retrieval`: computes retrieval metrics and MLflow logs.
- `generate`: runs retrieval + generation, with optional reranker.
- `eval_generation`: evaluates generated answers with heuristic or LLM judge.
- `judge_generation_sample`: runs LLM-as-a-Judge on existing generation outputs
  without regenerating answers.
- `experiment_matrix`: orchestrates index, retrieval, generation, and summary
  sweeps.
- `train`: trains a Cross-Encoder reranker.
- `prep_retriever_training_data` and `train_retriever`: prepare and train a
  dense bi-encoder retriever.
- `artifact_push` / `artifact_pull`: sync runtime artifacts through DVC/S3.
- `submit_job`, `job_status`, `job_result`, `serve_jobs_api`: async inference
  job API.

Recent changes included in the current codebase:

- Remote OpenAI-compatible generation through vLLM / RunPod.
- Remote generation retries and exponential backoff.
- `GENERATION_REMOTE_CONCURRENCY` for concurrent remote generation requests.
- Prompt truncation through `generation.max_input_tokens`.
- Incremental raw JSONL writes and resume-by-`query_id` for long generation
  runs.
- Qdrant-backed dense retrieval and Qdrant as the dense backend inside hybrid,
  graph, and RAG-fusion methods.
- `judge_generation_sample` for cheap LLM-as-a-Judge checks on `N` examples per
  generation run.
- LLM judge retry, parse retry, compact JSON output instructions, and
  single-user-message mode for vLLM/Mistral compatibility.
- Experiment report files are Git-visible, while large indexes, raw data, and
  model artifacts remain DVC-managed or generated locally.

## Repository Layout

```text
medical-rag-reranker/
├── medical_rag_reranker/
│   ├── commands/             # Fire CLI entrypoints
│   ├── data/                 # data download, DVC helpers, prep scripts
│   ├── retrieval/            # BM25, dense, hybrid, graph, RAG-fusion
│   ├── inference/            # generation, reranking, single-pair inference
│   ├── evaluation/           # heuristic and LLM judge evaluation
│   ├── experiments/          # experiment matrix orchestration
│   ├── artifacts/            # artifact registry/sync helpers
│   ├── jobs/                 # async job storage/API
│   ├── models/               # Lightning Cross-Encoder model
│   ├── training/             # reranker/retriever training
│   └── utils/
├── configs/
│   ├── config.yaml           # root Hydra config
│   ├── data/
│   ├── generation/
│   ├── retrieval/
│   ├── run/
│   ├── jobs/
│   ├── logging/
│   ├── model/
│   └── train/
├── artifacts/experiments/    # committed reports + DVC-managed heavy artifacts
├── data/                     # DVC-managed local data
├── docs/                     # longer walkthroughs
├── reports/                  # ad hoc local reports
├── tests/
├── docker/
├── k8s/
├── airflow/
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
└── README.md
```

## Setup

Prerequisites:

- Python `>=3.11,<3.15`
- Poetry
- Git
- Optional: Docker, Docker Compose, Minikube, kubectl
- Optional: Qdrant endpoint for remote dense retrieval
- Optional: vLLM or RunPod OpenAI-compatible endpoint for remote generation and
  LLM-as-a-Judge

Install:

```bash
git clone https://github.com/terrylimax/medical-rag-reranker.git
cd medical-rag-reranker

poetry install --with dev
poetry run pre-commit install
```

Run checks:

```bash
poetry run pytest
poetry run pre-commit run -a
```

If a Python build installed through `pyenv` fails with
`ModuleNotFoundError: No module named '_lzma'`, reinstall Python with `xz`
support or switch Poetry to a Python build that includes `lzma`.

## CLI Overview

All commands are exposed through:

```bash
poetry run python -m medical_rag_reranker.commands <command>
```

Main commands:

| Command                                  | Purpose                                                                             |
| ---------------------------------------- | ----------------------------------------------------------------------------------- |
| `download_data`                          | Ensure raw data exists locally or through DVC.                                      |
| `prep_data`                              | Build `qa.jsonl`, `corpus.jsonl`, `splits.json`, `eval_queries.jsonl`, `qrels.tsv`. |
| `index`                                  | Build retrieval indexes/manifests.                                                  |
| `retrieval_run`                          | Run the configured retriever and write a TREC run.                                  |
| `eval_retrieval`                         | Compute retrieval metrics and log them.                                             |
| `generate`                               | Retrieve context and generate answers.                                              |
| `eval_generation`                        | Generate/evaluate answers with heuristic or LLM judge.                              |
| `judge_generation_sample`                | Judge existing generation raw JSONL files on a small sample.                        |
| `eval_reranked_retrieval`                | Compare retrieval before/after Cross-Encoder reranking.                             |
| `rag_demo`                               | Compact RAG demo for 1 to 5 questions.                                              |
| `train`                                  | Train Cross-Encoder reranker.                                                       |
| `prep_retriever_training_data`           | Build dense retriever training rows.                                                |
| `train_retriever`                        | Fine-tune dense bi-encoder retriever.                                               |
| `experiment_matrix`                      | Run or plan retrieval/generation experiment sweeps.                                 |
| `artifact_push` / `artifact_pull`        | Publish/download runtime artifacts through DVC/S3.                                  |
| `submit_job`, `job_status`, `job_result` | Async inference job flow.                                                           |
| `migrate_jobs_schema`                    | Apply Alembic migrations for Postgres job storage.                                  |
| `serve_jobs_api`                         | Run HTTP producer API for jobs.                                                     |

Hydra overrides can be passed as one string, a comma-separated string, or a JSON
list:

```bash
poetry run python -m medical_rag_reranker.commands index \
  --overrides "retrieval=bm25,data.use_dvc=false"

poetry run python -m medical_rag_reranker.commands train \
  --overrides '["train.max_epochs=2","train.batch_size=16"]'
```

## Minimal Local RAG Path

This path runs a small local baseline with BM25 and the default local generation
model.

```bash
poetry run python -m medical_rag_reranker.commands prep_data \
  --overrides "data.use_dvc=false"

poetry run python -m medical_rag_reranker.commands index \
  --overrides "data.use_dvc=false,retrieval=bm25"

poetry run python -m medical_rag_reranker.commands eval_retrieval \
  --overrides "data.use_dvc=false,retrieval=bm25"

poetry run python -m medical_rag_reranker.commands generate \
  --question "What is metformin used for?" \
  --overrides "data.use_dvc=false,retrieval=bm25,generation.local_files_only=true"

poetry run python -m medical_rag_reranker.commands eval_generation \
  --overrides "data.use_dvc=false,retrieval=bm25,generation.local_files_only=true"
```

Generation can use a cached Hugging Face model, a local model path, or a remote
OpenAI-compatible endpoint.

If the local model is not cached, either allow Hugging Face downloads or set
`generation.llm_model_name` to a local model path.

## Data and Artifacts

Data is not stored directly in Git. The project uses DVC for larger data and
runtime artifacts. The default DVC remote is a local folder `dvc_storage/`, which
is ignored by Git.

Core processed files:

- `data/processed/qa.jsonl`
- `data/processed/corpus.jsonl`
- `data/processed/splits.json`
- `data/processed/eval_queries.jsonl`
- `data/processed/qrels.tsv`
- `data/processed/medquad_graph.json` when graph retrieval is used

Runtime artifact sync:

```bash
export ARTIFACT_REMOTE_URI=s3://your-bucket/medical-rag/medquad-v1
export ARTIFACT_LOCAL_ROOT=.
export ARTIFACT_DVC_REMOTE=artifact_s3
export AWS_REGION=eu-central-1
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...

poetry run python -m medical_rag_reranker.commands artifact_push
poetry run python -m medical_rag_reranker.commands artifact_pull
```

The current `.gitignore` keeps heavy files ignored but allows experiment reports
to be committed:

- committed: markdown reports, CSV summaries, JSON summary files, matrix result
  manifests;
- ignored/DVC-managed: model weights, local data, dense indexes, raw runtime
  files, cache directories.

## Retrieval Methods

Retrieval is selected through `retrieval=<config_name>` and method-specific
overrides.

| Method family | Config / matrix method                         | Notes                                                 |
| ------------- | ---------------------------------------------- | ----------------------------------------------------- |
| BM25          | `retrieval=bm25`, `bm25`                       | Sparse lexical baseline.                              |
| Dense         | `retrieval=dense`, `dense_qdrant`              | Local dense index or Qdrant vector backend.           |
| Hybrid        | `retrieval=hybrid`, `hybrid_qdrant`            | BM25 + dense fusion.                                  |
| RAG fusion    | `rag_fusion_bm25`, `rag_fusion_dense`          | Deterministic query expansion + RRF.                  |
| Graph BM25    | `graph_bm25`                                   | Seed retrieval plus MedQuAD metadata graph expansion. |
| Graph hybrid  | `graph_hybrid`, `graph_hybrid_medcpt`          | Graph expansion over hybrid/MedCPT seed retrieval.    |
| MedCPT        | `medcpt`, `medcpt_zero_shot`, `medcpt_trained` | Biomedical bi-encoder retrieval.                      |

Qdrant examples:

```bash
export QDRANT_URL=https://your-qdrant-endpoint
export QDRANT_API_KEY=your-token
export QDRANT_COLLECTION=medical_rag_docs

poetry run python -m medical_rag_reranker.commands index \
  --overrides "retrieval=dense,retrieval.vector_backend=qdrant"

poetry run python -m medical_rag_reranker.commands eval_retrieval \
  --overrides "retrieval=dense,retrieval.vector_backend=qdrant"
```

Qdrant as a dense backend inside other methods:

```bash
# Local BM25 + Qdrant-backed dense retrieval
retrieval=hybrid,retrieval.dense_backend=qdrant

# Qdrant dense seed retrieval + graph expansion
retrieval=graph_bm25,retrieval.base_retriever=dense,retrieval.vector_backend=qdrant

# RAG fusion over Qdrant-backed dense retrieval
retrieval=rag_fusion_dense,retrieval.vector_backend=qdrant
```

## Reranking

The project supports optional Cross-Encoder reranking after initial retrieval.
Generation uses `generation.retrieve_top_k` candidates, optionally reranks them,
and sends `generation.top_k` documents into the final prompt.

Single-question generation with reranking:

```bash
poetry run python -m medical_rag_reranker.commands generate \
  --question "What is metformin used for?" \
  --overrides "retrieval=bm25,generation.use_reranker=true,generation.reranker_checkpoint_path=/absolute/path/to/reranker.ckpt,generation.retrieve_top_k=20,generation.top_k=5"
```

Retrieval-only reranker comparison:

```bash
poetry run python -m medical_rag_reranker.commands eval_reranked_retrieval \
  --overrides "retrieval=bm25,retrieval_run.top_k=20,run.eval_reranked_retrieval.reranker_checkpoint_path=/absolute/path/to/reranker.ckpt"
```

If `generation.use_reranker=true`, `generation.reranker_checkpoint_path` is
required.

## Generation

Generation backend is configured in `configs/generation/generation.yaml`.

Important settings:

| Setting / env var                                         | Meaning                                                           |
| --------------------------------------------------------- | ----------------------------------------------------------------- |
| `generation.backend` / `GENERATION_BACKEND`               | `local` or `openai_compatible`.                                   |
| `generation.llm_model_name` / `GENERATION_LLM_MODEL_NAME` | Local HF model or remote served model ID.                         |
| `generation.corpus_path` / `GENERATION_CORPUS_PATH`       | Corpus JSONL; set `null` only if retriever returns text payloads. |
| `generation.index` / `GENERATION_INDEX_PATH`              | Index or retriever manifest path.                                 |
| `generation.top_k`                                        | Number of docs used in final prompt.                              |
| `generation.retrieve_top_k`                               | Candidate count before reranking.                                 |
| `generation.max_input_tokens`                             | Prompt truncation budget.                                         |
| `generation.max_new_tokens`                               | Generation output budget.                                         |
| `VLLM_BASE_URL`                                           | OpenAI-compatible `/v1` endpoint.                                 |
| `VLLM_API_KEY`                                            | Remote generation API key.                                        |
| `VLLM_API_TYPE`                                           | `chat` or completion-style API.                                   |
| `VLLM_TIMEOUT_SECONDS`                                    | Per-request timeout.                                              |
| `GENERATION_REMOTE_CONCURRENCY`                           | Concurrent remote generation requests.                            |
| `GENERATION_REMOTE_MAX_RETRIES`                           | Retry count for transient remote failures.                        |

Remote vLLM / RunPod generation:

```bash
export GENERATION_BACKEND=openai_compatible
export VLLM_BASE_URL=https://your-vllm-endpoint/openai/v1
export VLLM_API_KEY=your-token
export VLLM_API_TYPE=chat
export VLLM_TIMEOUT_SECONDS=180
export GENERATION_REMOTE_CONCURRENCY=4
export GENERATION_REMOTE_MAX_RETRIES=3
export GENERATION_REMOTE_RETRY_BACKOFF_SECONDS=2

poetry run python -m medical_rag_reranker.commands generate \
  --question "What is metformin used for?" \
  --overrides "retrieval=bm25,generation.llm_model_name=mistralai/mistral-small-3.2-24b-instruct-2506"
```

For Qdrant-backed generation where retrieved points include document text:

```bash
export GENERATION_CORPUS_PATH=null
```

Do not set `GENERATION_CORPUS_PATH=null` for local BM25/dense/hybrid indexes
unless the active retriever returns document text in its results.

Long generation runs now write raw JSONL incrementally and can resume by
`query_id`, which is important for RunPod timeouts or interrupted runs.

## Evaluation Metrics

Retrieval metrics:

- `Hit@1`, `Hit@3`, `Hit@5`
- `MRR@10`
- `NDCG@10`
- latency p50/p95

Generation heuristic metrics:

- answer relevance
- context relevance
- groundedness
- citation presence rate
- supported and unsupported citation rates
- insufficient context rate
- empty answer rate
- retrieval, rerank, generation, and end-to-end latency

LLM-as-a-Judge metrics:

- `faithfulness`
- `relevance`
- `completeness`
- `safety`
- `verdict`
- `rationale`
- aggregate pass/fail rates

Latency metrics are reported as raw per-question latency in the main summary and
as throughput-normalized latency in the audited comparison table.

## LLM-as-a-Judge

The LLM judge uses a separate OpenAI-compatible endpoint from the answer
generator. This lets the project generate answers once, then evaluate them
later with a different model or a smaller sample.

Recommended RunPod/vLLM environment for Mistral:

```bash
export LLM_JUDGE_BASE_URL=https://api.runpod.ai/v2/<endpoint-id>/openai/v1
export LLM_JUDGE_API_KEY=your-token
export LLM_JUDGE_MODEL=mistralai/mistral-small-3.2-24b-instruct-2506
export LLM_JUDGE_SINGLE_USER_MESSAGE=true
export LLM_JUDGE_MAX_TOKENS=192
export LLM_JUDGE_MAX_CONTEXT_DOCS=2
export LLM_JUDGE_MAX_DOC_CHARS=500
export LLM_JUDGE_TIMEOUT_SECONDS=180
export LLM_JUDGE_MAX_RETRIES=3
export LLM_JUDGE_PARSE_MAX_RETRIES=1
export LLM_JUDGE_RETRY_BACKOFF_SECONDS=2
```

Check the served model before judging:

```bash
python - <<'PY'
import json, os, urllib.request

base = os.environ["LLM_JUDGE_BASE_URL"].rstrip("/")
key = os.environ.get("LLM_JUDGE_API_KEY") or ""
headers = {"Authorization": f"Bearer {key}"} if key else {}
req = urllib.request.Request(base + "/models", headers=headers)
print(urllib.request.urlopen(req, timeout=180).read().decode())
PY
```

Judge an existing generation matrix on 5 examples per run:

```bash
poetry run python -m medical_rag_reranker.commands judge_generation_sample \
  --run_id e2e_practical_20260512 \
  --examples_limit 5 \
  --concurrency 4
```

Judge 50 examples per run:

```bash
poetry run python -m medical_rag_reranker.commands judge_generation_sample \
  --run_id e2e_practical_20260512 \
  --examples_limit 50 \
  --concurrency 4
```

Output files:

- `artifacts/experiments/<run_id>/llm_judge_sample/*.llm_judge_n<N>.jsonl`
- `artifacts/experiments/<run_id>/llm_judge_sample/*.llm_judge_n<N>.summary.json`
- `artifacts/experiments/<run_id>/llm_judge_sample/*.llm_judge_n<N>.md`
- `artifacts/experiments/<run_id>/llm_judge_sample/llm_judge_sample_n<N>_summary.csv`
- `artifacts/experiments/<run_id>/llm_judge_sample/llm_judge_sample_n<N>_ranking.md`

The command supports resume. Existing judged rows are reused and incomplete
files continue from pending examples.

## Experiment Matrix

The matrix runner can plan or run multiple methods across index, retrieval, and
generation stages.

Smoke dry run:

```bash
poetry run python -m medical_rag_reranker.commands experiment_matrix \
  --profile smoke \
  --stage all \
  --run_id smoke_matrix \
  --training_mode colab_artifacts \
  --dry_run true
```

Practical remote run used for the latest report:

```bash
poetry run python -m medical_rag_reranker.commands experiment_matrix \
  --profile practical_remote \
  --stage index \
  --run_id e2e_practical_20260512 \
  --training_mode colab_artifacts \
  --resume true

poetry run python -m medical_rag_reranker.commands experiment_matrix \
  --profile practical_remote \
  --stage retrieval \
  --run_id e2e_practical_20260512 \
  --training_mode colab_artifacts \
  --resume true

GENERATION_REMOTE_CONCURRENCY=4 \
poetry run python -m medical_rag_reranker.commands experiment_matrix \
  --profile practical_remote \
  --stage generation \
  --run_id e2e_practical_20260512 \
  --training_mode colab_artifacts \
  --resume true
```

Profile summary:

- `smoke`: small BM25 sanity run.
- `practical_remote`: 12 method families, 16 retrieval/index jobs, 32
  generation jobs with reranker on/off.
- `full_remote`: larger sweep over hybrid, graph, RAG-fusion, generation, and
  judge dimensions.

Important matrix outputs:

- `artifacts/experiments/<run_id>/experiment_matrix.result.json`
- `artifacts/experiments/<run_id>/e2e_summary.csv`
- `artifacts/experiments/<run_id>/best_configs.json`
- `artifacts/experiments/<run_id>/comparison_report.md`
- `artifacts/experiments/<run_id>/generation/*.summary.json`
- `artifacts/experiments/<run_id>/generation/*.raw.jsonl`
- `artifacts/experiments/<run_id>/llm_judge_sample/*`

## Latest Experiment: e2e_practical_20260512

Committed report files are under:

```text
artifacts/experiments/e2e_practical_20260512/
```

The run contains:

- 16 index/retrieval configurations.
- 32 generation runs (`top_k=5`, `retrieve_top_k=20`, reranker on/off).
- Heuristic generation evaluation over the full eval set.
- LLM-as-a-Judge sample evaluation with `n=5` and `n=50` examples per run.
- Latency concurrency audit for mixed sequential/concurrent generation runs.

Key files:

- `e2e_summary.csv`: primary retrieval + generation summary.
- `e2e_summary_with_llm_judge_n50.csv`: merged summary with LLM judge sample.
- `e2e_summary_with_llm_judge_n50_latency_audited.csv`: raw and
  throughput-normalized latency columns.
- `llm_judge_sample/llm_judge_sample_n50_summary.csv`: aggregate LLM judge
  metrics for 32 runs.
- `llm_judge_sample/llm_judge_sample_n50_ranking.md`: ranking by pass rate.
- `llm_judge_sample/e2e_summary_with_llm_judge_n50_top.md`: top combined table.
- `llm_judge_sample/latency_concurrency_audit.md`: concurrency metadata audit.

Top n50 LLM judge runs from the latest report:

| Rank | Method              | Rerank | LLM pass | Faithfulness | Hit@5 | MRR@10 |
| ---: | ------------------- | -----: | -------: | -----------: | ----: | -----: |
|    1 | `rag_fusion_qdrant` |  false |     0.86 |         4.64 | 0.570 |  0.415 |
|    2 | `medcpt_zero_shot`  |   true |     0.80 |         4.32 | 0.513 |  0.329 |
|    3 | `rag_fusion_qdrant` |  false |     0.78 |         4.38 | 0.607 |  0.469 |
|    4 | `rag_fusion_bm25`   |  false |     0.76 |         4.34 | 0.590 |  0.388 |
|    5 | `rag_fusion_bm25`   |   true |     0.76 |         4.30 | 0.587 |  0.387 |

Retrieval-only leaders in the practical run:

- `hybrid_medcpt_trained`: `Hit@5=0.7567`, `MRR@10=0.6080`.
- `graph_hybrid_medcpt_trained`: `Hit@5=0.7567`, `MRR@10=0.6093`.
- `medcpt_trained`: `Hit@5=0.7267`, `MRR@10=0.5522`.
- `hybrid_qdrant`: `Hit@5=0.6733`, `MRR@10=0.4904`.

Result interpretation:

- Retrieval metrics and LLM judge metrics provide complementary views of system
  quality.
- RAG-fusion methods produced the strongest n50 LLM judge pass rates in this
  run, while MedCPT hybrid methods produced the strongest retrieval metrics.
- The n50 judge ranking includes reranker deltas in
  `llm_judge_sample_n50_ranking.md`.

## Latency and Concurrency

Remote generation supports `GENERATION_REMOTE_CONCURRENCY` for faster
OpenAI-compatible inference. The latest audited table keeps both latency views:

- raw latency: observed per-question request latency;
- throughput-normalized latency: raw latency divided by inferred concurrency,
  useful only for throughput/cost comparison.

For user-facing per-question latency, prefer raw latency. For comparing how much
wall-clock time a batched remote setup consumes, use throughput-normalized
columns with the caveat that this is an approximation.

Audit file:

```text
artifacts/experiments/e2e_practical_20260512/llm_judge_sample/latency_concurrency_audit.md
```

Runs without concurrency metadata are assumed to have concurrency `1` in the
audited table.

## MLflow

Training, retrieval, and generation evaluation can log to MLflow.

Start local UI:

```bash
poetry run mlflow ui --host 127.0.0.1 --port 8080
```

MLflow settings live in:

- `configs/logging/mlflow.yaml`
- per-command `configs/run/*.yaml`

Logged data includes config parameters, Git commit metadata, retrieval metrics,
generation metrics, and selected latency metrics.

## Training

Cross-Encoder reranker training:

```bash
poetry run python -m medical_rag_reranker.commands train
```

Example overrides:

```bash
poetry run python -m medical_rag_reranker.commands train \
  --overrides "train.max_epochs=2,train.batch_size=16"
```

Dense retriever training:

```bash
poetry run python -m medical_rag_reranker.commands prep_retriever_training_data
poetry run python -m medical_rag_reranker.commands train_retriever
```

The matrix profile `training_mode=colab_artifacts` assumes trained artifacts
were produced externally, for example in Colab, and already exist under
`artifacts/retriever/...`.

## Docker Compose

Docker Compose runs:

- `nginx`: public gateway.
- `app`: FastAPI producer.
- `static`: static site.
- `worker`: background consumer.
- `rabbitmq`: broker.
- `postgres`: job storage.

Start:

```bash
cp .env.example .env
mkdir -p data artifacts runs reports .hf_cache
docker compose up -d --build
docker compose ps
```

Run migrations:

```bash
docker compose exec app \
  python -m medical_rag_reranker.commands migrate_jobs_schema
```

Health and job API:

```bash
curl -sS "http://127.0.0.1:${NGINX_PORT:-8080}/api/health"

curl -sS -X POST "http://127.0.0.1:${NGINX_PORT:-8080}/api/jobs" \
  -H "Content-Type: application/json" \
  -d '{"question":"What is metformin used for?"}'
```

For remote artifact sync in Docker Compose, put the artifact, AWS, Qdrant, and
vLLM variables in `.env`.

## Kubernetes / Minikube

Kubernetes manifests live in `k8s/` and use namespace `medical-rag`.

Cluster components:

- ingress-nginx with Basic Auth.
- `nginx` internal gateway.
- `app` FastAPI producer.
- `worker` consumer.
- `rabbitmq` broker.
- `postgres-master` and `postgres-slave` StatefulSets.
- `static` site.

Local Minikube flow:

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

Expose ingress locally:

```bash
kubectl -n ingress-nginx port-forward service/ingress-nginx-controller 8081:80
```

Verify:

```bash
export APP_URL=http://127.0.0.1:8081
export BASIC_AUTH=reviewer:reviewer

curl -sS -u "$BASIC_AUTH" "$APP_URL/api/health"
```

The Kubernetes flow is primarily for deployment validation. The image also
ships small demo assets for basic job checks.

## Airflow

An Airflow DAG exists at:

```text
airflow/dags/full_experiment_matrix.py
```

It orchestrates the same high-level flow as the CLI:

1. artifact pull
2. preflight
3. training/artifact availability checks
4. index
5. retrieval evaluation
6. generation evaluation
7. summary/report writing
8. artifact push

## Useful Result Views

Show top LLM judge runs:

```bash
column -s, -t < artifacts/experiments/e2e_practical_20260512/llm_judge_sample/llm_judge_sample_n50_summary.csv | less -S
```

Open the combined top report:

```bash
less artifacts/experiments/e2e_practical_20260512/llm_judge_sample/e2e_summary_with_llm_judge_n50_top.md
```

Inspect audited latency:

```bash
less artifacts/experiments/e2e_practical_20260512/llm_judge_sample/latency_concurrency_audit.md
```

List generation summaries:

```bash
find artifacts/experiments/e2e_practical_20260512/generation \
  -maxdepth 1 \
  -name "*.summary.json" \
  -print
```

## Troubleshooting

`FileNotFoundError: Corpus file does not exist: /app/data/processed/corpus.jsonl`

- Local BM25/dense/hybrid generation needs `generation.corpus_path`.
- Run `prep_data` or set `GENERATION_CORPUS_PATH` to the correct local path.
- Use `GENERATION_CORPUS_PATH=null` only for remote retrievers that return text
  payloads.

vLLM HTTP 500 with context length error:

- Reduce `generation.top_k`.
- Reduce `generation.max_input_tokens`.
- Reduce `generation.max_new_tokens`.
- Use fewer or shorter context docs for LLM judge:
  `LLM_JUDGE_MAX_CONTEXT_DOCS`, `LLM_JUDGE_MAX_DOC_CHARS`.

RunPod endpoint serves the wrong model:

- Check `LLM_JUDGE_BASE_URL` or `VLLM_BASE_URL`.
- Query `/models`.
- Make sure shell env was reloaded:

```bash
set -a
source .env
set +a
```

`generation.use_reranker=true requires generation.reranker_checkpoint_path`

- Provide `GENERATION_RERANKER_CHECKPOINT_PATH` or set
  `generation.use_reranker=false`.

Remote generation timeout or disconnect:

- Increase `VLLM_TIMEOUT_SECONDS`.
- Lower `GENERATION_REMOTE_CONCURRENCY`.
- Keep `--resume true` for matrix runs.
- Check RunPod worker logs and max context settings.

Untracked old experiment directories after report ignore changes:

- The current `.gitignore` allows experiment report files under
  `artifacts/experiments/**`.
- Old report directories may become visible as untracked.
- Add only the run directories you intend to commit.

## Tests

Run all tests:

```bash
poetry run pytest
```

Targeted tests for recent generation/judge changes:

```bash
poetry run pytest \
  tests/test_generate_helpers.py \
  tests/test_llm_judge.py \
  tests/test_judge_generation.py \
  tests/test_experiment_matrix.py
```

Run lint:

```bash
poetry run ruff check medical_rag_reranker tests
```

## License

MIT. See `pyproject.toml`.
