# Colab cells for GPU retriever training

Copy each fenced block into a separate Google Colab code cell and run from top to bottom.

## Cell 0. GPU check

```python
import torch

print("CUDA:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise RuntimeError("Enable T4 GPU: Runtime -> Change runtime type.")

print("GPU:", torch.cuda.get_device_name(0))
```

## Cell 1. Imports and run settings

```python
from pathlib import Path
import json
import os
import subprocess
import sys

REPO_OWNER = "terrylimax"
REPO_NAME = "medical-rag-reranker"
REPO_URL = f"https://github.com/{REPO_OWNER}/{REPO_NAME}.git"
BRANCH = "colab-gpu-retriever"
PROJECT_DIR = Path("/content") / REPO_NAME

USE_DRIVE = True
TOP_K = 10
EPOCHS = 2
BATCH_SIZE = 8
NEGATIVES_PER_QUERY = 4
HARD_NEGATIVE_POOL_SIZE = 50
```

## Cell 2. Drive and output paths

```python
def build_output_root(use_drive=True):
    if not use_drive:
        return Path("/content") / "medical-rag-reranker-colab"

    try:
        from google.colab import drive

        print("Authorize Drive if Colab asks.")
        drive.mount("/content/drive", force_remount=False)
        base = Path("/content/drive/MyDrive")
        return base / "medical-rag-reranker-colab"
    except Exception as exc:
        print("Drive mount failed; using /content.")
        print(type(exc).__name__, str(exc))
        return Path("/content") / "medical-rag-reranker-colab"


DRIVE_ROOT = build_output_root(USE_DRIVE)
ARTIFACT_ROOT = DRIVE_ROOT / "artifacts"
RUN_ROOT = DRIVE_ROOT / "runs"
VAL_DIR = RUN_ROOT / "validation_full"
TRAINING_DIR = ARTIFACT_ROOT / "retriever_training_full"
RETRIEVER_DIR = ARTIFACT_ROOT / "retriever"
RETRIEVER_DIR = RETRIEVER_DIR / "medcpt-bi-encoder-full"

for path in [
    DRIVE_ROOT,
    ARTIFACT_ROOT,
    RUN_ROOT,
    VAL_DIR,
    TRAINING_DIR,
    RETRIEVER_DIR,
]:
    path.mkdir(parents=True, exist_ok=True)

print("Output root:", DRIVE_ROOT)
```

## Cell 3. Shell helper

```python
def sh(*args, cwd=None):
    args = [str(arg) for arg in args]
    print("+", " ".join(args))
    subprocess.run(
        args,
        cwd=None if cwd is None else str(cwd),
        check=True,
    )
```

## Cell 4. Clone branch and install package

```python
if not (PROJECT_DIR / ".git").exists():
    sh(
        "git",
        "clone",
        "--branch",
        BRANCH,
        REPO_URL,
        PROJECT_DIR,
    )
else:
    sh("git", "fetch", "origin", BRANCH, cwd=PROJECT_DIR)
    sh("git", "checkout", BRANCH, cwd=PROJECT_DIR)
    sh("git", "pull", "--ff-only", cwd=PROJECT_DIR)

os.chdir(PROJECT_DIR)

sh(
    sys.executable,
    "-m",
    "pip",
    "install",
    "-q",
    "-e",
    ".",
    "--no-deps",
)

print("Project dir:", PROJECT_DIR)
```

## Cell 5. Install runtime dependencies

```python
packages = [
    "transformers",
    "datasets",
    "tokenizers",
    "hydra-core",
    "omegaconf",
    "fire",
    "rank-bm25",
    "pytrec-eval",
    "mlflow",
    "sentence-transformers",
    "pandas",
    "pyarrow",
    "scikit-learn",
    "scipy",
    "tqdm",
]

sh(
    sys.executable,
    "-m",
    "pip",
    "install",
    "-q",
    *packages,
)

from datasets import load_dataset
from IPython.display import display
import pandas as pd

print("CUDA:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
```

## Cell 6. Project command helpers

```python
os.chdir(PROJECT_DIR)


def run_project(command, overrides):
    payload = json.dumps(overrides, ensure_ascii=False)
    cmd = [
        sys.executable,
        "-m",
        "medical_rag_reranker.commands",
        command,
        "--overrides",
        payload,
    ]
    sh(*cmd, cwd=PROJECT_DIR)


def common_overrides():
    return [
        "data.use_dvc=false",
        f"paths.artifacts_dir={ARTIFACT_ROOT}",
        f"paths.runs_dir={RUN_ROOT}",
        "run.prep_data.out_dir=data/processed",
    ]


print("Helpers are ready")
```

## Cell 7. Download MedQuAD and prepare data

```python
raw_dir = PROJECT_DIR / "data/raw/medquad"
raw_path = raw_dir / "train.parquet"

if not raw_path.exists():
    raw_dir.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("lavita/MedQuAD")
    ds["train"].to_parquet(str(raw_path))
    print("Saved MedQuAD to", raw_path)
else:
    print("MedQuAD already exists:", raw_path)

raw_cfg = "run.prep_data.raw_nih_path="
raw_cfg += "data/raw/medquad/train.parquet"
prep_overrides = common_overrides()
prep_overrides += [
    raw_cfg,
    "run.prep_data.out_dir=data/processed",
]

run_project("prep_data", prep_overrides)
```

## Cell 8. Build validation queries and qrels

```python
processed = PROJECT_DIR / "data/processed"


def read_jsonl(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


qa_rows = read_jsonl(processed / "qa.jsonl")
corpus_rows = read_jsonl(processed / "corpus.jsonl")
splits_path = processed / "splits.json"
splits = json.loads(splits_path.read_text(encoding="utf-8"))

qa_by_qid = {
    str(row["question_id"]): row
    for row in qa_rows
    if row.get("question_id") is not None
}
corpus_by_qid = {
    str(row["question_id"]): row
    for row in corpus_rows
    if row.get("question_id") is not None
}

num_val = 0
queries_path = VAL_DIR / "val_queries.jsonl"
qrels_path = VAL_DIR / "val_qrels.tsv"

with queries_path.open("w", encoding="utf-8") as fq:
    with qrels_path.open("w", encoding="utf-8") as fr:
        for qid in map(str, splits.get("val", [])):
            qa = qa_by_qid.get(qid)
            doc = corpus_by_qid.get(qid)
            if not qa or not doc:
                continue

            question = str(qa.get("question") or "").strip()
            doc_id = str(doc.get("doc_id") or "").strip()
            if not question or not doc_id:
                continue

            record = {"query_id": qid, "question": question}
            fq.write(json.dumps(record, ensure_ascii=False) + "\n")
            fr.write(f"{qid}\t0\t{doc_id}\t1\n")
            num_val += 1

print("qa rows:", len(qa_rows))
print("corpus rows:", len(corpus_rows))
print("val queries:", num_val)
print("validation dir:", VAL_DIR)
```

## Cell 9. Retrieval evaluation helper

```python
def eval_retriever(run_name, retrieval_config, extra=None):
    query_file = VAL_DIR / "val_queries.jsonl"
    qrels_file = VAL_DIR / "val_qrels.tsv"
    run_file = VAL_DIR / f"{run_name}.trec"

    overrides = common_overrides()
    overrides += [
        f"retrieval={retrieval_config}",
        "run.retrieval_index.corpus=data/processed/corpus.jsonl",
        f"retrieval_run.queries={query_file}",
        f"retrieval_run.out={run_file}",
        f"retrieval_run.top_k={TOP_K}",
        f"retrieval_run.run_name={run_name}",
        f"run.eval_retrieval.eval_queries={query_file}",
        f"run.eval_retrieval.qrels={qrels_file}",
        f"run.eval_retrieval.out_run={run_file}",
        'run.eval_retrieval.ks="5,10"',
        f"run.eval_retrieval.run_name={run_name}",
        f"run.eval_retrieval.retriever={run_name}",
    ]
    if extra:
        overrides += extra

    run_project("index", overrides)
    run_project("eval_retrieval", overrides)

    metrics_path = run_file.with_suffix(".trec.metrics.json")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["run_name"] = run_name
    metrics["run_file"] = str(run_file)
    return metrics
```

## Cell 10. Baseline: BM25

```python
metrics_rows = []
metrics_rows.append(eval_retriever("bm25", "bm25"))
display(pd.DataFrame(metrics_rows))
```

## Cell 11. Baseline: MiniLM dense

```python
metrics_rows.append(eval_retriever("dense_minilm", "dense"))
display(pd.DataFrame(metrics_rows))
```

## Cell 12. Baseline: MiniLM hybrid

```python
metrics_rows.append(eval_retriever("hybrid_minilm", "hybrid"))
display(pd.DataFrame(metrics_rows))
```

## Cell 13. Build hard-negative training data

```python
train_data_overrides = common_overrides()
train_data_overrides += [
    f"run.retriever_training_data.out_dir={TRAINING_DIR}",
]
neg_cfg = "run.retriever_training_data.negatives_per_query="
neg_cfg += str(NEGATIVES_PER_QUERY)
pool_cfg = "run.retriever_training_data.hard_negative_pool_size="
pool_cfg += str(HARD_NEGATIVE_POOL_SIZE)
train_data_overrides += [neg_cfg, pool_cfg]

run_project(
    "prep_retriever_training_data",
    train_data_overrides,
)

for name in ["train_retriever.jsonl", "val_retriever.jsonl"]:
    path = TRAINING_DIR / name
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            rows = sum(1 for line in handle if line.strip())
        print(name, "rows:", rows)
```

## Cell 14. Train MedCPT bi-encoder on GPU

```python
train_path = TRAINING_DIR / "train_retriever.jsonl"
val_path = TRAINING_DIR / "val_retriever.jsonl"

train_overrides = common_overrides()
train_overrides += [
    f"run.retriever_training_data.out_dir={TRAINING_DIR}",
    f"run.train_retriever.train_path={train_path}",
    f"run.train_retriever.val_path={val_path}",
    f"run.train_retriever.output_dir={RETRIEVER_DIR}",
    "run.train_retriever.device=cuda",
    f"run.train_retriever.batch_size={BATCH_SIZE}",
    f"run.train_retriever.epochs={EPOCHS}",
    "run.train_retriever.lr=2e-5",
    "run.train_retriever.log_every_n_steps=20",
]

run_project("train_retriever", train_overrides)
print("trained retriever dir:", RETRIEVER_DIR)
```

## Cell 15. Evaluate trained retrievers

```python
trained_query_encoder = RETRIEVER_DIR / "query_encoder"
trained_doc_encoder = RETRIEVER_DIR / "doc_encoder"

model_overrides = [
    f"retrieval.query_model_name={trained_query_encoder}",
    f"retrieval.doc_model_name={trained_doc_encoder}",
    f"run.retrieval_index.query_model={trained_query_encoder}",
    f"run.retrieval_index.doc_model={trained_doc_encoder}",
]

medcpt_extra = model_overrides + [
    "retrieval.index_file=trained_medcpt_index.pkl",
]
metrics_rows.append(
    eval_retriever("trained_medcpt", "medcpt", medcpt_extra)
)

hybrid_extra = model_overrides + [
    "retrieval.index_file=hybrid_trained_medcpt/hybrid_index.json",
]
metrics_rows.append(
    eval_retriever(
        "hybrid_trained_medcpt",
        "hybrid_medcpt",
        hybrid_extra,
    )
)

display(pd.DataFrame(metrics_rows))
```

## Cell 16. Final comparison table

```python
run_names = [
    "bm25",
    "dense_minilm",
    "hybrid_minilm",
    "trained_medcpt",
    "hybrid_trained_medcpt",
]

loaded_rows = []
for run_name in run_names:
    metrics_path = VAL_DIR / f"{run_name}.trec.metrics.json"
    run_file = VAL_DIR / f"{run_name}.trec"

    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        metrics["run_name"] = run_name
        metrics["run_file"] = str(run_file)
        loaded_rows.append(metrics)

summary = pd.DataFrame(loaded_rows)
metric_cols = [
    "run_name",
    "NDCG@10",
    "R@10",
    "P@10",
    "NDCG@5",
    "R@5",
    "P@5",
    "num_queries_eval",
    "run_file",
]
cols = [col for col in metric_cols if col in summary.columns]
summary = summary[cols]

if "NDCG@10" in summary.columns:
    summary = summary.sort_values("NDCG@10", ascending=False)

summary_csv = VAL_DIR / "retrieval_comparison.csv"
summary_json = VAL_DIR / "retrieval_comparison.json"

summary.to_csv(summary_csv, index=False)
summary.to_json(
    summary_json,
    orient="records",
    force_ascii=False,
    indent=2,
)

print("comparison csv:", summary_csv)
print("comparison json:", summary_json)
print("trained retriever dir:", RETRIEVER_DIR)
display(summary)
```
