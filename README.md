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

## Recent Changes (Dec 2025)

- Config-driven CLI: training/inference now run via Hydra configs in `configs/` with a Fire-based entrypoint (`python -m medical_rag_reranker.commands ...`).
- Robust overrides: `--overrides` supports a single override, comma/space-separated overrides, or a JSON list.
- DVC integration: `train` and `infer` try `dvc pull` first, then fall back to downloading open datasets, then `dvc add/push` to the local remote.
- MLflow logging: Lightning logs hyperparameters (including `git.commit_id`) and metrics to the tracking server configured in `configs/logging/mlflow.yaml` (default `http://127.0.0.1:8080`).
- Extra metrics: validation now logs `val/auroc` and `val/f1` in addition to `train/loss` and `val/loss` (>= 3 charts in MLflow).

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

Training is implemented using Lightning and configured with Hydra.

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
- Logged metrics include at least: `train/loss`, `val/loss`, `val/auroc`, `val/f1`.

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

## License

This project is provided for educational purposes.
