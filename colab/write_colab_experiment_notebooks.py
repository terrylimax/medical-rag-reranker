from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]


def md(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": dedent(source).strip().splitlines(keepends=True),
    }


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": dedent(source).strip().splitlines(keepends=True),
    }


def write_notebook(path: Path, title: str, cells: list[dict]) -> None:
    notebook = {
        "cells": cells,
        "metadata": {
            "colab": {
                "provenance": [],
                "gpuType": "T4",
            },
            "kernelspec": {
                "display_name": "Python 3",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.11",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=2) + "\n", "utf-8")
    print(f"Wrote {path.relative_to(ROOT)}: {title}")


NOTEBOOK_01 = [
    md(
        """
        # 01: Train Retriever And Build Indices

        Цель: подготовить данные, дообучить MedCPT bi-encoder на hard negatives, построить все базовые индексы retrieval, сохранить промежуточные результаты в Google Drive как Colab cache и выгрузить финальные артефакты в S3/DVC через `artifact_push`.

        Основные выходы:
        - `data/processed`: `qa.jsonl`, `corpus.jsonl`, `splits.json`, `eval_queries.jsonl`, `qrels.tsv`
        - `artifacts/models/retriever/trained_medcpt_biencoder`: `query_encoder`, `doc_encoder`, `training_summary.json`
        - `artifacts/indices`: BM25, MiniLM dense, MedCPT, hybrid, trained MedCPT, graph metadata index
        - `manifest.json`: единая карта путей для остальных ноутбуков
        """
    ),
    code(
        """
        import platform
        import torch

        print("Python:", platform.python_version())
        print("Torch:", torch.__version__)
        print("CUDA available:", torch.cuda.is_available())
        if not torch.cuda.is_available():
            raise RuntimeError("Enable a GPU in Colab: Runtime -> Change runtime type -> T4/A100.")
        print("GPU:", torch.cuda.get_device_name(0))
        """
    ),
    code(
        """
        from pathlib import Path
        import datetime as dt
        import json
        import os
        import shutil
        import subprocess
        import sys
        import time

        REPO_OWNER = "terrylimax"
        REPO_NAME = "medical-rag-reranker"
        REPO_URL = f"https://github.com/{REPO_OWNER}/{REPO_NAME}.git"
        BRANCH = "main"

        RUN_ID = os.environ.get("COLAB_RUN_ID", "medquad_full_v1")
        USE_DRIVE = True

        EVAL_SIZE = 300
        TOP_K = 50
        KS = "1,5,10,20,50"

        TRAIN_BI_ENCODER = True
        BI_ENCODER_EPOCHS = 2
        BI_ENCODER_BATCH_SIZE = 8
        ENCODE_BATCH_SIZE = 32
        NEGATIVES_PER_QUERY = 4
        HARD_NEGATIVE_POOL_SIZE = 50

        PROJECT_DIR = Path("/content") / REPO_NAME
        PROJECT_PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
        RAW_MEDQUAD_PATH = PROJECT_DIR / "data" / "raw" / "medquad" / "train.parquet"

        print("RUN_ID:", RUN_ID)
        """
    ),
    code(
        """
        def mount_drive_or_content(use_drive: bool) -> Path:
            if not use_drive:
                return Path("/content") / "medical-rag-reranker-colab"

            try:
                from google.colab import drive

                drive.mount("/content/drive", force_remount=False)
                return Path("/content/drive/MyDrive") / "medical-rag-reranker-colab"
            except Exception as exc:
                print("Drive mount failed; falling back to /content.")
                print(type(exc).__name__, str(exc))
                return Path("/content") / "medical-rag-reranker-colab"


        DRIVE_BASE = mount_drive_or_content(USE_DRIVE)
        DRIVE_ROOT = DRIVE_BASE / RUN_ID
        DATA_DRIVE_DIR = DRIVE_ROOT / "data"
        PROCESSED_DRIVE_DIR = DATA_DRIVE_DIR / "processed"
        ARTIFACT_ROOT = DRIVE_ROOT / "artifacts"
        INDEX_ROOT = ARTIFACT_ROOT / "indices"
        MODEL_ROOT = ARTIFACT_ROOT / "models"
        TRAINING_DATA_DIR = ARTIFACT_ROOT / "retriever_training"
        RETRIEVER_DIR = MODEL_ROOT / "retriever" / "trained_medcpt_biencoder"
        RUN_ROOT = DRIVE_ROOT / "runs"

        for path in [
            DRIVE_ROOT,
            DATA_DRIVE_DIR,
            PROCESSED_DRIVE_DIR,
            ARTIFACT_ROOT,
            INDEX_ROOT,
            MODEL_ROOT,
            TRAINING_DATA_DIR,
            RETRIEVER_DIR,
            RUN_ROOT,
        ]:
            path.mkdir(parents=True, exist_ok=True)

        MANIFEST_PATH = DRIVE_ROOT / "manifest.json"
        print("Drive root:", DRIVE_ROOT)
        print("Manifest:", MANIFEST_PATH)
        """
    ),
    code(
        """
        def sh(args, *, cwd: Path | None = None) -> None:
            args = [str(arg) for arg in args]
            print("+", " ".join(args))
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            env["MEDICAL_RAG_PROGRESS"] = "1"
            env.setdefault("TQDM_MININTERVAL", "1")
            env.setdefault("TQDM_DYNAMIC_NCOLS", "1")

            proc = subprocess.Popen(
                args,
                cwd=None if cwd is None else str(cwd),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=0,
                env=env,
            )
            assert proc.stdout is not None
            buffer: list[str] = []
            while True:
                ch = proc.stdout.read(1)
                if ch == "" and proc.poll() is not None:
                    break
                if not ch:
                    time.sleep(0.05)
                    continue
                if ch == "\\r":
                    if buffer:
                        print("".join(buffer), end="\\r", flush=True)
                        buffer = []
                elif ch == "\\n":
                    print("".join(buffer), flush=True)
                    buffer = []
                else:
                    buffer.append(ch)
            if buffer:
                print("".join(buffer), flush=True)

            returncode = proc.wait()
            if returncode:
                raise subprocess.CalledProcessError(returncode, args)


        def run_project(command: str, overrides: list[str]) -> None:
            payload = json.dumps(overrides, ensure_ascii=False)
            sh(
                [
                    sys.executable,
                    "-u",
                    "-m",
                    "medical_rag_reranker.commands",
                    command,
                    "--overrides",
                    payload,
                ],
                cwd=PROJECT_DIR,
            )


        def base_overrides(*, artifacts_dir: Path | None = None) -> list[str]:
            if artifacts_dir is None:
                artifacts_dir = ARTIFACT_ROOT
            return [
                "data.use_dvc=false",
                "data.processed_dir=data/processed",
                f"paths.artifacts_dir={artifacts_dir}",
                f"paths.runs_dir={RUN_ROOT}",
                "run.prep_data.out_dir=data/processed",
            ]
        """
    ),
    code(
        """
        if not (PROJECT_DIR / ".git").exists():
            sh(["git", "clone", "--branch", BRANCH, REPO_URL, PROJECT_DIR])
        else:
            sh(["git", "fetch", "origin", BRANCH], cwd=PROJECT_DIR)
            sh(["git", "checkout", BRANCH], cwd=PROJECT_DIR)
            sh(["git", "pull", "--ff-only"], cwd=PROJECT_DIR)

        os.chdir(PROJECT_DIR)
        sh([sys.executable, "-m", "pip", "install", "-q", "-e", ".", "--no-deps"], cwd=PROJECT_DIR)

        packages = [
            "transformers",
            "datasets",
            "tokenizers",
            "hydra-core",
            "omegaconf",
            "fire",
            "rank-bm25",
            "mlflow",
            "sentence-transformers",
            "pandas",
            "pyarrow",
            "scikit-learn",
            "scipy",
            "tqdm",
            "matplotlib",
            "sentencepiece",
            "accelerate",
            "lightning",
            "torchmetrics",
        ]
        sh([sys.executable, "-m", "pip", "install", "-q", *packages], cwd=PROJECT_DIR)
        print("Project is ready:", PROJECT_DIR)
        """
    ),
    md(
        """
        ## Prepare MedQuAD artifacts

        `prep_data` пишет reproducible split: `train/val/test`, retrieval queries и qrels. Для честных retrieval-метрик дальше используем `eval_queries.jsonl` и `qrels.tsv` из test split.
        """
    ),
    code(
        """
        from datasets import load_dataset

        RAW_MEDQUAD_PATH.parent.mkdir(parents=True, exist_ok=True)
        if not RAW_MEDQUAD_PATH.exists():
            ds = load_dataset("lavita/MedQuAD")
            ds["train"].to_parquet(str(RAW_MEDQUAD_PATH))
            print("Saved raw MedQuAD:", RAW_MEDQUAD_PATH)
        else:
            print("Raw MedQuAD already exists:", RAW_MEDQUAD_PATH)

        prep_overrides = base_overrides()
        prep_overrides += [
            f"run.prep_data.raw_nih_path={RAW_MEDQUAD_PATH.relative_to(PROJECT_DIR)}",
            f"run.prep_data.eval_size={EVAL_SIZE}",
            "run.prep_data.seed=42",
        ]
        run_project("prep_data", prep_overrides)

        if PROCESSED_DRIVE_DIR.exists():
            shutil.rmtree(PROCESSED_DRIVE_DIR)
        shutil.copytree(PROJECT_PROCESSED_DIR, PROCESSED_DRIVE_DIR)
        print("Backed up processed data to:", PROCESSED_DRIVE_DIR)
        """
    ),
    md(
        """
        ## Build hard-negative data and train the bi-encoder

        Training writes full Hugging Face encoder directories to Drive. These directories are the reusable model artifact, not just a notebook output.
        """
    ),
    code(
        """
        train_data_overrides = base_overrides()
        train_data_overrides += [
            f"run.retriever_training_data.out_dir={TRAINING_DATA_DIR}",
            f"run.retriever_training_data.negatives_per_query={NEGATIVES_PER_QUERY}",
            f"run.retriever_training_data.hard_negative_pool_size={HARD_NEGATIVE_POOL_SIZE}",
        ]
        run_project("prep_retriever_training_data", train_data_overrides)

        for filename in ["train_retriever.jsonl", "val_retriever.jsonl", "retriever_training_data_summary.json"]:
            path = TRAINING_DATA_DIR / filename
            print(filename, "exists:", path.exists(), "path:", path)
        """
    ),
    code(
        """
        if TRAIN_BI_ENCODER:
            train_overrides = base_overrides()
            train_overrides += [
                f"run.retriever_training_data.out_dir={TRAINING_DATA_DIR}",
                f"run.train_retriever.train_path={TRAINING_DATA_DIR / 'train_retriever.jsonl'}",
                f"run.train_retriever.val_path={TRAINING_DATA_DIR / 'val_retriever.jsonl'}",
                f"run.train_retriever.output_dir={RETRIEVER_DIR}",
                "run.train_retriever.device=cuda",
                f"run.train_retriever.batch_size={BI_ENCODER_BATCH_SIZE}",
                f"run.train_retriever.epochs={BI_ENCODER_EPOCHS}",
                "run.train_retriever.lr=2e-5",
                "run.train_retriever.log_every_n_steps=20",
            ]
            run_project("train_retriever", train_overrides)
        else:
            print("TRAIN_BI_ENCODER=False; using existing retriever dir:", RETRIEVER_DIR)

        print("Query encoder:", RETRIEVER_DIR / "query_encoder")
        print("Document encoder:", RETRIEVER_DIR / "doc_encoder")
        """
    ),
    md(
        """
        ## Check the training curve

        `train_retriever` сохраняет loss по эпохам в `training_summary.json`. Для отчета этого достаточно, чтобы показать, что fine-tuning не был "чёрным ящиком". Если нужен более детальный график по шагам, надо дополнительно логировать step-level history в тренировочном коде.
        """
    ),
    code(
        """
        import matplotlib.pyplot as plt
        import pandas as pd

        summary_path = RETRIEVER_DIR / "training_summary.json"
        if not summary_path.exists():
            print("Training summary is not available yet:", summary_path)
        else:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            history = summary.get("history", [])
            if not history:
                print("Training summary has no history:", summary_path)
            else:
                loss_df = pd.DataFrame(history)
                display(loss_df)

                y_cols = [
                    col
                    for col in ["train_loss", "val_loss"]
                    if col in loss_df.columns and loss_df[col].notna().any()
                ]
                ax = loss_df.plot(
                    x="epoch",
                    y=y_cols,
                    marker="o",
                    figsize=(7, 4),
                    grid=True,
                    title="Bi-encoder training loss",
                )
                ax.set_xlabel("Epoch")
                ax.set_ylabel("Loss")
                plt.show()
        """
    ),
    md(
        """
        ## Build and save retrieval indices

        Индексы кладутся в `artifacts/indices/<method>/...`. Hybrid-индекс сохраняется как manifest JSON плюс BM25/dense компоненты рядом с ним.
        """
    ),
    code(
        """
        def build_index(method: str, retrieval_config: str, index_file: str, extra: list[str] | None = None) -> dict:
            out_path = INDEX_ROOT / index_file
            overrides = base_overrides(artifacts_dir=INDEX_ROOT)
            overrides += [
                f"retrieval={retrieval_config}",
                "run.retrieval_index.corpus=data/processed/corpus.jsonl",
                f"retrieval.index_file={index_file}",
                f"run.retrieval_index.batch_size={ENCODE_BATCH_SIZE}",
            ]
            if extra:
                overrides += extra
            run_project("index", overrides)
            return {
                "method": method,
                "retrieval_config": retrieval_config,
                "index_file": index_file,
                "path": str(out_path),
                "extra_overrides": extra or [],
            }


        trained_query_encoder = RETRIEVER_DIR / "query_encoder"
        trained_doc_encoder = RETRIEVER_DIR / "doc_encoder"
        trained_encoder_overrides = [
            f"retrieval.query_model_name={trained_query_encoder}",
            f"retrieval.doc_model_name={trained_doc_encoder}",
            f"run.retrieval_index.query_model={trained_query_encoder}",
            f"run.retrieval_index.doc_model={trained_doc_encoder}",
            "retrieval.local_files_only=true",
            "run.retrieval_index.local_files_only=true",
        ]

        index_registry: dict[str, dict] = {}

        specs = [
            ("bm25", "bm25", "bm25/bm25_index.json.gz", []),
            (
                "dense_minilm",
                "dense",
                "dense_minilm/dense_index.pkl",
                [
                    "run.retrieval_index.model=sentence-transformers/all-MiniLM-L6-v2",
                ],
            ),
            ("medcpt_zero_shot", "medcpt", "medcpt_zero_shot/medcpt_index.pkl", []),
            ("hybrid_minilm", "hybrid", "hybrid_minilm/hybrid_index.json", []),
            (
                "hybrid_medcpt_zero_shot",
                "hybrid_medcpt",
                "hybrid_medcpt_zero_shot/hybrid_index.json",
                [],
            ),
        ]

        if trained_query_encoder.exists() and trained_doc_encoder.exists():
            specs.extend(
                [
                    (
                        "trained_medcpt",
                        "medcpt",
                        "trained_medcpt/medcpt_index.pkl",
                        trained_encoder_overrides,
                    ),
                    (
                        "hybrid_trained_medcpt",
                        "hybrid_medcpt",
                        "hybrid_trained_medcpt/hybrid_index.json",
                        trained_encoder_overrides,
                    ),
                ]
            )
        else:
            print("Trained encoder dirs are absent; trained_medcpt indices will be skipped.")

        for method, retrieval_config, index_file, extra in specs:
            print("\\n== Building", method, "==")
            index_registry[method] = build_index(method, retrieval_config, index_file, extra)

        print("Built indices:", sorted(index_registry))
        """
    ),
    md(
        """
        ## Build graph metadata index

        Это лёгкий graph-aware baseline: граф строится по metadata связям документов (`diagnosis_or_topic`, `question_intent`, `group_id`). В следующем ноутбуке он будет расширять кандидатов из BM25/hybrid run-файлов и оцениваться теми же TREC-метриками.
        """
    ),
    code(
        """
        from collections import defaultdict
        import re


        def read_jsonl(path: Path) -> list[dict]:
            rows = []
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        rows.append(json.loads(line))
            return rows


        def normalize_entity(value: object) -> str:
            text = re.sub(r"\\s+", " ", str(value or "").strip().lower())
            text = text.strip(" .,:;!?()[]{}")
            return text


        def build_graph_metadata_index(corpus_path: Path, out_path: Path) -> dict:
            rows = read_jsonl(corpus_path)
            entity_to_doc_ids: dict[str, list[str]] = defaultdict(list)
            doc_to_entities: dict[str, list[str]] = {}

            fields = ["diagnosis_or_topic", "question_intent", "group_id"]
            for row in rows:
                doc_id = str(row.get("doc_id") or "").strip()
                if not doc_id:
                    continue
                entities = []
                for field in fields:
                    value = normalize_entity(row.get(field))
                    if not value:
                        continue
                    entity = f"{field}:{value}"
                    entities.append(entity)
                    entity_to_doc_ids[entity].append(doc_id)
                doc_to_entities[doc_id] = sorted(set(entities))

            payload = {
                "format": "medical-rag-reranker.metadata-graph-index",
                "version": 1,
                "corpus_path": str(corpus_path),
                "fields": fields,
                "relation_weights": {
                    "diagnosis_or_topic": 0.45,
                    "group_id": 0.35,
                    "question_intent": 0.20,
                },
                "doc_to_entities": doc_to_entities,
                "entity_to_doc_ids": {key: sorted(set(value)) for key, value in entity_to_doc_ids.items()},
            }
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            return payload


        graph_index_path = INDEX_ROOT / "graph_metadata" / "graph_index.json"
        graph_payload = build_graph_metadata_index(PROJECT_PROCESSED_DIR / "corpus.jsonl", graph_index_path)
        index_registry["graph_metadata"] = {
            "method": "graph_metadata",
            "retrieval_config": "run_file_graph_expansion",
            "index_file": "graph_metadata/graph_index.json",
            "path": str(graph_index_path),
            "num_docs": len(graph_payload["doc_to_entities"]),
            "num_entities": len(graph_payload["entity_to_doc_ids"]),
        }
        print("Graph index:", graph_index_path)
        print("Graph docs/entities:", len(graph_payload["doc_to_entities"]), len(graph_payload["entity_to_doc_ids"]))
        """
    ),
    code(
        """
        manifest = {
            "run_id": RUN_ID,
            "created_at_utc": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "repo_url": REPO_URL,
            "branch": BRANCH,
            "drive_root": str(DRIVE_ROOT),
            "project_dir_expected_in_colab": str(PROJECT_DIR),
            "data": {
                "processed_project_dir": str(PROJECT_PROCESSED_DIR),
                "processed_drive_dir": str(PROCESSED_DRIVE_DIR),
                "corpus": str(PROJECT_PROCESSED_DIR / "corpus.jsonl"),
                "eval_queries": str(PROJECT_PROCESSED_DIR / "eval_queries.jsonl"),
                "qrels": str(PROJECT_PROCESSED_DIR / "qrels.tsv"),
            },
            "models": {
                "trained_medcpt_biencoder": {
                    "root": str(RETRIEVER_DIR),
                    "query_encoder": str(RETRIEVER_DIR / "query_encoder"),
                    "doc_encoder": str(RETRIEVER_DIR / "doc_encoder"),
                    "training_summary": str(RETRIEVER_DIR / "training_summary.json"),
                }
            },
            "indices": index_registry,
            "runs_dir": str(RUN_ROOT),
            "notes": [
                "Keep this manifest with the Drive folder; later notebooks use it as the source of truth.",
                "BM25 index stores a corpus path. Later notebooks restore data/processed into /content/medical-rag-reranker before loading it.",
            ],
        }
        MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        print("Saved manifest:", MANIFEST_PATH)
        print(json.dumps({k: v["path"] for k, v in index_registry.items()}, ensure_ascii=False, indent=2))
        """
    ),
]


NOTEBOOK_02 = [
    md(
        """
        # 02: Benchmark Retrieval, Reranker, Graph

        Цель: прогнать единый benchmark для всех retrieval-методов на одном `eval_queries.jsonl`/`qrels.tsv`, включая:
        - BM25
        - dense MiniLM
        - MedCPT zero-shot
        - trained MedCPT bi-encoder
        - hybrid variants
        - RAG Fusion variants
        - graph-aware expansion поверх run-файлов
        - optional Cross-Encoder reranking для выбранных baseline runs
        """
    ),
    code(
        """
        from pathlib import Path
        import json
        import os
        import shutil
        import subprocess
        import sys
        import time

        import pandas as pd

        REPO_OWNER = "terrylimax"
        REPO_NAME = "medical-rag-reranker"
        REPO_URL = f"https://github.com/{REPO_OWNER}/{REPO_NAME}.git"
        BRANCH = "main"

        RUN_ID = os.environ.get("COLAB_RUN_ID", "medquad_full_v1")
        USE_DRIVE = True
        TOP_K = 50
        KS = "1,5,10,20,50"

        PROJECT_DIR = Path("/content") / REPO_NAME
        PROJECT_PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
        """
    ),
    code(
        """
        def mount_drive_or_content(use_drive: bool) -> Path:
            if not use_drive:
                return Path("/content") / "medical-rag-reranker-colab"

            try:
                from google.colab import drive

                drive.mount("/content/drive", force_remount=False)
                return Path("/content/drive/MyDrive") / "medical-rag-reranker-colab"
            except Exception as exc:
                print("Drive mount failed; falling back to /content.")
                print(type(exc).__name__, str(exc))
                return Path("/content") / "medical-rag-reranker-colab"


        DRIVE_BASE = mount_drive_or_content(USE_DRIVE)
        DRIVE_ROOT = DRIVE_BASE / RUN_ID
        MANIFEST_PATH = DRIVE_ROOT / "manifest.json"
        if not MANIFEST_PATH.exists():
            raise FileNotFoundError(f"Run notebook 01 first; missing manifest: {MANIFEST_PATH}")

        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        ARTIFACT_ROOT = DRIVE_ROOT / "artifacts"
        INDEX_ROOT = ARTIFACT_ROOT / "indices"
        RUN_ROOT = DRIVE_ROOT / "runs"
        RETRIEVAL_RUN_DIR = RUN_ROOT / "retrieval"
        RERANK_RUN_DIR = RUN_ROOT / "reranked"
        SUMMARY_DIR = RUN_ROOT / "summaries"
        PROCESSED_DRIVE_DIR = Path(manifest["data"]["processed_drive_dir"])

        for path in [RETRIEVAL_RUN_DIR, RERANK_RUN_DIR, SUMMARY_DIR]:
            path.mkdir(parents=True, exist_ok=True)

        print("Drive root:", DRIVE_ROOT)
        print("Manifest loaded:", MANIFEST_PATH)
        """
    ),
    code(
        """
        def sh(args, *, cwd: Path | None = None) -> None:
            args = [str(arg) for arg in args]
            print("+", " ".join(args))
            subprocess.run(args, cwd=None if cwd is None else str(cwd), check=True)


        def run_project(command: str, overrides: list[str]) -> None:
            payload = json.dumps(overrides, ensure_ascii=False)
            sh(
                [
                    sys.executable,
                    "-m",
                    "medical_rag_reranker.commands",
                    command,
                    "--overrides",
                    payload,
                ],
                cwd=PROJECT_DIR,
            )


        def base_overrides() -> list[str]:
            return [
                "data.use_dvc=false",
                "data.processed_dir=data/processed",
                f"paths.artifacts_dir={INDEX_ROOT}",
                f"paths.runs_dir={RUN_ROOT}",
                "run.prep_data.out_dir=data/processed",
            ]
        """
    ),
    code(
        """
        if not (PROJECT_DIR / ".git").exists():
            sh(["git", "clone", "--branch", BRANCH, REPO_URL, PROJECT_DIR])
        else:
            sh(["git", "fetch", "origin", BRANCH], cwd=PROJECT_DIR)
            sh(["git", "checkout", BRANCH], cwd=PROJECT_DIR)
            sh(["git", "pull", "--ff-only"], cwd=PROJECT_DIR)

        sh([sys.executable, "-m", "pip", "install", "-q", "-e", ".", "--no-deps"], cwd=PROJECT_DIR)
        packages = [
            "transformers",
            "datasets",
            "tokenizers",
            "hydra-core",
            "omegaconf",
            "fire",
            "rank-bm25",
            "mlflow",
            "sentence-transformers",
            "pandas",
            "pyarrow",
            "scikit-learn",
            "scipy",
            "tqdm",
            "sentencepiece",
            "accelerate",
            "lightning",
            "torchmetrics",
        ]
        sh([sys.executable, "-m", "pip", "install", "-q", *packages], cwd=PROJECT_DIR)

        if not (PROJECT_DIR / "medical_rag_reranker").exists():
            raise FileNotFoundError(
                "Project package directory was not found. "
                f"Expected: {PROJECT_DIR / 'medical_rag_reranker'}"
            )
        if str(PROJECT_DIR) not in sys.path:
            sys.path.insert(0, str(PROJECT_DIR))
        print("Project package import path:", PROJECT_DIR)

        if PROJECT_PROCESSED_DIR.exists():
            shutil.rmtree(PROJECT_PROCESSED_DIR)
        shutil.copytree(PROCESSED_DRIVE_DIR, PROJECT_PROCESSED_DIR)
        print("Restored processed data:", PROJECT_PROCESSED_DIR)
        """
    ),
    md(
        """
        ## Retrieval benchmark

        Все методы пишут TREC run-файл и `*.metrics.json`. Для RAG Fusion используется тот же индекс, что и у базового retriever, но retrieval layer расширяет запросы и агрегирует ранги через RRF.
        """
    ),
    code(
        """
        indices = manifest["indices"]


        def metric_path_for_run(run_file: Path) -> Path:
            return run_file.with_suffix(run_file.suffix + ".metrics.json")


        def load_metrics(run_file: Path) -> dict:
            path = metric_path_for_run(run_file)
            if not path.exists():
                raise FileNotFoundError(path)
            return json.loads(path.read_text(encoding="utf-8"))


        def eval_retrieval_method(
            *,
            method: str,
            retrieval_config: str,
            index_path: Path,
            extra: list[str] | None = None,
        ) -> dict:
            run_file = RETRIEVAL_RUN_DIR / f"{method}.trec"
            overrides = base_overrides()
            overrides += [
                f"retrieval={retrieval_config}",
                f"retrieval_run.index={index_path}",
                f"retrieval_run.queries={PROJECT_PROCESSED_DIR / 'eval_queries.jsonl'}",
                f"retrieval_run.out={run_file}",
                f"retrieval_run.top_k={TOP_K}",
                f"retrieval_run.run_name={method}",
                f"run.eval_retrieval.eval_queries={PROJECT_PROCESSED_DIR / 'eval_queries.jsonl'}",
                f"run.eval_retrieval.qrels={PROJECT_PROCESSED_DIR / 'qrels.tsv'}",
                f"run.eval_retrieval.out_run={run_file}",
                f"run.eval_retrieval.run_path={run_file}",
                f'run.eval_retrieval.ks="{KS}"',
                f"run.eval_retrieval.run_name={method}",
                f"run.eval_retrieval.retriever={method}",
            ]
            if extra:
                overrides += extra

            run_project("retrieval_run", overrides)
            run_project("eval_retrieval", overrides)

            metrics = load_metrics(run_file)
            metrics.update(
                {
                    "method": method,
                    "retrieval_config": retrieval_config,
                    "index_path": str(index_path),
                    "run_file": str(run_file),
                    "stage": "retrieval",
                }
            )
            return metrics
        """
    ),
    code(
        """
        method_specs = [
            {
                "method": "bm25",
                "retrieval_config": "bm25",
                "index_path": Path(indices["bm25"]["path"]),
            },
            {
                "method": "dense_minilm",
                "retrieval_config": "dense",
                "index_path": Path(indices["dense_minilm"]["path"]),
                "extra": ["retrieval.name=dense"],
            },
            {
                "method": "medcpt_zero_shot",
                "retrieval_config": "medcpt",
                "index_path": Path(indices["medcpt_zero_shot"]["path"]),
            },
            {
                "method": "hybrid_minilm",
                "retrieval_config": "hybrid",
                "index_path": Path(indices["hybrid_minilm"]["path"]),
            },
            {
                "method": "hybrid_medcpt_zero_shot",
                "retrieval_config": "hybrid_medcpt",
                "index_path": Path(indices["hybrid_medcpt_zero_shot"]["path"]),
            },
            {
                "method": "rag_fusion_bm25",
                "retrieval_config": "rag_fusion_bm25",
                "index_path": Path(indices["bm25"]["path"]),
            },
            {
                "method": "rag_fusion_dense_minilm",
                "retrieval_config": "rag_fusion_dense",
                "index_path": Path(indices["dense_minilm"]["path"]),
                "extra": ["retrieval.name=rag_fusion_dense_minilm"],
            },
            {
                "method": "rag_fusion_medcpt_zero_shot",
                "retrieval_config": "rag_fusion_medcpt_pilot",
                "index_path": Path(indices["medcpt_zero_shot"]["path"]),
                "extra": [
                    "retrieval.name=rag_fusion_medcpt_zero_shot",
                    "retrieval.base_retriever=bi_encoder",
                ],
            },
        ]

        if "trained_medcpt" in indices:
            method_specs.extend(
                [
                    {
                        "method": "trained_medcpt",
                        "retrieval_config": "medcpt",
                        "index_path": Path(indices["trained_medcpt"]["path"]),
                    },
                    {
                        "method": "hybrid_trained_medcpt",
                        "retrieval_config": "hybrid_medcpt",
                        "index_path": Path(indices["hybrid_trained_medcpt"]["path"]),
                    },
                    {
                        "method": "rag_fusion_trained_medcpt",
                        "retrieval_config": "rag_fusion_medcpt_pilot",
                        "index_path": Path(indices["trained_medcpt"]["path"]),
                        "extra": [
                            "retrieval.name=rag_fusion_trained_medcpt",
                            "retrieval.base_retriever=bi_encoder",
                        ],
                    },
                ]
            )

        retrieval_rows = []
        for spec in method_specs:
            print("\\n== Benchmark", spec["method"], "==")
            retrieval_rows.append(eval_retrieval_method(**spec))

        retrieval_df = pd.DataFrame(retrieval_rows)
        display_cols = [
            "method",
            "NDCG@10",
            "R@10",
            "P@10",
            "Hit@10",
            "MRR@10",
            "latency_mean_ms",
            "latency_p95_ms",
            "index_size_mb",
            "run_file",
        ]
        display(retrieval_df[[col for col in display_cols if col in retrieval_df.columns]].sort_values("NDCG@10", ascending=False))
        """
    ),
    md(
        """
        ## Graph-aware expansion

        Метод берёт готовый TREC run от base retriever, расширяет кандидатов соседними документами из metadata graph и агрегирует score через RRF. Это не заменяет BM25/dense, а проверяет, помогает ли графовая связность по теме/intent/group вытянуть релевантный документ выше.
        """
    ),
    code(
        """
        from collections import defaultdict


        def read_trec_run(path: Path) -> dict[str, list[tuple[str, float]]]:
            runs: dict[str, list[tuple[str, float]]] = defaultdict(list)
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    qid, _q0, doc_id, _rank, score, _run_name = line.split()[:6]
                    runs[qid].append((doc_id, float(score)))
            for qid in list(runs):
                runs[qid].sort(key=lambda item: item[1], reverse=True)
            return dict(runs)


        def write_trec_run(path: Path, runs: dict[str, list[tuple[str, float]]], run_name: str) -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8") as handle:
                for qid, docs in runs.items():
                    for rank, (doc_id, score) in enumerate(docs, start=1):
                        handle.write(f"{qid}\\tQ0\\t{doc_id}\\t{rank}\\t{score:.8f}\\t{run_name}\\n")


        def load_graph_index(path: Path) -> dict:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("format") != "medical-rag-reranker.metadata-graph-index":
                raise ValueError("Unsupported graph index format")
            return payload


        def relation_weight(entity: str, weights: dict[str, float]) -> float:
            relation = entity.split(":", 1)[0]
            return float(weights.get(relation, 0.05))


        def graph_expand_run(
            *,
            base_run_path: Path,
            graph_index_path: Path,
            out_path: Path,
            run_name: str,
            top_k: int = 50,
            seed_top_n: int = 20,
            graph_weight: float = 0.65,
            rrf_k: int = 60,
            max_neighbors_per_entity: int = 80,
        ) -> Path:
            base_runs = read_trec_run(base_run_path)
            graph = load_graph_index(graph_index_path)
            doc_to_entities = graph["doc_to_entities"]
            entity_to_doc_ids = graph["entity_to_doc_ids"]
            weights = graph.get("relation_weights", {})

            expanded_runs: dict[str, list[tuple[str, float]]] = {}
            started = time.perf_counter()
            latencies_ms = []

            for qid, docs in base_runs.items():
                q_started = time.perf_counter()
                scores: dict[str, float] = {}
                for rank, (doc_id, _base_score) in enumerate(docs[:seed_top_n], start=1):
                    base_rrf = 1.0 / (float(rrf_k) + float(rank))
                    scores[doc_id] = scores.get(doc_id, 0.0) + base_rrf
                    for entity in doc_to_entities.get(doc_id, []):
                        boost = graph_weight * relation_weight(entity, weights) * base_rrf
                        for neighbor_doc_id in entity_to_doc_ids.get(entity, [])[:max_neighbors_per_entity]:
                            scores[neighbor_doc_id] = scores.get(neighbor_doc_id, 0.0) + boost

                ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
                expanded_runs[qid] = [(doc_id, float(score)) for doc_id, score in ranked]
                latencies_ms.append((time.perf_counter() - q_started) * 1000.0)

            write_trec_run(out_path, expanded_runs, run_name)
            latency_path = out_path.with_suffix(out_path.suffix + ".latency.json")
            latency_path.write_text(
                json.dumps(
                    {
                        "run_path": str(out_path),
                        "base_run_path": str(base_run_path),
                        "graph_index_path": str(graph_index_path),
                        "num_queries": len(expanded_runs),
                        "latencies_ms": latencies_ms,
                        "total_latency_ms": (time.perf_counter() - started) * 1000.0,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            return out_path
        """
    ),
    code(
        """
        def eval_existing_run(method: str, run_file: Path, index_path: Path) -> dict:
            overrides = base_overrides()
            overrides += [
                f"retrieval_run.index={index_path}",
                f"run.eval_retrieval.eval_queries={PROJECT_PROCESSED_DIR / 'eval_queries.jsonl'}",
                f"run.eval_retrieval.qrels={PROJECT_PROCESSED_DIR / 'qrels.tsv'}",
                f"run.eval_retrieval.out_run={run_file}",
                f"run.eval_retrieval.run_path={run_file}",
                f'run.eval_retrieval.ks="{KS}"',
                f"run.eval_retrieval.run_name={method}",
                f"run.eval_retrieval.retriever={method}",
            ]
            run_project("eval_retrieval", overrides)
            metrics = load_metrics(run_file)
            metrics.update(
                {
                    "method": method,
                    "retrieval_config": "graph_expansion",
                    "index_path": str(index_path),
                    "run_file": str(run_file),
                    "stage": "graph_retrieval",
                }
            )
            return metrics


        graph_index_path = Path(indices["graph_metadata"]["path"])
        graph_rows = []
        graph_bases = ["bm25", "hybrid_minilm"]
        if "hybrid_trained_medcpt" in indices:
            graph_bases.append("hybrid_trained_medcpt")

        for base_method in graph_bases:
            base_run = RETRIEVAL_RUN_DIR / f"{base_method}.trec"
            if not base_run.exists():
                print("Skipping graph expansion; missing base run:", base_run)
                continue
            method = f"graph_{base_method}"
            graph_run = RETRIEVAL_RUN_DIR / f"{method}.trec"
            graph_expand_run(
                base_run_path=base_run,
                graph_index_path=graph_index_path,
                out_path=graph_run,
                run_name=method,
                top_k=TOP_K,
            )
            graph_rows.append(eval_existing_run(method, graph_run, graph_index_path))

        graph_df = pd.DataFrame(graph_rows)
        if not graph_df.empty:
            display(graph_df[[col for col in display_cols if col in graph_df.columns]].sort_values("NDCG@10", ascending=False))
        """
    ),
    md(
        """
        ## Optional Cross-Encoder reranking

        Укажите путь к checkpoint `*.ckpt`. Если checkpoint не задан, notebook честно пропускает reranker-секцию, но retrieval/graph результаты остаются пригодными для анализа.
        """
    ),
    code(
        """
        RERANKER_CHECKPOINT_PATH = os.environ.get("RERANKER_CHECKPOINT_PATH", "")
        RERANK_TOP_N = 20
        RERANKER_BATCH_SIZE = 16
        RERANK_BASE_METHODS = ["bm25", "hybrid_minilm"]
        if "hybrid_trained_medcpt" in indices:
            RERANK_BASE_METHODS.append("hybrid_trained_medcpt")
        if (RETRIEVAL_RUN_DIR / "graph_hybrid_trained_medcpt.trec").exists():
            RERANK_BASE_METHODS.append("graph_hybrid_trained_medcpt")

        print("Reranker checkpoint:", RERANKER_CHECKPOINT_PATH or "<not set>")
        print("Rerank base methods:", RERANK_BASE_METHODS)
        """
    ),
    code(
        """
        def eval_reranked_run(base_method: str, checkpoint_path: Path) -> dict:
            base_run = RETRIEVAL_RUN_DIR / f"{base_method}.trec"
            if not base_run.exists():
                raise FileNotFoundError(base_run)

            method = f"{base_method}__reranked"
            out_run = RERANK_RUN_DIR / f"{method}.trec"
            comparison_report = RERANK_RUN_DIR / f"{method}.md"
            comparison_jsonl = RERANK_RUN_DIR / f"{method}.jsonl"

            overrides = base_overrides()
            overrides += [
                "retrieval=bm25",
                f"run.eval_reranked_retrieval.run_path={base_run}",
                f"run.eval_reranked_retrieval.eval_queries={PROJECT_PROCESSED_DIR / 'eval_queries.jsonl'}",
                f"run.eval_reranked_retrieval.qrels={PROJECT_PROCESSED_DIR / 'qrels.tsv'}",
                f"run.eval_reranked_retrieval.corpus_path={PROJECT_PROCESSED_DIR / 'corpus.jsonl'}",
                f"run.eval_reranked_retrieval.out_run={out_run}",
                f"run.eval_reranked_retrieval.reranker_checkpoint_path={checkpoint_path}",
                f"run.eval_reranked_retrieval.rerank_top_n={RERANK_TOP_N}",
                f"run.eval_reranked_retrieval.reranker_batch_size={RERANKER_BATCH_SIZE}",
                f'run.eval_reranked_retrieval.ks="{KS}"',
                f"run.eval_reranked_retrieval.run_name={method}",
                f"run.eval_reranked_retrieval.retriever={base_method}",
                f"run.eval_reranked_retrieval.comparison_report_path={comparison_report}",
                f"run.eval_reranked_retrieval.comparison_jsonl_path={comparison_jsonl}",
            ]
            run_project("eval_reranked_retrieval", overrides)

            comparison_path = out_run.with_suffix(out_run.suffix + ".comparison.json")
            comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
            row = {f"rerank_{key}": value for key, value in comparison.items()}
            row.update(
                {
                    "method": method,
                    "base_method": base_method,
                    "run_file": str(out_run),
                    "stage": "reranked_retrieval",
                    "reranker_checkpoint": str(checkpoint_path),
                }
            )
            return row


        rerank_rows = []
        if RERANKER_CHECKPOINT_PATH:
            checkpoint = Path(RERANKER_CHECKPOINT_PATH)
            if checkpoint.exists():
                for base_method in RERANK_BASE_METHODS:
                    print("\\n== Rerank", base_method, "==")
                    rerank_rows.append(eval_reranked_run(base_method, checkpoint))
            else:
                print("Checkpoint path does not exist; skipping reranking:", checkpoint)
        else:
            print("No reranker checkpoint configured; skipping reranking.")

        rerank_df = pd.DataFrame(rerank_rows)
        if not rerank_df.empty:
            display(rerank_df)
        """
    ),
    code(
        """
        all_rows = []
        if not retrieval_df.empty:
            all_rows.extend(retrieval_df.to_dict("records"))
        if not graph_df.empty:
            all_rows.extend(graph_df.to_dict("records"))
        if not rerank_df.empty:
            all_rows.extend(rerank_df.to_dict("records"))

        summary_df = pd.DataFrame(all_rows)
        summary_csv = SUMMARY_DIR / "retrieval_benchmark_summary.csv"
        summary_json = SUMMARY_DIR / "retrieval_benchmark_summary.json"
        summary_df.to_csv(summary_csv, index=False)
        summary_df.to_json(summary_json, orient="records", force_ascii=False, indent=2)

        print("Saved summary CSV:", summary_csv)
        print("Saved summary JSON:", summary_json)
        if "NDCG@10" in summary_df.columns:
            cols = [col for col in display_cols if col in summary_df.columns]
            display(summary_df[cols].sort_values("NDCG@10", ascending=False))
        else:
            display(summary_df)
        """
    ),
]


NOTEBOOK_03 = [
    md(
        """
        # 03: Evaluate Generation Quality

        Цель: проверить качество ответов RAG поверх сохранённых retrieval run-файлов. Notebook читает TREC run, берёт top-k документов из `corpus.jsonl`, генерирует ответ и считает:
        - bridge-метрики `gold_in_context`, `top1_is_gold`, `gold_rank`;
        - reference-based метрики относительно gold answer из MedQuAD;
        - reference-free groundedness/relevance;
        - optional LLM-as-a-Judge.
        """
    ),
    code(
        """
        from pathlib import Path
        import json
        import os
        import re
        import shutil
        import subprocess
        import sys
        import time

        import pandas as pd
        import torch

        REPO_OWNER = "terrylimax"
        REPO_NAME = "medical-rag-reranker"
        REPO_URL = f"https://github.com/{REPO_OWNER}/{REPO_NAME}.git"
        BRANCH = "main"

        RUN_ID = os.environ.get("COLAB_RUN_ID", "medquad_full_v1")
        USE_DRIVE = True
        MAX_EXAMPLES = 100
        CONTEXT_TOP_K_VALUES = [1, 3, 5]
        LLM_MODEL_NAME = "google/flan-t5-small"
        MAX_INPUT_TOKENS = 1024
        MAX_NEW_TOKENS = 192
        GENERATION_SEED = 42

        USE_LLM_JUDGE = os.environ.get("USE_LLM_JUDGE", "false").strip().lower() in {"1", "true", "yes", "y", "on"}
        JUDGE_BASE_URL = os.environ.get("LLM_JUDGE_BASE_URL", "http://localhost:8000/v1")
        JUDGE_MODEL = os.environ.get("LLM_JUDGE_MODEL", "mistralai/Mistral-7B-Instruct-v0.3")
        JUDGE_API_KEY = os.environ.get("LLM_JUDGE_API_KEY", "EMPTY")
        JUDGE_MAX_CONTEXT_DOCS = int(os.environ.get("LLM_JUDGE_MAX_CONTEXT_DOCS", "5"))

        PROJECT_DIR = Path("/content") / REPO_NAME
        PROJECT_PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
        """
    ),
    code(
        """
        def mount_drive_or_content(use_drive: bool) -> Path:
            if not use_drive:
                return Path("/content") / "medical-rag-reranker-colab"

            try:
                from google.colab import drive

                drive.mount("/content/drive", force_remount=False)
                return Path("/content/drive/MyDrive") / "medical-rag-reranker-colab"
            except Exception as exc:
                print("Drive mount failed; falling back to /content.")
                print(type(exc).__name__, str(exc))
                return Path("/content") / "medical-rag-reranker-colab"


        DRIVE_BASE = mount_drive_or_content(USE_DRIVE)
        DRIVE_ROOT = DRIVE_BASE / RUN_ID
        MANIFEST_PATH = DRIVE_ROOT / "manifest.json"
        if not MANIFEST_PATH.exists():
            raise FileNotFoundError(f"Run notebook 01 first; missing manifest: {MANIFEST_PATH}")

        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        RUN_ROOT = DRIVE_ROOT / "runs"
        RETRIEVAL_RUN_DIR = RUN_ROOT / "retrieval"
        RERANK_RUN_DIR = RUN_ROOT / "reranked"
        GENERATION_ROOT = RUN_ROOT / "generation"
        SUMMARY_DIR = RUN_ROOT / "summaries"
        PROCESSED_DRIVE_DIR = Path(manifest["data"]["processed_drive_dir"])

        for path in [GENERATION_ROOT, SUMMARY_DIR]:
            path.mkdir(parents=True, exist_ok=True)

        print("Drive root:", DRIVE_ROOT)
        print("Generation output:", GENERATION_ROOT)
        """
    ),
    code(
        """
        def sh(args, *, cwd: Path | None = None) -> None:
            args = [str(arg) for arg in args]
            print("+", " ".join(args))
            subprocess.run(args, cwd=None if cwd is None else str(cwd), check=True)


        if not (PROJECT_DIR / ".git").exists():
            sh(["git", "clone", "--branch", BRANCH, REPO_URL, PROJECT_DIR])
        else:
            sh(["git", "fetch", "origin", BRANCH], cwd=PROJECT_DIR)
            sh(["git", "checkout", BRANCH], cwd=PROJECT_DIR)
            sh(["git", "pull", "--ff-only"], cwd=PROJECT_DIR)

        sh([sys.executable, "-m", "pip", "install", "-q", "-e", ".", "--no-deps"], cwd=PROJECT_DIR)
        packages = [
            "transformers",
            "tokenizers",
            "omegaconf",
            "hydra-core",
            "fire",
            "pandas",
            "tqdm",
            "sentencepiece",
            "accelerate",
            "torchmetrics",
        ]
        sh([sys.executable, "-m", "pip", "install", "-q", *packages], cwd=PROJECT_DIR)

        if PROJECT_PROCESSED_DIR.exists():
            shutil.rmtree(PROJECT_PROCESSED_DIR)
        shutil.copytree(PROCESSED_DRIVE_DIR, PROJECT_PROCESSED_DIR)
        print("Restored processed data:", PROJECT_PROCESSED_DIR)
        """
    ),
    md(
        """
        ## Select methods

        По умолчанию берём BM25 baseline, лучший hybrid/trained вариант, graph-aware вариант и reranked вариант, если их run-файлы существуют. Список можно переопределить вручную.
        """
    ),
    code(
        """
        def existing_run(method: str) -> Path | None:
            candidates = [
                RETRIEVAL_RUN_DIR / f"{method}.trec",
                RERANK_RUN_DIR / f"{method}.trec",
            ]
            for path in candidates:
                if path.exists():
                    return path
            return None


        preferred_methods = [
            "bm25",
            "hybrid_minilm",
            "hybrid_trained_medcpt",
            "graph_hybrid_trained_medcpt",
            "rag_fusion_trained_medcpt",
            "hybrid_trained_medcpt__reranked",
            "graph_hybrid_trained_medcpt__reranked",
        ]

        GENERATION_METHODS = [
            method for method in preferred_methods if existing_run(method) is not None
        ]
        if not GENERATION_METHODS:
            raise RuntimeError("No retrieval run files found. Run notebook 02 first.")

        print("Generation methods:", GENERATION_METHODS)
        for method in GENERATION_METHODS:
            print(method, "->", existing_run(method))
        """
    ),
    code(
        """
        if str(PROJECT_DIR) not in sys.path:
            sys.path.insert(0, str(PROJECT_DIR))

        from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, set_seed
        from medical_rag_reranker.evaluation.reference_free import (
            evaluate_generation_result,
            summarize_generation_evaluations,
        )
        from medical_rag_reranker.evaluation.llm_judge import LocalOpenAICompatibleJudge

        set_seed(GENERATION_SEED)


        def read_jsonl(path: Path, limit: int | None = None) -> list[dict]:
            rows = []
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    rows.append(json.loads(line))
                    if limit is not None and len(rows) >= limit:
                        break
            return rows


        def write_jsonl(path: Path, rows: list[dict]) -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\\n")


        def load_docstore(corpus_path: Path) -> dict[str, dict]:
            rows = read_jsonl(corpus_path)
            return {str(row["doc_id"]): row for row in rows if row.get("doc_id")}


        def read_qrels(path: Path) -> dict[str, list[str]]:
            qrels: dict[str, list[str]] = {}
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    qid, _iter, doc_id, rel = line.split()[:4]
                    if int(rel) > 0:
                        qrels.setdefault(str(qid), []).append(str(doc_id))
            return qrels


        def read_trec_run(path: Path) -> dict[str, list[tuple[str, float]]]:
            runs: dict[str, list[tuple[str, float]]] = {}
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    qid, _q0, doc_id, _rank, score, _run_name = line.split()[:6]
                    runs.setdefault(qid, []).append((doc_id, float(score)))
            for qid in list(runs):
                runs[qid].sort(key=lambda item: item[1], reverse=True)
            return runs


        def query_id(row: dict, idx: int) -> str:
            return str(row.get("query_id") or row.get("question_id") or f"query-{idx}")


        def query_text(row: dict) -> str:
            value = row.get("question") or row.get("text")
            if not value:
                raise ValueError("Query row must contain question or text")
            return str(value)


        def tokenize_reference(text: str) -> list[str]:
            normalized = "".join(ch.lower() if ch.isalnum() else " " for ch in str(text or ""))
            return [token for token in normalized.split() if token]


        def lcs_len(left: list[str], right: list[str]) -> int:
            if not left or not right:
                return 0
            prev = [0] * (len(right) + 1)
            for token_left in left:
                curr = [0] * (len(right) + 1)
                for j, token_right in enumerate(right, start=1):
                    if token_left == token_right:
                        curr[j] = prev[j - 1] + 1
                    else:
                        curr[j] = max(prev[j], curr[j - 1])
                prev = curr
            return prev[-1]


        def rouge_l_f1(candidate: str, reference: str) -> float:
            cand = tokenize_reference(candidate)
            ref = tokenize_reference(reference)
            if not cand or not ref:
                return 0.0
            lcs = lcs_len(cand, ref)
            precision = lcs / float(len(cand))
            recall = lcs / float(len(ref))
            if precision + recall == 0.0:
                return 0.0
            return 2.0 * precision * recall / (precision + recall)


        def lexical_cosine(candidate: str, reference: str) -> float:
            from collections import Counter
            import math

            cand = Counter(tokenize_reference(candidate))
            ref = Counter(tokenize_reference(reference))
            if not cand or not ref:
                return 0.0
            shared = set(cand) & set(ref)
            dot = sum(cand[token] * ref[token] for token in shared)
            cand_norm = math.sqrt(sum(value * value for value in cand.values()))
            ref_norm = math.sqrt(sum(value * value for value in ref.values()))
            return dot / float(cand_norm * ref_norm + 1e-12)


        def find_rank(ranked_docs: list[tuple[str, float]], gold_doc_ids: set[str]) -> int | None:
            for rank, (doc_id, _score) in enumerate(ranked_docs, start=1):
                if doc_id in gold_doc_ids:
                    return rank
            return None


        def read_latency_profile(run_file: Path) -> list[float]:
            latency_path = run_file.with_suffix(run_file.suffix + ".latency.json")
            if not latency_path.exists():
                return []
            payload = json.loads(latency_path.read_text(encoding="utf-8"))
            return [float(value) for value in payload.get("latencies_ms", [])]


        def build_prompt(question: str, docs: list[dict]) -> str:
            if docs:
                context = "\\n\\n".join(f"[{doc['doc_id']}] {doc['text']}" for doc in docs)
            else:
                context = "(no retrieved documents)"
            return (
                "You are a medical QA assistant.\\n"
                "Rules:\\n"
                "1) Answer strictly using only the provided context.\\n"
                "2) If context is insufficient, say so explicitly.\\n"
                "3) Cite supporting sources using [doc_id] format.\\n"
                "4) Do not invent citations.\\n\\n"
                f"Context:\\n{context}\\n\\n"
                f"Question:\\n{question}\\n\\n"
                "Answer:"
            )
        """
    ),
    code(
        """
        class NotebookGenerator:
            def __init__(self, model_name: str) -> None:
                self.model_name = model_name
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.is_seq2seq = True
                try:
                    self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
                except Exception:
                    self.is_seq2seq = False
                    self.model = AutoModelForCausalLM.from_pretrained(model_name)
                if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                self.model.to(self.device)
                self.model.eval()

            def generate(self, prompt: str) -> str:
                encoded = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=MAX_INPUT_TOKENS,
                ).to(self.device)
                kwargs = {
                    "max_new_tokens": MAX_NEW_TOKENS,
                    "do_sample": False,
                    "pad_token_id": self.tokenizer.pad_token_id,
                }
                with torch.no_grad():
                    output = self.model.generate(**encoded, **kwargs)
                if self.is_seq2seq:
                    return self.tokenizer.decode(output[0], skip_special_tokens=True).strip()
                prompt_len = encoded["input_ids"].shape[1]
                return self.tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True).strip()


        generator = NotebookGenerator(LLM_MODEL_NAME)
        print("Loaded generator:", LLM_MODEL_NAME, "on", generator.device)

        judge = None
        if USE_LLM_JUDGE:
            judge = LocalOpenAICompatibleJudge(
                base_url=JUDGE_BASE_URL,
                model=JUDGE_MODEL,
                api_key=JUDGE_API_KEY,
                temperature=0.0,
                timeout_seconds=60.0,
                max_tokens=512,
                max_context_docs=JUDGE_MAX_CONTEXT_DOCS,
                max_doc_chars=1500,
            )
            print("LLM-as-a-Judge enabled:", JUDGE_BASE_URL, JUDGE_MODEL)
        else:
            print("LLM-as-a-Judge disabled. Set USE_LLM_JUDGE=true to enable it.")
        """
    ),
    code(
        """
        CITATION_PATTERN = re.compile(r"\\[([^\\[\\]]+)\\]")


        def detect_citations(answer: str) -> list[str]:
            seen = set()
            out = []
            for raw in CITATION_PATTERN.findall(answer):
                citation = raw.strip()
                if citation and citation not in seen:
                    seen.add(citation)
                    out.append(citation)
            return out


        def summarize_gold_aware_rows(rows: list[dict]) -> dict:
            if not rows:
                raise ValueError("Cannot summarize empty generation rows.")

            summary = summarize_generation_evaluations(rows)
            numeric_keys = [
                "gold_in_context",
                "top1_is_gold",
                "citation_points_to_gold",
                "rouge_l_f1_to_gold",
                "lexical_cosine_to_gold",
                "gold_rank_found",
                "reciprocal_gold_rank",
                "gold_in_top_1",
                "gold_in_top_3",
                "gold_in_top_5",
                "gold_in_top_10",
            ]
            for key in numeric_keys:
                values = [float(row.get(key, 0.0)) for row in rows]
                summary[f"avg_{key}"] = sum(values) / float(max(1, len(values)))

            ranks = [
                int(row["gold_rank"])
                for row in rows
                if row.get("gold_rank") is not None
            ]
            summary["gold_rank_not_found_rate"] = 1.0 - summary["avg_gold_rank_found"]
            if ranks:
                ordered = sorted(ranks)
                mid = len(ordered) // 2
                summary["gold_rank_mean"] = sum(ordered) / float(len(ordered))
                summary["gold_rank_median"] = (
                    float(ordered[mid])
                    if len(ordered) % 2
                    else (ordered[mid - 1] + ordered[mid]) / 2.0
                )
            else:
                summary["gold_rank_mean"] = 0.0
                summary["gold_rank_median"] = 0.0

            judged = [row for row in rows if isinstance(row.get("llm_judge"), dict)]
            summary["llm_judge_coverage_rate"] = len(judged) / float(max(1, len(rows)))
            summary["llm_judge_error_rate"] = (
                sum(1 for row in rows if row.get("llm_judge_error"))
                / float(max(1, len(rows)))
            )
            for key in ["faithfulness", "relevance", "completeness", "safety"]:
                values = [
                    float(row["llm_judge"][key])
                    for row in judged
                    if key in row["llm_judge"]
                ]
                if values:
                    summary[f"avg_judge_{key}"] = sum(values) / float(len(values))
            verdicts = [
                str(row["llm_judge"].get("verdict", "")).strip().lower()
                for row in judged
                if row["llm_judge"].get("verdict")
            ]
            if verdicts:
                summary["judge_pass_rate"] = (
                    sum(1 for verdict in verdicts if verdict == "pass")
                    / float(len(verdicts))
                )
            return summary


        def generate_for_run(method: str, run_file: Path, context_top_k: int, judge=None) -> dict:
            queries = read_jsonl(PROJECT_PROCESSED_DIR / "eval_queries.jsonl", limit=MAX_EXAMPLES)
            docstore = load_docstore(PROJECT_PROCESSED_DIR / "corpus.jsonl")
            qrels = read_qrels(PROJECT_PROCESSED_DIR / "qrels.tsv")
            run = read_trec_run(run_file)
            latencies_ms = read_latency_profile(run_file)

            rows = []
            for idx, query_row in enumerate(queries, start=1):
                qid = query_id(query_row, idx)
                question = query_text(query_row)
                ranked_for_query = run.get(qid, [])
                ranked_docs = ranked_for_query[: int(context_top_k)]
                gold_doc_ids = set(qrels.get(qid, []))
                explicit_gold = query_row.get("gold_doc_id")
                if explicit_gold:
                    gold_doc_ids.add(str(explicit_gold))
                gold_doc_id = next(iter(sorted(gold_doc_ids)), None)
                gold_doc = docstore.get(gold_doc_id, {}) if gold_doc_id else {}
                reference_answer = str(gold_doc.get("text") or "")
                gold_rank = find_rank(ranked_for_query, gold_doc_ids)

                docs = []
                for doc_id, score in ranked_docs:
                    doc = docstore.get(doc_id, {})
                    docs.append(
                        {
                            "doc_id": doc_id,
                            "score": float(score),
                            "text": str(doc.get("text") or ""),
                            "source": doc.get("source"),
                        }
                    )

                prompt = build_prompt(question, docs)
                started = time.perf_counter()
                answer = generator.generate(prompt)
                generation_latency_ms = (time.perf_counter() - started) * 1000.0
                citations = detect_citations(answer)
                retrieved_doc_ids = {doc["doc_id"] for doc in docs}
                retrieval_latency_ms = (
                    float(latencies_ms[idx - 1])
                    if idx - 1 < len(latencies_ms)
                    else 0.0
                )
                gold_in_context = bool(gold_doc_ids & retrieved_doc_ids)
                top1_doc_id = docs[0]["doc_id"] if docs else None

                row = {
                    "query_id": qid,
                    "question": question,
                    "method": method,
                    "run_file": str(run_file),
                    "context_top_k": int(context_top_k),
                    "gold_doc_id": gold_doc_id,
                    "gold_doc_ids": sorted(gold_doc_ids),
                    "gold_rank": gold_rank,
                    "reference_answer": reference_answer,
                    "top1_doc_id": top1_doc_id,
                    "top1_is_gold": float(top1_doc_id in gold_doc_ids) if top1_doc_id else 0.0,
                    "gold_in_context": float(gold_in_context),
                    "gold_rank_found": float(gold_rank is not None),
                    "reciprocal_gold_rank": 0.0 if gold_rank is None else 1.0 / float(gold_rank),
                    "gold_in_top_1": float(gold_rank is not None and gold_rank <= 1),
                    "gold_in_top_3": float(gold_rank is not None and gold_rank <= 3),
                    "gold_in_top_5": float(gold_rank is not None and gold_rank <= 5),
                    "gold_in_top_10": float(gold_rank is not None and gold_rank <= 10),
                    "retrieved": docs,
                    "answer": answer,
                    "citations_detected": citations,
                    "supported_citations_detected": [c for c in citations if c in retrieved_doc_ids],
                    "unsupported_citations_detected": [c for c in citations if c not in retrieved_doc_ids],
                    "citation_points_to_gold": float(any(c in gold_doc_ids for c in citations)),
                    "rouge_l_f1_to_gold": rouge_l_f1(answer, reference_answer),
                    "lexical_cosine_to_gold": lexical_cosine(answer, reference_answer),
                    "retrieval_latency_ms": retrieval_latency_ms,
                    "generation_latency_ms": float(generation_latency_ms),
                    "end_to_end_latency_ms": float(retrieval_latency_ms + generation_latency_ms),
                    "reranker_enabled": "reranked" in method,
                }
                row["evaluation"] = evaluate_generation_result(row)
                if judge is not None:
                    try:
                        row["llm_judge"] = judge.evaluate(row)
                    except Exception as exc:
                        row["llm_judge_error"] = f"{type(exc).__name__}: {exc}"
                rows.append(row)

            summary = summarize_gold_aware_rows(rows)
            summary.update(
                {
                    "method": method,
                    "run_file": str(run_file),
                    "llm_model_name": LLM_MODEL_NAME,
                    "context_top_k": int(context_top_k),
                    "max_examples": MAX_EXAMPLES,
                    "use_llm_judge": bool(judge is not None),
                    "judge_base_url": JUDGE_BASE_URL if judge is not None else None,
                    "judge_model": JUDGE_MODEL if judge is not None else None,
                    "generation_latency_mean_ms": sum(row["generation_latency_ms"] for row in rows) / max(1, len(rows)),
                }
            )
            return {"rows": rows, "summary": summary}
        """
    ),
    code(
        """
        def truncate(text: str, max_chars: int = 240) -> str:
            clean = " ".join(str(text or "").split())
            if len(clean) <= max_chars:
                return clean
            return clean[: max_chars - 3] + "..."


        def write_generation_report(path: Path, summary: dict, rows: list[dict]) -> None:
            lines = [
                f"# Generation Evaluation: {summary['method']}",
                "",
                "## Summary",
                "",
            ]
            for key in sorted(summary):
                value = summary[key]
                if isinstance(value, float):
                    lines.append(f"- `{key}`: {value:.4f}")
                else:
                    lines.append(f"- `{key}`: {value}")
            lines.append("")
            lines.append("## Examples")
            lines.append("")
            for idx, row in enumerate(rows, start=1):
                lines.append(f"### Example {idx} (`{row['query_id']}`)")
                lines.append("")
                lines.append(f"**Question**: {row['question']}")
                lines.append("")
                ev = row["evaluation"]
                lines.append(
                    f"**Scores**: context_relevance={ev['context_relevance']:.3f}, "
                    f"groundedness={ev['groundedness']:.3f}, "
                    f"answer_relevance={ev['answer_relevance']:.3f}"
                )
                lines.append(
                    f"**Gold**: gold_doc_id=`{row.get('gold_doc_id')}`, "
                    f"gold_rank={row.get('gold_rank')}, "
                    f"gold_in_context={row.get('gold_in_context')}, "
                    f"rouge_l_f1={row.get('rouge_l_f1_to_gold', 0.0):.3f}, "
                    f"lexical_cosine={row.get('lexical_cosine_to_gold', 0.0):.3f}"
                )
                if isinstance(row.get("llm_judge"), dict):
                    judge_row = row["llm_judge"]
                    lines.append(
                        "**LLM judge**: "
                        f"faithfulness={float(judge_row.get('faithfulness', 0.0)):.1f}, "
                        f"relevance={float(judge_row.get('relevance', 0.0)):.1f}, "
                        f"completeness={float(judge_row.get('completeness', 0.0)):.1f}, "
                        f"safety={float(judge_row.get('safety', 0.0)):.1f}, "
                        f"verdict={judge_row.get('verdict')}"
                    )
                elif row.get("llm_judge_error"):
                    lines.append(f"**LLM judge error**: {row['llm_judge_error']}")
                lines.append("")
                lines.append("**Top docs**:")
                for rank, doc in enumerate(row["retrieved"], start=1):
                    marker = " [GOLD]" if doc["doc_id"] in set(row.get("gold_doc_ids") or []) else ""
                    lines.append(f"{rank}. `{doc['doc_id']}`{marker} (score={doc['score']:.4f}) - {truncate(doc['text'])}")
                lines.append("")
                lines.append("**Answer:**")
                lines.append("")
                lines.append(row["answer"])
                lines.append("")
            path.write_text("\\n".join(lines), encoding="utf-8")


        generation_summaries = []
        for context_top_k in CONTEXT_TOP_K_VALUES:
            for method in GENERATION_METHODS:
                run_file = existing_run(method)
                print(f"\\n== Generate {method} | context_top_k={context_top_k} ==")
                result = generate_for_run(
                    method,
                    run_file,
                    context_top_k=context_top_k,
                    judge=judge,
                )
                out_dir = GENERATION_ROOT / method / f"topk_{context_top_k}"
                out_dir.mkdir(parents=True, exist_ok=True)

                raw_jsonl = out_dir / "generation_eval.jsonl"
                summary_json = out_dir / "summary.json"
                report_md = out_dir / "report.md"

                write_jsonl(raw_jsonl, result["rows"])
                summary_json.write_text(json.dumps(result["summary"], ensure_ascii=False, indent=2), encoding="utf-8")
                write_generation_report(report_md, result["summary"], result["rows"])
                generation_summaries.append(result["summary"])

                print("Saved:", raw_jsonl)
                print("Saved:", summary_json)
                print("Saved:", report_md)

        gen_df = pd.DataFrame(generation_summaries)
        comparison_csv = SUMMARY_DIR / "generation_quality_summary.csv"
        comparison_json = SUMMARY_DIR / "generation_quality_summary.json"
        gen_df.to_csv(comparison_csv, index=False)
        gen_df.to_json(comparison_json, orient="records", force_ascii=False, indent=2)

        print("Saved generation comparison:", comparison_csv)
        sort_key = "avg_rouge_l_f1_to_gold" if "avg_rouge_l_f1_to_gold" in gen_df.columns else "avg_groundedness"
        display(gen_df.sort_values(sort_key, ascending=False) if sort_key in gen_df.columns else gen_df)
        """
    ),
]


def main() -> None:
    write_notebook(
        ROOT / "colab" / "01_train_retriever_and_build_indices.ipynb",
        "01 Train Retriever And Build Indices",
        NOTEBOOK_01,
    )
    write_notebook(
        ROOT / "colab" / "02_benchmark_retrieval_reranker_graph.ipynb",
        "02 Benchmark Retrieval Reranker Graph",
        NOTEBOOK_02,
    )
    write_notebook(
        ROOT / "colab" / "03_evaluate_generation_quality.ipynb",
        "03 Evaluate Generation Quality",
        NOTEBOOK_03,
    )


if __name__ == "__main__":
    main()
