from __future__ import annotations

import os
from datetime import datetime

try:
    from airflow import DAG
    from airflow.operators.bash import BashOperator
except Exception as exc:  # pragma: no cover - imported only by Airflow runtime
    raise RuntimeError(
        "This DAG must be imported by an Airflow environment with apache-airflow "
        "installed."
    ) from exc


PROJECT_DIR = os.getenv("MEDICAL_RAG_PROJECT_DIR", "/opt/medical-rag-reranker")
PROFILE = os.getenv("EXPERIMENT_PROFILE", "full_remote")
RUN_ID = os.getenv("EXPERIMENT_RUN_ID", "medquad_full_remote")
TRAINING_MODE = os.getenv("EXPERIMENT_TRAINING_MODE", "colab_artifacts")


def _command(stage: str) -> str:
    return (
        f"cd {PROJECT_DIR} && "
        "python -m medical_rag_reranker.commands experiment_matrix "
        f"--profile {PROFILE} "
        f"--stage {stage} "
        f"--run_id {RUN_ID} "
        f"--training_mode {TRAINING_MODE} "
        "--resume true "
        "--dry_run false"
    )


with DAG(
    dag_id="full_experiment_matrix",
    description="Full remote retrieval/generation experiment matrix.",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    tags=["medical-rag", "experiments"],
) as dag:
    artifact_pull = BashOperator(
        task_id="artifact_pull",
        bash_command=(
            f"cd {PROJECT_DIR} && python -m medical_rag_reranker.commands artifact_pull"
        ),
    )
    preflight = BashOperator(
        task_id="preflight",
        bash_command=_command("preflight"),
    )
    train = BashOperator(
        task_id="train",
        bash_command=_command("train"),
    )
    index = BashOperator(
        task_id="index",
        bash_command=_command("index"),
    )
    retrieval_eval = BashOperator(
        task_id="retrieval_eval",
        bash_command=_command("retrieval"),
    )
    generation_eval = BashOperator(
        task_id="generation_eval",
        bash_command=_command("generation"),
    )
    summary = BashOperator(
        task_id="summary",
        bash_command=_command("summary"),
    )
    artifact_push = BashOperator(
        task_id="artifact_push",
        bash_command=(
            f"cd {PROJECT_DIR} && "
            "python -m medical_rag_reranker.commands artifact_push "
            f"--include 'artifacts/experiments/{RUN_ID}/**/*'"
        ),
    )

    (
        artifact_pull
        >> preflight
        >> train
        >> index
        >> retrieval_eval
        >> generation_eval
        >> summary
        >> artifact_push
    )
