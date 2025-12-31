from __future__ import annotations

from pathlib import Path

from datasets import load_dataset


def download_data(data_dir: str = "data/raw") -> None:
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    # 1) QA пары (question, answer)
    medquad = load_dataset(
        "lavita/MedQuAD"
    )  # https://huggingface.co/datasets/lavita/MedQuAD
    medquad_path = data_path / "medquad"
    medquad_path.mkdir(parents=True, exist_ok=True)
    medquad["train"].to_parquet(str(medquad_path / "train.parquet"))

    # 2) Корпус документов для негативов
    pubmed = load_dataset("MedRAG/pubmed")  # https://huggingface.co/MedRAG/datasets
    pubmed_path = data_path / "pubmed"
    pubmed_path.mkdir(parents=True, exist_ok=True)
    pubmed["train"].to_parquet(str(pubmed_path / "train.parquet"))

    # StatPearls может быть тяжёлым/особым, но обычно грузится через datasets
    statpearls = load_dataset(
        "MedRAG/statpearls"
    )  # https://huggingface.co/MedRAG/datasets
    statpearls_path = data_path / "statpearls"
    statpearls_path.mkdir(parents=True, exist_ok=True)
    statpearls["train"].to_parquet(str(statpearls_path / "train.parquet"))
