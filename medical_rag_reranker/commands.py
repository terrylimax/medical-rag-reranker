from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import fire
from omegaconf import DictConfig

from medical_rag_reranker.data.dvc_data import ensure_data
from medical_rag_reranker.inference.infer import infer_from_cfg
from medical_rag_reranker.commands.retrieval_run import run_from_cfg as retrieval_run_from_cfg
from medical_rag_reranker.training.train import train_from_cfg
from medical_rag_reranker.utils.hydra_cfg import load_cfg


def _load_cfg(
    config_dir: Optional[str] = None,
    config_name: str = "config",
    overrides: Optional[str | Sequence[str]] = None,
) -> DictConfig:
    """Backward-compatible wrapper around `medical_rag_reranker.utils.hydra_cfg.load_cfg`."""
    return load_cfg(config_dir=config_dir, config_name=config_name, overrides=overrides)


def main() -> None:
    """
    Fire exposes functions as CLI commands:
    - python -m medical_rag_reranker.commands download_data
    - python -m medical_rag_reranker.commands train
    - python -m medical_rag_reranker.commands infer

    Examples:
    - python -m medical_rag_reranker.commands download_data
    - python -m medical_rag_reranker.commands train
    - python -m medical_rag_reranker.commands train --overrides '["train.max_epochs=2","train.batch_size=16"]'
    - python -m medical_rag_reranker.commands infer --query "..." --document "..."
    """
    fire.Fire(
        {
            "download_data": cmd_download_data,
            "train": cmd_train,
            "infer": cmd_infer,
            "retrieval_run": cmd_retrieval_run,
        }
    )


def cmd_download_data(
    config_dir: Optional[str] = None, overrides: Optional[str] = None
) -> None:
    cfg = _load_cfg(config_dir=config_dir, overrides=overrides)
    ensure_data(cfg)


def cmd_train(
    config_dir: Optional[str] = None, overrides: Optional[str] = None
) -> None:
    cfg = _load_cfg(config_dir=config_dir, overrides=overrides)

    # Ensure data exists (DVC pull -> fallback download)
    ensure_data(cfg)

    # Run training
    train_from_cfg(cfg)


def cmd_infer(
    query: str,
    document: str,
    checkpoint_path: Optional[str] = None,
    config_dir: Optional[str] = None,
    overrides: Optional[str] = None,
) -> float:
    cfg = _load_cfg(config_dir=config_dir, overrides=overrides)

    # Requirement: integrate DVC data pulling into infer as well
    ensure_data(cfg)

    score = infer_from_cfg(
        cfg, query=query, document=document, checkpoint_path=checkpoint_path
    )
    print(score)
    return score


def cmd_retrieval_run(
    config_dir: Optional[str] = None, overrides: Optional[str] = None
) -> None:
    """Run the configured retriever over queries and write a TREC run file."""
    cfg = _load_cfg(config_dir=config_dir, overrides=overrides)

    # Ensure data exists (queries/corpus may be managed via DVC)
    ensure_data(cfg)

    retrieval_run_from_cfg(cfg)


if __name__ == "__main__":
    main()
