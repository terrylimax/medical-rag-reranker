from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Sequence

import fire
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

from medical_rag_reranker.data.dvc_data import ensure_data
from medical_rag_reranker.inference.infer import infer_from_cfg
from medical_rag_reranker.training.train import train_from_cfg


def _load_cfg(
    config_dir: Optional[str] = None,
    config_name: str = "config",
    overrides: Optional[str | Sequence[str]] = None,
) -> DictConfig:
    """Load Hydra config from ./configs (repo root) by default."""
    repo_root = Path(__file__).resolve().parents[1]
    cfg_dir = Path(config_dir) if config_dir else (repo_root / "configs")

    override_list: list[str] = []

    if overrides:
        if isinstance(overrides, (list, tuple)):
            override_list = [str(x) for x in overrides]
        else:
            overrides_str = str(overrides).strip()
            # Allow passing JSON list: --overrides '["train.max_epochs=2","train.batch_size=16"]'
            if overrides_str.startswith("["):
                override_list = [str(x) for x in json.loads(overrides_str)]
            else:
                # Allow simple comma-separated or space-separated values
                # Example: --overrides 'train.max_epochs=2,train.batch_size=16'
                # Example: --overrides 'train.max_epochs=2 train.batch_size=16'
                normalized = overrides_str.replace(",", " ")
                override_list = [
                    part for part in (p.strip() for p in normalized.split()) if part
                ]

    with initialize_config_dir(version_base=None, config_dir=str(cfg_dir)):
        cfg = compose(config_name=config_name, overrides=override_list)
    return cfg


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


if __name__ == "__main__":
    main()
