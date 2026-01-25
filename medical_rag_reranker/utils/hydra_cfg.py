from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Sequence

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig


def load_cfg(
    config_dir: Optional[str] = None,
    config_name: str = "config",
    overrides: Optional[str | Sequence[str]] = None,
) -> DictConfig:
    """Load Hydra config from ./configs (repo root) by default.

    Args:
        config_dir: Optional path to a configs directory. If omitted, uses
            `<repo_root>/configs`.
        config_name: Base config name to compose.
        overrides: Optional overrides. Can be:
            - a list/tuple of override strings
            - a JSON list string (e.g. '["train.max_epochs=2"]')
            - a comma/space separated string (e.g. 'a=1,b=2' or 'a=1 b=2')

    Returns:
        Composed Hydra `DictConfig`.
    """
    repo_root = Path(__file__).resolve().parents[2]
    cfg_dir = Path(config_dir) if config_dir else (repo_root / "configs")

    override_list: list[str] = []

    if overrides:
        if isinstance(overrides, (list, tuple)):
            override_list = [str(x) for x in overrides]
        else:
            overrides_str = str(overrides).strip()
            # Allow passing JSON list: --overrides '["a=1","b=2"]'
            if overrides_str.startswith("["):
                override_list = [str(x) for x in json.loads(overrides_str)]
            else:
                # Allow simple comma-separated or space-separated values
                normalized = overrides_str.replace(",", " ")
                override_list = [
                    part for part in (p.strip() for p in normalized.split()) if part
                ]

    with initialize_config_dir(version_base=None, config_dir=str(cfg_dir)):
        cfg = compose(config_name=config_name, overrides=override_list)
    return cfg
