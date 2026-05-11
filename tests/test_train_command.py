from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

from omegaconf import OmegaConf

from medical_rag_reranker.commands import __main__ as commands
from medical_rag_reranker.data.dvc_data import has_prepared_training_artifacts


def _write_required_processed_artifacts(processed_dir: Path) -> None:
    processed_dir.mkdir(parents=True, exist_ok=True)
    for name in ("qa.jsonl", "corpus.jsonl", "splits.json"):
        (processed_dir / name).write_text("{}\n", encoding="utf-8")


def test_has_prepared_training_artifacts_requires_all_files(tmp_path: Path) -> None:
    processed_dir = tmp_path / "processed"

    assert has_prepared_training_artifacts(processed_dir) is False

    processed_dir.mkdir(parents=True, exist_ok=True)
    for name in ("qa.jsonl", "corpus.jsonl", "splits.json"):
        (processed_dir / name).touch()

    assert has_prepared_training_artifacts(processed_dir) is False

    _write_required_processed_artifacts(processed_dir)

    assert has_prepared_training_artifacts(processed_dir) is True


def test_train_skips_raw_data_when_prepared_artifacts_exist(
    tmp_path: Path, monkeypatch
) -> None:
    processed_dir = tmp_path / "processed"
    _write_required_processed_artifacts(processed_dir)
    cfg = OmegaConf.create(
        {
            "data": {
                "processed_dir": str(processed_dir),
                "prefer_prepared_artifacts": True,
            }
        }
    )
    calls: list[str] = []
    fake_train_module = ModuleType("medical_rag_reranker.training.train")

    def fake_train_from_cfg(received_cfg) -> None:
        assert received_cfg is cfg
        calls.append("train")

    fake_train_module.train_from_cfg = fake_train_from_cfg
    monkeypatch.setitem(
        sys.modules,
        "medical_rag_reranker.training.train",
        fake_train_module,
    )
    monkeypatch.setattr(commands, "_load_cfg", lambda **_kwargs: cfg)
    monkeypatch.setattr(
        commands,
        "ensure_data",
        lambda _cfg: calls.append("ensure_data"),
    )

    commands.cmd_train()

    assert calls == ["train"]


def test_train_ensures_raw_data_when_prepared_artifacts_missing(
    tmp_path: Path, monkeypatch
) -> None:
    cfg = OmegaConf.create(
        {
            "data": {
                "processed_dir": str(tmp_path / "processed"),
                "prefer_prepared_artifacts": True,
            }
        }
    )
    calls: list[str] = []
    fake_train_module = ModuleType("medical_rag_reranker.training.train")

    def fake_train_from_cfg(received_cfg) -> None:
        assert received_cfg is cfg
        calls.append("train")

    fake_train_module.train_from_cfg = fake_train_from_cfg
    monkeypatch.setitem(
        sys.modules,
        "medical_rag_reranker.training.train",
        fake_train_module,
    )
    monkeypatch.setattr(commands, "_load_cfg", lambda **_kwargs: cfg)
    monkeypatch.setattr(
        commands,
        "ensure_data",
        lambda _cfg: calls.append("ensure_data"),
    )

    commands.cmd_train()

    assert calls == ["ensure_data", "train"]
