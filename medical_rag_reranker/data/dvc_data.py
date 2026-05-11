from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig

from medical_rag_reranker.data.download import download_data


def _has_required_raw_artifacts(raw_dir: Path) -> bool:
    """Check whether minimal QA raw artifacts are present.

    For `prep_data`, having at least one QA source locally is enough:
    - medquad parquet downloaded by `download_data`
    - or a manually prepared NIH QA jsonl file.
    """
    required_any = [
        raw_dir / "medquad" / "train.parquet",
        raw_dir / "nih_qa.jsonl",
    ]
    return any(p.exists() for p in required_any)


def has_prepared_training_artifacts(processed_dir: Path) -> bool:
    """Check whether reranker training can use prepared processed artifacts."""
    required = (
        processed_dir / "qa.jsonl",
        processed_dir / "corpus.jsonl",
        processed_dir / "splits.json",
    )
    return all(path.is_file() and path.stat().st_size > 0 for path in required)


def ensure_data(cfg: DictConfig) -> None:
    """Ensure data exists locally.

    Strategy (local DVC remote):
    1) If use_dvc=true, try `dvc pull` via Python API.
    2) If pull fails (e.g., first run), download from open sources.
    3) If use_dvc=true, `dvc add` and `dvc push` to populate the local remote.

    This makes `train`/`infer` idempotent: subsequent runs can restore data via DVC.
    """
    raw_dir = Path(str(cfg.data.raw_dir))

    use_dvc = bool(getattr(cfg.data, "use_dvc", False))
    dvc_remote = str(getattr(cfg.data, "dvc_remote", "local"))

    if use_dvc:
        try:
            from dvc.repo import Repo  # type: ignore

            repo_root = Path(__file__).resolve().parents[2]
            repo = Repo(str(repo_root))
            repo.pull(targets=[str(raw_dir)], remote=dvc_remote)
        except Exception:
            pass

    # If still missing, fall back to open-source download.
    # A pre-created empty directory is not enough; verify expected artifacts.
    if not _has_required_raw_artifacts(raw_dir):
        download_data(str(raw_dir))

    if use_dvc:
        try:
            from dvc.repo import Repo  # type: ignore

            repo_root = Path(__file__).resolve().parents[2]
            repo = Repo(str(repo_root))
            repo.add(str(raw_dir))
            repo.push(remote=dvc_remote)
        except Exception:
            # If DVC isn't initialized yet or any issue happens,
            # we still keep downloaded data available locally.
            return
