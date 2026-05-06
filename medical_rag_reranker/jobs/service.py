from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from medical_rag_reranker.jobs.models import InferenceJob, JobStatus
from medical_rag_reranker.jobs.ports import (
    JobProcessor,
    JobRepository,
    ResultRepository,
)


RETRIEVAL_METHOD_ALIASES = {
    "graph": "graph_bm25",
}


def _resolve_configs_root() -> Path:
    candidates = [
        Path.cwd() / "configs",
        Path(__file__).resolve().parents[2] / "configs",
    ]
    for candidate in candidates:
        if (candidate / "retrieval").is_dir():
            return candidate
    return candidates[0]


def available_retrieval_methods() -> list[str]:
    """Return configured retrieval Hydra group options."""
    retrieval_dir = _resolve_configs_root() / "retrieval"
    if not retrieval_dir.is_dir():
        return ["bm25"]

    methods = sorted(path.stem for path in retrieval_dir.glob("*.yaml"))
    return methods or ["bm25"]


def retrieval_method_options(cfg: DictConfig) -> list[dict[str, Any]]:
    """Return retrieval methods with resolved index readiness metadata."""
    options: list[dict[str, Any]] = []
    for method in available_retrieval_methods():
        try:
            retrieval_cfg = _load_retrieval_cfg(method)
            index_path = _selected_index_path(cfg, retrieval_cfg)
            index_ready = bool(index_path and Path(index_path).exists())
            if index_path is None:
                status = "unconfigured"
                message = "Retrieval config does not declare an index_file."
            elif index_ready:
                status = "ready"
                message = "Index is available."
            else:
                status = "missing_index"
                message = f"Index file is missing: {index_path}"

            options.append(
                {
                    "value": method,
                    "retriever": str(retrieval_cfg.get("name", method)),
                    "index_path": index_path,
                    "index_ready": index_ready,
                    "status": status,
                    "message": message,
                }
            )
        except Exception as exc:
            options.append(
                {
                    "value": method,
                    "retriever": method,
                    "index_path": None,
                    "index_ready": False,
                    "status": "invalid_config",
                    "message": str(exc),
                }
            )
    return options


def normalize_retrieval_method(value: object | None) -> str | None:
    """Validate and normalize a user-selected retrieval method."""
    if value is None:
        return None

    method = str(value).strip()
    if not method:
        return None

    method = RETRIEVAL_METHOD_ALIASES.get(method, method)
    if not method.replace("_", "").replace("-", "").isalnum():
        raise ValueError(f"Invalid retrieval method: {method!r}")

    allowed = set(available_retrieval_methods())
    if method not in allowed:
        choices = ", ".join(sorted(allowed))
        raise ValueError(f"Unsupported retrieval method `{method}`. Choices: {choices}")

    return method


def _load_retrieval_cfg(method: str) -> DictConfig:
    path = _resolve_configs_root() / "retrieval" / f"{method}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Retrieval config does not exist: {path}")
    loaded = OmegaConf.load(path)
    if not isinstance(loaded, DictConfig):
        raise TypeError(f"Retrieval config must be a mapping: {path}")
    return loaded


def _selected_index_path(cfg: DictConfig, retrieval_cfg: DictConfig) -> str | None:
    index_file = retrieval_cfg.get("index_file")
    if index_file is None:
        return None

    index_path = Path(str(index_file))
    if index_path.is_absolute():
        return str(index_path)

    artifacts_dir = OmegaConf.select(cfg, "paths.artifacts_dir")
    if artifacts_dir is None:
        return str(index_path)
    return str(Path(str(artifacts_dir)) / index_path)


def _set_selected_index_paths(cfg: DictConfig, index_path: str | None) -> None:
    if index_path is None:
        return

    for key in (
        "generation.index",
        "retrieval_run.index",
        "run.retrieval_index.out",
    ):
        OmegaConf.update(cfg, key, index_path, merge=False, force_add=True)


def _with_retrieval_method(cfg: DictConfig, method: str | None) -> DictConfig:
    if method is None:
        return cfg

    retrieval_cfg = _load_retrieval_cfg(method)
    next_cfg = OmegaConf.merge(cfg, {})
    next_cfg.retrieval = retrieval_cfg
    _set_selected_index_paths(next_cfg, _selected_index_path(next_cfg, retrieval_cfg))
    return next_cfg


def _job_retrieval_method(metadata: dict[str, Any]) -> str | None:
    return normalize_retrieval_method(
        metadata.get("retrieval_method") or metadata.get("retrieval")
    )


class InferenceJobService(JobProcessor):
    """Application service for creating and processing inference jobs."""

    def __init__(
        self,
        *,
        cfg: DictConfig,
        job_repository: JobRepository,
        result_repository: ResultRepository,
    ) -> None:
        self._cfg = cfg
        self._job_repository = job_repository
        self._result_repository = result_repository

    def create_job(
        self,
        *,
        question: str,
        metadata: dict[str, Any] | None = None,
    ) -> InferenceJob:
        if not str(question).strip():
            raise ValueError("Question must be a non-empty string.")
        return self._job_repository.create(
            question=str(question).strip(),
            metadata=metadata,
        )

    def get_job(self, job_id: str) -> InferenceJob | None:
        return self._job_repository.get(job_id)

    def get_result(self, job_id: str) -> dict[str, Any] | None:
        return self._result_repository.get(job_id)

    def process(self, job_id: str) -> None:
        job = self._job_repository.get(job_id)
        if job is None:
            raise KeyError(f"Job not found: {job_id}")

        self._job_repository.update_status(
            job_id,
            JobStatus.RUNNING,
            error_message=None,
        )

        try:
            # Keep import local: heavy dependencies load only when a job is executed.
            from medical_rag_reranker.inference.generate import generate_from_cfg

            retrieval_method = _job_retrieval_method(job.metadata)
            cfg = _with_retrieval_method(self._cfg, retrieval_method)
            result = generate_from_cfg(
                cfg=cfg,
                question=job.question,
                queries_path=None,
                output_path=None,
            )
            if not isinstance(result, dict):
                raise RuntimeError(
                    "Job execution expected a single-result dict, but got batch output."
                )

            if retrieval_method is not None:
                result["retrieval_method"] = retrieval_method

            self._result_repository.save(job_id, result)
            self._job_repository.update_status(
                job_id,
                JobStatus.SUCCEEDED,
                error_message=None,
            )
        except Exception as exc:
            self._job_repository.update_status(
                job_id,
                JobStatus.FAILED,
                error_message=str(exc),
            )
            raise
