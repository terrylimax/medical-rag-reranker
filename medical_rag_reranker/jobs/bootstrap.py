from __future__ import annotations

from dataclasses import dataclass

from omegaconf import DictConfig

from medical_rag_reranker.jobs.dispatchers import SyncDispatcher
from medical_rag_reranker.jobs.ports import TaskDispatcher
from medical_rag_reranker.jobs.repositories.file import (
    FileJobRepository,
    FileResultRepository,
)
from medical_rag_reranker.jobs.repositories.postgres import (
    PostgresJobRepository,
    PostgresResultRepository,
    build_postgres_engine,
    ensure_postgres_schema,
)
from medical_rag_reranker.jobs.service import InferenceJobService


@dataclass
class JobRuntime:
    service: InferenceJobService
    dispatcher: TaskDispatcher


def _as_optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if text in ("", "None", "null"):
        return None
    return text


def _as_bool(value: object, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value

    text = str(value).strip().lower()
    if text in ("1", "true", "yes", "y", "on"):
        return True
    if text in ("0", "false", "no", "n", "off", ""):
        return False
    raise ValueError(f"Cannot parse boolean value from: {value!r}")


def _build_repositories(cfg: DictConfig):
    storage = str(getattr(cfg.jobs, "storage", "file")).strip().lower()
    if storage == "file":
        store_dir = str(cfg.jobs.store_dir)
        return (
            FileJobRepository(store_dir=store_dir),
            FileResultRepository(store_dir=store_dir),
        )

    if storage == "postgres":
        dsn = _as_optional_str(getattr(cfg.jobs.postgres, "dsn", None))
        if dsn is None:
            raise ValueError(
                "jobs.storage=postgres requires jobs.postgres.dsn to be set."
            )
        engine = build_postgres_engine(dsn)
        init_schema = _as_bool(getattr(cfg.jobs.postgres, "init_schema", False))
        if init_schema:
            ensure_postgres_schema(engine)
        return (
            PostgresJobRepository(engine=engine, init_schema=False),
            PostgresResultRepository(engine=engine, init_schema=False),
        )

    raise ValueError(f"Unsupported jobs.storage: {storage}. Use file or postgres.")


def build_job_runtime(cfg: DictConfig) -> JobRuntime:
    job_repository, result_repository = _build_repositories(cfg)

    service = InferenceJobService(
        cfg=cfg,
        job_repository=job_repository,
        result_repository=result_repository,
    )

    dispatch_mode = str(getattr(cfg.jobs, "dispatch", "sync")).strip().lower()
    if dispatch_mode == "sync":
        dispatcher: TaskDispatcher = SyncDispatcher(processor=service)
    elif dispatch_mode == "broker":
        raise NotImplementedError(
            "dispatch=broker is reserved for a future message-broker adapter."
        )
    else:
        raise ValueError(
            f"Unsupported jobs.dispatch mode: {dispatch_mode}. Use sync or broker."
        )

    return JobRuntime(service=service, dispatcher=dispatcher)


def init_jobs_schema(cfg: DictConfig) -> None:
    """Create required Postgres tables for job storage."""
    storage = str(getattr(cfg.jobs, "storage", "file")).strip().lower()
    if storage != "postgres":
        raise ValueError("jobs.storage must be `postgres` to initialize DB schema.")

    dsn = _as_optional_str(getattr(cfg.jobs.postgres, "dsn", None))
    if dsn is None:
        raise ValueError("jobs.postgres.dsn is required to initialize DB schema.")

    engine = build_postgres_engine(dsn)
    ensure_postgres_schema(engine)
