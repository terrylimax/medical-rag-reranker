from __future__ import annotations

from dataclasses import dataclass

from omegaconf import DictConfig

from medical_rag_reranker.jobs.dispatchers import BrokerDispatcher, SyncDispatcher
from medical_rag_reranker.jobs.migrations import upgrade_jobs_schema
from medical_rag_reranker.jobs.ports import TaskDispatcher
from medical_rag_reranker.jobs.repositories.file import (
    FileJobRepository,
    FileResultRepository,
)
from medical_rag_reranker.jobs.repositories.postgres import (
    PostgresJobRepository,
    PostgresResultRepository,
    build_postgres_engine,
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
            upgrade_jobs_schema(dsn)
        return (
            PostgresJobRepository(engine=engine),
            PostgresResultRepository(engine=engine),
        )

    raise ValueError(f"Unsupported jobs.storage: {storage}. Use file or postgres.")


def build_job_service(cfg: DictConfig) -> InferenceJobService:
    job_repository, result_repository = _build_repositories(cfg)

    return InferenceJobService(
        cfg=cfg,
        job_repository=job_repository,
        result_repository=result_repository,
    )


def build_job_runtime(cfg: DictConfig) -> JobRuntime:
    service = build_job_service(cfg)

    dispatch_mode = str(getattr(cfg.jobs, "dispatch", "sync")).strip().lower()
    if dispatch_mode == "sync":
        dispatcher: TaskDispatcher = SyncDispatcher(processor=service)
    elif dispatch_mode == "broker":
        from medical_rag_reranker.jobs.celery_app import build_celery_app_from_cfg

        celery_app, broker_cfg = build_celery_app_from_cfg(cfg)
        dispatcher = BrokerDispatcher(
            celery_app=celery_app,
            task_name=broker_cfg.task_name,
            queue=broker_cfg.queue,
            routing_key=broker_cfg.routing_key,
        )
    else:
        raise ValueError(
            f"Unsupported jobs.dispatch mode: {dispatch_mode}. Use sync or broker."
        )

    return JobRuntime(service=service, dispatcher=dispatcher)


def migrate_jobs_schema(cfg: DictConfig) -> None:
    """Apply Alembic migrations for Postgres-backed job storage."""
    storage = str(getattr(cfg.jobs, "storage", "file")).strip().lower()
    if storage != "postgres":
        raise ValueError("jobs.storage must be `postgres` to apply DB migrations.")

    dsn = _as_optional_str(getattr(cfg.jobs.postgres, "dsn", None))
    if dsn is None:
        raise ValueError("jobs.postgres.dsn is required to apply DB migrations.")

    upgrade_jobs_schema(dsn)
