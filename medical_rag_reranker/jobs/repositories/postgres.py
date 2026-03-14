from __future__ import annotations

import json
from typing import Any, Mapping
from uuid import uuid4

from sqlalchemy import Engine, create_engine, text
from sqlalchemy.exc import NoSuchModuleError

from medical_rag_reranker.jobs.models import InferenceJob, JobStatus, utc_now_iso
from medical_rag_reranker.jobs.ports import JobRepository, ResultRepository


def build_postgres_engine(dsn: str) -> Engine:
    try:
        return create_engine(
            dsn,
            future=True,
            pool_pre_ping=True,
        )
    except (NoSuchModuleError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            "Failed to create Postgres engine. Install a DB driver, e.g. "
            '`poetry add "psycopg[binary]"`, and use DSN '
            "`postgresql+psycopg://...`."
        ) from exc


def _normalize_json_field(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except Exception:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


class PostgresJobRepository(JobRepository):
    """PostgreSQL-backed storage for inference jobs."""

    def __init__(self, *, engine: Engine) -> None:
        self._engine = engine

    def create(
        self, *, question: str, metadata: dict[str, Any] | None = None
    ) -> InferenceJob:
        job = InferenceJob(
            job_id=str(uuid4()),
            question=str(question),
            status=JobStatus.PENDING,
            created_at=utc_now_iso(),
            metadata=metadata or {},
        )

        with self._engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO inference_jobs (
                        job_id, question, status, created_at, metadata
                    ) VALUES (
                        :job_id, :question, :status, :created_at,
                        CAST(:metadata_json AS jsonb)
                    )
                    """
                ),
                {
                    "job_id": job.job_id,
                    "question": job.question,
                    "status": job.status.value,
                    "created_at": job.created_at,
                    "metadata_json": json.dumps(job.metadata, ensure_ascii=False),
                },
            )

        return job

    def get(self, job_id: str) -> InferenceJob | None:
        with self._engine.begin() as conn:
            row = (
                conn.execute(
                    text(
                        """
                    SELECT
                        job_id,
                        question,
                        status,
                        created_at::text AS created_at,
                        started_at::text AS started_at,
                        finished_at::text AS finished_at,
                        error_message,
                        metadata
                    FROM inference_jobs
                    WHERE job_id = :job_id
                    """
                    ),
                    {"job_id": job_id},
                )
                .mappings()
                .first()
            )

        if row is None:
            return None

        payload = dict(row)
        payload["metadata"] = _normalize_json_field(payload.get("metadata"))
        return InferenceJob.from_dict(payload)

    def update_status(
        self,
        job_id: str,
        status: JobStatus,
        *,
        error_message: str | None = None,
    ) -> InferenceJob:
        now = utc_now_iso()
        with self._engine.begin() as conn:
            row = (
                conn.execute(
                    text(
                        """
                    UPDATE inference_jobs
                    SET
                        status = :status,
                        error_message = :error_message,
                        started_at = CASE
                            WHEN :status = 'RUNNING' AND started_at IS NULL
                            THEN :now_ts
                            ELSE started_at
                        END,
                        finished_at = CASE
                            WHEN :status = 'SUCCEEDED' OR :status = 'FAILED'
                            THEN :now_ts
                            ELSE finished_at
                        END
                    WHERE job_id = :job_id
                    RETURNING
                        job_id,
                        question,
                        status,
                        created_at::text AS created_at,
                        started_at::text AS started_at,
                        finished_at::text AS finished_at,
                        error_message,
                        metadata
                    """
                    ),
                    {
                        "job_id": job_id,
                        "status": status.value,
                        "error_message": error_message,
                        "now_ts": now,
                    },
                )
                .mappings()
                .first()
            )

        if row is None:
            raise KeyError(f"Job not found: {job_id}")

        payload = dict(row)
        payload["metadata"] = _normalize_json_field(payload.get("metadata"))
        return InferenceJob.from_dict(payload)


class PostgresResultRepository(ResultRepository):
    """PostgreSQL-backed storage for inference results."""

    def __init__(self, *, engine: Engine) -> None:
        self._engine = engine

    def save(self, job_id: str, result: Mapping[str, Any]) -> None:
        now = utc_now_iso()
        with self._engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO inference_results (job_id, result, created_at, updated_at)
                    VALUES (
                        :job_id,
                        CAST(:result_json AS jsonb),
                        :created_at,
                        :updated_at
                    )
                    ON CONFLICT (job_id)
                    DO UPDATE SET
                        result = EXCLUDED.result,
                        updated_at = EXCLUDED.updated_at
                    """
                ),
                {
                    "job_id": job_id,
                    "result_json": json.dumps(dict(result), ensure_ascii=False),
                    "created_at": now,
                    "updated_at": now,
                },
            )

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self._engine.begin() as conn:
            row = (
                conn.execute(
                    text(
                        """
                    SELECT result
                    FROM inference_results
                    WHERE job_id = :job_id
                    """
                    ),
                    {"job_id": job_id},
                )
                .mappings()
                .first()
            )

        if row is None:
            return None
        result = row.get("result")
        if isinstance(result, dict):
            return result
        if isinstance(result, str):
            parsed = json.loads(result)
            return parsed if isinstance(parsed, dict) else {"value": parsed}
        return None
