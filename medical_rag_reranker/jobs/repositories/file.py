from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

from medical_rag_reranker.jobs.models import InferenceJob, JobStatus, utc_now_iso
from medical_rag_reranker.jobs.ports import JobRepository, ResultRepository


class FileJobRepository(JobRepository):
    """File-based storage for inference jobs."""

    def __init__(self, store_dir: str | Path) -> None:
        self._store_dir = Path(store_dir)
        self._jobs_dir = self._store_dir / "jobs"
        self._jobs_dir.mkdir(parents=True, exist_ok=True)

    def _job_path(self, job_id: str) -> Path:
        return self._jobs_dir / f"{job_id}.json"

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
        self._write_job(job)
        return job

    def get(self, job_id: str) -> InferenceJob | None:
        path = self._job_path(job_id)
        if not path.exists():
            return None
        row = json.loads(path.read_text(encoding="utf-8"))
        return InferenceJob.from_dict(row)

    def update_status(
        self,
        job_id: str,
        status: JobStatus,
        *,
        error_message: str | None = None,
    ) -> InferenceJob:
        job = self.get(job_id)
        if job is None:
            raise KeyError(f"Job not found: {job_id}")

        now = utc_now_iso()
        job.status = status
        job.error_message = error_message

        if status == JobStatus.RUNNING and job.started_at is None:
            job.started_at = now
        if status in (JobStatus.SUCCEEDED, JobStatus.FAILED):
            job.finished_at = now

        self._write_job(job)
        return job

    def _write_job(self, job: InferenceJob) -> None:
        path = self._job_path(job.job_id)
        payload = json.dumps(job.to_dict(), ensure_ascii=False, indent=2)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(payload, encoding="utf-8")
        tmp.replace(path)


class FileResultRepository(ResultRepository):
    """File-based storage for inference job results."""

    def __init__(self, store_dir: str | Path) -> None:
        self._store_dir = Path(store_dir)
        self._results_dir = self._store_dir / "results"
        self._results_dir.mkdir(parents=True, exist_ok=True)

    def _result_path(self, job_id: str) -> Path:
        return self._results_dir / f"{job_id}.json"

    def save(self, job_id: str, result: Mapping[str, Any]) -> None:
        path = self._result_path(job_id)
        payload = json.dumps(dict(result), ensure_ascii=False, indent=2)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(payload, encoding="utf-8")
        tmp.replace(path)

    def get(self, job_id: str) -> dict[str, Any] | None:
        path = self._result_path(job_id)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))
