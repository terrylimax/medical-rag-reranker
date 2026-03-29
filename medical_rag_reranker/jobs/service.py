from __future__ import annotations

from typing import Any

from omegaconf import DictConfig

from medical_rag_reranker.jobs.models import InferenceJob, JobStatus
from medical_rag_reranker.jobs.ports import (
    JobProcessor,
    JobRepository,
    ResultRepository,
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

            result = generate_from_cfg(
                cfg=self._cfg,
                question=job.question,
                queries_path=None,
                output_path=None,
            )
            if not isinstance(result, dict):
                raise RuntimeError(
                    "Job execution expected a single-result dict, but got batch output."
                )

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
