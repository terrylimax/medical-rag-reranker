from __future__ import annotations

from typing import Any, Mapping, Protocol

from medical_rag_reranker.jobs.models import InferenceJob, JobStatus


class JobRepository(Protocol):
    def create(
        self, *, question: str, metadata: dict[str, Any] | None = None
    ) -> InferenceJob: ...

    def get(self, job_id: str) -> InferenceJob | None: ...

    def update_status(
        self,
        job_id: str,
        status: JobStatus,
        *,
        error_message: str | None = None,
    ) -> InferenceJob: ...


class ResultRepository(Protocol):
    def save(self, job_id: str, result: Mapping[str, Any]) -> None: ...

    def get(self, job_id: str) -> dict[str, Any] | None: ...


class JobProcessor(Protocol):
    def process(self, job_id: str) -> None: ...


class TaskDispatcher(Protocol):
    def dispatch(self, job_id: str) -> None: ...
