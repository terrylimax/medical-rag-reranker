from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from medical_rag_reranker.jobs.ports import JobProcessor, TaskDispatcher


@dataclass
class SyncDispatcher(TaskDispatcher):
    """Dispatcher that executes jobs immediately in the current process."""

    processor: JobProcessor

    def dispatch(self, job_id: str) -> None:
        self.processor.process(job_id)


@dataclass
class BrokerDispatcher(TaskDispatcher):
    """Dispatcher that publishes jobs to a message broker via Celery."""

    celery_app: Any
    task_name: str
    queue: str | None = None
    routing_key: str | None = None

    def dispatch(self, job_id: str) -> None:
        options: dict[str, str] = {}
        if self.queue:
            options["queue"] = self.queue
        if self.routing_key:
            options["routing_key"] = self.routing_key

        self.celery_app.send_task(
            self.task_name,
            args=[job_id],
            kwargs={},
            **options,
        )
