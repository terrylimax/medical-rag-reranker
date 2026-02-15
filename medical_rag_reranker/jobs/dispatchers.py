from __future__ import annotations

from dataclasses import dataclass

from medical_rag_reranker.jobs.ports import JobProcessor, TaskDispatcher


@dataclass
class SyncDispatcher(TaskDispatcher):
    """Dispatcher that executes jobs immediately in the current process."""

    processor: JobProcessor

    def dispatch(self, job_id: str) -> None:
        self.processor.process(job_id)
