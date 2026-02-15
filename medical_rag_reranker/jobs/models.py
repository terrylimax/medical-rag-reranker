from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


def utc_now_iso() -> str:
    """Return current UTC timestamp in ISO-8601 format."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class JobStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"


@dataclass
class InferenceJob:
    job_id: str
    question: str
    status: JobStatus
    created_at: str
    metadata: dict[str, Any] = field(default_factory=dict)
    started_at: str | None = None
    finished_at: str | None = None
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "question": self.question,
            "status": self.status.value,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "error_message": self.error_message,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "InferenceJob":
        metadata = row.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}

        return cls(
            job_id=str(row["job_id"]),
            question=str(row["question"]),
            status=JobStatus(str(row["status"])),
            created_at=str(row["created_at"]),
            started_at=(
                None if row.get("started_at") is None else str(row["started_at"])
            ),
            finished_at=(
                None if row.get("finished_at") is None else str(row["finished_at"])
            ),
            error_message=(
                None if row.get("error_message") is None else str(row["error_message"])
            ),
            metadata=metadata,
        )
