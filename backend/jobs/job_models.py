from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime, timezone


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class PipelineStage(str, Enum):
    QUEUED = "queued"
    PLANNING = "planning"
    RETRIEVING = "retrieving"
    SUMMARIZING = "summarizing"
    EVALUATING = "evaluating"
    REFLECTING = "reflecting"
    GENERATING_REPORT = "generating_report"
    DONE = "done"
    ERROR = "error"


class SSEEvent(BaseModel):
    """
    Single Server-Sent Event payload.
    Every field is JSON-serialisable — ready for Redis pub/sub if needed.
    """
    event: str                          # SSE event type name
    job_id: str
    stage: PipelineStage
    message: str
    data: Dict[str, Any] = Field(default_factory=dict)
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class JobRecord(BaseModel):
    """
    Full job record stored in JobManager.
    Immutable fields set at creation; mutable fields updated during execution.
    """
    job_id: str
    query: str
    model: str
    collection_name: str
    status: JobStatus = JobStatus.PENDING
    stage: PipelineStage = PipelineStage.QUEUED
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    updated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    events: List[Dict[str, Any]] = Field(default_factory=list)
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    def to_status_response(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "status": self.status,
            "stage": self.stage,
            "query": self.query,
            "model": self.model,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "event_count": len(self.events),
            "has_result": self.result is not None,
            "error": self.error,
        }