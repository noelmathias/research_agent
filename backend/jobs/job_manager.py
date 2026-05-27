import asyncio
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, AsyncGenerator
from backend.jobs.job_models import (
    JobRecord,
    JobStatus,
    PipelineStage,
    SSEEvent,
)
from backend.core.logger import get_logger

logger = get_logger(__name__)

# ── In-process store ──────────────────────────────────────────────────────────
# Swap _jobs and _queues backends for Redis to scale horizontally.
# All public methods keep identical signatures.

_jobs: Dict[str, JobRecord] = {}
_queues: Dict[str, asyncio.Queue] = {}

# Sentinel pushed to queue when job finishes — closes SSE connection cleanly
_STREAM_DONE_SENTINEL = "__DONE__"


class JobManager:
    """
    Manages research job lifecycle and SSE event distribution.

    Thread-safety: asyncio single-threaded — no locks needed.
    Redis upgrade path: replace _jobs dict with Redis hash operations,
    replace _queues with Redis Streams or pub/sub channels.
    """

    # ── Job lifecycle ─────────────────────────────────────────────────────

    def create_job(
        self,
        query: str,
        model: str,
        collection_name: str,
    ) -> JobRecord:
        job_id = str(uuid.uuid4())[:12]
        job = JobRecord(
            job_id=job_id,
            query=query,
            model=model,
            collection_name=collection_name,
        )
        _jobs[job_id] = job
        _queues[job_id] = asyncio.Queue()
        logger.info("Job created | job_id=%s | query_len=%d", job_id, len(query))
        return job

    def get_job(self, job_id: str) -> Optional[JobRecord]:
        return _jobs.get(job_id)

    def list_jobs(self) -> List[Dict]:
        return [j.to_status_response() for j in _jobs.values()]

    def update_job_status(
        self,
        job_id: str,
        status: JobStatus,
        stage: PipelineStage,
        error: Optional[str] = None,
    ) -> None:
        job = _jobs.get(job_id)
        if not job:
            return
        job.status = status
        job.stage = stage
        job.updated_at = datetime.now(timezone.utc).isoformat()
        if error:
            job.error = error
        logger.debug(
            "Job updated | job_id=%s | status=%s | stage=%s",
            job_id, status, stage,
        )

    def set_job_result(self, job_id: str, result: Dict) -> None:
        job = _jobs.get(job_id)
        if not job:
            return
        job.result = result
        job.updated_at = datetime.now(timezone.utc).isoformat()

    # ── SSE event distribution ────────────────────────────────────────────

    def emit(self, event: SSEEvent) -> None:
        """
        Push an event into the job's queue and append to event log.
        Safe to call from within the running asyncio task.
        """
        job = _jobs.get(event.job_id)
        if not job:
            return

        event_dict = event.model_dump()
        job.events.append(event_dict)
        job.updated_at = datetime.now(timezone.utc).isoformat()

        queue = _queues.get(event.job_id)
        if queue:
            queue.put_nowait(event_dict)

        logger.debug(
            "SSE event | job_id=%s | stage=%s | event=%s",
            event.job_id, event.stage, event.event,
        )

    def close_stream(self, job_id: str) -> None:
        """Push sentinel to unblock any waiting SSE consumers."""
        queue = _queues.get(job_id)
        if queue:
            queue.put_nowait(_STREAM_DONE_SENTINEL)

    async def stream_events(
        self, job_id: str
    ) -> AsyncGenerator[Dict, None]:
        """
        Async generator consumed by the SSE endpoint.
        Yields event dicts until sentinel received or job not found.
        Replays already-emitted events first so late subscribers
        don't miss early pipeline stages.
        """
        job = _jobs.get(job_id)
        if not job:
            return

        # Replay historical events for late subscribers
        for event in job.events:
            yield event

        # If job already finished, nothing more to stream
        if job.status in (JobStatus.COMPLETED, JobStatus.FAILED):
            return

        queue = _queues.get(job_id)
        if not queue:
            return

        while True:
            try:
                item = await asyncio.wait_for(queue.get(), timeout=30.0)
            except asyncio.TimeoutError:
                # Yield keepalive comment — prevents proxy timeout
                yield {"event": "keepalive", "data": {}}
                continue

            if item == _STREAM_DONE_SENTINEL:
                break

            yield item


# Module-level singleton
job_manager = JobManager()