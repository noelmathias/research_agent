import asyncio
import json
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from backend.jobs.job_manager import job_manager
from backend.jobs.job_models import JobStatus
from backend.jobs.pipeline_runner import execute_research_job
from shared.schemas import ResearchRunRequest
from backend.core.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)


@router.post("/research/run_async", summary="Submit async research job")
async def run_research_async(request: ResearchRunRequest):
    """
    Submits a research job and returns job_id immediately.
    Pipeline runs in the background.
    Poll /research/job/{job_id} for status.
    Stream /research/stream/{job_id} for live events.
    """
    from shared.config import get_settings
    settings = get_settings()

    job = job_manager.create_job(
        query=request.query,
        model=request.model or settings.ollama_model,
        collection_name=request.collection_name or settings.chroma_collection_name,
    )

    # Fire and forget — pipeline runs concurrently
    asyncio.create_task(
        execute_research_job(
            job_id=job.job_id,
            query=request.query,
            model=request.model or settings.ollama_model,
            collection_name=request.collection_name,
        )
    )

    logger.info(
        "Async job submitted | job_id=%s | query_len=%d",
        job.job_id, len(request.query),
    )

    return {
        "job_id": job.job_id,
        "status": job.status,
        "message": "Job submitted. Connect to /research/stream/{job_id} for live updates.",
        "stream_url": f"/api/v1/research/stream/{job.job_id}",
        "status_url": f"/api/v1/research/job/{job.job_id}",
    }


@router.get("/research/job/{job_id}", summary="Poll job status and result")
async def get_job_status(job_id: str):
    """
    Returns current job status.
    When status is 'completed', result field contains full pipeline output.
    """
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(
            status_code=404,
            detail=f"Job '{job_id}' not found.",
        )

    response = job.to_status_response()

    # Include full result only when complete
    if job.status == JobStatus.COMPLETED and job.result:
        response["result"] = job.result

    return response


@router.get("/research/jobs", summary="List all submitted jobs")
async def list_jobs():
    jobs = job_manager.list_jobs()
    return {"total": len(jobs), "jobs": jobs}


@router.get(
    "/research/stream/{job_id}",
    summary="SSE stream for live pipeline events",
)
async def stream_job_events(job_id: str):
    """
    Server-Sent Events endpoint.

    Streams pipeline events as they are emitted by LangGraph nodes.
    Replays historical events for late subscribers.
    Closes automatically when pipeline completes or fails.

    Client usage:
        const es = new EventSource('/api/v1/research/stream/{job_id}');
        es.addEventListener('node_complete', e => console.log(e.data));
    """
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(
            status_code=404,
            detail=f"Job '{job_id}' not found.",
        )

    logger.info("SSE connection opened | job_id=%s", job_id)

    async def event_generator():
        try:
            async for event_dict in job_manager.stream_events(job_id):
                # Keepalive — no data payload
                if event_dict.get("event") == "keepalive":
                    yield ": keepalive\n\n"
                    continue

                event_type = event_dict.get("event", "message")
                payload = json.dumps(event_dict)
                yield f"event: {event_type}\ndata: {payload}\n\n"

        except asyncio.CancelledError:
            logger.info("SSE connection closed by client | job_id=%s", job_id)
        finally:
            logger.info("SSE generator done | job_id=%s", job_id)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",       # disables nginx buffering
            "Connection": "keep-alive",
        },
    )