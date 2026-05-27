import asyncio
from typing import Optional
from backend.jobs.job_manager import job_manager
from backend.jobs.job_models import JobStatus, PipelineStage
from backend.graph.streaming_graph import run_streaming_pipeline
from backend.services.research_service import (
    _build_result_response,
)
from backend.core.logger import get_logger

logger = get_logger(__name__)


async def execute_research_job(
    job_id: str,
    query: str,
    model: str,
    collection_name: Optional[str],
) -> None:
    """
    Background task entry point.

    Called via asyncio.create_task() — runs concurrently with
    the HTTP response that returned the job_id to the client.

    On completion: stores result in job record.
    On failure: marks job as FAILED, stores error.
    Always: closes SSE stream via sentinel.
    """
    logger.info(
        "Pipeline runner | job=%s | starting | query_len=%d",
        job_id, len(query),
    )

    try:
        final_state = await run_streaming_pipeline(
            job_id=job_id,
            query=query,
            model=model,
            collection_name=collection_name,
        )

        # Build serialisable result from final state
        result = _build_result_response(final_state, query, model)
        job_manager.set_job_result(job_id, result)
        job_manager.update_job_status(
            job_id, JobStatus.COMPLETED, PipelineStage.DONE
        )
        logger.info("Pipeline runner | job=%s | completed", job_id)

    except Exception as e:
        logger.error(
            "Pipeline runner | job=%s | failed | %s", job_id, e
        )
        job_manager.update_job_status(
            job_id,
            JobStatus.FAILED,
            PipelineStage.ERROR,
            error=str(e),
        )
    finally:
        # Always close stream — prevents client hanging
        job_manager.close_stream(job_id)
        logger.info(
            "Pipeline runner | job=%s | stream closed", job_id
        )