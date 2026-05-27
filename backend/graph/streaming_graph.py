from typing import Optional
from backend.graph.state import ResearchState
from backend.graph.research_graph import build_research_graph
from backend.jobs.job_manager import job_manager
from backend.jobs.job_models import (
    JobStatus,
    PipelineStage,
    SSEEvent,
)
from backend.core.logger import get_logger

logger = get_logger(__name__)

# Map LangGraph node names → (PipelineStage, human label, SSE event name)
_NODE_META = {
    "planner": (
        PipelineStage.PLANNING,
        "Planner Agent",
        "node_start",
    ),
    "retriever": (
        PipelineStage.RETRIEVING,
        "Retriever Agent",
        "node_start",
    ),
    "summarizer": (
        PipelineStage.SUMMARIZING,
        "Summarizer Agent",
        "node_start",
    ),
    "evaluator": (
        PipelineStage.EVALUATING,
        "Evaluator Agent",
        "node_start",
    ),
    "reflection": (
        PipelineStage.REFLECTING,
        "Reflection Agent",
        "node_start",
    ),
    "report": (
        PipelineStage.GENERATING_REPORT,
        "Report Generator",
        "node_start",
    ),
}


def _emit(job_id: str, stage: PipelineStage, event: str, message: str, data: dict = {}) -> None:
    job_manager.emit(SSEEvent(
        event=event,
        job_id=job_id,
        stage=stage,
        message=message,
        data=data,
    ))


def _node_complete_data(node: str, state: ResearchState) -> dict:
    """Extract lightweight summary data from state after each node."""
    if node == "planner":
        return {"plan_steps": len(state.get("plan", []))}
    if node == "retriever":
        return {"chunks_retrieved": len(state.get("retrieved_chunks", []))}
    if node == "summarizer":
        return {"summary_len": len(state.get("summary", ""))}
    if node == "evaluator":
        return {
            "confidence_score": round(state.get("confidence_score", 0.0), 4),
            "passed": state.get("evaluation_passed", False),
            "flags": len(state.get("hallucination_flags", [])),
        }
    if node == "reflection":
        return {
            "retry_count": state.get("retry_count", 0),
            "reflection_notes": state.get("reflection_notes", ""),
        }
    if node == "report":
        return {
            "report_id": state.get("report_id", ""),
            "citations": len(state.get("citations", [])),
        }
    return {}


async def run_streaming_pipeline(
    job_id: str,
    query: str,
    model: str,
    collection_name: Optional[str] = None,
) -> ResearchState:
    """
    Executes the research graph with per-node SSE event emission.

    Uses LangGraph's astream_events API to intercept node
    entry and exit without modifying any agent logic.

    Returns final ResearchState for job result storage.
    """
    from backend.services.research_service import _build_initial_state

    initial_state = _build_initial_state(query, model)
    if collection_name:
        initial_state["tool_calls"] = [
            {"preferred_collection": collection_name}
        ]

    graph = build_research_graph()

    # ── Emit: job started ─────────────────────────────────────────────────
    job_manager.update_job_status(
        job_id, JobStatus.RUNNING, PipelineStage.PLANNING
    )
    _emit(
        job_id,
        PipelineStage.PLANNING,
        "pipeline_start",
        "Research pipeline started.",
        {"query": query, "model": model},
    )

    final_state: ResearchState = initial_state
    active_node: Optional[str] = None

    try:
        # astream_events yields granular LangGraph lifecycle events
        async for event in graph.astream_events(
            initial_state, version="v2"
        ):
            kind = event.get("event", "")
            name = event.get("name", "")

            # ── Node entered ──────────────────────────────────────────────
            if kind == "on_chain_start" and name in _NODE_META:
                stage, label, _ = _NODE_META[name]
                active_node = name
                job_manager.update_job_status(
                    job_id, JobStatus.RUNNING, stage
                )
                _emit(
                    job_id,
                    stage,
                    "node_start",
                    f"{label} started.",
                    {"node": name},
                )
                logger.info(
                    "Streaming | job=%s | node=%s | entered", job_id, name
                )

            # ── Node completed ────────────────────────────────────────────
            elif kind == "on_chain_end" and name in _NODE_META:
                stage, label, _ = _NODE_META[name]
                output = event.get("data", {}).get("output", {})

                # Merge node output into running final_state
                if isinstance(output, dict):
                    final_state = {**final_state, **output}

                summary_data = _node_complete_data(name, final_state)
                _emit(
                    job_id,
                    stage,
                    "node_complete",
                    f"{label} complete.",
                    {"node": name, **summary_data},
                )
                logger.info(
                    "Streaming | job=%s | node=%s | complete", job_id, name
                )

    except Exception as e:
        logger.error("Streaming pipeline error | job=%s | %s", job_id, e)
        _emit(
            job_id,
            PipelineStage.ERROR,
            "pipeline_error",
            f"Pipeline failed: {str(e)}",
            {"error": str(e)},
        )
        job_manager.update_job_status(
            job_id,
            JobStatus.FAILED,
            PipelineStage.ERROR,
            error=str(e),
        )
        raise

    # ── Emit: pipeline complete ───────────────────────────────────────────
    report_id = final_state.get("report_id", "")
    confidence = round(final_state.get("confidence_score", 0.0), 4)
    passed = final_state.get("evaluation_passed", False)

    _emit(
        job_id,
        PipelineStage.DONE,
        "pipeline_complete",
        "Research pipeline complete.",
        {
            "report_id": report_id,
            "confidence_score": confidence,
            "evaluation_passed": passed,
            "retry_count": final_state.get("retry_count", 0),
        },
    )

    return final_state