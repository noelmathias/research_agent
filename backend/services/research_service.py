from backend.graph.research_graph import research_graph
from backend.graph.state import ResearchState
from shared.schemas import (
    ResearchPlanResponse,
    ResearchRunResponse,
    ResearchSubtask,
    RetrievalInfo,
    EvaluationInfo,
    ReportInfo,
)
from shared.config import get_settings
from typing import Optional, Dict, Any

settings = get_settings()


def _build_initial_state(query: str, model: str) -> ResearchState:
    """Centralised state initialiser — all keys always present."""
    return ResearchState(
        query=query,
        model=model,
        raw_plan="",
        plan=[],
        tool_calls=[],
        tool_results=[],
        retrieved_chunks=[],
        retrieved_metadatas=[],
        summary="",
        confidence_score=0.0,
        hallucination_flags=[],
        evaluation_passed=False,
        evaluation_reasoning="",
        retry_count=0,
        reflection_notes="",
        final_report="",
        report_id="",
        citations=[],
        error=None,
        current_step="init",
    )


def _build_result_response(
    result: ResearchState,
    query: str,
    model: str,
) -> Dict[str, Any]:
    """
    Converts a final ResearchState into a JSON-serialisable dict.
    Used by both the sync run endpoint and the async job store.
    """
    subtasks = [
        {"step": item["step"], "task": item["task"], "reason": item["reason"]}
        for item in result.get("plan", [])
    ]
    return {
        "query": query,
        "model": model,
        "plan": subtasks,
        "retrieval": {
            "chunks_retrieved": len(result.get("retrieved_chunks", [])),
            "tool_calls": result.get("tool_calls", []),
            "tool_results": result.get("tool_results", []),
        },
        "summary": result.get("summary", ""),
        "evaluation": {
            "confidence_score": result.get("confidence_score", 0.0),
            "passed": result.get("evaluation_passed", False),
            "hallucination_flags": result.get("hallucination_flags", []),
            "reasoning": result.get("evaluation_reasoning", ""),
            "retry_count": result.get("retry_count", 0),
            "reflection_notes": result.get("reflection_notes", ""),
        },
        "report": {
            "report_id": result.get("report_id", ""),
            "citations": result.get("citations", []),
            "final_report": result.get("final_report", ""),
        },
        "current_step": result.get("current_step", "unknown"),
        "error": result.get("error"),
    }


async def generate_research_plan(
    query: str,
    model: Optional[str] = None,
) -> ResearchPlanResponse:
    resolved_model = model or settings.ollama_model
    initial_state = _build_initial_state(query, resolved_model)
    result: ResearchState = await research_graph.ainvoke(initial_state)

    if result.get("error") and not result.get("plan"):
        raise RuntimeError(result["error"])

    subtasks = [
        ResearchSubtask(
            step=item["step"],
            task=item["task"],
            reason=item["reason"],
        )
        for item in result["plan"]
    ]
    return ResearchPlanResponse(
        query=query,
        plan=subtasks,
        raw_plan=result["raw_plan"],
        model=resolved_model,
    )


async def run_research(
    query: str,
    model: Optional[str] = None,
    collection_name: Optional[str] = None,
) -> ResearchRunResponse:
    resolved_model = model or settings.ollama_model
    initial_state = _build_initial_state(query, resolved_model)

    if collection_name:
        initial_state["tool_calls"] = [
            {"preferred_collection": collection_name}
        ]

    result: ResearchState = await research_graph.ainvoke(initial_state)

    subtasks = [
        ResearchSubtask(
            step=item["step"],
            task=item["task"],
            reason=item["reason"],
        )
        for item in result.get("plan", [])
    ]

    return ResearchRunResponse(
        query=query,
        model=resolved_model,
        plan=subtasks,
        retrieval=RetrievalInfo(
            chunks_retrieved=len(result.get("retrieved_chunks", [])),
            tool_calls=result.get("tool_calls", []),
            tool_results=result.get("tool_results", []),
        ),
        summary=result.get("summary", ""),
        evaluation=EvaluationInfo(
            confidence_score=result.get("confidence_score", 0.0),
            passed=result.get("evaluation_passed", False),
            hallucination_flags=result.get("hallucination_flags", []),
            reasoning=result.get("evaluation_reasoning", ""),
            retry_count=result.get("retry_count", 0),
            reflection_notes=result.get("reflection_notes", ""),
        ),
        report=ReportInfo(
            report_id=result.get("report_id", ""),
            citations=result.get("citations", []),
            final_report=result.get("final_report", ""),
        ),
        current_step=result.get("current_step", "unknown"),
        error=result.get("error"),
    )