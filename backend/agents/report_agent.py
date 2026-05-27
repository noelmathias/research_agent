import uuid
from pathlib import Path
from backend.graph.state import ResearchState
from backend.pipelines.report_pipeline import build_citations, assemble_report
from shared.config import get_settings

settings = get_settings()

from backend.core.logger import get_logger
logger = get_logger(__name__)

async def run_report(state: ResearchState) -> ResearchState:
    """
    Report Agent node.

    Triggered after evaluator passes (or after max retries exhausted).
    Assembles the final structured markdown report by:
      1. Building deduplicated citations from retrieval metadata
      2. Calling assemble_report() to render full markdown
      3. Persisting the report to data/reports/<report_id>.md
      4. Populating state['final_report'], state['report_id'],
         state['citations']

    Does not call Ollama — pure assembly from existing state.
    The summary produced by the summarizer IS the executive summary.
    """
    query = state["query"]
    model = state.get("model", "llama3")
    plan = state.get("plan", [])
    summary = state.get("summary", "")
    retrieved_chunks = state.get("retrieved_chunks", [])
    retrieved_metadatas = state.get("retrieved_metadatas", [])

    evaluation = {
        "confidence_score": state.get("confidence_score", 0.0),
        "passed": state.get("evaluation_passed", False),
        "hallucination_flags": state.get("hallucination_flags", []),
        "reasoning": state.get("evaluation_reasoning", ""),
        "retry_count": state.get("retry_count", 0),
    }

    try:
        report_id = str(uuid.uuid4())[:8]

        # Build citations from retrieval metadata
        citations = build_citations(
            retrieved_metadatas=retrieved_metadatas,
            retrieved_chunks=retrieved_chunks,
        )

        # Assemble full markdown report
        final_report = assemble_report(
            query=query,
            plan=plan,
            summary=summary,
            citations=citations,
            evaluation=evaluation,
            model=model,
            report_id=report_id,
        )
        logger.info("Report | id=%s | citations=%d", report_id, len(citations))

        # Persist to disk
        reports_dir = Path(settings.reports_dir)
        reports_dir.mkdir(parents=True, exist_ok=True)
        report_path = reports_dir / f"{report_id}.md"
        report_path.write_text(final_report, encoding="utf-8")

        state["final_report"] = final_report
        state["report_id"] = report_id
        state["citations"] = citations
        state["current_step"] = "report_complete"
        state["error"] = None
        logger.info("Report | saved to %s", str(report_path))


    except Exception as e:
        logger.error("Report assembly failed: %s", e)
        state["final_report"] = summary  # fallback to raw summary
        state["report_id"] = ""
        state["citations"] = []
        state["current_step"] = "report_error"
        state["error"] = f"Report assembly failed: {str(e)}"

    return state