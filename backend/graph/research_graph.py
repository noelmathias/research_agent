from langgraph.graph import StateGraph, END
from backend.graph.state import ResearchState
from backend.agents.planner_agent import run_planner
from backend.agents.retriever_agent import run_retriever
from backend.agents.summarizer_agent import run_summarizer
from backend.agents.evaluator_agent import run_evaluator
from backend.agents.reflection_agent import run_reflection
from backend.agents.report_agent import run_report
from shared.config import get_settings

settings = get_settings()


def _should_reflect(state: ResearchState) -> str:
    """
    Conditional edge router after evaluator.

    Returns:
        "reflect"  — failed AND retries remaining
        "report"   — passed OR retry limit reached
    """
    passed = state.get("evaluation_passed", True)
    retry_count = state.get("retry_count", 0)
    max_retries = settings.max_retries

    if not passed and retry_count < max_retries:
        return "reflect"
    return "report"


def build_research_graph() -> StateGraph:
    """
    Phase 8 topology:

        planner → retriever → summarizer → evaluator
                                               ↓
                                  ┌── reflect (failed + retries left)
                                  │       ↓
                                  │   summarizer
                                  │       ↓
                                  │   evaluator
                                  │       ↓ (loop until pass or limit)
                                  └──> report → END
    """
    graph = StateGraph(ResearchState)

    # ── Nodes ──────────────────────────────────────────────────────────────
    graph.add_node("planner", run_planner)
    graph.add_node("retriever", run_retriever)
    graph.add_node("summarizer", run_summarizer)
    graph.add_node("evaluator", run_evaluator)
    graph.add_node("reflection", run_reflection)
    graph.add_node("report", run_report)

    # ── Linear edges ───────────────────────────────────────────────────────
    graph.set_entry_point("planner")
    graph.add_edge("planner", "retriever")
    graph.add_edge("retriever", "summarizer")
    graph.add_edge("summarizer", "evaluator")

    # ── Conditional edge — reflect or report ───────────────────────────────
    graph.add_conditional_edges(
        "evaluator",
        _should_reflect,
        {
            "reflect": "reflection",
            "report": "report",
        },
    )

    # ── Reflection loops back to summarizer ────────────────────────────────
    graph.add_edge("reflection", "summarizer")

    # ── Report always exits ────────────────────────────────────────────────
    graph.add_edge("report", END)

    return graph.compile()


research_graph = build_research_graph()