from typing import TypedDict, List, Optional, Any, Dict


class ResearchState(TypedDict):
    # Input
    query: str
    model: str

    # Planner outputs
    raw_plan: str
    plan: List[dict]

    # MCP tool tracking
    tool_calls: List[Dict[str, Any]]
    tool_results: List[Dict[str, Any]]

    # Retriever outputs
    retrieved_chunks: List[str]
    retrieved_metadatas: List[Dict[str, Any]]

    # Summarizer outputs
    summary: str

    # Evaluator outputs
    confidence_score: float
    hallucination_flags: List[str]
    evaluation_passed: bool
    evaluation_reasoning: str

    # Reflection / retry loop
    retry_count: int
    reflection_notes: str

    # Report outputs (Phase 8)
    final_report: str
    report_id: str
    citations: List[Dict[str, Any]]

    # Internal
    error: Optional[str]
    current_step: str