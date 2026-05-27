import pytest
from unittest.mock import AsyncMock, patch
from backend.graph.state import ResearchState


# ── Planner Agent ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_planner_returns_valid_plan(minimal_state):
    mock_response = (
        '[{"step": 1, "task": "Define RAG", "reason": "Foundation"}, '
        '{"step": 2, "task": "Survey benchmarks", "reason": "Evidence"}]'
    )
    with patch(
        "backend.agents.planner_agent.ollama.generate",
        new_callable=AsyncMock,
        return_value=mock_response,
    ):
        from backend.agents.planner_agent import run_planner
        result = await run_planner(minimal_state)

    assert result["current_step"] == "planner_complete"
    assert len(result["plan"]) == 2
    assert result["plan"][0]["task"] == "Define RAG"
    assert result["error"] is None


@pytest.mark.asyncio
async def test_planner_handles_malformed_json(minimal_state):
    """Planner sets error state on unparseable LLM output."""
    with patch(
        "backend.agents.planner_agent.ollama.generate",
        new_callable=AsyncMock,
        return_value="This is not JSON at all.",
    ):
        from backend.agents.planner_agent import run_planner
        result = await run_planner(minimal_state)

    assert result["current_step"] == "planner_error"
    assert result["error"] is not None
    assert result["plan"] == []


@pytest.mark.asyncio
async def test_planner_handles_json_with_fences(minimal_state):
    """Planner strips markdown fences before parsing."""
    fenced = (
        "```json\n"
        '[{"step": 1, "task": "Task A", "reason": "Reason A"}]\n'
        "```"
    )
    with patch(
        "backend.agents.planner_agent.ollama.generate",
        new_callable=AsyncMock,
        return_value=fenced,
    ):
        from backend.agents.planner_agent import run_planner
        result = await run_planner(minimal_state)

    assert result["current_step"] == "planner_complete"
    assert len(result["plan"]) == 1


# ── Retriever Agent ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_retriever_populates_chunks(minimal_state):
    minimal_state["plan"] = [
        {"step": 1, "task": "Retrieve RAG papers", "reason": "Evidence"}
    ]
    mock_result = AsyncMock()
    mock_result.success = True
    mock_result.result = [
        {"text": "RAG chunk 1", "score": 0.9, "metadata": {"source": "doc.pdf"}, "id": "id1"},
        {"text": "RAG chunk 2", "score": 0.7, "metadata": {"source": "doc.pdf"}, "id": "id2"},
    ]
    mock_result.error = None

    with patch(
        "backend.agents.retriever_agent.mcp_router.route",
        new_callable=AsyncMock,
        return_value=mock_result,
    ):
        from backend.agents.retriever_agent import run_retriever
        result = await run_retriever(minimal_state)

    assert result["current_step"] == "retriever_complete"
    assert len(result["retrieved_chunks"]) == 2
    assert len(result["retrieved_metadatas"]) == 2


@pytest.mark.asyncio
async def test_retriever_handles_empty_results(minimal_state):
    mock_result = AsyncMock()
    mock_result.success = True
    mock_result.result = []
    mock_result.error = None

    with patch(
        "backend.agents.retriever_agent.mcp_router.route",
        new_callable=AsyncMock,
        return_value=mock_result,
    ):
        from backend.agents.retriever_agent import run_retriever
        result = await run_retriever(minimal_state)

    assert result["retrieved_chunks"] == []
    assert result["current_step"] == "retriever_complete"


# ── Summarizer Agent ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_summarizer_with_chunks(minimal_state):
    minimal_state["retrieved_chunks"] = [
        "RAG uses a retriever and generator.",
        "The retriever fetches documents from a vector store.",
    ]
    minimal_state["plan"] = [
        {"step": 1, "task": "Define RAG", "reason": "Foundation"}
    ]

    with patch(
        "backend.agents.summarizer_agent.ollama.generate",
        new_callable=AsyncMock,
        return_value="## Summary\n\nRAG combines retrieval with generation.\n\n## Key Findings\n- Efficient retrieval",
    ):
        from backend.agents.summarizer_agent import run_summarizer
        result = await run_summarizer(minimal_state)

    assert result["current_step"] == "summarizer_complete"
    assert len(result["summary"]) > 0
    assert result["error"] is None


@pytest.mark.asyncio
async def test_summarizer_without_chunks(minimal_state):
    """Summarizer uses no-context prompt when chunks are empty."""
    minimal_state["retrieved_chunks"] = []

    with patch(
        "backend.agents.summarizer_agent.ollama.generate",
        new_callable=AsyncMock,
        return_value="No source documents. General overview: RAG is a technique...",
    ):
        from backend.agents.summarizer_agent import run_summarizer
        result = await run_summarizer(minimal_state)

    assert result["current_step"] == "summarizer_complete"
    assert result["summary"] != ""


# ── Evaluator Agent ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_evaluator_passing_score(minimal_state):
    minimal_state["summary"] = "RAG combines retrieval with generation effectively."
    minimal_state["retrieved_chunks"] = ["RAG uses retrieval."]

    mock_json = (
        '{"confidence_score": 0.82, "passed": true, '
        '"hallucination_flags": [], '
        '"reasoning": "Summary is well-grounded."}'
    )
    with patch(
        "backend.agents.evaluator_agent.ollama.generate",
        new_callable=AsyncMock,
        return_value=mock_json,
    ):
        from backend.agents.evaluator_agent import run_evaluator
        result = await run_evaluator(minimal_state)

    assert result["evaluation_passed"] is True
    assert result["confidence_score"] == pytest.approx(0.82)
    assert result["hallucination_flags"] == []


@pytest.mark.asyncio
async def test_evaluator_failing_score(minimal_state):
    minimal_state["summary"] = "Some ungrounded claim about fusion reactors."
    minimal_state["retrieved_chunks"] = ["RAG uses retrieval."]

    mock_json = (
        '{"confidence_score": 0.3, "passed": false, '
        '"hallucination_flags": ["Fusion reactor claim has no basis in context."], '
        '"reasoning": "Summary contains unsupported claims."}'
    )
    with patch(
        "backend.agents.evaluator_agent.ollama.generate",
        new_callable=AsyncMock,
        return_value=mock_json,
    ):
        from backend.agents.evaluator_agent import run_evaluator
        result = await run_evaluator(minimal_state)

    assert result["evaluation_passed"] is False
    assert result["confidence_score"] == pytest.approx(0.3)
    assert len(result["hallucination_flags"]) == 1


@pytest.mark.asyncio
async def test_evaluator_empty_summary_auto_fails(minimal_state):
    """Empty summary is caught before LLM call and auto-fails."""
    minimal_state["summary"] = ""

    from backend.agents.evaluator_agent import run_evaluator
    result = await run_evaluator(minimal_state)

    assert result["evaluation_passed"] is False
    assert result["confidence_score"] == 0.0


@pytest.mark.asyncio
async def test_evaluator_defaults_to_pass_on_parse_error(minimal_state):
    """Evaluator defaults to pass if JSON parse fails."""
    minimal_state["summary"] = "Valid summary text."

    with patch(
        "backend.agents.evaluator_agent.ollama.generate",
        new_callable=AsyncMock,
        return_value="This is not valid JSON output.",
    ):
        from backend.agents.evaluator_agent import run_evaluator
        result = await run_evaluator(minimal_state)

    assert result["evaluation_passed"] is True
    assert result["confidence_score"] == 0.5