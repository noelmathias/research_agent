import pytest
import asyncio
from typing import Generator
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.fixture(scope="session")
def event_loop():
    """Single event loop for all async tests in session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def test_client() -> Generator:
    """
    FastAPI TestClient. Uses real app with mocked Ollama.
    Ollama is patched so tests run without a live Ollama server.
    """
    with patch(
        "backend.core.ollama_client.OllamaClient.generate",
        new_callable=AsyncMock,
        return_value='[{"step": 1, "task": "Test task", "reason": "Test reason"}]',
    ), patch(
        "backend.core.ollama_client.OllamaClient.is_reachable",
        new_callable=AsyncMock,
        return_value=True,
    ):
        from backend.main import app
        with TestClient(app) as client:
            yield client


@pytest.fixture
def sample_texts():
    return [
        "Retrieval-Augmented Generation combines a retrieval mechanism "
        "with a generative language model.",
        "The retriever fetches relevant documents from a vector store "
        "based on the input query.",
        "The generator uses the retrieved documents as context to produce "
        "a grounded response.",
    ]


@pytest.fixture
def sample_metadatas():
    return [
        {"source": "test_doc_1.txt", "type": "text"},
        {"source": "test_doc_1.txt", "type": "text"},
        {"source": "test_doc_2.txt", "type": "text"},
    ]


@pytest.fixture
def sample_query():
    return "What is retrieval-augmented generation?"


@pytest.fixture
def minimal_state():
    """Returns a fully initialised ResearchState for agent unit tests."""
    from backend.graph.state import ResearchState
    return ResearchState(
        query="What is RAG?",
        model="llama3",
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