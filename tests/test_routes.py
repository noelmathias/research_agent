import pytest
from unittest.mock import AsyncMock, patch


def test_health_endpoint(test_client):
    response = test_client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "ollama_reachable" in data
    assert "model" in data


def test_root_endpoint(test_client):
    response = test_client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()


def test_research_plan_endpoint(test_client):
    with patch(
        "backend.services.research_service.research_graph.ainvoke",
        new_callable=AsyncMock,
        return_value={
            "query": "What is RAG?",
            "raw_plan": '[{"step":1,"task":"Define RAG","reason":"Foundation"}]',
            "plan": [{"step": 1, "task": "Define RAG", "reason": "Foundation"}],
            "tool_calls": [],
            "tool_results": [],
            "retrieved_chunks": [],
            "retrieved_metadatas": [],
            "summary": "",
            "confidence_score": 0.0,
            "hallucination_flags": [],
            "evaluation_passed": False,
            "evaluation_reasoning": "",
            "retry_count": 0,
            "reflection_notes": "",
            "final_report": "",
            "report_id": "",
            "citations": [],
            "error": None,
            "current_step": "planner_complete",
            "model": "llama3",
        },
    ):
        response = test_client.post(
            "/api/v1/research/plan",
            json={"query": "What is RAG?"},
        )

    assert response.status_code == 200
    data = response.json()
    assert "plan" in data
    assert isinstance(data["plan"], list)


def test_vector_count_endpoint(test_client):
    with patch(
        "backend.api.routes.vector.store.collection_count",
        return_value=42,
    ):
        response = test_client.get(
            "/api/v1/vector/count",
            params={"collection_name": "research_docs"},
        )
    assert response.status_code == 200
    assert response.json()["count"] == 42


def test_vector_search_empty_collection(test_client):
    with patch(
        "backend.api.routes.vector.store.collection_count",
        return_value=0,
    ):
        response = test_client.post(
            "/api/v1/vector/search",
            json={"query": "test query", "top_k": 5},
        )
    assert response.status_code == 200
    data = response.json()
    assert data["total_results"] == 0
    assert data["results"] == []


def test_vector_ingest_endpoint(test_client):
    with patch(
        "backend.api.routes.vector.ingest_texts",
        return_value={"chunks_stored": 3, "ids": ["a", "b", "c"]},
    ):
        response = test_client.post(
            "/api/v1/vector/ingest",
            json={"texts": ["text one", "text two"]},
        )
    assert response.status_code == 200
    data = response.json()
    assert data["chunks_stored"] == 3


def test_mcp_tools_list(test_client):
    response = test_client.get("/api/v1/mcp/tools")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 3
    tool_names = [t["name"] for t in data["tools"]]
    assert "search_vector" in tool_names
    assert "parse_pdf" in tool_names
    assert "web_search" in tool_names


def test_mcp_tool_not_found(test_client):
    response = test_client.get("/api/v1/mcp/tools/nonexistent_tool")
    assert response.status_code == 404


def test_report_list_empty(test_client):
    with patch(
        "backend.api.routes.report.list_reports",
        return_value=[],
    ):
        response = test_client.get("/api/v1/report/list")
    assert response.status_code == 200
    assert response.json()["total"] == 0


def test_report_not_found(test_client):
    with patch(
        "backend.api.routes.report.get_report_by_id",
        return_value=None,
    ):
        response = test_client.get("/api/v1/report/nonexistent123")
    assert response.status_code == 404