import pytest
from unittest.mock import AsyncMock, patch
from backend.mcp.server import MCPServer
from backend.mcp.router import MCPRouter
from backend.mcp.schemas import ToolName, ToolCallRequest


# ── Registry ──────────────────────────────────────────────────────────────────

def test_mcp_server_registers_three_tools():
    server = MCPServer()
    tools = server.list_tools()
    assert len(tools) == 3


def test_mcp_server_tool_names():
    server = MCPServer()
    names = server.tool_names()
    assert "search_vector" in names
    assert "parse_pdf" in names
    assert "web_search" in names


def test_mcp_server_get_existing_tool():
    server = MCPServer()
    tool = server.get(ToolName.VECTOR_SEARCH)
    assert tool is not None
    assert tool.name == ToolName.VECTOR_SEARCH


def test_mcp_server_get_missing_tool():
    server = MCPServer()
    result = server.get("nonexistent_tool")  # type: ignore
    assert result is None


# ── Router ────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_router_dispatches_vector_search():
    router = MCPRouter()
    mock_result = AsyncMock()
    mock_result.success = True
    mock_result.result = []

    with patch(
        "backend.mcp.router.run_vector_search",
        new_callable=AsyncMock,
        return_value=mock_result,
    ):
        request = ToolCallRequest(
            tool_name=ToolName.VECTOR_SEARCH,
            parameters={"query": "test", "top_k": 3},
        )
        result = await router.route(request)

    assert result.success is True


@pytest.mark.asyncio
async def test_router_dispatches_web_search():
    router = MCPRouter()

    request = ToolCallRequest(
        tool_name=ToolName.WEB_SEARCH,
        parameters={"query": "LLM benchmarks", "max_results": 3},
    )
    result = await router.route(request)

    assert result.success is True
    assert result.tool_name == ToolName.WEB_SEARCH
    assert len(result.result) == 3


@pytest.mark.asyncio
async def test_router_returns_error_for_unknown_tool():
    """Router returns failure result for unregistered tool name."""
    router = MCPRouter()

    # Bypass enum validation to test router error handling directly
    request = ToolCallRequest(
        tool_name=ToolName.WEB_SEARCH,
        parameters={"query": "test"},
    )
    request.tool_name = "unknown_tool"  # type: ignore

    result = await router.route(request)
    assert result.success is False
    assert result.error is not None


# ── Web search stub ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_web_search_stub_returns_results():
    from backend.mcp.tools.web_tool import run_web_search
    result = await run_web_search(query="test query", max_results=5)

    assert result.success is True
    assert len(result.result) == 5
    assert result.metadata["provider"] == "stub"
    assert result.metadata["live"] is False


@pytest.mark.asyncio
async def test_web_search_stub_result_structure():
    from backend.mcp.tools.web_tool import run_web_search
    result = await run_web_search(query="AI research", max_results=2)

    for item in result.result:
        assert "title" in item
        assert "url" in item
        assert "snippet" in item