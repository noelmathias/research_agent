from typing import Any, Dict
from backend.mcp.schemas import ToolCallResult, ToolName


async def run_web_search(query: str, max_results: int = 5) -> ToolCallResult:
    """
    MCP web search tool — stub implementation.

    Returns structured placeholder results so the agent pipeline
    can be fully exercised end-to-end before a live search API
    is wired in Phase 9.

    Replace the stub body with a real provider
    (SerpAPI, Tavily, DuckDuckGo) without changing the signature.
    """
    stub_results = [
        {
            "title": f"Search result {i + 1} for: {query}",
            "url": f"https://example.com/result-{i + 1}",
            "snippet": (
                f"This is a placeholder result {i + 1} for query '{query}'. "
                "Replace this stub with a live search provider in Phase 9."
            ),
        }
        for i in range(max_results)
    ]

    return ToolCallResult(
        tool_name=ToolName.WEB_SEARCH,
        success=True,
        result=stub_results,
        metadata={
            "query": query,
            "max_results": max_results,
            "provider": "stub",
            "live": False,
        },
    )