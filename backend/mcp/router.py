from typing import Any, Dict, Optional
from backend.mcp.schemas import ToolCallRequest, ToolCallResult, ToolName
from backend.mcp.tools.vector_tool import run_vector_search
from backend.mcp.tools.pdf_tool import run_pdf_parse
from backend.mcp.tools.web_tool import run_web_search


class MCPRouter:
    """
    Dispatches ToolCallRequests to the correct tool handler.

    Agents call route() with a ToolCallRequest.
    The router resolves the tool name to a handler and returns
    a standardised ToolCallResult.

    Adding a new tool = add one entry to _DISPATCH_TABLE.
    No other changes needed.
    """

    async def route(self, request: ToolCallRequest) -> ToolCallResult:
        """Main dispatch entry point. Called by agents."""
        handler = self._DISPATCH_TABLE.get(request.tool_name)

        if handler is None:
            return ToolCallResult(
                tool_name=request.tool_name,
                success=False,
                result=None,
                error=f"Unknown tool: '{request.tool_name}'. "
                      f"Registered tools: {list(self._DISPATCH_TABLE.keys())}",
            )

        return await handler(request)

    # ── Dispatch table ────────────────────────────────────────────────────────
    # Maps ToolName → private async handler method.
    # Each handler unpacks parameters and delegates to the tool module.

    async def _handle_vector_search(
        self, request: ToolCallRequest
    ) -> ToolCallResult:
        params = request.parameters
        return await run_vector_search(
            query=params["query"],
            top_k=int(params.get("top_k", 5)),
            collection_name=request.collection_name
            or params.get("collection_name"),
        )

    async def _handle_pdf_parse(
        self, request: ToolCallRequest
    ) -> ToolCallResult:
        params = request.parameters
        return await run_pdf_parse(
            filename=params["filename"],
            collection_name=request.collection_name,
        )

    async def _handle_web_search(
        self, request: ToolCallRequest
    ) -> ToolCallResult:
        params = request.parameters
        return await run_web_search(
            query=params["query"],
            max_results=int(params.get("max_results", 5)),
        )

    @property
    def _DISPATCH_TABLE(self):
        return {
            ToolName.VECTOR_SEARCH: self._handle_vector_search,
            ToolName.PDF_PARSE: self._handle_pdf_parse,
            ToolName.WEB_SEARCH: self._handle_web_search,
        }


# Module-level singleton
mcp_router = MCPRouter()