from fastapi import APIRouter, HTTPException
from backend.mcp.server import mcp_server
from backend.mcp.router import mcp_router
from backend.mcp.schemas import ToolCallRequest, ToolCallResult, ToolName
from typing import List

router = APIRouter()


@router.get("/mcp/tools", summary="List all registered MCP tools")
async def list_tools():
    """Returns the full tool registry with definitions and parameters."""
    tools = mcp_server.list_tools()
    return {
        "total": len(tools),
        "tools": [t.model_dump() for t in tools],
    }


@router.get("/mcp/tools/{tool_name}", summary="Get a specific tool definition")
async def get_tool(tool_name: str):
    """Returns definition for a single tool by name."""
    try:
        name = ToolName(tool_name)
    except ValueError:
        raise HTTPException(
            status_code=404,
            detail=f"Tool '{tool_name}' not found. "
                   f"Available: {mcp_server.tool_names()}",
        )

    tool = mcp_server.get(name)
    if not tool:
        raise HTTPException(status_code=404, detail=f"Tool '{tool_name}' not registered.")

    return tool.model_dump()


@router.post("/mcp/invoke", response_model=ToolCallResult, summary="Invoke an MCP tool")
async def invoke_tool(request: ToolCallRequest):
    """
    Dispatches a tool call through the MCP router.
    Returns a standardised ToolCallResult envelope.
    """
    try:
        result = await mcp_router.route(request)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))