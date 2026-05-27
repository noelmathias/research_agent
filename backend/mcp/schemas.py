from pydantic import BaseModel
from typing import Any, Dict, List, Optional
from enum import Enum


class ToolName(str, Enum):
    """
    Canonical tool names used across the MCP layer.
    Agents reference these constants — never raw strings.
    """
    VECTOR_SEARCH = "search_vector"
    PDF_PARSE = "parse_pdf"
    WEB_SEARCH = "web_search"


class ToolParameter(BaseModel):
    name: str
    type: str
    description: str
    required: bool = True


class ToolDefinition(BaseModel):
    """
    Describes a registered MCP tool.
    Returned by the registry for agent introspection.
    """
    name: ToolName
    description: str
    parameters: List[ToolParameter]
    version: str = "1.0.0"


class ToolCallRequest(BaseModel):
    """Payload to invoke a tool via the MCP router."""
    tool_name: ToolName
    parameters: Dict[str, Any]
    collection_name: Optional[str] = None


class ToolCallResult(BaseModel):
    """Standardised result envelope returned by every tool."""
    tool_name: ToolName
    success: bool
    result: Any
    error: Optional[str] = None
    metadata: Dict[str, Any] = {}