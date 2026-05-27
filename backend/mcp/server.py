from typing import Dict, List
from backend.mcp.schemas import ToolDefinition, ToolName, ToolParameter
from typing import Optional

class MCPServer:
    """
    MCP tool registry.

    Maintains the canonical list of available tools with their
    definitions. Agents query this to understand what tools exist
    before deciding which to invoke via the router.

    Singleton — one registry per process.
    """

    _instance: Optional["MCPServer"] = None

    def __new__(cls) -> "MCPServer":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._registry: Dict[ToolName, ToolDefinition] = {}
            cls._instance._initialised = False
        return cls._instance

    def _register_defaults(self) -> None:
        """Register all built-in tools on first access."""

        self.register(ToolDefinition(
            name=ToolName.VECTOR_SEARCH,
            description=(
                "Performs semantic similarity search over the ChromaDB vector store. "
                "Use for retrieving relevant text chunks from ingested documents."
            ),
            parameters=[
                ToolParameter(
                    name="query",
                    type="string",
                    description="Natural language search query",
                    required=True,
                ),
                ToolParameter(
                    name="top_k",
                    type="integer",
                    description="Number of results to return (default 5)",
                    required=False,
                ),
                ToolParameter(
                    name="collection_name",
                    type="string",
                    description="Target ChromaDB collection (default: research_docs)",
                    required=False,
                ),
            ],
        ))

        self.register(ToolDefinition(
            name=ToolName.PDF_PARSE,
            description=(
                "Parses an uploaded PDF by filename and returns "
                "per-page extracted text with metadata."
            ),
            parameters=[
                ToolParameter(
                    name="filename",
                    type="string",
                    description="Name of the uploaded PDF file",
                    required=True,
                ),
            ],
        ))

        self.register(ToolDefinition(
            name=ToolName.WEB_SEARCH,
            description=(
                "Searches the web for current information. "
                "Returns title, URL, and snippet per result."
            ),
            parameters=[
                ToolParameter(
                    name="query",
                    type="string",
                    description="Web search query string",
                    required=True,
                ),
                ToolParameter(
                    name="max_results",
                    type="integer",
                    description="Maximum results to return (default 5)",
                    required=False,
                ),
            ],
        ))

    def register(self, tool: ToolDefinition) -> None:
        self._registry[tool.name] = tool

    def get(self, name: ToolName) -> ToolDefinition | None:
        return self._registry.get(name)

    def list_tools(self) -> List[ToolDefinition]:
        return list(self._registry.values())

    def tool_names(self) -> List[str]:
        return [t.value for t in self._registry.keys()]

    @property
    def registry(self) -> Dict[ToolName, ToolDefinition]:
        if not self._initialised:
            self._register_defaults()
            self._initialised = True
        return self._registry


# Module-level singleton
mcp_server = MCPServer()
# Trigger default registration at import time
_ = mcp_server.registry