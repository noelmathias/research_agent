from typing import Any, Dict, Optional
from backend.pipelines.retrieval_pipeline import search_documents
from backend.mcp.schemas import ToolCallResult, ToolName


async def run_vector_search(
    query: str,
    top_k: int = 5,
    collection_name: Optional[str] = None,
) -> ToolCallResult:
    """
    MCP vector search tool.
    Embeds query and retrieves semantically similar chunks from ChromaDB.
    """
    try:
        results = await search_documents(
            query=query,
            top_k=top_k,
            collection_name=collection_name,
        )

        serialised = [
            {
                "id": r.id,
                "text": r.text,
                "score": r.score,
                "metadata": r.metadata,
            }
            for r in results.results
        ]

        return ToolCallResult(
            tool_name=ToolName.VECTOR_SEARCH,
            success=True,
            result=serialised,
            metadata={
                "query": query,
                "top_k": top_k,
                "results_returned": len(serialised),
                "collection": collection_name or "research_docs",
            },
        )

    except Exception as e:
        return ToolCallResult(
            tool_name=ToolName.VECTOR_SEARCH,
            success=False,
            result=[],
            error=str(e),
        )