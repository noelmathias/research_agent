from backend.graph.state import ResearchState
from backend.mcp.router import mcp_router
from backend.mcp.schemas import ToolCallRequest, ToolName
from backend.core.logger import get_logger
logger = get_logger(__name__)

async def run_retriever(state: ResearchState) -> ResearchState:
    """
    Retriever Agent node.

    Reads the research plan produced by the Planner Agent.
    Constructs a consolidated retrieval query from all plan tasks.
    Invokes the MCP search_vector tool to fetch relevant chunks.
    Populates state['retrieved_chunks'], state['retrieved_metadatas'],
    and state['tool_results'].
    """
    query = state["query"]
    plan = state.get("plan", [])

    if plan:
        task_summary = " ".join(item["task"] for item in plan)
        retrieval_query = f"{query} {task_summary}"
    else:
        retrieval_query = query

    retrieval_query = retrieval_query[:512]

    logger.info("Retriever | query_len=%d", len(retrieval_query))

    try:
        request = ToolCallRequest(
            tool_name=ToolName.VECTOR_SEARCH,
            parameters={
                "query": retrieval_query,
                "top_k": 4,
            },
        )

        tool_result = await mcp_router.route(request)

        state["tool_calls"] = state.get("tool_calls", []) + [
            {
                "tool": ToolName.VECTOR_SEARCH.value,
                "query": retrieval_query,
            }
        ]

        if tool_result.success and tool_result.result:
            chunks = [
                item["text"]
                for item in tool_result.result
                if item.get("text", "").strip()
            ]
            metadatas = [
                item.get("metadata", {})
                for item in tool_result.result
                if item.get("text", "").strip()
            ]
            state["retrieved_chunks"] = chunks
            state["retrieved_metadatas"] = metadatas
            state["tool_results"] = state.get("tool_results", []) + [
                {
                    "tool": ToolName.VECTOR_SEARCH.value,
                    "chunks_retrieved": len(chunks),
                    "success": True,
                }
            ]

            logger.info("Retriever | chunks_retrieved=%d", len(chunks))

        else:
            state["retrieved_chunks"] = []
            state["retrieved_metadatas"] = []
            state["tool_results"] = state.get("tool_results", []) + [
                {
                    "tool": ToolName.VECTOR_SEARCH.value,
                    "chunks_retrieved": 0,
                    "success": False,
                    "reason": tool_result.error or "empty result",
                }
            ]

        state["current_step"] = "retriever_complete"
        state["error"] = None

    except Exception as e:

        logger.error("Retriever failed: %s", e)
        state["retrieved_chunks"] = []
        state["retrieved_metadatas"] = []
        state["error"] = f"Retriever failed: {str(e)}"
        state["current_step"] = "retriever_error"

    return state