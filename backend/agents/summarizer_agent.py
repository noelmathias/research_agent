from backend.graph.state import ResearchState
from backend.core.ollama_client import OllamaClient

ollama = OllamaClient()

from backend.core.logger import get_logger
logger = get_logger(__name__)

# Maximum characters of retrieved context passed to the LLM.
# Keeps prompt size manageable for local models.
MAX_CONTEXT_CHARS = 6000

SUMMARIZER_PROMPT_TEMPLATE = """You are a technical research summarizer.

Your task is to synthesize the retrieved context below into a clear,
well-structured research summary that directly answers the research query.

Research Query:
{query}

Research Plan:
{plan}

Retrieved Context:
{context}

Instructions:
- Write a structured summary with clear sections
- Ground every claim in the retrieved context
- If the context is insufficient, state what is missing
- Use technical language appropriate for the domain
- End with a "Key Findings" section listing 3-5 bullet points

Write the summary now:
"""

NO_CONTEXT_PROMPT_TEMPLATE = """You are a technical research summarizer.

No relevant documents were found in the knowledge base for the query below.
Provide a concise general overview based on your training knowledge.
Clearly note at the start that no source documents were available.

Research Query:
{query}

Write the summary now:
"""


def _format_plan(plan: list) -> str:
    if not plan:
        return "No plan available."
    return "\n".join(
        f"  Step {item['step']}: {item['task']}" for item in plan
    )


def _build_context(chunks: list[str]) -> str:
    """
    Joins retrieved chunks into a single context block.
    Truncates to MAX_CONTEXT_CHARS to stay within local LLM limits.
    """
    if not chunks:
        return ""

    joined = "\n\n---\n\n".join(
        f"[Chunk {i + 1}]\n{chunk.strip()}"
        for i, chunk in enumerate(chunks)
    )
    return joined[:MAX_CONTEXT_CHARS]


async def run_summarizer(state: ResearchState) -> ResearchState:
    """
    Summarizer Agent node.

    Reads retrieved_chunks and plan from state.
    Constructs a grounded summarization prompt.
    Calls Ollama to generate the research summary.
    Populates state['summary'].
    """
    query = state["query"]
    model = state.get("model", "mistral")
    plan = state.get("plan", [])
    chunks = state.get("retrieved_chunks", [])

    try:
        if chunks:
            context = _build_context(chunks)
            prompt = SUMMARIZER_PROMPT_TEMPLATE.format(
                query=query,
                plan=_format_plan(plan),
                context=context,
            )
        else:
            prompt = NO_CONTEXT_PROMPT_TEMPLATE.format(query=query)

        logger.info(
    "Summarizer | chunks=%d | has_context=%s | model=%s",
    len(chunks), bool(chunks), model,
)
        summary = await ollama.generate(prompt=prompt, model=model)

        state["summary"] = summary.strip()
        state["current_step"] = "summarizer_complete"
        state["error"] = None

        logger.info("Summarizer | summary_len=%d", len(summary))

    except Exception as e:

        logger.error("Summarizer failed: %s", e)       
        state["summary"] = ""
        state["error"] = f"Summarizer failed: {str(e)}"
        state["current_step"] = "summarizer_error"

    return state

