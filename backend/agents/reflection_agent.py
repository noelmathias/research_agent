from backend.graph.state import ResearchState
from backend.core.ollama_client import OllamaClient
from shared.config import get_settings
from typing import List
from backend.core.logger import get_logger
logger = get_logger(__name__)


ollama = OllamaClient()
settings = get_settings()

# Maximum characters of retrieved context in reflection prompt
MAX_CONTEXT_CHARS = 5000

REFLECTION_PROMPT_TEMPLATE = """You are a research summarizer performing a self-correction pass.

Your previous summary was evaluated and did not meet the quality threshold.

Original Query:
{query}

Evaluation Feedback:
- Confidence Score: {confidence_score}
- Evaluation Reasoning: {reasoning}
- Flagged Issues:
{flags}

Retrieved Source Context:
{context}

Previous Summary (to improve upon):
{previous_summary}

Instructions:
- Address every flagged issue explicitly
- Remove or correct any unsupported claims
- Ground every statement in the retrieved context
- Maintain clear structure with sections
- End with a revised "Key Findings" section

Write the improved summary now:
"""


def _format_flags(flags: List[str]) -> str:
    if not flags:
        return "  - No specific flags raised."
    return "\n".join(f"  - {flag}" for flag in flags)


def _build_context(chunks: List[str]) -> str:
    joined = "\n\n---\n\n".join(
        f"[Chunk {i + 1}]\n{chunk.strip()}"
        for i, chunk in enumerate(chunks)
    )
    return joined[:MAX_CONTEXT_CHARS]


async def run_reflection(state: ResearchState) -> ResearchState:
    """
    Reflection Agent node.

    Triggered only when evaluator returns evaluation_passed = False
    AND retry_count < max_retries.

    Builds a corrective prompt that:
      1. Presents the evaluator's feedback
      2. Lists hallucination flags explicitly
      3. Provides the full retrieved context
      4. Requests a revised summary

    Increments retry_count.
    Overwrites state['summary'] with the improved version.
    Graph then routes back to evaluator for re-scoring.
    """
    query = state["query"]
    model = state.get("model", "llama3")
    previous_summary = state.get("summary", "")
    chunks = state.get("retrieved_chunks", [])
    confidence = state.get("confidence_score", 0.0)
    reasoning = state.get("evaluation_reasoning", "")
    flags = state.get("hallucination_flags", [])
    retry_count = state.get("retry_count", 0)

    context = _build_context(chunks) if chunks else (
        "No source documents were retrieved. "
        "Base the summary on the query alone and be explicit about this."
    )

    prompt = REFLECTION_PROMPT_TEMPLATE.format(
        query=query,
        confidence_score=confidence,
        reasoning=reasoning,
        flags=_format_flags(flags),
        context=context,
        previous_summary=previous_summary[:2000],
    )

    try:
        logger.info(
    "Reflection | retry=%d | flags=%d | model=%s",
    retry_count + 1, len(flags), model,
)
        improved_summary = await ollama.generate(prompt=prompt, model=model)

        state["summary"] = improved_summary.strip()
        state["retry_count"] = retry_count + 1
        state["reflection_notes"] = (
            f"Retry {retry_count + 1}: addressed flags — {', '.join(flags[:3])}"
            if flags else f"Retry {retry_count + 1}: general quality improvement"
        )
        state["current_step"] = "reflection_complete"
        state["error"] = None
        logger.info("Reflection | improved_summary_len=%d", len(improved_summary))

    except Exception as e:
        logger.error("Reflection failed: %s", e)
        # Keep existing summary, increment retry to prevent infinite loop
        state["retry_count"] = retry_count + 1
        state["reflection_notes"] = f"Reflection failed: {str(e)}"
        state["current_step"] = "reflection_error"
        state["error"] = f"Reflection failed: {str(e)}"

    return state