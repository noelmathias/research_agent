import json
import re
from backend.graph.state import ResearchState
from backend.core.ollama_client import OllamaClient
from shared.config import get_settings
from typing import List

from backend.core.logger import get_logger
logger = get_logger(__name__)

ollama = OllamaClient()
settings = get_settings()

EVALUATOR_PROMPT_TEMPLATE = """You are a research quality evaluator.

Assess the research summary below against the original query and
retrieved context. Your job is to detect hallucinations, unsupported
claims, and gaps in coverage.

Original Query:
{query}

Retrieved Context (source material):
{context}

Generated Summary:
{summary}

Evaluate the summary on these criteria:
1. Factual grounding — are claims supported by the retrieved context?
2. Completeness — does the summary address the query adequately?
3. Hallucination — are there specific claims with no basis in the context?
4. Coherence — is the summary logically structured?

Scoring guidance:
- 1.00: Near-perfect. Every meaningful claim is directly grounded in the retrieved context, with no extrapolation or filler.
- 0.85-0.95: Strong. Mostly grounded, but may contain minor abstraction or compression.
- 0.70-0.84: Mixed. Useful overall, but includes generic filler, weakly supported interpretation, or modest gaps.
- 0.50-0.69: Weak. Several unsupported claims, omissions, or vague generalizations.
- 0.00-0.49: Poor. Major hallucinations, lack of grounding, or failure to answer the query.

Important:
- Be skeptical. Do not award 1.0 unless the summary is exceptionally well-grounded.
- Generic academic filler such as "further research is needed" should be flagged unless explicitly supported by the retrieved context.
- If the summary goes beyond the retrieved context, reduce the score.

Respond ONLY with a valid JSON object. No preamble. No markdown fences.

Required keys:
{{
  "confidence_score": <float between 0.0 and 1.0>,
  "passed": <true or false>,
  "hallucination_flags": [<list of specific flagged claims as strings>],
  "reasoning": <one concise sentence explaining the score>
}}

Rules:
- confidence_score >= {threshold} means passed = true
- hallucination_flags must be empty list [] if no issues found
- reasoning must be a single sentence
- Return ONLY the JSON object
"""


def _extract_json_object(text: str) -> dict:
    """Robustly extract JSON object from LLM output."""
    text = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON object found in evaluator output:\n{text}")
    return json.loads(text[start: end + 1])


def _build_context_snippet(chunks: List[str], max_chars: int = 3000) -> str:
    """Truncated context for the evaluator prompt."""
    joined = "\n\n".join(
        f"[Chunk {i + 1}]: {c.strip()}"
        for i, c in enumerate(chunks)
    )
    return joined[:max_chars]


def _apply_evaluation_penalties(
    summary: str,
    context: str,
    confidence: float,
    flags: List[str],
) -> tuple[float, List[str]]:
    lowered_summary = summary.lower()
    lowered_context = context.lower()
    updated_flags = list(flags)
    penalty = 0.0

    heuristic_phrases = {
        "further research is needed": 0.12,
        "future research is needed": 0.12,
        "more research is needed": 0.10,
        "it can be concluded": 0.08,
        "this demonstrates that": 0.05,
    }

    for phrase, amount in heuristic_phrases.items():
        if phrase in lowered_summary and phrase not in lowered_context:
            penalty += amount
            updated_flags.append(
                f"Potential unsupported filler or extrapolation: '{phrase}'."
            )

    if confidence >= 0.99 and updated_flags:
        penalty += 0.08

    adjusted = max(0.0, min(1.0, confidence - penalty))
    deduped_flags = list(dict.fromkeys(updated_flags))
    return adjusted, deduped_flags


from typing import List


async def run_evaluator(state: ResearchState) -> ResearchState:
    """
    Evaluator Agent node.

    Reads the generated summary and retrieved chunks.
    Prompts Ollama to score the summary for:
      - confidence (0.0 – 1.0)
      - hallucination flags (list of flagged claims)
      - pass/fail against configured threshold
      - one-sentence reasoning

    Populates:
      state['confidence_score']
      state['hallucination_flags']
      state['evaluation_passed']
      state['evaluation_reasoning']
    """
    query = state["query"]
    model = state.get("model", "llama3")
    summary = state.get("summary", "")
    chunks = state.get("retrieved_chunks", [])
    threshold = settings.confidence_threshold

    # Cannot evaluate an empty summary
    if not summary.strip():
        state["confidence_score"] = 0.0
        state["hallucination_flags"] = ["Summary is empty — nothing to evaluate."]
        state["evaluation_passed"] = False
        state["evaluation_reasoning"] = "Empty summary produced by summarizer."
        state["current_step"] = "evaluator_complete"
        return state

    context_snippet = _build_context_snippet(chunks)

    # If no context was retrieved, evaluate summary standalone
    if not context_snippet:
        context_snippet = (
            "No source documents were retrieved. "
            "Evaluation is based on internal coherence only."
        )

    prompt = EVALUATOR_PROMPT_TEMPLATE.format(
        query=query,
        context=context_snippet,
        summary=summary[:3000],
        threshold=threshold,
    )

    try:
        raw_output = await ollama.generate(prompt=prompt, model=model)
        parsed = _extract_json_object(raw_output)

        confidence = float(parsed.get("confidence_score", 0.0))
        confidence = max(0.0, min(1.0, confidence))  # clamp to [0, 1]

        flags = parsed.get("hallucination_flags", [])
        if not isinstance(flags, list):
            flags = []
        reasoning = str(parsed.get("reasoning", "No reasoning provided."))
        confidence, flags = _apply_evaluation_penalties(
            summary=summary,
            context=context_snippet,
            confidence=confidence,
            flags=flags,
        )
        passed = confidence >= threshold and len(flags) == 0

        state["confidence_score"] = round(confidence, 4)
        state["hallucination_flags"] = flags
        state["evaluation_passed"] = passed
        state["evaluation_reasoning"] = reasoning
        state["current_step"] = "evaluator_complete"
        state["error"] = None

        logger.info(
    "Evaluator | confidence=%.4f | passed=%s | flags=%d",
    confidence, passed, len(flags),
)

    except Exception as e:
        logger.warning("Evaluator parse error — defaulting to cautious fail: %s", e)
        state["confidence_score"] = max(0.0, round(threshold - 0.05, 4))
        state["hallucination_flags"] = ["Evaluator could not verify grounding reliably."]
        state["evaluation_passed"] = False
        state["evaluation_reasoning"] = (
            f"Evaluator parse error — defaulting to cautious fail: {str(e)}"
        )
        state["current_step"] = "evaluator_error_defaulted"

    return state
