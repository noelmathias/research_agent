import json
import re
from typing import Any
from backend.core.ollama_client import OllamaClient
from backend.graph.state import ResearchState

ollama = OllamaClient()

from backend.core.logger import get_logger
logger = get_logger(__name__)

PLANNER_PROMPT_TEMPLATE = """You are a research planning assistant.

Given a research query, decompose it into a structured list of research subtasks.
Each subtask should be a concrete, actionable step a researcher would take.

Query: {query}

Respond ONLY with a valid JSON array. Each item must have exactly these keys:
- "step": integer (1-based index)
- "task": string (what to do)
- "reason": string (why this step matters)

Example format:
[
  {{"step": 1, "task": "Search for foundational papers on transformers", "reason": "Establishes baseline knowledge"}},
  {{"step": 2, "task": "Retrieve recent benchmarks comparing transformer variants", "reason": "Provides current performance data"}}
]

Return ONLY the JSON array. No preamble. No explanation. No markdown.
"""


def _extract_json_array(text: str) -> list:
    """Extract JSON array from model output robustly."""
    # Strip markdown code fences if present
    text = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()

    # Find first [ and last ]
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON array found in response:\n{text}")

    return json.loads(text[start : end + 1])


async def run_planner(state: ResearchState) -> ResearchState:
    """
    Planner node: takes query, produces structured research plan.
    Mutates and returns state.
    """
    query = state["query"]
    model = state.get("model", "llama3")

    logger.info("Planner | query_len=%d | model=%s", len(query), model)

    prompt = PLANNER_PROMPT_TEMPLATE.format(query=query)

    try:
        raw_output = await ollama.generate(prompt=prompt, model=model)
        plan_list = _extract_json_array(raw_output)

        # Validate structure — ensure required keys exist
        validated = []
        for item in plan_list:
            validated.append(
                {
                    "step": int(item.get("step", len(validated) + 1)),
                    "task": str(item.get("task", "")),
                    "reason": str(item.get("reason", "")),
                }
            )

        state["raw_plan"] = raw_output
        state["plan"] = validated
        state["current_step"] = "planner_complete"
        state["error"] = None

        logger.info("Planner | steps=%d", len(validated))

    except Exception as e:

        logger.error("Planner failed: %s", e)

        state["raw_plan"] = ""
        state["plan"] = []
        state["error"] = f"Planner failed: {str(e)}"
        state["current_step"] = "planner_error"

    

    return state
