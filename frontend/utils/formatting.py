from datetime import datetime, timezone
from typing import Any, Dict, List


def score_badge(score: float) -> str:
    """Returns coloured emoji badge for confidence scores."""
    if score >= 0.7:
        return f"🟢 {score:.4f}"
    elif score >= 0.5:
        return f"🟡 {score:.4f}"
    return f"🔴 {score:.4f}"


def score_color_label(score: float) -> str:
    if score >= 0.7:
        return "High"
    elif score >= 0.5:
        return "Medium"
    return "Low"


def format_timestamp(value: float | int | str) -> str:
    try:
        if isinstance(value, str):
            try:
                parsed = datetime.fromisoformat(
                    value.replace("Z", "+00:00")
                )
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=timezone.utc)
                return parsed.astimezone(timezone.utc).strftime(
                    "%Y-%m-%d %H:%M UTC"
                )
            except ValueError:
                return "Unknown"
        return datetime.fromtimestamp(
            float(value), tz=timezone.utc
        ).strftime("%Y-%m-%d %H:%M UTC")
    except Exception:
        return "Unknown"


def format_kb(size_bytes: int) -> str:
    return f"{size_bytes / 1024:.1f} KB"


def truncate(text: str, max_len: int = 120) -> str:
    return text if len(text) <= max_len else text[:max_len] + "..."


def format_plan(plan: List[Dict[str, Any]]) -> str:
    lines = []
    for item in plan:
        lines.append(
            f"**Step {item['step']}:** {item['task']}\n"
            f"_{item['reason']}_"
        )
    return "\n\n".join(lines)


def pipeline_step_label(step: str) -> str:
    labels = {
        "init": "⏳ Initialising",
        "planner_complete": "✅ Plan ready",
        "planner_error": "❌ Planner failed",
        "retriever_complete": "✅ Retrieval done",
        "retriever_error": "❌ Retriever failed",
        "summarizer_complete": "✅ Summary ready",
        "summarizer_error": "❌ Summarizer failed",
        "evaluator_complete": "✅ Evaluation done",
        "evaluator_error_defaulted": "⚠️ Evaluator defaulted",
        "reflection_complete": "🔄 Reflection done",
        "reflection_error": "⚠️ Reflection failed",
        "report_complete": "✅ Report ready",
        "report_error": "⚠️ Report assembly failed",
        "unknown": "❓ Unknown",
    }
    return labels.get(step, f"📍 {step}")
