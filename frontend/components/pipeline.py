import streamlit as st
import httpx
import json
import os
import time
from frontend.utils.api import run_research_pipeline
from frontend.utils.formatting import (
    score_badge,
    pipeline_step_label,
    format_plan,
    truncate,
)

_DEFAULT_BACKEND_URL = "http://localhost" + ":8000"
_BACKEND_URL = os.environ.get("BACKEND_URL", _DEFAULT_BACKEND_URL)
API_BASE = f"{_BACKEND_URL}/api/v1"

# ── Stage display config ──────────────────────────────────────────────────────
_STAGE_CONFIG = {
    "queued":            ("⏳", "Queued",             "grey"),
    "planning":          ("🧠", "Planning",           "blue"),
    "retrieving":        ("🔍", "Retrieving",         "blue"),
    "summarizing":       ("✍️", "Summarizing",        "blue"),
    "evaluating":        ("🔬", "Evaluating",         "blue"),
    "reflecting":        ("🔄", "Reflecting",         "orange"),
    "generating_report": ("📄", "Generating Report",  "blue"),
    "done":              ("✅", "Complete",            "green"),
    "error":             ("❌", "Error",               "red"),
}

_NODE_LABELS = {
    "planner":   "🧠 Planner Agent",
    "retriever": "🔍 Retriever Agent",
    "summarizer":"✍️  Summarizer Agent",
    "evaluator": "🔬 Evaluator Agent",
    "reflection":"🔄 Reflection Agent",
    "report":    "📄 Report Generator",
}


def _format_node_complete(d: dict) -> str:
    node = d.get("node", "")
    if node == "planner":
        return f"Plan ready — {d.get('plan_steps', 0)} steps"
    if node == "retriever":
        return f"Retrieved {d.get('chunks_retrieved', 0)} chunks"
    if node == "summarizer":
        return f"Summary generated — {d.get('summary_len', 0)} chars"
    if node == "evaluator":
        score = d.get("confidence_score", 0)
        passed = d.get("passed", False)
        flags = d.get("flags", 0)
        return (
            f"Score: `{score:.2f}` — "
            f"{'✅ Pass' if passed else '❌ Fail'} — "
            f"{flags} flag(s)"
        )
    if node == "reflection":
        return f"Retry {d.get('retry_count', 0)} complete"
    if node == "report":
        return (
            f"Report `{d.get('report_id', '')}` — "
            f"{d.get('citations', 0)} citation(s)"
        )
    return "Complete"


_EVENT_DETAIL = {
    "node_start":        lambda d: f"Starting `{d.get('node','')}`...",
    "node_complete":     _format_node_complete,
    "pipeline_start":    lambda d: f"Query received. Model: `{d.get('model','')}`",
    "pipeline_complete": lambda d: (
        f"Done — confidence: `{d.get('confidence_score', 0):.2f}` — "
        f"report: `{d.get('report_id','none')}`"
    ),
    "pipeline_error":    lambda d: f"❌ {d.get('error', 'Unknown error')}",
    "keepalive":         lambda d: "",
}



def _get_detail(event_type: str, data: dict) -> str:
    fn = _EVENT_DETAIL.get(event_type)
    if fn:
        try:
            return fn(data)
        except Exception:
            return ""
    return ""


def _submit_job(query: str, model: str, collection: str) -> dict | None:
    try:
        r = httpx.post(
            f"{API_BASE}/research/run_async",
            json={
                "query": query,
                "model": model,
                "collection_name": collection,
            },
            timeout=15,
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"Job submission failed: {e}")
        return None


def _poll_job(job_id: str) -> dict | None:
    try:
        r = httpx.get(
            f"{API_BASE}/research/job/{job_id}",
            timeout=10,
        )
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def _fetch_events(job_id: str) -> list[dict]:
    """
    Fetch all accumulated events from the job record.
    Used for final result rendering after stream closes.
    """
    job = _poll_job(job_id)
    if not job:
        return []
    return job.get("events", [])


def _render_event_card(event: dict, index: int) -> None:
    """Render a single timeline event as a compact card."""
    event_type = event.get("event", "")
    stage = event.get("stage", "")
    message = event.get("message", "")
    data = event.get("data", {})
    ts = event.get("timestamp", "")[:19].replace("T", " ")

    if event_type == "keepalive":
        return

    icon, label, _ = _STAGE_CONFIG.get(stage, ("📍", stage, "grey"))

    detail = _get_detail(event_type, data)

    with st.container():
        cols = st.columns([0.5, 3, 5, 2])
        cols[0].markdown(f"**{icon}**")
        cols[1].markdown(f"**{label}**")
        cols[2].markdown(detail or message)
        cols[3].caption(ts)

        # Expand node_complete events with data
        if event_type == "node_complete" and data:
            with st.expander("details", expanded=False):
                st.json(data)


def render_pipeline_tab(config: dict) -> None:
    model = config["model"]
    collection_name = config["collection_name"]

    st.subheader("🔬 Research Pipeline")
    st.caption(
        "Async execution with live progress — "
        "Planner → Retriever → Summarizer → Evaluator → Report"
    )

    # ── Query input ───────────────────────────────────────────────────────
    query = st.text_area(
        "Research query",
        placeholder=(
            "e.g. What are the key architectural differences between "
            "transformer models and state space models?"
        ),
        height=90,
        key="pipeline_query",
    )

    col1, col2 = st.columns([3, 1])
    with col2:
        run_collection = st.text_input(
            "Collection",
            value=collection_name,
            key="pipeline_collection",
            label_visibility="collapsed",
        )

    if st.button(
        "🚀 Start Research Pipeline",
        type="primary",
        use_container_width=True,
    ):
        if not query.strip():
            st.warning("Enter a research query.")
            return

        # ── Submit job ────────────────────────────────────────────────────
        with st.spinner("Submitting job..."):
            job_response = _submit_job(query, model, run_collection)

        if not job_response:
            return

        job_id = job_response["job_id"]
        st.session_state["active_job_id"] = job_id
        st.session_state["job_events"] = []
        st.session_state["job_complete"] = False

        st.success(
            f"Job submitted — ID: `{job_id}` — "
            f"Streaming live updates below."
        )

    # ── Live timeline ─────────────────────────────────────────────────────
    job_id = st.session_state.get("active_job_id")
    if not job_id:
        return

    st.divider()
    st.markdown(f"**Job:** `{job_id}`")

    # ── Stream events via polling ─────────────────────────────────────────
    # Streamlit doesn't support native EventSource, so we poll the
    # job status endpoint and re-render events on each Streamlit rerun.
    # This gives near-real-time updates without websockets.

    timeline_placeholder = st.empty()
    status_placeholder = st.empty()

    if not st.session_state.get("job_complete", False):
        job_data = _poll_job(job_id)

        if job_data:
            status = job_data.get("status", "pending")
            stage = job_data.get("stage", "queued")
            icon, label, _ = _STAGE_CONFIG.get(
                stage, ("⏳", stage, "grey")
            )

            status_placeholder.info(
                f"{icon} **{label}** — status: `{status}` — "
                f"events: `{job_data.get('event_count', 0)}`"
            )

            if status in ("completed", "failed"):
                st.session_state["job_complete"] = True
                st.session_state["job_final"] = job_data

        # Auto-rerun while job is running
        if not st.session_state.get("job_complete", False):
            time.sleep(2)
            st.rerun()

    # ── Render event timeline ─────────────────────────────────────────────
    job_data = _poll_job(job_id)
    events = []
    if job_data:
        events = job_data.get("events", [])

    with timeline_placeholder.container():
        st.markdown("#### ⏱️ Execution Timeline")
        if not events:
            st.caption("Waiting for first event...")
        else:
            for i, event in enumerate(events):
                _render_event_card(event, i)

    # ── Final result ──────────────────────────────────────────────────────
    if st.session_state.get("job_complete"):
        final_job = st.session_state.get("job_final", {})
        status = final_job.get("status", "unknown")

        st.divider()

        if status == "failed":
            st.error(
                f"Pipeline failed: {final_job.get('error', 'Unknown error')}"
            )
            return

        result = final_job.get("result", {})
        if not result:
            # Result may need one more poll after completion
            refreshed = _poll_job(job_id)
            if refreshed:
                result = refreshed.get("result", {})

        if not result:
            st.warning("Job complete but result not yet available. Refresh.")
            return

        # ── Metrics ───────────────────────────────────────────────────────
        ev = result.get("evaluation", {})
        rep = result.get("report", {})
        ret = result.get("retrieval", {})
        confidence = ev.get("confidence_score", 0.0)
        passed = ev.get("passed", False)
        retries = ev.get("retry_count", 0)
        report_id = rep.get("report_id", "")
        citations = rep.get("citations", [])

        if passed:
            st.success(
                f"✅ Pipeline complete — "
                f"confidence: `{confidence:.2f}` — "
                f"report: `{report_id}`"
            )
        else:
            st.warning(
                f"⚠️ Below threshold — confidence: `{confidence:.2f}`"
            )

        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Steps", len(result.get("plan", [])))
        m2.metric("Chunks", ret.get("chunks_retrieved", 0))
        m3.metric("Confidence", f"{confidence:.2f}")
        m4.metric("Quality", "✅ Pass" if passed else "❌ Fail")
        m5.metric("Retries", retries)
        m6.metric("Citations", len(citations))

        # ── Evaluation ────────────────────────────────────────────────────
        with st.expander("🔍 Evaluation Details", expanded=not passed):
            c1, c2 = st.columns([1, 2])
            with c1:
                st.markdown(f"**Score:** {score_badge(confidence)}")
                st.markdown(
                    f"**Verdict:** "
                    f"{'✅ Passed' if passed else '❌ Failed'}"
                )
                st.markdown(
                    f"**Reasoning:** {ev.get('reasoning', 'N/A')}"
                )
                if ev.get("reflection_notes"):
                    st.caption(f"Reflection: {ev['reflection_notes']}")
            with c2:
                flags = ev.get("hallucination_flags", [])
                if flags:
                    st.markdown("**⚠️ Hallucination Flags**")
                    for flag in flags:
                        st.warning(flag)
                else:
                    st.success("✅ No hallucination flags.")

        # ── Plan ──────────────────────────────────────────────────────────
        plan = result.get("plan", [])
        if plan:
            with st.expander(
                f"📋 Research Plan ({len(plan)} steps)", expanded=False
            ):
                for item in plan:
                    st.markdown(
                        f"**Step {item['step']}:** {item['task']}"
                    )
                    st.caption(item["reason"])

        # ── Citations ─────────────────────────────────────────────────────
        if citations:
            with st.expander(
                f"📚 Citations ({len(citations)} sources)", expanded=False
            ):
                for c in citations:
                    pages = c.get("pages", [])
                    page_str = (
                        f" · pp. {', '.join(str(p) for p in sorted(pages))}"
                        if pages else ""
                    )
                    st.markdown(
                        f"**[{c['citation_number']}]** "
                        f"`{c['source']}`{page_str}"
                    )
                    st.caption(truncate(c.get("excerpt", ""), 180))

        # ── Report ────────────────────────────────────────────────────────
        st.subheader("📄 Final Research Report")
        final_report = rep.get("final_report", "")
        if final_report:
            st.markdown(final_report)
            st.divider()
            st.download_button(
                label="⬇️ Download Report (.md)",
                data=final_report,
                file_name=f"report_{report_id}.md",
                mime="text/markdown",
                use_container_width=True,
            )
        else:
            summary = result.get("summary", "")
            if summary:
                st.markdown(summary)
            else:
                st.info("No report content available.")

        # ── Clear job ─────────────────────────────────────────────────────
        if st.button("🔁 Start New Research", use_container_width=False):
            for key in [
                "active_job_id", "job_events",
                "job_complete", "job_final",
            ]:
                st.session_state.pop(key, None)
            st.rerun()
