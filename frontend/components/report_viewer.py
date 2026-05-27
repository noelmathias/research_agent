import streamlit as st
from frontend.utils.api import list_reports, get_report, delete_report
from frontend.utils.formatting import format_timestamp, score_badge


def render_report_viewer_tab() -> None:
    st.subheader("📑 Report Library")
    st.caption(
        "Browse, view, download, or delete previously generated reports."
    )

    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button(
            "🔄 Refresh Reports", use_container_width=True
        ):
            try:
                data = list_reports()
                st.session_state["reports_list"] = data.get("reports", [])
            except Exception as e:
                st.error(f"Could not load reports: {e}")

    reports = st.session_state.get("reports_list", None)

    # Auto-load on first visit
    if reports is None:
        try:
            data = list_reports()
            reports = data.get("reports", [])
            st.session_state["reports_list"] = reports
        except Exception:
            reports = []

    if not reports:
        st.info(
            "No reports found. Run the full research pipeline to generate one."
        )
        return

    st.success(f"{len(reports)} report(s) in library.")
    st.divider()

    for rep in reports:
        rid = rep["report_id"]
        created = format_timestamp(rep.get("created_at", 0))
        size = rep.get("size_kb", 0)

        with st.expander(
            f"📑 `{rid}` — {created} — {size} KB", expanded=False
        ):
            col_meta, col_actions = st.columns([3, 1])

            with col_meta:
                st.caption(f"**File:** `{rep['filename']}`")
                st.caption(f"**Size:** {size} KB")
                st.caption(f"**Created:** {created}")

            with col_actions:
                view_key = f"view_{rid}"
                del_key = f"del_{rid}"

                if st.button("📖 View", key=view_key, use_container_width=True):
                    try:
                        content = get_report(rid)
                        st.session_state[f"report_content_{rid}"] = content
                    except Exception as e:
                        st.error(str(e))

                if st.button(
                    "🗑️ Delete", key=del_key, use_container_width=True
                ):
                    try:
                        delete_report(rid)
                        st.session_state["reports_list"] = [
                            r for r in reports if r["report_id"] != rid
                        ]
                        st.success(f"Report `{rid}` deleted.")
                        st.rerun()
                    except Exception as e:
                        st.error(str(e))

            # Render report content if loaded
            content_key = f"report_content_{rid}"
            if content_key in st.session_state:
                st.divider()
                st.markdown(st.session_state[content_key])
                st.download_button(
                    label="⬇️ Download (.md)",
                    data=st.session_state[content_key],
                    file_name=rep["filename"],
                    mime="text/markdown",
                    key=f"dl_{rid}",
                    use_container_width=True,
                )