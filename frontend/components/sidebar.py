import streamlit as st
from frontend.utils.api import get_collection_count, list_pdfs
from typing import Dict

def render_sidebar() -> Dict:
    """
    Renders the sidebar configuration panel.
    Returns config dict consumed by all components.
    """
    from typing import Dict

    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        st.divider()

        # ── Model selection ───────────────────────────────────────────────
        st.markdown("**LLM Model**")
        model = st.selectbox(
            "Ollama model",
            ["llama3", "mistral", "deepseek-r1"],
            index=0,
            label_visibility="collapsed",
        )

        st.divider()

        # ── Collection ────────────────────────────────────────────────────
        st.markdown("**Vector Collection**")
        collection_name = st.text_input(
            "Collection name",
            value="research_docs",
            label_visibility="collapsed",
        )

        col1, col2 = st.columns(2)
        with col1:
            if st.button("📊 Count", use_container_width=True):
                try:
                    data = get_collection_count(collection_name)
                    st.session_state["chunk_count"] = data["document_count"]
                except Exception as e:
                    st.error(str(e))

        with col2:
            count = st.session_state.get("chunk_count", "—")
            st.metric("Chunks", count)

        st.divider()

        # ── PDF list ──────────────────────────────────────────────────────
        st.markdown("**Uploaded PDFs**")
        if st.button("📂 Refresh", use_container_width=True):
            try:
                data = list_pdfs()
                st.session_state["pdf_list"] = data.get("files", [])
            except Exception as e:
                st.error(str(e))

        pdf_list = st.session_state.get("pdf_list", [])
        if pdf_list:
            for f in pdf_list:
                st.caption(f"📄 {f['filename']} ({f['size_kb']} KB)")
        else:
            st.caption("No PDFs uploaded yet.")

        st.divider()

        # ── Pipeline config display ───────────────────────────────────────
        st.markdown("**Pipeline Config**")
        st.caption("Configure via `.env`")
        st.caption(
            "• Confidence threshold: `CONFIDENCE_THRESHOLD`\n"
            "• Max retries: `MAX_RETRIES`\n"
            "• Chunk size: `CHUNK_SIZE`"
        )

        st.divider()
        st.caption("🔬 Research Agent v0.9.0")

    return {
        "model": model,
        "collection_name": collection_name,
    }
