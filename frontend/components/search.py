import streamlit as st
from frontend.utils.api import search_vector
from frontend.utils.formatting import score_badge, truncate


def render_search_tab(config: dict) -> None:
    collection_name = config["collection_name"]

    st.subheader("🔍 Semantic Search")
    st.caption(
        "Query the vector store directly. "
        "Results ranked by cosine similarity score."
    )

    col1, col2, col3 = st.columns([4, 1, 1])
    with col1:
        query = st.text_input(
            "Search query:",
            placeholder="What is retrieval-augmented generation?",
            key="search_query_input",
        )
    with col2:
        top_k = st.number_input(
            "Top K", min_value=1, max_value=20, value=5, key="search_top_k"
        )
    with col3:
        search_collection = st.text_input(
            "Collection",
            value=collection_name,
            key="search_collection",
            label_visibility="collapsed",
        )

    if st.button("🔍 Search", type="primary", use_container_width=True):
        if not query.strip():
            st.warning("Enter a search query.")
            return

        with st.spinner("Querying vector store..."):
            try:
                data = search_vector(
                    query=query,
                    top_k=top_k,
                    collection_name=search_collection,
                )
            except Exception as e:
                st.error(f"Search failed: {e}")
                return

        total = data.get("total_results", 0)

        if total == 0:
            st.info(
                "No results found. "
                "Ingest documents via the Ingestion tab first."
            )
            return

        st.success(
            f"Found **{total}** result(s) in "
            f"collection `{data['collection']}`"
        )
        st.divider()

        for i, result in enumerate(data["results"], 1):
            score = result["score"]
            badge = score_badge(score)
            meta = result.get("metadata", {})
            source = meta.get("source", "unknown")
            page = meta.get("page_number", "")
            page_str = f" · page {page}" if page else ""

            with st.expander(
                f"Result {i} — {badge} — `{source}`{page_str}",
                expanded=(i == 1),
            ):
                st.markdown(result["text"])

                col_meta1, col_meta2, col_meta3 = st.columns(3)
                col_meta1.caption(f"**ID:** `{result['id'][:16]}...`")
                col_meta2.caption(f"**Source:** `{source}`")
                col_meta3.caption(f"**Score:** `{score}`")

                if meta:
                        st.markdown("Full metadata:")
                        st.json(meta)
                        