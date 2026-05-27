import streamlit as st
from frontend.utils.api import upload_pdf, ingest_text, get_collection_count


def render_ingestion_tab(config: dict) -> None:
    collection_name = config["collection_name"]

    st.subheader("📥 Document Ingestion")
    st.caption(
        "Add documents to the vector store. "
        "Supports PDF upload and raw text paste."
    )

    sub_tab1, sub_tab2 = st.tabs(["📄 Upload PDF", "📝 Paste Text"])

    # ── PDF Upload ────────────────────────────────────────────────────────
    with sub_tab1:
        st.markdown("#### PDF Upload & Ingestion")
        st.caption(
            "Parsed page-by-page with PyMuPDF. "
            "Chunked, embedded, and stored in ChromaDB automatically."
        )

        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=["pdf"],
            key="pdf_uploader",
        )

        pdf_target = st.text_input(
            "Target collection:",
            value=collection_name,
            key="pdf_target_collection",
        )

        if uploaded_file is not None:
            col_info1, col_info2 = st.columns(2)
            col_info1.info(f"**File:** {uploaded_file.name}")
            col_info2.info(
                f"**Size:** {round(uploaded_file.size / 1024, 1)} KB"
            )

            if st.button(
                "📤 Upload & Process PDF",
                type="primary",
                use_container_width=True,
                key="upload_pdf_btn",
            ):
                with st.spinner(
                    "Parsing PDF → chunking → embedding → storing..."
                ):
                    try:
                        data = upload_pdf(
                            file_bytes=uploaded_file.getvalue(),
                            filename=uploaded_file.name,
                            collection_name=pdf_target,
                        )

                        st.success("✅ PDF ingested successfully!")

                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("Pages", data["total_pages"])
                        m2.metric("Characters", f"{data['total_chars']:,}")
                        m3.metric("Chunks Stored", data["chunks_stored"])
                        m4.metric("Collection", data["collection"])

                        with st.expander(
                            "📊 Page Breakdown", expanded=False
                        ):
                            st.dataframe(
                                [
                                    {
                                        "Page": p["page_number"],
                                        "Characters": p["char_count"],
                                        "Words": p["word_count"],
                                    }
                                    for p in data["page_info"]
                                ],
                                use_container_width=True,
                            )

                        with st.expander("🔑 Chunk IDs", expanded=False):
                            shown = data["ids"][:10]
                            st.code("\n".join(shown))
                            if len(data["ids"]) > 10:
                                st.caption(
                                    f"... and {len(data['ids']) - 10} more"
                                )

                    except Exception as e:
                        st.error(f"Upload failed: {e}")

    # ── Text Paste ────────────────────────────────────────────────────────
    with sub_tab2:
        st.markdown("#### Paste Text for Ingestion")
        st.caption(
            "Paste any text content — articles, notes, abstracts. "
            "Automatically chunked and embedded."
        )

        text_input = st.text_area(
            "Text content:",
            placeholder=(
                "Paste research content, article text, "
                "or document excerpts here..."
            ),
            height=220,
            key="ingest_text_input",
        )

        col_src, col_type = st.columns(2)
        with col_src:
            source_label = st.text_input(
                "Source label:",
                placeholder="arxiv:2305.12345 or manual",
                key="ingest_source_label",
            )
        with col_type:
            doc_type = st.selectbox(
                "Document type:",
                ["article", "paper", "note", "webpage", "other"],
                key="ingest_doc_type",
            )

        text_target = st.text_input(
            "Target collection:",
            value=collection_name,
            key="text_target_collection",
        )

        if st.button(
            "📥 Ingest Text",
            type="primary",
            use_container_width=True,
            key="ingest_text_btn",
        ):
            if not text_input.strip():
                st.warning("Paste some text before ingesting.")
            else:
                with st.spinner("Chunking and embedding text..."):
                    try:
                        data = ingest_text(
                            texts=[text_input],
                            metadatas=[
                                {
                                    "source": source_label or "manual_input",
                                    "type": doc_type,
                                }
                            ],
                            collection_name=text_target,
                        )

                        st.success(
                            f"✅ Stored **{data['chunks_stored']}** chunks "
                            f"in collection `{data['collection']}`"
                        )

                        # Refresh chunk count
                        try:
                            count_data = get_collection_count(text_target)
                            st.session_state["chunk_count"] = count_data[
                                "document_count"
                            ]
                        except Exception:
                            pass

                        with st.expander("🔑 Chunk IDs", expanded=False):
                            st.json(data["ids"])

                    except Exception as e:
                        st.error(f"Ingestion failed: {e}")
