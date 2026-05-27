import streamlit as st
from frontend.utils.api import get_health, test_ollama


def render_health_tab() -> None:
    st.subheader("🩺 System Health")

    # ── Backend + Ollama status ───────────────────────────────────────────
    st.markdown("#### Service Status")

    if st.button("🔄 Check Health", use_container_width=False):
        with st.spinner("Pinging backend..."):
            try:
                data = get_health()
                c1, c2, c3, c4 = st.columns(4)
                c1.metric(
                    "Backend",
                    "🟢 Online" if data["status"] == "ok" else "🔴 Error",
                )
                c2.metric(
                    "Ollama",
                    "🟢 Online" if data["ollama_reachable"] else "🔴 Offline",
                )
                c3.metric("Model", data["model"])
                c4.metric("API Version", data.get("version", "—"))

                if not data["ollama_reachable"]:
                    st.error(
                        "Ollama is not reachable. "
                        "Run `ollama serve` in a separate terminal."
                    )
                else:
                    st.success("All systems operational.")

            except Exception as e:
                st.error(f"Backend unreachable: {e}")
                st.info(
                    "Ensure FastAPI is running: "
                    "`uvicorn backend.main:app --reload --port 8000`"
                )

    st.divider()

    # ── Ollama LLM test ───────────────────────────────────────────────────
    st.markdown("#### Test Ollama Generation")
    st.caption("Send a direct prompt to the Ollama LLM.")

    model = st.selectbox(
        "Model for test:",
        ["llama3", "mistral", "deepseek-r1"],
        key="health_model_select",
    )

    prompt = st.text_area(
        "Prompt:",
        placeholder="Explain the difference between RAG and fine-tuning in 3 sentences.",
        height=100,
        key="health_test_prompt",
    )

    if st.button(
        "▶️ Run Prompt",
        type="secondary",
        use_container_width=False,
        key="health_run_prompt",
    ):
        if not prompt.strip():
            st.warning("Enter a prompt.")
        else:
            with st.spinner(f"Generating with `{model}`..."):
                try:
                    data = test_ollama(prompt=prompt, model=model)
                    st.success(f"Response from `{data['model']}`:")
                    st.markdown(data["response"])
                except Exception as e:
                    st.error(f"Generation failed: {e}")

    st.divider()

    # ── Stack info ────────────────────────────────────────────────────────
    st.markdown("#### Tech Stack")
    stack = {
        "Frontend": "Streamlit",
        "Backend": "FastAPI",
        "Agent Framework": "LangGraph",
        "LLM": "Ollama (local)",
        "Embeddings": "all-MiniLM-L6-v2 (sentence-transformers)",
        "Vector DB": "ChromaDB (persistent)",
        "PDF Parsing": "PyMuPDF",
        "Tool Layer": "MCP (Python)",
    }
    for k, v in stack.items():
        st.caption(f"**{k}:** {v}")