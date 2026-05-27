import streamlit as st
from frontend.utils.api import list_mcp_tools, invoke_mcp_tool
from frontend.utils.formatting import score_badge, truncate


def render_mcp_tab(config: dict) -> None:
    collection_name = config["collection_name"]

    st.subheader("🛠️ MCP Tool Console")
    st.caption(
        "Inspect the MCP tool registry and invoke tools directly "
        "outside the agent pipeline."
    )

    col_reg, col_invoke = st.columns([1, 2])

    # ── Registry panel ────────────────────────────────────────────────────
    with col_reg:
        st.markdown("#### Registered Tools")

        if st.button("🔄 Load Registry", use_container_width=True):
            try:
                data = list_mcp_tools()
                st.session_state["mcp_tools"] = data["tools"]
                st.success(f"{data['total']} tools loaded.")
            except Exception as e:
                st.error(str(e))

        tools = st.session_state.get("mcp_tools", [])

        if not tools:
            st.info("Click **Load Registry** to inspect tools.")
        else:
            for tool in tools:
                with st.expander(
                    f"🔧 `{tool['name']}`", expanded=False
                ):
                    st.markdown(f"**v{tool.get('version', '1.0.0')}**")
                    st.markdown(tool["description"])
                    st.markdown("**Parameters:**")
                    for p in tool["parameters"]:
                        req_label = (
                            "required" if p["required"] else "optional"
                        )
                        st.caption(
                            f"• `{p['name']}` ({p['type']}, "
                            f"{req_label}) — {p['description']}"
                        )

    # ── Invocation panel ──────────────────────────────────────────────────
    with col_invoke:
        st.markdown("#### Invoke Tool")

        tool_choice = st.selectbox(
            "Select tool:",
            ["search_vector", "parse_pdf", "web_search"],
            key="mcp_tool_select",
        )

        params: dict = {}

        if tool_choice == "search_vector":
            params["query"] = st.text_input(
                "Query:",
                placeholder="transformer self-attention",
                key="mcp_vec_query",
            )
            params["top_k"] = st.slider(
                "Top K", 1, 10, 5, key="mcp_vec_topk"
            )

        elif tool_choice == "parse_pdf":
            params["filename"] = st.text_input(
                "PDF filename:",
                placeholder="paper.pdf",
                key="mcp_pdf_filename",
            )

        elif tool_choice == "web_search":
            params["query"] = st.text_input(
                "Search query:",
                placeholder="latest LLM benchmark results",
                key="mcp_web_query",
            )
            params["max_results"] = st.slider(
                "Max results", 1, 10, 5, key="mcp_web_max"
            )

        invoke_collection = st.text_input(
            "Collection override (optional):",
            value="",
            key="mcp_invoke_collection",
        )

        if st.button(
            f"⚡ Invoke `{tool_choice}`",
            type="primary",
            use_container_width=True,
            key="mcp_invoke_btn",
        ):
            clean_params = {
                k: v for k, v in params.items()
                if v not in ("", None)
            }
            if not clean_params:
                st.warning("Fill in at least one parameter.")
                return

            with st.spinner(f"Calling `{tool_choice}`..."):
                try:
                    result = invoke_mcp_tool(
                        tool_name=tool_choice,
                        parameters=clean_params,
                        collection_name=invoke_collection.strip() or None,
                    )
                except Exception as e:
                    st.error(f"Invocation failed: {e}")
                    return

            if result.get("success"):
                st.success(
                    f"✅ `{result['tool_name']}` executed successfully."
                )

                meta = result.get("metadata", {})
                if meta:
                    st.json(meta)

                st.markdown("**Result:**")

                if tool_choice == "search_vector":
                    items = result.get("result", [])
                    if not items:
                        st.info(
                            "No results — ingest documents first."
                        )
                    else:
                        for i, item in enumerate(items, 1):
                            score = item["score"]
                            with st.expander(
                                f"Result {i} — {score_badge(score)}",
                                expanded=(i == 1),
                            ):
                                st.markdown(item["text"])
                                st.json(item["metadata"])

                elif tool_choice == "parse_pdf":
                    pages = result.get("result", [])
                    if not pages:
                        st.info("No pages returned.")
                    else:
                        for page in pages[:5]:
                            with st.expander(
                                f"Page {page['page_number']} "
                                f"({page['word_count']} words)"
                            ):
                                st.text(
                                    truncate(page["text"], 600)
                                )

                elif tool_choice == "web_search":
                    for item in result.get("result", []):
                        with st.expander(
                            truncate(item["title"], 80)
                        ):
                            st.caption(item["url"])
                            st.markdown(item["snippet"])

            else:
                st.error(
                    f"Tool failed: "
                    f"{result.get('error', 'Unknown error')}"
                )