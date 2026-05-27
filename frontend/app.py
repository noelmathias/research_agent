import streamlit as st

# Page config must be first Streamlit call
st.set_page_config(
    page_title="Research Agent",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

from frontend.components.sidebar import render_sidebar
from frontend.components.pipeline import render_pipeline_tab
from frontend.components.report_viewer import render_report_viewer_tab
from frontend.components.ingestion import render_ingestion_tab
from frontend.components.search import render_search_tab
from frontend.components.mcp_panel import render_mcp_tab
from frontend.components.health import render_health_tab

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div style='padding: 0.5rem 0 1rem 0;'>
        <h1 style='margin: 0;'>🔬 Research Agent</h1>
        <p style='color: grey; margin: 0;'>
            Autonomous Multi-Agent Research Assistant
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
config = render_sidebar()

# ── Navigation tabs ───────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🔬 Research",
    "📑 Reports",
    "📥 Ingestion",
    "🔍 Search",
    "🛠️ MCP Tools",
    "🩺 Health",
])

with tab1:
    render_pipeline_tab(config)

with tab2:
    render_report_viewer_tab()

with tab3:
    render_ingestion_tab(config)

with tab4:
    render_search_tab(config)

with tab5:
    render_mcp_tab(config)

with tab6:
    render_health_tab()