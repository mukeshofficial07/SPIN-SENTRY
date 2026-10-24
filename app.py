"""
app.py
------
SPIN-SENTRY — Research Intelligence Platform
Multi-page Streamlit application entry point.

The Tri-Review Architecture: A Multi-Pass Framework
for Intra-Document Discrepancy Detection.

Upload a clinical-trial PDF → receive a structured spin-detection report.

IMPORTANT DISCLAIMERS:
  - Rule-based approximations are clearly labeled throughout.
  - The NLI model (cross-encoder/nli-MiniLM2-L6-H768) is used as-is,
    without biomedical fine-tuning.
  - Performance metrics reported in the paper (89% / 81% / F1=0.93)
    are from the authors' evaluation on their corpus. They are NOT
    reproduced or validated by this implementation.
  - WE < 0.5 as spin flag threshold is an engineering choice for this
    prototype, not a validated operating point.
"""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="SPIN-SENTRY · Research Intelligence",
    page_icon="S",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Multi-page navigation — clean labels, minimal icons
# ---------------------------------------------------------------------------

home_page        = st.Page("pages/1_Home.py",             title="Home",              default=True)
workspace_page   = st.Page("pages/2_Research_Workspace.py", title="Research Workspace")
results_page     = st.Page("pages/3_Analysis_Results.py", title="Analysis Results")
lab_page         = st.Page("pages/4_Dataset_Model_Lab.py", title="Dataset & Model Lab")
methodology_page = st.Page("pages/5_Methodology.py",      title="Methodology & About")

pg = st.navigation(
    {
        "Explore":   [home_page],
        "Analysis":  [workspace_page, results_page],
        "Research":  [lab_page, methodology_page],
    }
)

pg.run()
