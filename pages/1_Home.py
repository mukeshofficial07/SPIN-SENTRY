"""
pages/1_Home.py
---------------
SPIN-SENTRY — Home / Project Overview.
Professional landing page for the research intelligence platform.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
from components.theme import (
    inject_custom_css, render_hero, render_feature_card,
    render_arch_step, render_stat_card, render_sidebar_nav,
    render_disclaimer, COLORS,
)
inject_custom_css()
render_sidebar_nav()

# ---------------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------------

render_hero(
    title="SPIN-SENTRY",
    subtitle=(
        "A multi-pass forward alignment pipeline that detects rhetorical spin "
        "in clinical trial papers by cross-checking abstract claims against "
        "Results, Harms, Limitations, and Methods sections."
    ),
    badge="TRI-REVIEW ARCHITECTURE · FORWARD ALIGNMENT PIPELINE · REVIEW I",
)

# CTA button — navigate to Research Workspace
cta_col, _ = st.columns([1, 3])
with cta_col:
    if st.button("Start Analysis →", type="primary", width="stretch"):
        st.switch_page("pages/2_Research_Workspace.py")

# ---------------------------------------------------------------------------
# Quick stats row
# ---------------------------------------------------------------------------

cols = st.columns(5)
with cols[0]:
    render_stat_card("4", "Pipeline Stages")
with cols[1]:
    render_stat_card("22", "Research Papers")
with cols[2]:
    render_stat_card("3,737", "Sentence Candidates")
with cols[3]:
    render_stat_card("6", "Claim Categories")
with cols[4]:
    render_stat_card("225", "Tests Passing", COLORS["green"])

st.divider()

# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------

st.markdown("## Forward Alignment Architecture")
st.caption(
    "Each abstract claim passes through four stages. "
    "The pipeline is modular — stages can be upgraded independently."
)

col_left, col_right = st.columns([1.1, 0.9])

with col_left:
    render_arch_step(
        1, "HCP-BEA — Hybrid Claim Parser",
        "Segment abstract into sentences, detect structural roles "
        "(Objective / Methods / Finding / Conclusion), classify into 6 claim types "
        "(Efficacy, Safety, Generalizability, Comparative, Novelty, Recommendation), "
        "anchor to biomedical entities, and score hedging language.",
        COLORS["cyan"],
    )
    render_arch_step(
        2, "MZEM — Multi-Zone Evidence Mapper",
        "Partition the full-text body into four evidence zones using regex "
        "and fuzzy header matching: Results, Harms, Limitations, Methods. "
        "Route each claim to its primary evidence zone.",
        COLORS["blue"],
    )
    render_arch_step(
        3, "DC-NLI — Directional Claim-Evidence NLI",
        "Score each claim against top-K evidence sentences using a pretrained "
        "NLI cross-encoder. Compute forward entailment (Eq. 1), reverse entailment "
        "(Eq. 2), and generalizability gap (Eq. 3). "
        "Prioritize contradictory evidence during retrieval.",
        COLORS["violet"],
    )
    render_arch_step(
        4, "ESWE — Evidence-Strength Weighted Entailment",
        "Combine NLI score with Evidence Strength Index (study design × sample size × "
        "analysis type) and zone–type relevance: WE = NLI × ESI × relevance (Eq. 4). "
        "Claims with WE < 0.50 are flagged for reviewer attention.",
        COLORS["green"],
    )

with col_right:
    st.markdown(f"""
    <div class="spin-card">
        <h4>Core Equations</h4>
        <div style="font-family:'SFMono-Regular',Consolas,monospace;
             font-size:0.85rem; color:{COLORS["text_secondary"]}; line-height:2.1;">
            <div><span style="color:{COLORS["cyan"]};font-weight:600;">Eq. 1</span>
                &nbsp; forward = P(entailment | e, c)
            </div>
            <div><span style="color:{COLORS["cyan"]};font-weight:600;">Eq. 2</span>
                &nbsp; reverse = P(entailment | c, e)
            </div>
            <div><span style="color:{COLORS["cyan"]};font-weight:600;">Eq. 3</span>
                &nbsp; gap = 1 − reverse
            </div>
            <div style="margin-top:0.75rem; padding:0.6rem 0.75rem;
                 background:{COLORS["bg_primary"]}; border-radius:6px;
                 border:1px solid {COLORS["border"]};">
                <span style="color:{COLORS["green"]};font-weight:600;">Eq. 4</span>
                &nbsp; WE(c, e) = NLI × ESI × relevance
            </div>
        </div>
    </div>

    <div class="spin-card">
        <h4>Spin Flag Rule</h4>
        <p>Claims with <code>WE &lt; 0.50</code> are flagged for reviewer attention.</p>
        <p style="font-size:0.8rem;color:{COLORS["text_muted"]};">
            This threshold is an engineering heuristic for this prototype,
            not a validated clinical operating point.
        </p>
    </div>

    <div class="spin-card">
        <h4>NLI Model</h4>
        <p><code>cross-encoder/nli-MiniLM2-L6-H768</code></p>
        <p style="font-size:0.8rem;color:{COLORS["text_muted"]};">
            General-domain pretrained model. Not fine-tuned on biomedical text.
            ~90 MB, runs on CPU.
        </p>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# ---------------------------------------------------------------------------
# Platform features
# ---------------------------------------------------------------------------

st.markdown("## Platform Capabilities")

feat_cols = st.columns(4)
with feat_cols[0]:
    render_feature_card(
        "PDF",
        "PDF Analysis",
        "Upload any clinical trial PDF. The pipeline extracts text, "
        "identifies claims, maps evidence zones, and scores each claim against "
        "trial data.",
    )
with feat_cols[1]:
    render_feature_card(
        "NLI",
        "Claim Verification",
        "Six claim types cross-checked against four evidence zones using "
        "directional NLI with forward, reverse, and gap scoring.",
    )
with feat_cols[2]:
    render_feature_card(
        "Lab",
        "Dataset Laboratory",
        "22-paper corpus with 3,737 candidates. "
        "TF-IDF classifier with paper-grouped evaluation and transparent metrics.",
    )
with feat_cols[3]:
    render_feature_card(
        "JSON",
        "Transparent Reporting",
        "Every score is explained. Approximations are labeled throughout. "
        "Full JSON export for reproducibility and further analysis.",
    )

st.divider()

# ---------------------------------------------------------------------------
# Quick start
# ---------------------------------------------------------------------------

st.markdown("## Quick Start")

qs_cols = st.columns(3)
with qs_cols[0]:
    st.markdown(f"""
    <div class="spin-card">
        <h4>1 — Upload a PDF</h4>
        <p>Go to <strong>Research Workspace</strong> and upload a clinical trial PDF,
        or select one of the included demo papers.</p>
    </div>
    """, unsafe_allow_html=True)

with qs_cols[1]:
    st.markdown(f"""
    <div class="spin-card">
        <h4>2 — Run Analysis</h4>
        <p>The pipeline extracts abstract claims and cross-checks each one against
        Results, Harms, Limitations, and Methods sections.</p>
    </div>
    """, unsafe_allow_html=True)

with qs_cols[2]:
    st.markdown(f"""
    <div class="spin-card">
        <h4>3 — Review & Export</h4>
        <p>Inspect claim-level evidence scores, explanations, and spin flags.
        Download the JSON report for further analysis.</p>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# ---------------------------------------------------------------------------
# Disclaimer
# ---------------------------------------------------------------------------

render_disclaimer(
    "Automated spin flags are <strong>heuristic screening signals</strong> for human "
    "reviewer exploration. They are not proof of scientific misconduct or clinical "
    "trial invalidity. Rule-based approximations are clearly labeled. "
    "Paper-reported metrics (89% / 81% / F1=0.93) are not claimed by this "
    "implementation."
)

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.markdown(f"""
<div style="text-align:center; padding:2rem 0 1rem; color:{COLORS["text_muted"]};">
    <div style="font-size:0.82rem; font-weight:600; margin-bottom:0.35rem;">SPIN-SENTRY</div>
    <div style="font-size:0.75rem;">
        The Tri-Review Architecture · Multi-Pass Intra-Document Discrepancy Detection<br>
        University Text Mining Project · Review I: Forward Alignment Pipeline (Stages 1–4)
    </div>
</div>
""", unsafe_allow_html=True)
