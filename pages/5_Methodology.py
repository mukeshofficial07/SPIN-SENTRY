"""
pages/5_Methodology.py
----------------------
SPIN-SENTRY — Methodology & About.
Architecture documentation, mathematical formulations, ESI table,
known limitations, and technology stack.
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

from components.theme import (
    inject_custom_css, render_sidebar_nav, render_arch_step,
    render_badge, render_disclaimer, render_section_heading, COLORS,
)

inject_custom_css()
render_sidebar_nav()

# ---------------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------------

st.markdown(
    f'<div class="ss-page-header">'
    f'<h1 class="ss-page-title">Methodology & About</h1>'
    f'<p class="ss-page-description">'
    f'Architecture documentation, mathematical formulations, evidence weighting tables, '
    f'and scientific disclaimers for the SPIN-SENTRY alignment engine.'
    f'</p></div>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Section A: Paper reference
# ---------------------------------------------------------------------------

render_section_heading("Theoretical Framework & Paper Reference")

st.markdown(f"""
<div class="spin-card" style="border-left:3px solid {COLORS["blue"]};">
    <div style="display:flex;gap:0.5rem;flex-wrap:wrap;margin-bottom:0.75rem;">
        {render_badge("UNIVERSITY TEXT MINING PROJECT", "protocol")}
        {render_badge("REVIEW I: FORWARD ALIGNMENT", "supported")}
        {render_badge("STAGES 1–4 IMPLEMENTED", "neutral")}
    </div>
    <h4>The Tri-Review Architecture: A Multi-Pass Framework for Intra-Document Discrepancy Detection</h4>
    <p>
        The <em>Tri-Review Architecture</em> establishes an automated multi-pass framework to detect
        rhetorical spin and selective reporting in clinical trial literature.
        The complete design specifies three interconnected review passes:
    </p>
    <ul style="color:{COLORS["text_secondary"]};font-size:0.88rem;margin-bottom:0.75rem;line-height:1.7;">
        <li><strong>Review I (Forward Alignment):</strong> Abstract-to-body consistency audit —
            verifies that abstract claims match empirical section findings.</li>
        <li><strong>Review II (Protocol Alignment):</strong> Abstract-to-registry discrepancy
            check against ClinicalTrials.gov.</li>
        <li><strong>Review III (Peer Consistency):</strong> Cross-trial replication and
            systematic synthesis validation.</li>
    </ul>
    <p style="margin-bottom:0;">
        <strong>Prototype scope:</strong> This implementation covers
        <strong>Review I: Forward Alignment Pipeline (Stages 1–4)</strong>,
        engineered as an applied University Text Mining project.
    </p>
</div>
""", unsafe_allow_html=True)

st.divider()

# ---------------------------------------------------------------------------
# Section B: Four-Stage Architecture
# ---------------------------------------------------------------------------

render_section_heading("Four-Stage Architecture (Review I)")
st.caption(
    "Review I executes four sequential stages to parse claims, map full-text zones, "
    "compute directional inference, and apply evidence-strength weighting."
)

render_arch_step(
    1,
    "HCP-BEA — Hybrid Claim Parser with Biomedical Entity Anchoring",
    "<strong>Input:</strong> Abstract text (unstructured or structured).<br>"
    "<strong>Output:</strong> Typed, entity-anchored claims with lexical hedging metrics.<br>"
    "<strong>Claim types:</strong> Efficacy, Safety, Generalizability, Comparative, Novelty, Recommendation.<br>"
    "<strong>Implementation:</strong> NLTK sentence tokenizer, structural role tagging "
    "(Objective / Methods / Finding / Conclusion), keyword lexicon classifier, "
    "spaCy NER (<code>en_core_web_sm</code>), hedging density scorer.",
    COLORS["cyan"],
)

render_arch_step(
    2,
    "MZEM — Multi-Zone Evidence Mapper",
    "<strong>Input:</strong> Full-text body extracted from clinical trial publication.<br>"
    "<strong>Output:</strong> 4 evidence zones: Results, Harms / Adverse Events, "
    "Limitations / Discussion, Methods / Study Design.<br>"
    "<strong>Implementation:</strong> Dual-tier regex + RapidFuzz fuzzy header matching, "
    "terminating boundary detection, keyword density fallback.",
    COLORS["blue"],
)

render_arch_step(
    3,
    "DC-NLI — Directional Claim-Evidence NLI",
    "<strong>Input:</strong> Abstract claim + top-ranked candidate evidence sentences.<br>"
    "<strong>Output:</strong> Forward entailment, reverse entailment, gap score.<br>"
    "<strong>Model:</strong> <code>cross-encoder/nli-MiniLM2-L6-H768</code> "
    "(pretrained, general-domain, <em>not biomedical fine-tuned</em>).<br>"
    "<strong>Mechanics:</strong> forward = entailment(e→c), reverse = entailment(c→e), "
    "gap = 1 − reverse.",
    COLORS["violet"],
)

render_arch_step(
    4,
    "ESWE — Evidence-Strength Weighted Entailment",
    "<strong>Input:</strong> NLI scores, study design metadata (ESI), zone-type relevance.<br>"
    "<strong>Output:</strong> WE score, qualitative support tier, spin flag.<br>"
    "<strong>Equation:</strong> <code>WE(c, e) = NLI(c, e) × ESI(e) × relevance(t, z)</code><br>"
    "<strong>Decision rule:</strong> Claims with <strong>WE &lt; 0.50</strong> are flagged "
    "for reviewer attention.",
    COLORS["amber"],
)

with st.expander("Stage 1 & Stage 2 Routing Specifications"):
    r_col1, r_col2 = st.columns(2)
    with r_col1:
        st.markdown(f"""
        <div class="spin-card">
            <h4>6 Abstract Claim Categories</h4>
            <ul style="color:{COLORS["text_secondary"]};font-size:0.88rem;line-height:1.7;">
                <li><strong>Efficacy:</strong> Primary / secondary therapeutic efficacy assertions.</li>
                <li><strong>Safety:</strong> Adverse events, toxicity, tolerability, mortality.</li>
                <li><strong>Comparative:</strong> Superiority or non-inferiority vs. controls.</li>
                <li><strong>Generalizability:</strong> Extrapolation to broader populations.</li>
                <li><strong>Novelty:</strong> First-in-class or unprecedented discovery claims.</li>
                <li><strong>Recommendation:</strong> Direct clinical practice advice.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    with r_col2:
        st.markdown(f"""
        <div class="spin-card">
            <h4>4 Full-Text Evidence Zones</h4>
            <ul style="color:{COLORS["text_secondary"]};font-size:0.88rem;line-height:1.7;">
                <li><strong>Results:</strong> Quantitative primary/secondary outcomes.</li>
                <li><strong>Harms:</strong> Adverse events, safety withdrawals, toxicity.</li>
                <li><strong>Limitations:</strong> Study constraints, biases, power issues.</li>
                <li><strong>Methods:</strong> Protocol, blinding, sample size, ITT.</li>
            </ul>
            <p style="font-size:0.78rem;color:{COLORS["text_muted"]};margin:0.5rem 0 0 0;">
                Efficacy → Results &nbsp;·&nbsp; Safety → Harms &nbsp;·&nbsp;
                Generalizability → Limitations &nbsp;·&nbsp; Methods → Methods
            </p>
        </div>
        """, unsafe_allow_html=True)

st.divider()

# ---------------------------------------------------------------------------
# Section C: Equations
# ---------------------------------------------------------------------------

render_section_heading("Mathematical Formulations")
st.caption("Four foundational equations govern directional inference and evidence-strength calibration.")

eq_col1, eq_col2 = st.columns(2)

with eq_col1:
    st.markdown(f"""
    <div class="spin-card">
        <h4>Equation 1 — Forward Entailment</h4>
        <p style="font-size:0.87rem;">
            Whether full-text evidence <em>e</em> entails abstract claim <em>c</em>:
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.latex(r"\text{forward} = P(\text{entailment} \mid e, c)")

    st.markdown(f"""
    <div class="spin-card" style="margin-top:1rem;">
        <h4>Equation 3 — Information Gap</h4>
        <p style="font-size:0.87rem;">
            Unstated qualifications or overgeneralization relative to evidence:
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.latex(r"\text{gap} = 1 - \text{reverse}")

with eq_col2:
    st.markdown(f"""
    <div class="spin-card">
        <h4>Equation 2 — Reverse Entailment</h4>
        <p style="font-size:0.87rem;">
            Whether abstract claim <em>c</em> adequately captures nuances of evidence <em>e</em>:
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.latex(r"\text{reverse} = P(\text{entailment} \mid c, e)")

    st.markdown(f"""
    <div class="spin-card" style="margin-top:1rem;">
        <h4>Equation 4 — Weighted Entailment (ESWE)</h4>
        <p style="font-size:0.87rem;">
            Discounts raw NLI by study design strength and zone–type alignment:
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.latex(r"WE(c, e) = \text{NLI}(c, e) \times \text{ESI}(e) \times \text{relevance}(t, z)")

st.markdown(f"""
<div style="background:{COLORS["bg_card"]};border:1px solid {COLORS["border"]};
     border-radius:6px;padding:0.75rem 1rem;margin-top:0.75rem;
     font-size:0.85rem;color:{COLORS["text_secondary"]};">
    <strong style="color:{COLORS["text_primary"]};">Variable definitions:</strong>
    &nbsp;&nbsp;
    <code>c</code> = abstract claim &bull;
    <code>e</code> = best evidence sentence &bull;
    <code>NLI(c,e)</code> = forward entailment score &bull;
    <code>ESI(e)</code> = Evidence Strength Index &bull;
    <code>relevance(t,z)</code> = claim-type <em>t</em> to evidence-zone <em>z</em> weight
</div>
""", unsafe_allow_html=True)

st.divider()

# ---------------------------------------------------------------------------
# Section D: ESI Table
# ---------------------------------------------------------------------------

render_section_heading("Evidence Strength Index (ESI) — Table I")
st.caption(
    "ESI(e) is a scalar [0.0 – 1.0] representing study design quality, "
    "derived from design type, sample size, and statistical protocol."
)

esi_data = [
    {"Study Design": "RCT",          "Sample Size": "> 1000", "Analysis Type": "Intention-to-treat", "ESI": 1.0,
     "Notes": "Gold-standard double-blind randomised trial"},
    {"Study Design": "RCT",          "Sample Size": "< 100",  "Analysis Type": "Per-protocol",       "ESI": 0.7,
     "Notes": "Small cohort trial with protocol deviation risk"},
    {"Study Design": "Observational", "Sample Size": "> 5000", "Analysis Type": "Adjusted",           "ESI": 0.6,
     "Notes": "Large registry with multivariate covariate control"},
    {"Study Design": "Case series",   "Sample Size": "N < 50", "Analysis Type": "Uncontrolled",        "ESI": 0.2,
     "Notes": "Hypothesis-generating report without control arm"},
]

df_esi = pd.DataFrame(esi_data)

def _style_esi(val):
    if val >= 0.9:
        return f"background-color:{COLORS['green']}22;color:{COLORS['green']};font-weight:700;"
    elif val >= 0.65:
        return f"background-color:{COLORS['cyan']}22;color:{COLORS['cyan']};font-weight:600;"
    elif val >= 0.5:
        return f"background-color:{COLORS['blue']}22;color:{COLORS['blue']};font-weight:600;"
    return f"background-color:{COLORS['red']}22;color:{COLORS['red']};font-weight:600;"

styled_esi = (
    df_esi.style
    .format({"ESI": "{:.1f}"})
    .map(_style_esi, subset=["ESI"])
)

st.dataframe(
    styled_esi, width="stretch",
    hide_index=True,
    column_config={
        "Study Design": st.column_config.TextColumn("Study Design", width="medium"),
        "Sample Size":  st.column_config.TextColumn("Sample Size", width="small"),
        "Analysis Type": st.column_config.TextColumn("Analysis Type", width="medium"),
        "ESI":          st.column_config.NumberColumn("ESI", format="%.1f", width="small"),
        "Notes":        st.column_config.TextColumn("Context", width="large"),
    },
)

st.caption(
    "When trial design cannot be parsed from Methods text by automated heuristics, "
    "SPIN-SENTRY defaults to ESI = 0.50 (unspecified empirical evidence baseline)."
)

st.divider()

# ---------------------------------------------------------------------------
# Section E: Known Limitations
# ---------------------------------------------------------------------------

render_section_heading("Known Limitations & Assumptions")

st.markdown(f"""
<div class="spin-card">
    <ul style="color:{COLORS["text_secondary"]};font-size:0.9rem;line-height:1.8;margin-bottom:0;">
        <li><strong>Rule-based claim classification:</strong> Keyword lexicon and regex heuristics,
            not a fine-tuned biomedical multi-label classifier.</li>
        <li><strong>General-domain NLI model:</strong>
            <code>cross-encoder/nli-MiniLM2-L6-H768</code> is trained on MNLI/SNLI
            (web/dialogue), not clinical trial literature.</li>
        <li><strong>Regex zone segmentation:</strong> May misroute sentences in non-standard
            PDF layouts; SciBERT section classifier not used.</li>
        <li><strong>PDF extraction:</strong> <code>pdfplumber</code> extracts linear text;
            complex tables, multi-column figures, and supplements may be missed.</li>
        <li><strong>Heuristic threshold:</strong> WE &lt; 0.50 spin flag is an engineering
            choice for this prototype, not clinically validated.</li>
        <li><strong>Experimental classifier:</strong> Optional ML classifier trained on
            AI-generated pseudo-labels, not human expert annotations.</li>
    </ul>
</div>
""", unsafe_allow_html=True)

st.divider()

# ---------------------------------------------------------------------------
# Section F: Critical Disclaimers
# ---------------------------------------------------------------------------

render_section_heading("Critical Disclaimers")

st.warning(
    "**WHAT THIS SYSTEM IS NOT:**\n\n"
    "- **Not a clinical decision tool.** Never use for patient care, therapeutic choices, "
    "or diagnostic evaluations.\n"
    "- **Not a replacement for peer review.** Human expert appraisal by clinical "
    "epidemiologists remains the gold standard.\n"
    "- **Does not validate paper quality.** A spin flag indicates rhetorical divergence "
    "between abstract and body text — not scientific misconduct or trial invalidity.\n"
    "- **Paper metrics (89%, 81%, F1=0.93) are not reproduced here.** Those figures are "
    "from the paper authors' own evaluation and are not claimed by this prototype."
)

st.divider()

# ---------------------------------------------------------------------------
# Section G: Annotation Process
# ---------------------------------------------------------------------------

render_section_heading("Annotation Process & Evaluation Framework")

ann_col1, ann_col2 = st.columns(2)

with ann_col1:
    st.markdown(f"""
    <div class="spin-card" style="border-left:3px solid {COLORS["cyan"]};">
        <h4>Training Data Provenance</h4>
        <div style="font-size:0.85rem;color:{COLORS["text_secondary"]};line-height:1.75;">
            <div><strong>Source corpus:</strong> 3,737 candidates from 22 Open Access papers</div>
            <div><strong>Genuine human reviews:</strong> 13 rows reviewed manually</div>
            <div><strong>AI-batch annotations:</strong> 3,720 rows processed automatically</div>
            <div><strong>Training set:</strong> 2,375 rows classified as claims</div>
            <div><strong>Annotator field note:</strong> "Mukesh" stamp does not imply
                manual review for AI-batch rows</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with ann_col2:
    st.markdown(f"""
    <div class="spin-card" style="border-left:3px solid {COLORS["amber"]};">
        <h4>Evaluation Caveats</h4>
        <div style="font-size:0.85rem;color:{COLORS["text_secondary"]};line-height:1.75;">
            <div><strong>Current evaluation:</strong>
                <code>agreement_with_ai_generated_batch_labels</code></div>
            <div><strong>Split:</strong> Paper-grouped 80/20 (17 train / 5 holdout)</div>
            <div><strong>Test set:</strong> 552 holdout sentences; 549 (99.5%) are
                AI-batch labels</div>
            <div><strong>Gold standard:</strong> Not yet available — only 13 genuine
                human reviews exist</div>
            <div><strong>"human_verified" model name:</strong> Historical label only;
                actual source is AI-batch annotations</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.info(
    "**Evaluation roadmap:** A genuine gold-standard test set requires ≥ 30 human-reviewed "
    "examples per category (≥ 180 total). The Annotation Review Tool at "
    "http://localhost:8503 progressively builds this gold standard. "
    "Until then, all reported metrics measure AI-label agreement, not human accuracy."
)

st.divider()

# ---------------------------------------------------------------------------
# Section H: Technology Stack
# ---------------------------------------------------------------------------

render_section_heading("Technology Stack")

tech_col1, tech_col2 = st.columns([1.3, 0.7])

with tech_col1:
    tech_data = [
        {"Component": "NLI Model",    "Technology": "cross-encoder/nli-MiniLM2-L6-H768",   "Role": "Directional claim-evidence inference"},
        {"Component": "NER",          "Technology": "spaCy en_core_web_sm",                 "Role": "Biomedical entity extraction"},
        {"Component": "PDF",          "Technology": "pdfplumber",                            "Role": "Full-text PDF extraction"},
        {"Component": "Classifier",   "Technology": "TF-IDF + OvR Logistic Regression",    "Role": "Machine-learned claim classification"},
        {"Component": "UI",           "Technology": "Streamlit",                             "Role": "Interactive web interface"},
        {"Component": "Language",     "Technology": "Python 3.13",                           "Role": "Core runtime"},
        {"Component": "Fuzzy Match",  "Technology": "RapidFuzz",                             "Role": "Zone header matching"},
        {"Component": "Tokenizer",    "Technology": "NLTK Punkt",                            "Role": "Sentence segmentation"},
    ]
    st.dataframe(
        pd.DataFrame(tech_data), width="stretch",
        hide_index=True,
        column_config={
            "Component":  st.column_config.TextColumn("Component", width="medium"),
            "Technology": st.column_config.TextColumn("Framework / Package", width="large"),
            "Role":       st.column_config.TextColumn("Role", width="large"),
        },
    )

with tech_col2:
    st.markdown(f"""
    <div class="spin-card">
        <h4>Runtime Specs</h4>
        <div style="font-size:0.85rem;color:{COLORS["text_secondary"]};line-height:1.8;">
            <div><strong>Inference:</strong> CPU cross-encoder</div>
            <div><strong>NLI model:</strong> ~90 MB</div>
            <div><strong>Vectorization:</strong> scikit-learn TF-IDF</div>
            <div><strong>Architecture:</strong> Tri-Review (Review I)</div>
            <div><strong>Tests:</strong> 225 passing</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown(f"""
<div style="text-align:center;color:{COLORS["text_muted"]};
     font-size:0.78rem;padding:2rem 0 1rem;">
    SPIN-SENTRY &bull; Research Intelligence Platform &bull;
    Tri-Review Architecture — Review I: Forward Alignment Pipeline
</div>
""", unsafe_allow_html=True)
