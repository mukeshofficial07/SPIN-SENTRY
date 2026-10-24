"""
pages/2_Research_Workspace.py
-----------------------------
PDF upload → full SPIN-SENTRY analysis pipeline.
"""

import sys
import time
import json
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd

from components.theme import (
    inject_custom_css, render_sidebar_nav, render_stat_card,
    render_badge, render_disclaimer, render_section_heading,
    render_risk_banner, COLORS,
)
from pipeline import run_pipeline, PipelineResult
from stages.stage3_nli import get_model_status
from stages.stage4_eswe import SPIN_FLAG_THRESHOLD

inject_custom_css()
render_sidebar_nav()

# ---------------------------------------------------------------------------
# Sidebar settings
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown(
        f'<div style="font-size:0.78rem;font-weight:700;color:{COLORS["text_muted"]}; '
        f'text-transform:uppercase;letter-spacing:0.8px;margin-bottom:0.75rem;">'
        f'Analysis Settings</div>',
        unsafe_allow_html=True,
    )
    top_k = st.slider("Evidence sentences per claim (top-K)", 1, 10, 5)

    from utils.trained_claim_classifier import MODEL_PATH
    _active_model_path = MODEL_PATH
    _active_model_name = _active_model_path.name
    _model_exists = _active_model_path.exists()

    classifier_mode = st.selectbox(
        "Stage 1 classifier",
        ["trained", "rules"],
        format_func=lambda v: f"ML Model ({_active_model_name})" if v == "trained"
                              else "Rule-based (keyword lexicon)",
        help=f"Trained mode loads `{_active_model_name}`. "
             "Rule-based uses keyword lexicons with no model dependency.",
    )
    if classifier_mode == "trained":
        if _model_exists:
            st.caption(
                f"Active: `{_active_model_name}`\n\n"
                "Evaluated on 197 source-verified claims (Macro-F1: 0.730). "
                "Predictions are automated routing signals; verify in evidence explorer."
            )
        else:
            st.error(
                f"Model not found: `{_active_model_path}`\n\n"
                "Switch to Rule-based or run `scripts/build_and_retrain.py`."
            )
    st.divider()

    st.markdown(
        f'<div style="font-size:0.78rem;font-weight:700;color:{COLORS["text_muted"]}; '
        f'text-transform:uppercase;letter-spacing:0.8px;margin-bottom:0.75rem;">'
        f'NLI Model</div>',
        unsafe_allow_html=True,
    )
    status = get_model_status()
    if status["loaded"]:
        st.success(f"Loaded: {status['model_name']}")
    else:
        st.error(f"Not loaded — {status['error']}")

    st.divider()
    st.markdown(
        f"""<div style="font-size:0.77rem;color:{COLORS["text_muted"]};line-height:1.6;">
            <strong style="color:{COLORS["text_secondary"]};">Screening Notice</strong><br>
            Spin flags are heuristic signals for reviewer exploration,
            not proof of misconduct.<br><br>
            Rule-based components are approximations.<br>
            Paper metrics are not reproduced here.<br>
            Spin flag: WE &lt; {SPIN_FLAG_THRESHOLD} (prototype threshold).
        </div>""",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------------

st.markdown(
    f'<div class="ss-page-header">'
    f'<h1 class="ss-page-title">Research Workspace</h1>'
    f'<p class="ss-page-description">'
    f'Upload a clinical trial PDF to cross-check abstract claims against '
    f'the paper\'s own evidence sections.'
    f'</p></div>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# File upload + demo selector
# ---------------------------------------------------------------------------

upload_col, demo_col = st.columns([2, 1])

with upload_col:
    uploaded_file = st.file_uploader(
        "Upload PDF",
        type=["pdf"],
        help="Open Access papers from PubMed Central work best.",
        label_visibility="collapsed",
    )
    if uploaded_file:
        size_kb = len(uploaded_file.getvalue()) / 1024
        st.markdown(
            f'<p style="font-size:0.8rem;color:{COLORS["text_secondary"]};margin-top:0.25rem;">'
            f'<strong style="color:{COLORS["text_primary"]};">{uploaded_file.name}</strong>'
            f' &nbsp;·&nbsp; {size_kb:.0f} KB</p>',
            unsafe_allow_html=True,
        )
    st.caption("Supported format: PDF. Open Access papers from PubMed Central recommended.")

sample_options = {
    "Safe Demo — QuietDesk (Low Risk)": Path("data/demo_papers/SPIN_SENTRY_Demo_Safe.pdf"),
    "Moderate Risk Demo — FocusBoard (Overstatement)": Path("data/demo_papers/SPIN_SENTRY_Demo_Risk.pdf"),
    "High Risk Demo — StudyFlow (Inconsistent)": Path("data/demo_papers/SPIN_SENTRY_Demo_High_Risk.pdf"),
    "Clinical Trial (Kinesio Taping RCT)": Path("data/sample_paper.pdf"),
}
existing_samples = {k: v for k, v in sample_options.items() if v.exists()}

with demo_col:
    st.markdown(
        f'<p style="font-size:0.82rem;color:{COLORS["text_secondary"]}; '
        f'margin-bottom:0.4rem;">Or run a demonstration paper:</p>',
        unsafe_allow_html=True,
    )
    if existing_samples:
        selected_sample_label = st.selectbox(
            "Demo paper",
            list(existing_samples.keys()),
            label_visibility="collapsed",
        )
        sample_path = existing_samples[selected_sample_label]
        use_sample = st.button(
            "Load & Run Selected Paper",
            type="secondary", width="stretch",
        )
    else:
        sample_path = Path("data/sample_paper.pdf")
        use_sample = False
        st.caption("No demo papers found.")

# ---------------------------------------------------------------------------
# Run button
# ---------------------------------------------------------------------------

can_run = uploaded_file or use_sample
run_button = st.button(
    "Run Analysis",
    type="primary",
    disabled=not can_run, width="stretch",
)

if not can_run:
    st.markdown(
        f'<p style="font-size:0.82rem;color:{COLORS["text_muted"]};margin-top:0.5rem;">'
        f'Upload a PDF or select a demo paper to enable analysis.</p>',
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Pipeline execution
# ---------------------------------------------------------------------------

if run_button:
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            pdf_path = Path(tmp.name)
        display_name = uploaded_file.name
    else:
        pdf_path = sample_path
        display_name = sample_path.name

    st.divider()

    progress_bar = st.progress(0, text="Initializing pipeline…")
    status_text = st.empty()

    def update_progress(msg: str, frac: float):
        progress_bar.progress(min(frac, 1.0), text=msg)

    try:
        result: PipelineResult = run_pipeline(
            pdf_path=pdf_path,
            top_k=top_k,
            classifier_mode=classifier_mode,
            progress_callback=update_progress,
        )
    except Exception as exc:
        progress_bar.empty()
        st.error(f"Pipeline error: {exc}")
        st.stop()

    progress_bar.empty()
    status_text.empty()

    st.session_state["last_result"] = result
    st.session_state["last_display_name"] = display_name

    # -----------------------------------------------------------------------
    # Warnings
    # -----------------------------------------------------------------------

    if result.warnings:
        with st.expander("Pipeline Extraction Warnings", expanded=False):
            for w in result.warnings:
                st.warning(w)

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------

    st.markdown(
        f'<div class="ss-page-header" style="padding-top:0.5rem;">'
        f'<h1 class="ss-page-title">Analysis Summary</h1>'
        f'<p class="ss-page-description">{display_name}</p>'
        f'</div>',
        unsafe_allow_html=True,
    )

    summary = result.summary

    # Risk color
    risk_val = summary["spin_risk"]
    if "Low" in risk_val:
        risk_color = COLORS["green"]
        risk_clean = "Low"
    elif "Moderate" in risk_val:
        risk_color = COLORS["amber"]
        risk_clean = "Moderate"
    else:
        risk_color = COLORS["red"]
        risk_clean = "High"

    render_risk_banner(
        risk_clean,
        summary.get("empirical_flagged", summary["flagged_claims"]),
        summary["supported_claims"],
        summary["avg_we_score"],
    )

    stat_cols = st.columns(6)
    with stat_cols[0]:
        render_stat_card(str(summary["total_claims"]), "Total Statements")
    with stat_cols[1]:
        render_stat_card(
            str(summary.get("empirical_claims", summary["total_claims"])),
            "Empirical Claims",
            COLORS["cyan"],
        )
    with stat_cols[2]:
        render_stat_card(str(summary.get("protocol_statements", 0)), "Protocol / Methods", COLORS["blue"])
    with stat_cols[3]:
        render_stat_card(str(summary["supported_claims"]), "Supported", COLORS["green"])
    with stat_cols[4]:
        render_stat_card(
            str(summary.get("empirical_flagged", summary["flagged_claims"])),
            "Flagged",
            COLORS["red"],
        )
    with stat_cols[5]:
        render_stat_card(risk_clean, "Spin Risk", risk_color)

    st.markdown("")
    score_cols = st.columns(3)
    with score_cols[0]:
        render_stat_card(
            f"{summary.get('empirical_avg_we', summary['avg_we_score']):.3f}",
            "Empirical Avg WE",
        )
    with score_cols[1]:
        render_stat_card(f"{summary['avg_we_score']:.3f}", "Overall Avg WE")
    with score_cols[2]:
        render_stat_card(f"{result.elapsed_seconds:.1f}s", "Pipeline Runtime")

    st.caption(
        f"NLI model: `{result.model_status['model_name']}` · "
        f"Fallback active: `{result.model_status['fallback_active']}`"
    )

    st.divider()

    # -----------------------------------------------------------------------
    # Stage 1: Claims table
    # -----------------------------------------------------------------------

    render_section_heading("Stage 1 — HCP-BEA: Abstract Claims & Structural Roles")
    classifier_note = (
        f"Classifier: `{_active_model_name}` (AI-batch trained, provisional labels)"
        if classifier_mode == "trained"
        else "Classifier: Rule-based keyword lexicon"
    )
    st.caption(
        f"{classifier_note} · "
        "Sentences segmented via NLTK, categorized by structural role, "
        "assigned claim types via keyword lexicon/model, "
        "anchored to entities using spaCy `en_core_web_sm`."
    )

    if result.claims:
        claims_data = []
        for i, c in enumerate(result.claims, 1):
            claims_data.append({
                "#":           i,
                "Role":        getattr(c, "role", "Finding / Claim"),
                "Empirical":   "Yes" if getattr(c, "is_empirical_claim", True) else "Protocol",
                "Sentence":    c.sentence[:120] + ("…" if len(c.sentence) > 120 else ""),
                "Types":       ", ".join(c.types),
                "Zone Route":  c.zone_route,
                "Hedging":     f"{c.hedging:.2f}",
            })
        st.dataframe(pd.DataFrame(claims_data), width="stretch", hide_index=True)

        with st.expander("View Extracted Abstract Text"):
            st.text(result.abstract_text)
    else:
        st.warning("No claims extracted.")

    st.divider()

    # -----------------------------------------------------------------------
    # Stage 2: Zone map
    # -----------------------------------------------------------------------

    render_section_heading("Stage 2 — MZEM: Evidence Zones")
    st.caption(
        "Zones detected via regex + fuzzy header matching. "
        "SciBERT section classifier is not used (approximation)."
    )

    zone_cols = st.columns(4)
    zone_names = ["Results", "Harms", "Limitations", "Methods"]
    zone_accents = {
        "Results":     COLORS["green"],
        "Harms":       COLORS["red"],
        "Limitations": COLORS["amber"],
        "Methods":     COLORS["blue"],
    }

    for col, zname in zip(zone_cols, zone_names):
        zone = result.zone_map.zones.get(zname)
        accent = zone_accents[zname]
        with col:
            if zone and not zone.is_empty:
                st.markdown(f"""
                <div class="stat-card" style="border-left:3px solid {accent};">
                    <div style="font-size:0.78rem;font-weight:700;color:{accent};
                         text-transform:uppercase;letter-spacing:0.5px;
                         margin-bottom:0.35rem;">{zname}</div>
                    <div class="stat-value" style="color:{COLORS["text_primary"]};
                         font-size:1.5rem;">{zone.sentence_count}</div>
                    <div class="stat-label">sentences</div>
                    <div style="font-size:0.72rem;color:{COLORS["text_muted"]};
                         margin-top:0.3rem;">
                        {zone.match_method} · conf {zone.confidence:.2f}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="stat-card" style="border-left:3px solid {COLORS["border"]};">
                    <div style="font-size:0.78rem;font-weight:700;color:{COLORS["text_muted"]};
                         text-transform:uppercase;letter-spacing:0.5px;
                         margin-bottom:0.35rem;">{zname}</div>
                    <div style="color:{COLORS["text_muted"]};font-size:0.88rem;
                         margin-top:0.5rem;">Not detected</div>
                </div>
                """, unsafe_allow_html=True)

    for zname in zone_names:
        zone = result.zone_map.zones.get(zname)
        if zone and not zone.is_empty:
            with st.expander(f"{zname} zone — {zone.sentence_count} sentences"):
                for i, sent in enumerate(zone.sentences[:15], 1):
                    st.markdown(f"**{i}.** {sent}")
                if zone.sentence_count > 15:
                    st.caption(f"…and {zone.sentence_count - 15} more sentences.")

    st.divider()

    # -----------------------------------------------------------------------
    # Stages 3+4: Claim reports
    # -----------------------------------------------------------------------

    render_section_heading("Stages 3+4 — DC-NLI + ESWE: Claim Verification")
    st.caption(
        f"Formula: `WE = NLI(forward) × ESI × relevance(type, zone)` (Eq. 4). "
        f"Flag rule: `WE < {SPIN_FLAG_THRESHOLD}` (prototype threshold)."
    )

    if not result.reports:
        st.warning("No reports generated.")
    else:
        for i, report in enumerate(result.reports, 1):
            is_empirical = getattr(report, "is_empirical_claim", True)
            role_label = getattr(report, "role", "Finding / Claim")

            if not is_empirical:
                badge_variant = "protocol"
                badge_text = "Protocol"
                accent = COLORS["blue"]
            elif not report.spin_flag:
                badge_variant = "supported"
                badge_text = "Supported"
                accent = COLORS["green"]
            elif "Weak" in report.support_label:
                badge_variant = "weak"
                badge_text = "Weak"
                accent = COLORS["amber"]
            else:
                badge_variant = "flagged"
                badge_text = "Flagged"
                accent = COLORS["red"]

            expander_label = (
                f"[{badge_text}] [{role_label}]  "
                f"Claim {i}: {report.claim[:85]}{'…' if len(report.claim) > 85 else ''}"
            )

            with st.expander(expander_label, expanded=report.spin_flag and is_empirical):
                expl = getattr(report, "explanation", "") or \
                       f"Direct textual entailment in {report.zone_route} was {report.forward:.2f}."

                badge_html = render_badge(report.support_label, badge_variant)
                st.markdown(f"""
                <div class="spin-card" style="border-left:3px solid {accent};">
                    {badge_html}
                    <p style="margin-top:0.5rem;font-size:0.9rem;">{expl}</p>
                </div>
                """, unsafe_allow_html=True)

                col_left, col_right = st.columns([1.8, 1.2])

                with col_left:
                    st.markdown(f"**Abstract sentence:** {report.claim}")
                    st.markdown(
                        f"**Structural role:** `{role_label}` "
                        f"({'Empirical Finding' if is_empirical else 'Protocol / Design'})"
                    )
                    st.markdown(f"**Claim types:** `{'` · `'.join(report.claim_types)}`")
                    st.markdown(f"**Evidence zone:** `{report.zone_route}`")
                    hw = ', '.join(report.hedging_words) if report.hedging_words else 'none'
                    st.markdown(f"**Hedging score:** `{report.hedging:.2f}` (words: {hw})")

                with col_right:
                    m1, m2 = st.columns(2)
                    m1.metric("WE Score", f"{report.we_score:.3f}")
                    m2.metric("Forward NLI", f"{report.forward:.3f}")
                    m3, m4 = st.columns(2)
                    m3.metric("Reverse NLI", f"{report.reverse:.3f}")
                    m4.metric("Gap (1−rev)", f"{report.gap:.3f}")
                    m5, m6 = st.columns(2)
                    m5.metric("ESI", f"{report.esi:.2f}")
                    m6.metric("Relevance", f"{report.relevance:.2f}")

                if report.esi_metadata:
                    meta = report.esi_metadata
                    meta_cols = st.columns(3)
                    meta_cols[0].caption(f"Study design: `{meta.study_design}`")
                    meta_cols[1].caption(f"Sample size: `{meta.sample_size or 'unknown'}`")
                    meta_cols[2].caption(f"Analysis type: `{meta.analysis_type}`")

                if report.evidence_items:
                    best_ev = report.evidence_items[0]
                    src = best_ev.get("source_section", report.zone_route)
                    st.markdown(f"**Best evidence (rank 1 from `{src}`):**")
                    st.markdown(f"""
                    <div class="evidence-item">
                        <div class="evidence-claim">{best_ev["evidence"]}</div>
                        <div class="evidence-score">
                            <span class="score-pill">fwd {best_ev["forward"]:.3f}</span>
                            &nbsp;
                            <span class="score-pill">rev {best_ev["reverse"]:.3f}</span>
                            &nbsp;
                            <span class="score-pill">gap {best_ev["gap"]:.3f}</span>
                            &nbsp;
                            <span class="score-pill">contra {best_ev.get("fwd_contradiction", 0.0):.3f}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    if len(report.evidence_items) > 1:
                        with st.expander(f"View {len(report.evidence_items) - 1} more evidence candidates"):
                            for ev in report.evidence_items[1:]:
                                st.markdown(
                                    f"**Rank {ev['rank']}** "
                                    f"(fwd: `{ev['forward']:.3f}`, gap: `{ev['gap']:.3f}`): "
                                    f"{ev['evidence']}"
                                )
                else:
                    st.warning(f"No evidence found in `{report.zone_route}` zone.")

                st.caption(f"Model: `{report.model_used}`")

    st.divider()

    # -----------------------------------------------------------------------
    # Export
    # -----------------------------------------------------------------------

    render_section_heading("Export Report")
    dl_cols = st.columns([1, 1, 2])
    with dl_cols[0]:
        report_dict = result.to_dict()
        report_json = json.dumps(report_dict, indent=2, ensure_ascii=False)
        st.download_button(
            label="Download JSON Report",
            data=report_json,
            file_name=f"spin_sentry_report_{int(time.time())}.json",
            mime="application/json", width="stretch",
        )
    with dl_cols[1]:
        st.caption(
            "Full structured report with all claims, scores, and evidence passages. "
            "Includes screening notice."
        )

    st.divider()

    # -----------------------------------------------------------------------
    # Disclaimer
    # -----------------------------------------------------------------------

    render_disclaimer(
        "<strong>About reported metrics:</strong> The paper reports 89% claim-type "
        "accuracy (n=20), Harms-zone F1=0.93 (n=100), Precision@5≈81% (in progress). "
        "These figures are from the paper authors' evaluation and are "
        "<strong>not reproduced by this implementation</strong>. "
        "This prototype uses rule-based approximations for claim typing and zone "
        "segmentation, and a general-domain NLI model without biomedical fine-tuning."
    )
