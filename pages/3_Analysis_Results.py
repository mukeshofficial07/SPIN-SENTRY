"""
pages/3_Analysis_Results.py
----------------------------
Displays stored analysis results from session state.
Multi-tab inspection: summary, claims table, evidence explorer, JSON export.
"""

import sys
import time
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd

from components.theme import (
    inject_custom_css, render_sidebar_nav, render_stat_card,
    render_badge, render_disclaimer, render_section_heading,
    render_risk_banner, COLORS,
)
from pipeline import PipelineResult

inject_custom_css()
render_sidebar_nav()

# ---------------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------------

st.markdown(
    f'<div class="ss-page-header">'
    f'<h1 class="ss-page-title">Analysis Results</h1>'
    f'<p class="ss-page-description">'
    f'Detailed inspection of pipeline findings, evidence alignment, and diagnostic metrics.'
    f'</p></div>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session state check
# ---------------------------------------------------------------------------

result = st.session_state.get("last_result")

if not result:
    st.info(
        "No analysis results in this session. "
        "Run an analysis from **Research Workspace** first."
    )
    st.markdown(f"""
    <div class="spin-card" style="border-left:3px solid {COLORS["blue"]};margin-top:1rem;">
        <h4>How to generate results</h4>
        <ol style="color:{COLORS["text_secondary"]};margin-left:1.2rem;line-height:2;">
            <li>Navigate to <strong>Research Workspace</strong> in the sidebar.</li>
            <li>Upload a clinical trial PDF or load a demo paper.</li>
            <li>Click <strong>Run Analysis</strong>.</li>
            <li>Return here to inspect the diagnostic report and evidence explorer.</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ---------------------------------------------------------------------------
# Result context
# ---------------------------------------------------------------------------

display_name = st.session_state.get(
    "last_display_name",
    getattr(result, "title", None) or "Clinical Trial Paper",
)
summary = getattr(result, "summary", {}) or {}

st.markdown(
    f'<p style="font-size:0.9rem;color:{COLORS["text_secondary"]};margin-top:-0.5rem;'
    f'margin-bottom:1.5rem;">Inspecting: '
    f'<strong style="color:{COLORS["text_primary"]};">{display_name}</strong></p>',
    unsafe_allow_html=True,
)

if getattr(result, "warnings", None):
    with st.expander("Pipeline Extraction Warnings", expanded=False):
        for w in result.warnings:
            st.warning(w)

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_summary, tab_table, tab_evidence, tab_json = st.tabs([
    "Summary",
    "Claims Table",
    "Evidence Explorer",
    "JSON Export",
])

# ===========================================================================
# TAB 1: Summary
# ===========================================================================

with tab_summary:
    st.markdown("")

    total_claims    = summary.get("total_claims", len(getattr(result, "claims", [])))
    empirical_claims = summary.get("empirical_claims", 0)
    protocol_stmts  = summary.get("protocol_statements", 0)
    supported_claims = summary.get("supported_claims", 0)
    flagged_claims  = summary.get("flagged_claims", 0)
    empirical_flagged = summary.get("empirical_flagged", flagged_claims)
    avg_we           = summary.get("avg_we_score", 0.0)
    empirical_avg_we = summary.get("empirical_avg_we", avg_we)
    spin_risk        = summary.get("spin_risk", "Unknown")
    flag_rate        = summary.get("flag_rate", 0.0)

    if "Low" in spin_risk:
        risk_color = COLORS["green"]
        risk_clean = "Low"
    elif "Moderate" in spin_risk:
        risk_color = COLORS["amber"]
        risk_clean = "Moderate"
    else:
        risk_color = COLORS["red"]
        risk_clean = "High"

    render_risk_banner(risk_clean, empirical_flagged, supported_claims, avg_we)

    render_section_heading("Statement Counts")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        render_stat_card(str(total_claims), "Total Statements")
    with c2:
        render_stat_card(str(empirical_claims), "Empirical Claims", COLORS["cyan"])
    with c3:
        render_stat_card(str(protocol_stmts), "Protocol / Design", COLORS["blue"])
    with c4:
        render_stat_card(str(supported_claims), "Supported", COLORS["green"])
    with c5:
        render_stat_card(str(empirical_flagged), "Flagged", COLORS["red"])
    with c6:
        render_stat_card(risk_clean, "Spin Risk", risk_color)

    st.markdown("")
    render_section_heading("Weighted Entailment & Performance")
    s1, s2, s3, s4, s5 = st.columns(5)
    with s1:
        render_stat_card(f"{flag_rate * 100:.1f}%", "Empirical Flag Rate",
                         COLORS["amber"] if flag_rate > 0 else COLORS["green"])
    with s2:
        render_stat_card(f"{empirical_avg_we:.3f}", "Empirical Avg WE")
    with s3:
        render_stat_card(f"{avg_we:.3f}", "Overall Avg WE")
    with s4:
        model_status = getattr(result, "model_status", {}) or {}
        render_stat_card(
            "Neural" if not model_status.get("fallback_active") else "Fallback",
            "NLI Mode",
            COLORS["green"] if not model_status.get("fallback_active") else COLORS["amber"],
        )
    with s5:
        render_stat_card(f"{result.elapsed_seconds:.1f}s", "Runtime")

    st.divider()

    col_info, col_model = st.columns([1.6, 1.0])

    with col_info:
        st.markdown(f"""
        <div class="spin-card" style="border-left:3px solid {COLORS["blue"]};">
            <h4>Pipeline Verification Methodology</h4>
            <p>
                Abstract sentences are parsed into discrete statements, tagged by structural
                role (Objective / Methods / Finding / Conclusion), and routed to corresponding
                trial evidence sections. Each claim is evaluated using
                <strong>Weighted Entailment</strong>:
            </p>
            <p style="font-family:'SFMono-Regular',Consolas,monospace;color:{COLORS["cyan"]};
                 background:{COLORS["bg_secondary"]};padding:0.4rem 0.8rem;border-radius:4px;
                 font-size:0.85rem;">
                WE(c, e) = NLI_forward(c, e) × ESI(e) × relevance(type, zone)
            </p>
            <p style="font-size:0.82rem;color:{COLORS["text_muted"]};margin-top:0.5rem;margin-bottom:0;">
                Claims with WE &lt; 0.5 are flagged for reviewer attention.
                A low WE score indicates limited direct textual support,
                not definitive evidence of intentional spin.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col_model:
        st.markdown(f"""
        <div class="spin-card" style="border-left:3px solid {COLORS["violet"]};">
            <h4>Inference Diagnostics</h4>
            <p><strong>NLI Model:</strong><br>
               <code>{model_status.get("model_name", "cross-encoder/nli-MiniLM2-L6-H768")}</code>
            </p>
            <p><strong>Model Loaded:</strong>
               {"Yes" if model_status.get("loaded") else "Fallback"}</p>
            <p><strong>Fallback Active:</strong>
               {"Yes (keyword-overlap)" if model_status.get("fallback_active") else "No (neural cross-encoder)"}</p>
            <p style="margin-bottom:0;"><strong>Execution Time:</strong>
               {result.elapsed_seconds:.2f}s</p>
        </div>
        """, unsafe_allow_html=True)

    render_disclaimer(
        "<strong>Screening signals only.</strong> Automated spin flags indicate textual "
        "divergence between abstract phrasing and full-text evidence sections. "
        "They are heuristic signals for reviewer triage, not proof of misconduct or clinical "
        "invalidity. Published paper metrics (89% / F1=0.93 / 81%) are from the original "
        "paper's evaluation and are <strong>not reproduced by this prototype</strong>. "
        "Rule-based approximations and a general-domain NLI model are used throughout."
    )

# ===========================================================================
# TAB 2: Claims Table
# ===========================================================================

with tab_table:
    st.markdown("")
    render_section_heading("All Abstract Statements — Stage 1 + Stage 4 Output")
    st.caption("Complete breakdown of extracted claims, structural roles, and verification scores.")

    reports = getattr(result, "reports", [])

    if not reports:
        st.warning("No claim reports available.")
    else:
        claims_rows = []
        for i, rep in enumerate(reports, 1):
            is_empirical = getattr(rep, "is_empirical_claim", True)
            types_str = (
                ", ".join(rep.claim_types)
                if isinstance(rep.claim_types, (list, tuple))
                else str(rep.claim_types or "")
            )
            claims_rows.append({
                "#":             i,
                "Role":          getattr(rep, "role", "Finding / Claim"),
                "Empirical":     "Yes" if is_empirical else "Protocol",
                "Sentence":      rep.claim,
                "Types":         types_str or "General",
                "Zone":          rep.zone_route,
                "WE":            round(float(rep.we_score), 3),
                "Support":       rep.support_label,
                "Fwd NLI":       round(float(rep.forward), 3),
                "Hedging":       round(float(rep.hedging), 2),
            })

        df_claims = pd.DataFrame(claims_rows)

        f_col1, f_col2, f_col3 = st.columns([1, 1, 1])
        with f_col1:
            label_filter = st.multiselect(
                "Filter by support label",
                options=sorted(df_claims["Support"].unique()),
                default=[],
                placeholder="All labels",
            )
        with f_col2:
            role_filter = st.multiselect(
                "Filter by structural role",
                options=sorted(df_claims["Role"].unique()),
                default=[],
                placeholder="All roles",
            )
        with f_col3:
            emp_filter = st.multiselect(
                "Filter by empirical status",
                options=sorted(df_claims["Empirical"].unique()),
                default=[],
                placeholder="All",
            )

        filtered_df = df_claims.copy()
        if label_filter:
            filtered_df = filtered_df[filtered_df["Support"].isin(label_filter)]
        if role_filter:
            filtered_df = filtered_df[filtered_df["Role"].isin(role_filter)]
        if emp_filter:
            filtered_df = filtered_df[filtered_df["Empirical"].isin(emp_filter)]

        st.caption(f"Showing **{len(filtered_df)}** of **{len(df_claims)}** claims.")

        st.dataframe(
            filtered_df,
            width="stretch", hide_index=True,
            column_config={
                "#":        st.column_config.NumberColumn("#", width="small"),
                "Role":     st.column_config.TextColumn("Role", width="medium"),
                "Empirical": st.column_config.TextColumn("Empirical", width="small"),
                "Sentence": st.column_config.TextColumn("Sentence", width="large"),
                "Types":    st.column_config.TextColumn("Types", width="medium"),
                "Zone":     st.column_config.TextColumn("Zone", width="small"),
                "WE":       st.column_config.NumberColumn("WE", format="%.3f", width="small"),
                "Support":  st.column_config.TextColumn("Support", width="medium"),
                "Fwd NLI":  st.column_config.NumberColumn("Fwd NLI", format="%.3f", width="small"),
                "Hedging":  st.column_config.NumberColumn("Hedging", format="%.2f", width="small"),
            },
        )

        csv_data = filtered_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Claims as CSV",
            data=csv_data,
            file_name=f"spin_sentry_claims_{int(time.time())}.csv",
            mime="text/csv",
        )

# ===========================================================================
# TAB 3: Evidence Explorer
# ===========================================================================

with tab_evidence:
    st.markdown("")
    render_section_heading("Claim–Evidence Alignment Explorer")
    st.caption(
        "Inspect directional entailment scores and top-3 evidence passages per claim. "
        "Flagged empirical claims are expanded by default."
    )

    reports = getattr(result, "reports", [])

    if not reports:
        st.warning("No reports generated.")
    else:
        for i, rep in enumerate(reports, 1):
            is_empirical = getattr(rep, "is_empirical_claim", True)
            role_label   = getattr(rep, "role", "Finding / Claim")

            if not is_empirical:
                badge_variant = "protocol"
                border_color  = COLORS["blue"]
            elif not rep.spin_flag:
                badge_variant = "supported"
                border_color  = COLORS["green"]
            elif "Weak" in rep.support_label:
                badge_variant = "weak"
                border_color  = COLORS["amber"]
            else:
                badge_variant = "flagged"
                border_color  = COLORS["red"]

            expander_label = (
                f"[{rep.support_label}]  Claim {i}: "
                f"{rep.claim[:80]}{'…' if len(rep.claim) > 80 else ''}"
            )
            should_expand = (i == 1) or (rep.spin_flag and is_empirical)

            with st.expander(expander_label, expanded=should_expand):
                badge_html = render_badge(rep.support_label, badge_variant)
                expl = getattr(rep, "explanation", "") or \
                       f"Forward entailment in {rep.zone_route} was {rep.forward:.2f}."

                st.markdown(f"""
                <div class="spin-card" style="border-left:3px solid {border_color};">
                    <div style="display:flex;justify-content:space-between;
                         align-items:center;margin-bottom:0.4rem;">
                        <div>{badge_html}</div>
                        <span style="font-size:0.78rem;color:{COLORS["text_muted"]};">
                            Zone: <strong>{rep.zone_route}</strong>
                        </span>
                    </div>
                    <p style="font-size:0.9rem;color:{COLORS["text_primary"]};
                         margin-bottom:0;">{expl}</p>
                </div>
                """, unsafe_allow_html=True)

                c_left, c_right = st.columns([1.8, 1.2])

                with c_left:
                    st.markdown("**Abstract sentence:**")
                    st.markdown(f"""
                    <div class="evidence-item" style="color:{COLORS["text_primary"]};">
                        {rep.claim}
                    </div>
                    """, unsafe_allow_html=True)
                    types_list = (
                        rep.claim_types
                        if isinstance(rep.claim_types, (list, tuple))
                        else [str(rep.claim_types)]
                    )
                    st.markdown(f"**Role:** `{role_label}` · **Types:** `{'` · `'.join(types_list)}`")
                    hedging_detail = (
                        f" ({', '.join(rep.hedging_words)})" if getattr(rep, "hedging_words", None) else ""
                    )
                    st.markdown(f"**Hedging:** `{rep.hedging:.2f}`{hedging_detail}")

                with c_right:
                    m1, m2 = st.columns(2)
                    m1.metric("WE Score",    f"{rep.we_score:.3f}")
                    m2.metric("Forward NLI", f"{rep.forward:.3f}")
                    m3, m4 = st.columns(2)
                    m3.metric("Reverse NLI", f"{rep.reverse:.3f}")
                    m4.metric("Gap (1−rev)", f"{rep.gap:.3f}")
                    m5, m6 = st.columns(2)
                    m5.metric("ESI",         f"{rep.esi:.2f}")
                    m6.metric("Relevance",   f"{rep.relevance:.2f}")

                st.markdown("")
                render_section_heading(f"Top Evidence from '{rep.zone_route}'")

                evidence_items = getattr(rep, "evidence_items", []) or []
                top3 = evidence_items[:3]

                if not top3:
                    st.warning(f"No evidence passages extracted from `{rep.zone_route}`.")
                else:
                    for rank_idx, ev in enumerate(top3, 1):
                        ev_text   = ev.get("evidence", "")
                        ev_fwd    = ev.get("forward", 0.0)
                        ev_rev    = ev.get("reverse", 0.0)
                        ev_gap    = ev.get("gap", 0.0)
                        ev_contra = ev.get("fwd_contradiction", 0.0)
                        src       = ev.get("source_section", rep.zone_route)

                        rank_colors = [COLORS["blue"], COLORS["cyan"], COLORS["text_muted"]]
                        rank_accent = rank_colors[rank_idx - 1]

                        st.markdown(f"""
                        <div style="background:{COLORS["bg_card"]};
                             border:1px solid {COLORS["border"]};
                             border-left:3px solid {rank_accent};
                             border-radius:6px;padding:0.85rem 1rem;
                             margin-bottom:0.6rem;">
                            <div style="display:flex;justify-content:space-between;
                                 align-items:center;margin-bottom:0.4rem;flex-wrap:wrap;gap:0.4rem;">
                                <span style="font-size:0.75rem;font-weight:700;
                                     color:{rank_accent};text-transform:uppercase;
                                     letter-spacing:0.5px;">
                                    Rank {rank_idx} · {src}
                                </span>
                                <div style="font-size:0.78rem;color:{COLORS["text_secondary"]};">
                                    <span class="score-pill">fwd {ev_fwd:.3f}</span>
                                    &nbsp;
                                    <span class="score-pill">rev {ev_rev:.3f}</span>
                                    &nbsp;
                                    <span class="score-pill">gap {ev_gap:.3f}</span>
                                    &nbsp;
                                    <span class="score-pill">contra {ev_contra:.3f}</span>
                                </div>
                            </div>
                            <div style="color:{COLORS["text_secondary"]};font-size:0.88rem;
                                 line-height:1.55;">{ev_text}</div>
                        </div>
                        """, unsafe_allow_html=True)

                st.caption(f"Inference model: `{rep.model_used}`")

# ===========================================================================
# TAB 4: JSON Export
# ===========================================================================

with tab_json:
    st.markdown("")
    render_section_heading("Machine-Readable JSON Export")
    st.caption(
        "Full structured output with all pipeline stages, claims, scores, "
        "and evidence passages."
    )

    try:
        report_dict = result.to_dict()
    except Exception:
        report_dict = {
            "title":           getattr(result, "title", ""),
            "pdf_path":        getattr(result, "pdf_path", ""),
            "summary":         summary,
            "reports":         [getattr(r, "to_dict", lambda: vars(r))() for r in getattr(result, "reports", [])],
            "warnings":        getattr(result, "warnings", []),
            "elapsed_seconds": getattr(result, "elapsed_seconds", 0.0),
            "model_status":    getattr(result, "model_status", {}),
        }

    report_json = json.dumps(report_dict, indent=2, ensure_ascii=False, default=str)

    col_dl, col_desc = st.columns([1, 2])
    with col_dl:
        st.download_button(
            label="Download JSON Report",
            data=report_json,
            file_name=f"spin_sentry_analysis_{int(time.time())}.json",
            mime="application/json",
            type="primary", width="stretch",
        )
    with col_desc:
        st.caption(
            "Contains all segmented abstract statements, mapped evidence zones, "
            "directional entailment metrics, ESI weights, and screening notice."
        )

    st.code(report_json, language="json")
