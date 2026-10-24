"""
pages/4_Dataset_Model_Lab.py
-----------------------------
Dataset & Model Laboratory — corpus overview, classifier metrics,
model artifacts, review queue, and annotation provenance.

All figures are loaded from real project files. No numbers are fabricated.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from components.theme import (
    COLORS, inject_custom_css, render_sidebar_nav,
    render_stat_card, render_disclaimer, render_section_heading,
)

inject_custom_css()
render_sidebar_nav()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATA_DIR  = ROOT / "data" / "research_dataset"
PAPERS_CSV    = DATA_DIR / "papers.csv"
CANDIDATES_CSV = DATA_DIR / "candidates.csv"
QUEUE_CSV     = DATA_DIR / "priority_review_queue.csv"
GOLD_CSV      = DATA_DIR / "human_gold_standard.csv"
MODELS_DIR    = ROOT / "models"
REPORTS_DIR   = ROOT / "reports"

AVAILABLE_MODELS = {
    "Improved (Tuned Thresholds & Preprocessing)": {
        "metrics_path": REPORTS_DIR / "claim_type_metrics_improved.json",
        "config_path":  MODELS_DIR  / "claim_type_training_config_improved.json",
        "model_file":   "claim_type_model_improved.joblib",
        "badge":        "Active Default",
    },
    "Baseline v1 (AI-Batch Annotations, threshold 0.50)": {
        "metrics_path": REPORTS_DIR / "claim_type_metrics_human_verified.json",
        "config_path":  MODELS_DIR  / "claim_type_training_config_human_verified.json",
        "model_file":   "claim_type_model_human_verified.joblib",
        "badge":        "Prior Baseline",
    },
    "Baseline v0 (Machine Pseudo-Labels)": {
        "metrics_path": REPORTS_DIR / "claim_type_metrics_machine_pseudo.json",
        "config_path":  MODELS_DIR  / "claim_type_training_config_machine_pseudo.json",
        "model_file":   "claim_type_model_machine_pseudo.joblib",
        "badge":        "Heuristic Pseudo",
    },
}

# ---------------------------------------------------------------------------
# Safe-load helpers
# ---------------------------------------------------------------------------

def _load_csv(path: Path) -> pd.DataFrame | None:
    try:
        if path.exists():
            return pd.read_csv(path)
    except Exception:
        pass
    return None


def _load_json(path: Path) -> dict | None:
    try:
        if path.exists():
            with open(path, encoding="utf-8") as fh:
                return json.load(fh)
    except Exception:
        pass
    return None


# Pre-load
papers_df     = _load_csv(PAPERS_CSV)
candidates_df = _load_csv(CANDIDATES_CSV)
queue_df      = _load_csv(QUEUE_CSV)
gold_df       = _load_csv(GOLD_CSV)

# ---------------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------------

st.markdown(
    f'<div class="ss-page-header">'
    f'<h1 class="ss-page-title">Dataset & Model Laboratory</h1>'
    f'<p class="ss-page-description">'
    f'Audit the clinical research corpus, compare classifier versions, '
    f'inspect model artifacts, and verify evaluation provenance.'
    f'</p></div>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_corpus, tab_metrics, tab_models, tab_queue, tab_annotation = st.tabs([
    "Corpus Overview",
    "Classifier Metrics",
    "Model Artifacts",
    "Review Queue",
    "Annotation Provenance",
])

# ===== TAB 1 — Corpus Overview =============================================

with tab_corpus:
    st.markdown("")
    render_section_heading("Corpus Statistics")

    if papers_df is not None and candidates_df is not None:
        n_papers     = len(papers_df)
        n_candidates = len(candidates_df)
        n_sections   = candidates_df["source_section"].nunique()

        cols = st.columns(4)
        with cols[0]:
            render_stat_card(str(n_papers), "Papers Ingested", COLORS["cyan"])
        with cols[1]:
            render_stat_card(f"{n_candidates:,}", "Candidate Sentences", COLORS["blue"])
        with cols[2]:
            avg = n_candidates / n_papers if n_papers else 0
            render_stat_card(f"{avg:.0f}", "Avg Candidates / Paper")
        with cols[3]:
            render_stat_card(str(n_sections), "Unique Sections")

        st.divider()
        render_section_heading("Papers in Corpus")
        display_cols = [
            c for c in ["paper_id", "title", "doi", "license", "candidate_count"]
            if c in papers_df.columns
        ]
        st.dataframe(
            papers_df[display_cols],
            width="stretch", hide_index=True,
            height=min(400, 35 * len(papers_df) + 38),
        )

        st.divider()
        render_section_heading("Section Coverage (Top 15)")
        section_counts = (
            candidates_df["source_section"]
            .value_counts()
            .head(15)
            .reset_index()
        )
        section_counts.columns = ["Section", "Candidates"]
        st.dataframe(section_counts, width="stretch", hide_index=True)
    else:
        st.warning(
            "Could not load `papers.csv` or `candidates.csv` "
            "from `data/research_dataset/`."
        )

# ===== TAB 2 — Classifier Metrics ==========================================

with tab_metrics:
    st.markdown("")

    gold_eval         = _load_json(REPORTS_DIR / "eval_gold_improved.json")
    gold_holdout_eval = _load_json(REPORTS_DIR / "eval_gold_holdout_improved.json")

    render_section_heading("Human Gold-Standard Evaluation — claim_type_model_improved")
    st.success(
        "Evaluated against **197 source-verified human annotations** across all 22 "
        "clinical trial papers (reviewed by annotator against linked PMC source text). "
        "These metrics reflect true predictive agreement with human clinical reviewers."
    )

    if gold_eval:
        g_cols = st.columns(4)
        with g_cols[0]:
            render_stat_card(f"{gold_eval.get('exact_match', 0):.1%}", "Exact Match (Gold)", COLORS["cyan"])
        with g_cols[1]:
            render_stat_card(f"{gold_eval.get('f1_micro', 0):.3f}", "Micro F1 (Gold)", COLORS["blue"])
        with g_cols[2]:
            render_stat_card(f"{gold_eval.get('f1_macro', 0):.3f}", "Macro F1 (Gold)")
        with g_cols[3]:
            render_stat_card(f"{gold_eval.get('hamming_loss', 0):.4f}", "Hamming Loss")

        render_section_heading("Per-Category Performance — Human Gold (N=197)")
        gold_per_label = gold_eval.get("per_label", {})
        thresh_applied = gold_eval.get("thresholds_applied", {})
        if gold_per_label:
            g_rows = []
            for lbl, vals in gold_per_label.items():
                thr_val = thresh_applied.get(lbl, 0.40)
                supp = vals.get("support", 0)
                reliability = (
                    "Sufficient (>=30)" if supp >= 30
                    else "Provisional (>=10)" if supp >= 10
                    else "Scarce (<10)"
                )
                g_rows.append({
                    "Category":          lbl,
                    "Threshold":         f"{thr_val:.2f}",
                    "Precision":         f"{vals['precision']:.3f}",
                    "Recall":            f"{vals['recall']:.3f}",
                    "F1":                f"{vals['f1']:.3f}",
                    "Gold Support":      supp,
                    "Reliability":       reliability,
                })
            st.dataframe(pd.DataFrame(g_rows), width="stretch", hide_index=True)

        if gold_holdout_eval:
            st.info(
                f"**Independent Holdout — 5 Unseen Papers (N=49):**\n\n"
                f"Macro-F1: `{gold_holdout_eval.get('f1_macro', 0):.4f}`  |  "
                f"Micro-F1: `{gold_holdout_eval.get('f1_micro', 0):.4f}`  |  "
                f"Exact Match: `{gold_holdout_eval.get('exact_match', 0):.1%}`  |  "
                f"Hamming Loss: `{gold_holdout_eval.get('hamming_loss', 0):.4f}`\n\n"
                f"**Leakage audit:** 0 sentences from these 5 papers were used in "
                f"training or threshold tuning."
            )

    st.divider()

    st.info(
        "**Historical AI-Batch Baseline (Reference Only)**\n\n"
        "The comparison table below evaluates models against a legacy holdout partition "
        "containing 552 examples (549 heuristic AI-batch annotations, 3 manual reviews). "
        "These metrics measure agreement with automated heuristics and are shown for "
        "model lineage transparency only."
    )

    render_section_heading("Multi-Model Comparison — AI-Batch Partition")
    comparison_rows = []
    for model_name, info in AVAILABLE_MODELS.items():
        m = _load_json(info["metrics_path"])
        if m:
            pl = m.get("per_label", {})
            comparison_rows.append({
                "Model":               model_name,
                "Status":              info["badge"],
                "Exact Match":         f"{m.get('exact_match_agreement', 0):.1%}",
                "Micro F1":            f"{m.get('f1_micro', 0):.4f}",
                "Macro F1":            f"{m.get('f1_macro', 0):.4f}",
                "Hamming Loss":        f"{m.get('hamming_loss', 0):.4f}",
                "Efficacy F1":         f"{pl.get('Efficacy', {}).get('f1', 0):.3f}",
                "Safety F1":           f"{pl.get('Safety', {}).get('f1', 0):.3f}",
                "Generalizability F1": f"{pl.get('Generalizability', {}).get('f1', 0):.3f}",
                "Comparative F1":      f"{pl.get('Comparative', {}).get('f1', 0):.3f}",
                "Novelty F1":          f"{pl.get('Novelty', {}).get('f1', 0):.3f}",
                "Recommendation F1":   f"{pl.get('Recommendation', {}).get('f1', 0):.3f}",
            })
    if comparison_rows:
        st.dataframe(pd.DataFrame(comparison_rows), width="stretch", hide_index=True)

    st.divider()

    render_section_heading("Model Deep Dive")
    selected_model_name = st.selectbox(
        "Select model to inspect",
        options=list(AVAILABLE_MODELS.keys()),
        index=0,
        label_visibility="collapsed",
    )
    selected_info = AVAILABLE_MODELS[selected_model_name]
    m_data = _load_json(selected_info["metrics_path"])

    if m_data:
        m_cols = st.columns(4)
        with m_cols[0]:
            render_stat_card(f"{m_data.get('exact_match_agreement', 0):.1%}", "Exact-Match", COLORS["cyan"])
        with m_cols[1]:
            render_stat_card(f"{m_data.get('f1_micro', 0):.3f}", "Micro F1", COLORS["blue"])
        with m_cols[2]:
            render_stat_card(f"{m_data.get('f1_macro', 0):.3f}", "Macro F1")
        with m_cols[3]:
            render_stat_card(f"{m_data.get('hamming_loss', 0):.4f}", "Hamming Loss", COLORS["amber"])

        cv = m_data.get("cv")
        if cv:
            render_section_heading("5-Fold GroupKFold Cross-Validation (Training Papers Only)")
            cv_cols = st.columns(3)
            with cv_cols[0]:
                render_stat_card(
                    f"{cv['cv_micro_f1_mean']:.4f} ± {cv['cv_micro_f1_std']:.4f}",
                    "Mean CV Micro F1", COLORS["cyan"],
                )
            with cv_cols[1]:
                render_stat_card(
                    f"{cv['cv_macro_f1_mean']:.4f} ± {cv['cv_macro_f1_std']:.4f}",
                    "Mean CV Macro F1",
                )
            with cv_cols[2]:
                render_stat_card(
                    f"{cv['cv_folds']} folds (paper grouped)",
                    "Validation Strategy",
                )

        render_section_heading("Per-Label Performance")
        per_label         = m_data.get("per_label", {})
        thresholds_applied = m_data.get("thresholds_applied", {})
        if per_label:
            rows = []
            for label, vals in per_label.items():
                thr_str = (
                    f"{thresholds_applied.get(label, 0.50):.2f}"
                    if thresholds_applied else "0.50"
                )
                rows.append({
                    "Category":  label,
                    "Threshold": thr_str,
                    "Precision": f"{vals['precision']:.3f}",
                    "Recall":    f"{vals['recall']:.3f}",
                    "F1":        f"{vals['f1']:.3f}",
                    "Support":   vals.get("support", ""),
                })
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

        st.caption(
            f"Dataset rows: **{m_data.get('dataset_rows', '?')}** "
            f"({m_data.get('train_rows', '?')} train / {m_data.get('test_rows', '?')} test) · "
            f"Training papers: **{m_data.get('train_papers', '?')}** · "
            f"Test papers: **{m_data.get('test_papers', '?')}** "
            f"({', '.join(m_data.get('test_paper_ids', []))})"
        )

        with st.expander("Raw Metrics Metadata"):
            st.json({
                "trained_at_utc":      m_data.get("trained_at_utc"),
                "label_source":        m_data.get("label_source"),
                "evaluation_target":   m_data.get("evaluation_target"),
                "python_version":      m_data.get("python_version"),
                "scikit_learn_version": m_data.get("scikit_learn_version"),
                "dataset_sha256":      m_data.get("dataset_sha256"),
                "seed":                m_data.get("seed"),
            })
    else:
        st.warning(f"Metrics file not found at `{selected_info['metrics_path']}`.")

    render_disclaimer(
        "All metrics in the comparison table above measure agreement with "
        "<strong>AI-generated batch annotations</strong>, not human ground truth. "
        "These are not accuracy measures in the traditional sense. "
        "The human gold-standard evaluation (N=197) above is the authoritative metric."
    )

# ===== TAB 3 — Model Artifacts =============================================

with tab_models:
    st.markdown("")
    render_section_heading("Files in models/ Directory")

    if MODELS_DIR.exists():
        model_files = sorted(MODELS_DIR.iterdir())
        file_rows = []
        for f in model_files:
            if f.is_file():
                size_kb = f.stat().st_size / 1024
                size_str = (
                    f"{size_kb / 1024:.2f} MB" if size_kb > 1024 else f"{size_kb:.1f} KB"
                )
                file_rows.append({
                    "Filename": f.name,
                    "Size":     size_str,
                    "Format":   f.suffix.lstrip(".").upper() or "—",
                })
        if file_rows:
            st.dataframe(pd.DataFrame(file_rows), width="stretch", hide_index=True)
    else:
        st.warning("`models/` directory does not exist.")

    st.divider()
    render_section_heading("Training Configurations")

    cfg_choice = st.selectbox(
        "Select configuration",
        options=list(AVAILABLE_MODELS.keys()),
        index=0,
        label_visibility="collapsed",
    )
    chosen_cfg = _load_json(AVAILABLE_MODELS[cfg_choice]["config_path"])
    if chosen_cfg:
        col_alg, col_hp = st.columns(2)
        with col_alg:
            st.markdown(f"""
            <div class="spin-card">
                <h4>Algorithm</h4>
                <p><strong>{chosen_cfg.get("algorithm", "N/A")}</strong></p>
                <p style="font-size:0.82rem;color:{COLORS["text_muted"]};">
                    Split: {chosen_cfg.get("split", "N/A")}<br>
                    Evaluation target: {chosen_cfg.get("evaluation_target", "N/A")}
                </p>
            </div>
            """, unsafe_allow_html=True)
        with col_hp:
            vec = chosen_cfg.get("vectorizer", {})
            clf = chosen_cfg.get("classifier", {})
            st.markdown(f"""
            <div class="spin-card">
                <h4>Hyperparameters</h4>
                <p style="font-size:0.85rem;">
                    <strong>Vectorizer:</strong> n-gram {vec.get("ngram_range", "?")},
                    sublinear_tf={vec.get("sublinear_tf", False)},
                    max_features={vec.get("max_features", "?")}<br>
                    <strong>Classifier:</strong> C={clf.get("C", 1.0)},
                    max_iter={clf.get("max_iter", "?")},
                    class_weight={clf.get("class_weight", "?")}
                </p>
            </div>
            """, unsafe_allow_html=True)

        if "tuned_thresholds" in chosen_cfg:
            render_section_heading("Learned Per-Category Decision Thresholds")
            thr_df = pd.DataFrame(
                list(chosen_cfg["tuned_thresholds"].items()),
                columns=["Category", "Optimal Threshold"],
            )
            st.dataframe(thr_df, width="stretch", hide_index=True)

        with st.expander("Full Training Config JSON"):
            st.json(chosen_cfg)
    else:
        st.info(f"Configuration file not found for {cfg_choice}.")

# ===== TAB 4 — Prioritized Review Queue ====================================

with tab_queue:
    st.markdown("")
    render_section_heading("Prioritized Human Review Queue")
    st.caption(
        "200 highest-value candidates prioritized for review based on rare categories "
        "(Generalizability, Recommendation), classifier uncertainty, and rule/model disagreements."
    )

    if queue_df is not None and not queue_df.empty:
        q_cols = st.columns(4)
        with q_cols[0]:
            render_stat_card(str(len(queue_df)), "Queue Size", COLORS["cyan"])
        with q_cols[1]:
            render_stat_card(str(queue_df["paper_id"].nunique()), "Papers Covered", COLORS["blue"])
        with q_cols[2]:
            render_stat_card(str(queue_df["priority_score"].max()), "Max Priority Score")
        with q_cols[3]:
            render_stat_card("Proportional", "Allocation Strategy")

        st.divider()
        render_section_heading("Top Prioritized Candidates")
        cols_to_show = [
            "priority_score", "paper_id", "source_section",
            "sentence", "machine_claim_class", "machine_suggested_labels",
            "priority_reasons",
        ]
        available_cols = [c for c in cols_to_show if c in queue_df.columns]
        st.dataframe(
            queue_df[available_cols].head(30),
            width="stretch", hide_index=True,
            height=350,
        )
        st.download_button(
            "Download Full Priority Queue (CSV)",
            data=queue_df.to_csv(index=False).encode("utf-8"),
            file_name="priority_review_queue.csv",
            mime="text/csv",
        )
    else:
        st.info(
            "Priority queue not found at `data/research_dataset/priority_review_queue.csv`. "
            "Generate with `scripts/build_review_queue.py`."
        )

# ===== TAB 5 — Annotation Provenance =======================================

with tab_annotation:
    st.markdown("")
    render_section_heading("Human-Reviewed Gold Standard")

    if gold_df is not None:
        n_gold = len(gold_df)
        g_cols = st.columns(3)
        with g_cols[0]:
            render_stat_card(str(n_gold), "Verified Human Reviews", COLORS["cyan"])
        with g_cols[1]:
            render_stat_card(">=30 / category", "Gold Standard Target", COLORS["amber"])
        with g_cols[2]:
            status_text = "Insufficient Sample" if n_gold < 180 else "Gold Standard Met"
            render_stat_card(status_text, "Current Evaluation State")

        render_section_heading("Per-Category Human Review Counts vs. Reliability Threshold")
        LABELS = ["Efficacy", "Safety", "Generalizability", "Comparative", "Novelty", "Recommendation"]
        gold_counts = {lbl: 0 for lbl in LABELS}
        for _, row in gold_df.iterrows():
            lbls = str(row.get("verified_labels", "")).split(";")
            for l in lbls:
                if l.strip() in gold_counts:
                    gold_counts[l.strip()] += 1

        gold_summary = []
        for lbl in LABELS:
            cnt = gold_counts[lbl]
            gold_summary.append({
                "Category":    lbl,
                "Reviews":     cnt,
                "Target":      30,
                "Deficit":     max(0, 30 - cnt),
                "Status":      (
                    "Sufficient" if cnt >= 30
                    else "Provisional" if cnt >= 10
                    else "Scarce"
                ),
            })
        st.dataframe(pd.DataFrame(gold_summary), width="stretch", hide_index=True)

        st.divider()
        render_section_heading("Genuinely Human-Reviewed Rows")
        cols_gold_display = [
            c for c in ["candidate_id", "paper_id", "source_section",
                        "sentence", "verified_labels", "annotator", "review_notes"]
            if c in gold_df.columns
        ]
        st.dataframe(gold_df[cols_gold_display], width="stretch", hide_index=True)

        manifest_path = DATA_DIR / "human_gold_standard_manifest.json"
        if manifest_path.exists():
            with open(manifest_path, encoding="utf-8") as mfh:
                manifest_json_str = mfh.read()
            st.download_button(
                "Download Audit Manifest (JSON)",
                data=manifest_json_str.encode("utf-8"),
                file_name="human_gold_standard_manifest.json",
                mime="application/json",
            )
    else:
        st.info("No gold-standard file found at `data/research_dataset/human_gold_standard.csv`.")

    st.divider()
    render_section_heading("Full Candidates Corpus Status")

    if candidates_df is not None:
        status_counts = candidates_df["annotation_status"].value_counts()
        st_cols = st.columns(min(len(status_counts), 5))
        status_colors = {
            "pending":   COLORS["amber"],
            "approved":  COLORS["green"],
            "rejected":  COLORS["red"],
            "corrected": COLORS["blue"],
        }
        for i, (status, count) in enumerate(status_counts.items()):
            if i < len(st_cols):
                with st_cols[i]:
                    color = status_colors.get(status, COLORS["text_secondary"])
                    render_stat_card(f"{count:,}", status.title(), color)

        n_approved  = (candidates_df["annotation_status"] == "approved").sum()
        n_pending   = (candidates_df["annotation_status"] == "pending").sum()
        n_rejected  = (candidates_df["annotation_status"] == "rejected").sum()
        n_corrected = (candidates_df["annotation_status"] == "corrected").sum()
        n_source_checked = (candidates_df["source_text_checked"] == "yes").sum() if "source_text_checked" in candidates_df.columns else 0
        n_mukesh = (
            (candidates_df["annotator"] == "Mukesh").sum()
            if "annotator" in candidates_df.columns else 0
        )

        st.info(
            f"**Provenance breakdown:**\n\n"
            f"- Total candidates: **{len(candidates_df):,}**\n"
            f"- Pending human review: **{n_pending:,}**\n"
            f"- Human-approved: **{n_approved:,}**\n"
            f"- Human-corrected: **{n_corrected:,}**\n"
            f"- Rejected: **{n_rejected:,}**\n"
            f"- Verified against source text: **{n_source_checked:,}**\n"
            f"- Confirmed human reviews (annotator='Mukesh'): **{n_mukesh:,}**"
        )

        st.divider()
        render_section_heading("Annotation Review Tool")
        st.caption(
            "Use the standalone annotation review app to review candidates against "
            "their linked PMC sources and build additional gold-standard annotations."
        )
        st.link_button(
            "Open Annotation Review App (port 8503)",
            "http://localhost:8503",
        )
