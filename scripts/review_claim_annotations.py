"""Batch human review for machine-suggested Stage 1 claim annotations.

Three batch tabs grouped by machine suggestion:
  Batch A — Likely Claim
  Batch B — Likely Non-claim
  Batch C — Uncertain

Provenance rules enforced here:
  - Only explicit UI actions create human_decision records.
  - Source-text verification checkbox required for human-confirmed export eligibility.
  - Undo reverts the last action in this session (single level, session-scoped).
  - No row is ever source-verified by default, config, reviewer name, or scripted bulk.
  - Page-level "verified" checkbox is sticky within a page; resets when page or batch changes.
  - Row-level "verified" checkbox always starts at False; never inherits page state automatically.
  - Effective source_verified = page_checkbox OR row_checkbox (explicit opt-in only).
"""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from scripts.claim_dataset import (
    CANDIDATE_FIELDS,
    CANDIDATES_CSV,
    LABELS,
    PAPER_FIELDS,
    PAPERS_CSV,
    read_rows,
    write_rows,
)
from scripts.export_claim_annotations import approved_rows
from utils.claim_annotation_review import (
    ANNOTATION_HISTORY_CSV,
    ANNOTATION_HISTORY_FIELDS,
    BATCH_A,
    BATCH_B,
    BATCH_C,
    DECISIONS,
    PAGE_SIZE,
    _ensure_history_file,
    append_history_row,
    apply_page_verification,
    calculate_batch_page_count,
    get_batch_class,
    get_fixed_page_slice,
    get_history_rows,
    get_status_counts,
    get_verification_method,
    get_visible_page_candidates,
    is_decided,
    is_eligible_human_annotation,
    is_pending_review,
    is_source_verified,
    load_merged_candidates,
    save_review_decision,
    save_review_decision_with_history,
    sort_batch_candidates,
)

PRIORITY_QUEUE_CSV = ROOT / "data" / "research_dataset" / "priority_review_queue.csv"
MACHINE_CSV = ROOT / "data" / "research_dataset" / "machine_suggested_annotations_current.csv"

PAGE_SIZE = 25


# ---------------------------------------------------------------------------
# Session-state key helpers
# ---------------------------------------------------------------------------

def _page_src_key(batch_class: str, page: int) -> str:
    """Session-state key for the page-level source-verification checkbox.

    Encodes batch+page so the value is naturally isolated per page and batch.
    A page change produces a new key → fresh False default.
    A batch change likewise produces a new key.
    Within the same page/batch the key is stable → sticky across st.rerun().
    """
    return f"page_src__{batch_class}__{page}"


def _row_src_key(candidate_id: str) -> str:
    return f"row_src__{candidate_id}"


def _sel_key(candidate_id: str) -> str:
    return f"sel__{candidate_id}"


def _note_key(candidate_id: str) -> str:
    return f"note__{candidate_id}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _csv_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def _save_candidates(candidates: list[dict]) -> None:
    write_rows(CANDIDATES_CSV, CANDIDATE_FIELDS, candidates)


def _load_priority_ids() -> set[str]:
    rows = _csv_rows(PRIORITY_QUEUE_CSV)
    return {r["candidate_id"] for r in rows if r.get("candidate_id")}


def _batch_label(batch_class: str) -> str:
    return {BATCH_A: "A: Likely Claim", BATCH_B: "B: Likely Non-claim", BATCH_C: "C: Uncertain"}.get(
        batch_class, "C: Uncertain"
    )


def _parse_labels(label_str: str) -> list[str]:
    """Parse semicolon or comma separated label string into list."""
    parts = [p.strip() for p in label_str.replace(",", ";").split(";") if p.strip()]
    return [p for p in parts if p in LABELS]


def _push_undo(candidate_id: str, snapshot: dict) -> None:
    if "undo_stack" not in st.session_state:
        st.session_state["undo_stack"] = []
    st.session_state["undo_stack"].append((candidate_id, snapshot))


def _pop_undo() -> tuple[str, dict] | None:
    stack = st.session_state.get("undo_stack", [])
    if not stack:
        return None
    item = stack.pop()
    st.session_state["undo_stack"] = stack
    return item


def _do_undo(candidates: list[dict], candidate_by_id: dict[str, dict]) -> str | None:
    """Undo last action: restore snapshot, remove last history row, save CSV."""
    item = _pop_undo()
    if item is None:
        return None
    candidate_id, snapshot = item
    row = candidate_by_id.get(candidate_id)
    if row is None:
        return None
    row.update(snapshot)
    _save_candidates(candidates)

    # Remove last history row for this candidate (best-effort)
    if ANNOTATION_HISTORY_CSV.exists():
        hist_rows = _csv_rows(ANNOTATION_HISTORY_CSV)
        for i in range(len(hist_rows) - 1, -1, -1):
            if hist_rows[i].get("candidate_id") == candidate_id:
                hist_rows.pop(i)
                break
        _ensure_history_file()
        with ANNOTATION_HISTORY_CSV.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=ANNOTATION_HISTORY_FIELDS)
            writer.writeheader()
            writer.writerows(hist_rows)

    return candidate_id


def _apply_decision(
    candidates: list[dict],
    candidate_by_id: dict[str, dict],
    candidate_id: str,
    action: str,
    reviewer: str,
    verified_labels: list[str],
    source_checked: bool,
    review_note: str,
    batch_name: str,
    verification_method: str = "",
) -> None:
    """Apply a single review decision: snapshot for undo, save, persist."""
    row = candidate_by_id.get(candidate_id)
    if row is None:
        return
    # Snapshot for undo
    snapshot = {
        "annotation_status": row.get("annotation_status", ""),
        "annotator": row.get("annotator", ""),
        "source_text_checked": row.get("source_text_checked", ""),
        "source_verified": row.get("source_verified", ""),
        "verified_labels": row.get("verified_labels", ""),
        "verified_sentence": row.get("verified_sentence", ""),
        "review_notes": row.get("review_notes", ""),
        "verification_method": row.get("verification_method", ""),
        "verified_at": row.get("verified_at", ""),
        "verified_by": row.get("verified_by", ""),
    }
    _push_undo(candidate_id, snapshot)

    save_review_decision_with_history(
        candidates=candidates,
        candidate_id=candidate_id,
        action=action,
        reviewer=reviewer,
        verified_labels=verified_labels,
        source_text_checked=source_checked,
        review_note=review_note,
        batch_name=batch_name,
        verification_method=verification_method,
    )
    _save_candidates(candidates)
    vm_tag = f" ({verification_method})" if (source_checked and verification_method) else ""
    st.session_state["last_action_msg"] = (
        f"✅ {action} saved for `{candidate_id[:12]}…`"
        + (f" 🔒 *source verified{vm_tag}*" if source_checked else " ⚠️ *source NOT verified — excluded from export*")
    )


def _mark_verified_only(
    candidate_id: str,
    reviewer: str,
    candidates: list[dict],
    candidate_by_id: dict[str, dict],
) -> None:
    """Write source_verified=true + reviewer timestamp WITHOUT changing annotation_status.

    This is the "Mark selected as verified" bulk action — it only touches verification
    fields and appends a provenance note. It does NOT change annotation_status or labels.
    """
    row = candidate_by_id.get(candidate_id)
    if row is None:
        return
    from datetime import datetime, timezone
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    old_notes = row.get("review_notes", "")
    row["source_text_checked"] = "yes"
    row["source_verified"] = "true"
    row["verified_at"] = ts
    row["verified_by"] = reviewer
    row["verification_method"] = "row_attestation"
    if not row.get("annotator"):
        row["annotator"] = reviewer
    verify_note = f"source_checked=yes; verified_by={reviewer}; verified_at={ts}; verification_method=row_attestation"
    row["review_notes"] = (old_notes + "; " + verify_note).lstrip("; ")
    append_history_row(
        candidate_id=candidate_id,
        reviewer=reviewer,
        action="mark_verified_only",
        old_status=row.get("annotation_status", ""),
        old_labels=row.get("verified_labels", ""),
        new_status=row.get("annotation_status", ""),
        new_labels=row.get("verified_labels", ""),
        source_checked=True,
        batch_name="bulk_verify",
        review_note="Source text verified retroactively via bulk-verify action",
    )


# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(page_title="SPIN-SENTRY Annotation Review", layout="wide", page_icon="🔬")

# Keyboard shortcuts:
#   C = Confirm, R = Reject, U = Uncertain, N = Next page
#   V = toggle page-level verified checkbox
#   Shift+V = toggle focused row verified checkbox
_KBD_JS = """
<script>
(function() {
  document.addEventListener('keydown', function(e) {
    // Don't fire inside text inputs / textareas
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
    var key = e.key;

    if (key === 'v' || key === 'V') {
      if (e.shiftKey) {
        // Shift+V: toggle the row-level verified checkbox closest to the focused element
        var focused = document.activeElement;
        var container = focused ? focused.closest('[data-testid="stVerticalBlock"]') : null;
        if (container) {
          var rowChk = container.querySelector('input[type="checkbox"][data-row-src="true"]');
          if (rowChk) { rowChk.click(); return; }
        }
        // Fallback: first visible row-level checkbox
        var allRowChk = document.querySelectorAll('input[type="checkbox"][data-row-src="true"]');
        if (allRowChk.length > 0) { allRowChk[0].click(); }
      } else {
        // V: toggle the page-level verified checkbox
        var pageChk = document.querySelector('input[type="checkbox"][data-page-src="true"]');
        if (pageChk) { pageChk.click(); }
      }
      return;
    }

    var mapping = {c: 'kbd_confirm', r: 'kbd_reject', u: 'kbd_uncertain', n: 'kbd_next'};
    var lkey = key.toLowerCase();
    if (mapping[lkey]) {
      var btn = document.querySelector('button[data-kbd="' + mapping[lkey] + '"]');
      if (btn) { btn.click(); }
    }
  });

  // Stamp data attributes on checkboxes by label text after each render
  function stampCheckboxes() {
    var labels = document.querySelectorAll('label');
    labels.forEach(function(label) {
      var txt = label.textContent.trim();
      var input = label.querySelector('input[type="checkbox"]');
      if (!input) return;
      if (txt.includes('I verified ALL sentences on this page')) {
        input.setAttribute('data-page-src', 'true');
      } else if (txt.includes('I verified this sentence')) {
        input.setAttribute('data-row-src', 'true');
      }
    });
  }

  // Run after DOM mutations
  var observer = new MutationObserver(stampCheckboxes);
  observer.observe(document.body, { childList: true, subtree: true });
  stampCheckboxes();
})();
</script>
"""
st.components.v1.html(_KBD_JS, height=0)

st.title("🔬 SPIN-SENTRY · Batch Human Annotation Review")
st.caption(
    "Review machine-suggested annotations grouped by batch. "
    "Only source-verified decisions qualify for the human-confirmed training export. "
    "AI suggestions are never auto-counted as human ground truth."
)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

candidates = read_rows(CANDIDATES_CSV, CANDIDATE_FIELDS)
if not candidates:
    st.error("No candidates found. Run the extraction pipeline first.")
    st.stop()

machine_rows = _csv_rows(MACHINE_CSV)
priority_ids = _load_priority_ids()

merged = load_merged_candidates(candidates, machine_rows, priority_ids)
candidate_by_id: dict[str, dict] = {r["candidate_id"]: r for r in candidates}
merged_by_id: dict[str, dict] = {r["candidate_id"]: r for r in merged}

# Full rows per batch, sorted deterministically to establish fixed page assignments
full_batch_rows: dict[str, list[dict]] = {BATCH_A: [], BATCH_B: [], BATCH_C: []}
for r in merged:
    bc = r.get("batch_class", BATCH_C)
    if bc not in full_batch_rows:
        bc = BATCH_C
    full_batch_rows[bc].append(r)

for bc in full_batch_rows:
    full_batch_rows[bc] = sort_batch_candidates(full_batch_rows[bc])

# Live pending counts per batch
pending_by_batch: dict[str, int] = {
    bc: sum(1 for r in full_batch_rows[bc] if is_pending_review(candidate_by_id.get(r["candidate_id"], r)))
    for bc in full_batch_rows
}

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("👤 Reviewer")
    reviewer_name = st.text_input("Reviewer name", value="Mukesh", key="global_reviewer_name")
    st.caption("Stamped on decisions you explicitly confirm or correct.")
    st.divider()

    counts = get_status_counts(candidates)
    total = len(candidates)
    st.subheader("📊 Status Counts")
    st.metric("Total candidates", total)
    col1, col2 = st.columns(2)
    col1.metric("⏳ Pending", counts["pending"])
    col2.metric("✅ Confirmed", counts["approved"] + counts["corrected"])
    col1.metric("❌ Rejected", counts["rejected"])
    col2.metric("❓ Uncertain", counts["uncertain"])
    assert (
        counts["pending"] + counts["approved"] + counts["corrected"]
        + counts["rejected"] + counts["uncertain"] + counts["other"]
    ) == total, "Counter mismatch"
    st.caption(f"Counters sum to {total} ✓")

    st.divider()
    eligible = [r for r in candidates if is_eligible_human_annotation(r)]
    st.metric("🏅 Export-eligible (human-verified)", len(eligible))
    st.caption("Only source-checked, non-AI rows.")

    st.divider()
    st.markdown(
        "**Keyboard shortcuts** *(outside text fields)*\n"
        "- **C** = Confirm/Approve\n"
        "- **R** = Reject\n"
        "- **U** = Mark Uncertain\n"
        "- **N** = Next page\n"
        "- **V** = Toggle page-level verified\n"
        "- **Shift+V** = Toggle focused row verified\n"
    )
    st.markdown(
        "**Integrity Rules**\n"
        "- No row is verified by default.\n"
        "- Source verification requires an explicit checkbox tick.\n"
        "- Only source-checked rows qualify for training export.\n"
        "- AI suggestions are never auto-counted as human ground truth."
    )

    st.divider()
    if st.session_state.get("last_action_msg"):
        st.success(st.session_state["last_action_msg"])
        if st.button("Clear message"):
            del st.session_state["last_action_msg"]

# ---------------------------------------------------------------------------
# Undo button (top level)
# ---------------------------------------------------------------------------

undo_col, _ = st.columns([2, 8])
with undo_col:
    if st.button("↩️ Undo last action"):
        reverted_id = _do_undo(candidates, candidate_by_id)
        if reverted_id:
            st.success(f"Undone for `{reverted_id[:12]}…`")
            st.rerun()
        else:
            st.warning("Nothing to undo in this session.")


# ---------------------------------------------------------------------------
# Helper: read effective source-verified state for a row
# ---------------------------------------------------------------------------

def _effective_source_checked(batch_class: str, page: int, candidate_id: str) -> bool:
    """Return effective source verification for a candidate.

    Page-level checkbox OR row-level checkbox — either being True means verified.
    Neither is pre-ticked; this just reads current session state.
    """
    page_checked = bool(st.session_state.get(_page_src_key(batch_class, page), False))
    row_checked = bool(st.session_state.get(_row_src_key(candidate_id), False))
    return page_checked or row_checked


# ---------------------------------------------------------------------------
# Helper: count how many rows on this page will be saved as verified
# ---------------------------------------------------------------------------

def _verified_count_on_page(batch_class: str, page: int, page_rows: list[dict]) -> int:
    return sum(
        1 for r in page_rows
        if _effective_source_checked(batch_class, page, r["candidate_id"])
    )


# ---------------------------------------------------------------------------
# Helper: render one row card
# ---------------------------------------------------------------------------

def _row_verification_badge(row: dict) -> tuple[str, str]:
    """Return (badge_text, hex_color) reflecting verified state and method."""
    if is_source_verified(row):
        vm = get_verification_method(row)
        if vm == "page_attestation":
            return ("🔒 verified (page)", "#10b981")
        return ("🔒 verified (row)", "#3b82f6")
    return ("🔓 not verified", "#f59e0b")


def _render_row_card(
    row: dict,
    batch_class: str,
    page: int,
    reviewer: str,
    candidates: list[dict],
    candidate_by_id: dict[str, dict],
    row_number: int = 0,
    global_number: int = 0,
) -> bool:
    """Render a single candidate row card. Returns True if select-checkbox is checked."""
    cid = row["candidate_id"]
    sentence = row.get("sentence", "")
    title = row.get("title", "")[:80]
    section = row.get("source_section", "")[:60]
    machine_class = row.get("machine_claim_class", "")
    machine_labels = row.get("machine_suggested_labels", "")
    is_priority = row.get("is_priority") == "yes"

    priority_badge = "🔴 " if is_priority else ""
    num_label = f"**#{global_number}**" if global_number else ""

    row_data = candidate_by_id.get(cid, row)
    verified_badge, verified_badge_color = _row_verification_badge(row_data)

    with st.container():
        header_cols = st.columns([0.3, 0.5, 6.5, 1.5, 1.5])
        with header_cols[0]:
            checked = st.checkbox(
                "Select",
                key=_sel_key(cid),
                label_visibility="collapsed",
            )
        with header_cols[1]:
            st.markdown(num_label)
        with header_cols[2]:
            st.markdown(f"{priority_badge}**{cid[:16]}…** · *{section}* · 📄 {title}")
        with header_cols[3]:
            unc = "⚠️ uncertain" if row.get("machine_uncertainty_flag") == "yes" else ""
            st.caption(f"`{machine_class}` {unc}")
        with header_cols[4]:
            st.markdown(
                f"<small><span style='color:{verified_badge_color}; font-weight:600;'>{verified_badge}</span></small>",
                unsafe_allow_html=True,
            )

        st.markdown(f"> {sentence}")

        # -- Machine suggested labels --
        _LABEL_COLORS = {
            "Efficacy": "#3b82f6", "Safety": "#ef4444",
            "Generalizability": "#8b5cf6", "Comparative": "#f59e0b",
            "Novelty": "#10b981", "Recommendation": "#ec4899",
        }
        if machine_labels:
            parsed_ml = _parse_labels(machine_labels)
            if parsed_ml:
                label_html = " ".join(
                    f"<span style='background:{_LABEL_COLORS.get(lbl, '#6b7280')}22; "
                    f"border:1px solid {_LABEL_COLORS.get(lbl, '#6b7280')}; "
                    f"color:{_LABEL_COLORS.get(lbl, '#6b7280')}; "
                    f"padding:2px 8px; border-radius:12px; font-size:0.78em; "
                    f"font-weight:600; margin-right:4px;'>{lbl}</span>"
                    for lbl in parsed_ml
                )
                prefix = "<span style='font-size:0.78em; color:#888;'>Machine labels:</span>"
                st.markdown(
                    f"<div style='margin:4px 0 6px 0;'>{prefix} {label_html}</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.caption(f"Machine labels (unparsed): {machine_labels}")
        else:
            st.caption("Machine labels: *none recorded*")

        with st.expander("📄 Show paper context"):
            ctx_before = row.get("machine_context_before", "")
            ctx_after = row.get("machine_context_after", "")
            rationale = row.get("machine_label_rationales", "")
            reasons = row.get("machine_review_reasons", "")
            if ctx_before:
                st.markdown(f"**Before:** {ctx_before}")
            st.markdown(f"**→ Sentence:** _{sentence}_")
            if ctx_after:
                st.markdown(f"**After:** {ctx_after}")
            if rationale:
                st.caption(f"Label rationale: {rationale}")
            if reasons:
                st.caption(f"Priority reasons: {reasons}")

        note_widget_key = _note_key(cid)
        row_note = st.text_input(
            "Optional note",
            key=note_widget_key,
            label_visibility="collapsed",
            placeholder="Optional review note…",
        )

        # Row-level source verification checkbox.
        # IMPORTANT: No value= default passed → always starts at False when first seen.
        # Session state preserves the tick across st.rerun() within the same page.
        st.checkbox(
            "✓ I verified this sentence against the source text",
            key=_row_src_key(cid),
            # Deliberately NO value= argument so default is always False on first render.
        )

        page_src_checked = bool(st.session_state.get(_page_src_key(batch_class, page), False))
        row_src_checked = bool(st.session_state.get(_row_src_key(cid), False))
        src = page_src_checked or row_src_checked
        method = "page_attestation" if page_src_checked else ("row_attestation" if row_src_checked else "")

        # Action buttons per batch type
        if batch_class == BATCH_A:  # Likely Claim
            btn_cols = st.columns([2, 3, 2, 2])
            with btn_cols[0]:
                if st.button("✅ Confirm", key=f"confirm_{cid}", help="C"):
                    ml = _parse_labels(machine_labels) or ["Efficacy"]
                    _apply_decision(
                        candidates, candidate_by_id, cid, "Confirm", reviewer, ml,
                        src, row_note, _batch_label(BATCH_A), verification_method=method,
                    )
                    st.rerun()
            with btn_cols[1]:
                with st.popover("✏️ Correct categories"):
                    new_labels = st.multiselect(
                        "Select correct labels", LABELS,
                        default=_parse_labels(machine_labels),
                        key=f"correct_labels_{cid}",
                    )
                    if st.button("Save correction", key=f"save_correct_{cid}"):
                        _apply_decision(
                            candidates, candidate_by_id, cid, "Correct", reviewer, new_labels,
                            src, row_note, _batch_label(BATCH_A), verification_method=method,
                        )
                        st.rerun()
            with btn_cols[2]:
                if st.button("❌ Reject", key=f"reject_{cid}", help="R"):
                    _apply_decision(
                        candidates, candidate_by_id, cid, "Reject", reviewer, [],
                        src, row_note, _batch_label(BATCH_A), verification_method=method,
                    )
                    st.rerun()
            with btn_cols[3]:
                if st.button("❓ Uncertain", key=f"uncertain_{cid}", help="U"):
                    _apply_decision(
                        candidates, candidate_by_id, cid, "Mark Uncertain", reviewer, [],
                        src, row_note, _batch_label(BATCH_A), verification_method=method,
                    )
                    st.rerun()

        elif batch_class == BATCH_B:  # Likely Non-claim
            btn_cols = st.columns([3, 3, 2])
            with btn_cols[0]:
                if st.button("✅ Confirm non-claim", key=f"confirm_nc_{cid}"):
                    _apply_decision(
                        candidates, candidate_by_id, cid, "Reject", reviewer, [],
                        src, row_note, _batch_label(BATCH_B), verification_method=method,
                    )
                    st.rerun()
            with btn_cols[1]:
                with st.popover("🔁 It IS a claim"):
                    claim_labels = st.multiselect(
                        "Assign categories", LABELS,
                        default=[],
                        key=f"claim_labels_{cid}",
                    )
                    if st.button("Save as claim", key=f"save_claim_{cid}"):
                        _apply_decision(
                            candidates, candidate_by_id, cid, "Correct", reviewer, claim_labels,
                            src, row_note, _batch_label(BATCH_B), verification_method=method,
                        )
                        st.rerun()
            with btn_cols[2]:
                if st.button("❓ Uncertain", key=f"uncertain_nc_{cid}", help="U"):
                    _apply_decision(
                        candidates, candidate_by_id, cid, "Mark Uncertain", reviewer, [],
                        src, row_note, _batch_label(BATCH_B), verification_method=method,
                    )
                    st.rerun()

        else:  # BATCH_C — Uncertain
            btn_cols = st.columns([3, 2, 2])
            with btn_cols[0]:
                with st.popover("✅ It IS a claim"):
                    unc_labels = st.multiselect(
                        "Assign categories", LABELS,
                        default=_parse_labels(machine_labels),
                        key=f"unc_labels_{cid}",
                    )
                    if st.button("Save as claim", key=f"save_unc_{cid}"):
                        _apply_decision(
                            candidates, candidate_by_id, cid, "Confirm", reviewer, unc_labels,
                            src, row_note, _batch_label(BATCH_C), verification_method=method,
                        )
                        st.rerun()
            with btn_cols[1]:
                if st.button("❌ Reject", key=f"reject_unc_{cid}", help="R"):
                    _apply_decision(
                        candidates, candidate_by_id, cid, "Reject", reviewer, [],
                        src, row_note, _batch_label(BATCH_C), verification_method=method,
                    )
                    st.rerun()
            with btn_cols[2]:
                if st.button("⏭️ Skip", key=f"skip_{cid}"):
                    pass  # No state change

        st.divider()
    return checked


# ---------------------------------------------------------------------------
# Helper: bulk apply panel
# ---------------------------------------------------------------------------

def _render_bulk_panel(
    page_rows: list[dict],
    selected_ids: set[str],
    batch_class: str,
    page: int,
    reviewer: str,
    candidates: list[dict],
    candidate_by_id: dict[str, dict],
) -> None:
    page_source_checked = bool(st.session_state.get(_page_src_key(batch_class, page), False))
    n_selected = sum(1 for r in page_rows if r["candidate_id"] in selected_ids)
    n_verified_on_page = _verified_count_on_page(batch_class, page, page_rows)

    # Summary bar
    info_col1, info_col2 = st.columns(2)
    info_col1.info(f"**{n_selected} of {len(page_rows)} rows selected** (current page only)")
    info_col2.info(
        f"**{n_verified_on_page} of {len(page_rows)} rows will be saved as source-verified**"
        " (page ✓ or row ✓ checked)"
    )

    if batch_class == BATCH_A:
        bulk_options = ["Confirm", "Reject", "Mark Uncertain", "Mark selected as verified"]
    elif batch_class == BATCH_B:
        bulk_options = ["Confirm non-claim", "Mark Uncertain", "Mark selected as verified"]
    else:
        bulk_options = ["Reject", "Mark Uncertain", "Mark selected as verified"]

    bulk_col1, bulk_col2, bulk_col3 = st.columns([2, 2, 3])
    with bulk_col1:
        bulk_action = st.selectbox("Bulk action", bulk_options, key=f"bulk_action_{batch_class}")
    with bulk_col2:
        bulk_labels: list[str] = []
        if bulk_action in {"Confirm", "Correct"}:
            bulk_labels = st.multiselect("Labels (bulk)", LABELS, key=f"bulk_labels_{batch_class}")
    with bulk_col3:
        bulk_note = st.text_input(
            "Bulk note", placeholder="Optional bulk note", key=f"bulk_note_{batch_class}"
        )

    n_apply = n_selected if bulk_action != "Mark selected as verified" else n_selected
    btn_label = (
        f"Apply '{bulk_action}' to {n_selected} selected rows"
        if n_selected > 0
        else "No rows selected"
    )
    apply_disabled = n_selected == 0

    if st.button(btn_label, key=f"bulk_apply_{batch_class}", disabled=apply_disabled):
        if bulk_action == "Mark selected as verified":
            # Only mark source_text_checked; does NOT change annotation_status
            for r in page_rows:
                cid = r["candidate_id"]
                if cid not in selected_ids:
                    continue
                _mark_verified_only(cid, reviewer, candidates, candidate_by_id)
            _save_candidates(candidates)
            st.success(
                f"Marked {n_selected} selected rows as source-verified "
                f"(reviewer={reviewer}). Annotation status unchanged."
            )
        else:
            action_map = {
                "Confirm": "Confirm",
                "Confirm non-claim": "Reject",
                "Reject": "Reject",
                "Mark Uncertain": "Mark Uncertain",
            }
            action = action_map.get(bulk_action, "Mark Uncertain")
            for r in page_rows:
                cid = r["candidate_id"]
                if cid not in selected_ids:
                    continue
                src = _effective_source_checked(batch_class, page, cid)
                labels = bulk_labels if action in {"Confirm", "Correct"} else []
                method = "page_attestation" if page_source_checked else ("row_attestation" if src else "")
                _apply_decision(
                    candidates, candidate_by_id, cid, action, reviewer, labels,
                    src, bulk_note, f"bulk_{_batch_label(batch_class)}",
                    verification_method=method,
                )
            st.success(f"Applied '{bulk_action}' to {n_selected} selected rows.")
        st.rerun()


# ---------------------------------------------------------------------------
# Helper: render one batch tab
# ---------------------------------------------------------------------------

def _render_batch_tab(
    batch_class: str,
    tab_label: str,
    sorted_batch_rows: list[dict],
    reviewer: str,
    candidates: list[dict],
    candidate_by_id: dict[str, dict],
) -> None:
    total_in_batch = len(sorted_batch_rows)
    pending_in_batch = sum(
        1 for r in sorted_batch_rows if is_pending_review(candidate_by_id.get(r["candidate_id"], r))
    )
    reviewed_in_batch = total_in_batch - pending_in_batch
    progress = reviewed_in_batch / max(total_in_batch, 1)

    st.subheader(f"Batch {tab_label}")
    prog_col1, prog_col2 = st.columns([3, 1])
    with prog_col1:
        st.progress(progress)
    with prog_col2:
        st.caption(f"{reviewed_in_batch}/{total_in_batch} reviewed ({pending_in_batch} pending)")

    if total_in_batch == 0:
        st.info("No candidates in this batch.")
        return

    # View mode / page size selector
    view_col1, view_col2 = st.columns([6, 4])
    with view_col1:
        size_choice = st.radio(
            "Display mode:",
            ["All in single page", "100 per page", "50 per page", "25 per page"],
            horizontal=True,
            key=f"size_choice__{batch_class}",
            help="Choose 'All in single page' to review all candidates together, or 50/100 per page for maximum speed.",
        )
    with view_col2:
        show_reviewed_key = f"show_reviewed__{batch_class}"
        show_reviewed = st.checkbox(
            "Show reviewed candidates",
            key=show_reviewed_key,
            help="Show already-decided candidates alongside pending ones",
        )

    # Determine effective page_size
    if size_choice == "All in single page":
        effective_page_size = max(total_in_batch, 1)
    elif size_choice == "100 per page":
        effective_page_size = 100
    elif size_choice == "50 per page":
        effective_page_size = 50
    else:
        effective_page_size = 25

    n_pages = calculate_batch_page_count(total_in_batch, effective_page_size)
    page_key = f"page_{batch_class}"
    if page_key not in st.session_state:
        st.session_state[page_key] = 1
    st.session_state[page_key] = max(1, min(st.session_state[page_key], n_pages))
    page = st.session_state[page_key]

    # Deterministic slice
    assigned_page_rows = get_fixed_page_slice(sorted_batch_rows, page, effective_page_size)
    visible_page_rows = get_visible_page_candidates(
        assigned_page_rows, candidate_by_id, show_reviewed=show_reviewed
    )
    pending_page_count = sum(
        1 for r in assigned_page_rows if is_pending_review(candidate_by_id.get(r["candidate_id"], r))
    )

    # Navigation bar (if more than 1 page)
    if n_pages > 1:
        nav_col1, nav_col2, nav_col3 = st.columns([1.5, 7, 1.5])
        with nav_col1:
            if st.button("◀ Prev", key=f"prev_{batch_class}", disabled=(page <= 1)):
                st.session_state[page_key] -= 1
                st.rerun()
        with nav_col2:
            st.caption(
                f"**Page {page} of {n_pages}** · {pending_page_count} pending of {len(assigned_page_rows)} assigned to this page"
            )
        with nav_col3:
            if st.button("Next ▶", key=f"next_{batch_class}", disabled=(page >= n_pages)):
                st.session_state[page_key] += 1
                st.rerun()
    else:
        st.caption(
            f"**Single Page View** · {pending_page_count} pending of {len(assigned_page_rows)} total candidates"
        )

    # -----------------------------------------------------------------------
    # Source verification checkbox
    # -----------------------------------------------------------------------
    src_key = _page_src_key(batch_class, page)
    page_source_checked = st.checkbox(
        "✓ I verified ALL sentences on this page against their source papers",
        key=src_key,
    )

    decided_unverified = [
        candidate_by_id[r["candidate_id"]]
        for r in assigned_page_rows
        if is_decided(candidate_by_id[r["candidate_id"]]) and not is_source_verified(candidate_by_id[r["candidate_id"]])
    ]
    n_decided_unverified = len(decided_unverified)

    if page_source_checked:
        btn_key = f"apply_page_attestation__{batch_class}__{page}"
        btn_label = f"⚡ Apply verification to {n_decided_unverified} decided rows on this page"
        if st.button(btn_label, key=btn_key, disabled=(n_decided_unverified == 0)):
            updated = apply_page_verification(
                page_rows=decided_unverified,
                reviewer=reviewer,
                candidates=candidates,
                batch_name=_batch_label(batch_class),
            )
            _save_candidates(candidates)
            new_export_count = sum(1 for r in candidates if is_eligible_human_annotation(r))
            st.session_state["last_action_msg"] = (
                f"✅ Applied page verification to {len(updated)} decided rows! "
                f"Export-eligible: {new_export_count}"
            )
            st.rerun()

    eligible_total = sum(1 for r in candidates if is_eligible_human_annotation(r))
    if page_source_checked:
        st.success(
            f"Page-level verification ON · **Export-eligible: {eligible_total}** · "
            f"Any new actions on this page will be recorded as **verified (page)**."
        )
    else:
        st.info(
            f"Page-level verification OFF · **Export-eligible: {eligible_total}** · "
            f"Use individual row checkboxes or press **V** to toggle verification."
        )

    st.divider()

    # Empty page state
    if len(visible_page_rows) == 0:
        if pending_page_count == 0:
            st.success(
                f"🎉 All {len(assigned_page_rows)} candidates assigned to this page have been reviewed!"
            )
        else:
            st.info("No visible candidates.")

    # Collect selected rows
    selected_ids: set[str] = {
        r["candidate_id"]
        for r in visible_page_rows
        if st.session_state.get(_sel_key(r["candidate_id"]), False)
    }

    # Select All checkbox above cards
    sel_all_key = f"sel_all__{batch_class}__{page}"
    sel_all = st.checkbox(
        f"Select all {len(visible_page_rows)} visible on this page",
        key=sel_all_key,
        help="Checks all visible rows on this page for bulk actions.",
    )
    if sel_all:
        for _r in visible_page_rows:
            st.session_state[_sel_key(_r["candidate_id"])] = True

    # Global position calculation: offset in full sorted batch
    page_start_global = (page - 1) * effective_page_size

    # Render each visible row card
    for row_idx, row in enumerate(visible_page_rows):
        cid = row["candidate_id"]
        global_pos = page_start_global + row_idx + 1
        checked = _render_row_card(
            row=row,
            batch_class=batch_class,
            page=page,
            reviewer=reviewer,
            candidates=candidates,
            candidate_by_id=candidate_by_id,
            row_number=row_idx + 1,
            global_number=global_pos,
        )
        if checked:
            selected_ids.add(cid)
        else:
            selected_ids.discard(cid)

    # Bulk actions panel
    if len(visible_page_rows) > 0:
        st.subheader("📦 Bulk Actions (current page only)")
        _render_bulk_panel(
            page_rows=visible_page_rows,
            selected_ids=selected_ids,
            batch_class=batch_class,
            page=page,
            reviewer=reviewer,
            candidates=candidates,
            candidate_by_id=candidate_by_id,
        )


# ---------------------------------------------------------------------------
# Main tabs (Lazy Navigation: only active batch executes, preventing lag/crashes)
# ---------------------------------------------------------------------------

tab_a_label = f"A: Likely Claim ({pending_by_batch[BATCH_A]} pending / {len(full_batch_rows[BATCH_A])} total)"
tab_b_label = f"B: Likely Non-claim ({pending_by_batch[BATCH_B]} pending / {len(full_batch_rows[BATCH_B])} total)"
tab_c_label = f"C: Uncertain ({pending_by_batch[BATCH_C]} pending / {len(full_batch_rows[BATCH_C])} total)"
tab_export_label = "📤 Export"

tab_options = [tab_a_label, tab_b_label, tab_c_label, tab_export_label]

# Horizontal radio tab bar for lazy execution
selected_tab = st.radio(
    "Active Tab",
    options=tab_options,
    horizontal=True,
    label_visibility="collapsed",
    key="main_batch_tab_selector",
)

st.markdown("<hr style='margin: 0.5rem 0 1.5rem 0;'>", unsafe_allow_html=True)

if selected_tab == tab_a_label:
    _render_batch_tab(
        batch_class=BATCH_A,
        tab_label="A: Likely Claim",
        sorted_batch_rows=full_batch_rows[BATCH_A],
        reviewer=reviewer_name,
        candidates=candidates,
        candidate_by_id=candidate_by_id,
    )
elif selected_tab == tab_b_label:
    _render_batch_tab(
        batch_class=BATCH_B,
        tab_label="B: Likely Non-claim",
        sorted_batch_rows=full_batch_rows[BATCH_B],
        reviewer=reviewer_name,
        candidates=candidates,
        candidate_by_id=candidate_by_id,
    )
elif selected_tab == tab_c_label:
    _render_batch_tab(
        batch_class=BATCH_C,
        tab_label="C: Uncertain",
        sorted_batch_rows=full_batch_rows[BATCH_C],
        reviewer=reviewer_name,
        candidates=candidates,
        candidate_by_id=candidate_by_id,
    )
elif selected_tab == tab_export_label:
    st.subheader("📤 Human-Verified Training Export")
    st.warning(
        "Only rows explicitly confirmed by a human AND source-text-verified qualify for export. "
        "AI-suggested annotations are exported separately and must NOT be used as ground truth."
    )

    eligible = [r for r in candidates if is_eligible_human_annotation(r)]
    st.metric("Human-verified export-eligible rows", len(eligible))

    if eligible:
        export_df = pd.DataFrame(eligible)
        st.dataframe(export_df, width="stretch")
        csv_bytes = export_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download human-verified CSV",
            data=csv_bytes,
            file_name="human_verified_annotations.csv",
            mime="text/csv",
        )
    else:
        st.info(
            "No human-verified rows available yet. "
            "Use Batch A/B/C tabs to confirm candidates with source-text verification."
        )

    st.divider()
    st.subheader("🤖 Machine Suggestion Export (separate provenance)")
    all_machine = _csv_rows(MACHINE_CSV)
    if all_machine:
        st.caption(f"{len(all_machine)} machine-suggested candidate records available.")
        machine_df = pd.DataFrame(all_machine)
        st.dataframe(machine_df.head(20), width="stretch")
        m_bytes = machine_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download machine suggestions CSV",
            data=m_bytes,
            file_name="machine_suggested_annotations.csv",
            mime="text/csv",
        )
