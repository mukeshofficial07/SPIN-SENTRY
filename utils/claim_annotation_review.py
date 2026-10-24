"""Small, UI-independent helpers for the human annotation review workflow."""

from __future__ import annotations

import csv
import json
import math
from collections.abc import MutableMapping, Sequence
from datetime import datetime, timezone
from pathlib import Path

DECISIONS = {
    "Approve": "approved",
    "Confirm": "approved",
    "Confirm non-claim": "rejected",
    "Confirm (Approve)": "approved",
    "Correct": "corrected",
    "Mark uncertain": "uncertain",
    "Mark Uncertain": "uncertain",
    "Reject": "rejected",
}

# Canonical label set
LABELS = ["Efficacy", "Safety", "Generalizability", "Comparative", "Novelty", "Recommendation"]

# Batch group mapping
BATCH_A = "claim"
BATCH_B = "non_claim"
BATCH_C = "uncertain"

DECIDED_STATUSES = {"approved", "confirmed", "corrected", "rejected", "uncertain"}
EXPORT_ELIGIBLE_STATUSES = {"approved", "confirmed", "corrected", "rejected"}

ANNOTATION_HISTORY_CSV = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "research_dataset"
    / "annotation_history.csv"
)
ANNOTATION_HISTORY_FIELDS = [
    "candidate_id",
    "timestamp_utc",
    "reviewer",
    "action",
    "old_status",
    "old_labels",
    "new_status",
    "new_labels",
    "source_checked",
    "batch_name",
    "review_note",
]


def is_source_verified(row: MutableMapping[str, str] | dict[str, str]) -> bool:
    """Return True if row has source-text verification recorded."""
    if str(row.get("source_verified", "")).strip().lower() in {"true", "yes", "1"}:
        return True
    if str(row.get("source_text_checked", "")).strip().lower() in {"true", "yes", "1"}:
        return True
    notes = str(row.get("review_notes", ""))
    if "source_checked=yes" in notes or "source_verified_by=" in notes:
        return True
    return False


def get_verification_method(row: MutableMapping[str, str] | dict[str, str]) -> str:
    """Return 'page_attestation', 'row_attestation', or ''."""
    vm = str(row.get("verification_method", "")).strip()
    if vm:
        return vm
    notes = str(row.get("review_notes", ""))
    if "verification_method=page_attestation" in notes:
        return "page_attestation"
    if "verification_method=row_attestation" in notes:
        return "row_attestation"
    if is_source_verified(row):
        return "row_attestation"
    return ""


def is_decided(row: MutableMapping[str, str] | dict[str, str]) -> bool:
    """Return True if candidate has a human decision (approved, confirmed, corrected, rejected, uncertain)."""
    status = str(row.get("annotation_status", "")).strip().casefold()
    if status in DECIDED_STATUSES:
        notes = str(row.get("review_notes", "")).strip()
        if notes.startswith("AI-assisted annotation") and "human_decision=" not in notes:
            return False
        return True
    return False


def is_eligible_human_annotation(row: MutableMapping[str, str] | dict[str, str]) -> bool:
    """Return True if row has status in (confirmed, corrected, rejected) AND source_verified = true AND human action."""
    status = str(row.get("annotation_status", "")).strip().casefold()
    if status not in EXPORT_ELIGIBLE_STATUSES:
        return False
    if not is_source_verified(row):
        return False
    # Exclude automated AI-assisted labels that were never confirmed by human
    notes = str(row.get("review_notes", "")).strip()
    if notes.startswith("AI-assisted annotation"):
        return False
    # Human action requires an annotator
    annotator = str(row.get("annotator", "")).strip()
    if not annotator:
        return False
    # For confirmed/corrected, verified_labels must be present
    if status in {"approved", "confirmed", "corrected"}:
        labels = [l.strip() for l in str(row.get("verified_labels", "")).split(";") if l.strip()]
        if not labels:
            return False
    return True


def is_pending_review(row: MutableMapping[str, str] | dict[str, str]) -> bool:
    """Return True if candidate has not yet been resolved by human review."""
    status = str(row.get("annotation_status", "pending")).strip().casefold()
    if status == "pending":
        return True
    # If previously marked approved/corrected by automated batch without source text checked:
    if status in {"approved", "corrected"} and str(row.get("source_text_checked", "no")).strip().casefold() != "yes":
        if str(row.get("review_notes", "")).strip().startswith("AI-assisted annotation"):
            return True
    return False


def save_review_decision(
    candidates: list[dict[str, str]],
    candidate_id: str,
    action: str,
    reviewer: str,
    verified_labels: Sequence[str] = (),
    verified_sentence: str = "",
    source_text_checked: bool = False,
    review_note: str = "",
    verification_method: str = "",
) -> dict[str, str]:
    """Update only human-review fields on one existing candidate row."""
    if action not in DECISIONS:
        raise ValueError(f"Unknown review decision: {action}")
    row = next((item for item in candidates if item.get("candidate_id") == candidate_id), None)
    if row is None:
        raise KeyError(candidate_id)

    decision = DECISIONS[action]
    row["annotation_status"] = decision
    row["annotator"] = reviewer.strip()
    row["source_text_checked"] = "yes" if source_text_checked else "no"

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    if source_text_checked:
        vm = verification_method or "row_attestation"
        row["source_verified"] = "true"
        row["verified_at"] = ts
        row["verified_by"] = reviewer.strip()
        row["verification_method"] = vm
    else:
        row["source_verified"] = "false"
        row["verification_method"] = ""
        row["verified_at"] = ""
        row["verified_by"] = ""

    # Record provenance, reviewer, decision, timestamp, and optional rationale note
    note_parts = [
        f"human_decision={decision}",
        f"reviewer={reviewer.strip()}",
        f"ts={ts}",
        f"source_checked={'yes' if source_text_checked else 'no'}",
    ]
    if source_text_checked:
        note_parts.append(f"verification_method={row['verification_method']}")
        note_parts.append(f"verified_by={reviewer.strip()}")
        note_parts.append(f"verified_at={ts}")
    if review_note.strip():
        note_parts.append(f"note={review_note.strip()}")
    row["review_notes"] = "; ".join(note_parts)

    if decision in {"approved", "corrected"}:
        row["verified_labels"] = ";".join(verified_labels)
        original = row.get("sentence", "")
        row["verified_sentence"] = (
            verified_sentence.strip() if verified_sentence.strip() != original.strip() else ""
        )
    elif decision in {"rejected", "uncertain"}:
        row["verified_labels"] = ""
        row["verified_sentence"] = ""

    return row


# ---------------------------------------------------------------------------
# Annotation history helpers
# ---------------------------------------------------------------------------


def _ensure_history_file() -> None:
    """Create annotation_history.csv with header if it doesn't exist."""
    if not ANNOTATION_HISTORY_CSV.exists():
        ANNOTATION_HISTORY_CSV.parent.mkdir(parents=True, exist_ok=True)
        with ANNOTATION_HISTORY_CSV.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(ANNOTATION_HISTORY_FIELDS)


def append_history_row(
    candidate_id: str,
    reviewer: str,
    action: str,
    old_status: str,
    old_labels: str,
    new_status: str,
    new_labels: str,
    source_checked: bool,
    batch_name: str = "",
    review_note: str = "",
) -> None:
    """Append one row to annotation_history.csv (append-only, thread-safe for single user)."""
    _ensure_history_file()
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    with ANNOTATION_HISTORY_CSV.open("a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            candidate_id, ts, reviewer, action,
            old_status, old_labels, new_status, new_labels,
            "yes" if source_checked else "no",
            batch_name, review_note,
        ])


def get_history_rows(candidate_id: str) -> list[dict]:
    """Return all history rows for a given candidate_id."""
    if not ANNOTATION_HISTORY_CSV.exists():
        return []
    with ANNOTATION_HISTORY_CSV.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return [r for r in reader if r.get("candidate_id") == candidate_id]


# ---------------------------------------------------------------------------
# Batch classification helpers
# ---------------------------------------------------------------------------


def get_batch_class(row: dict, machine_by_id: dict[str, dict]) -> str:
    """Return 'claim', 'non_claim', or 'uncertain' based on machine_claim_class."""
    mc = machine_by_id.get(row.get("candidate_id", ""), {})
    cls = str(mc.get("machine_claim_class", "")).strip().lower()
    if cls == "claim":
        return BATCH_A
    if cls == "non_claim":
        return BATCH_B
    return BATCH_C


def load_merged_candidates(
    candidates: list[dict],
    machine_rows: list[dict],
    priority_ids: set[str] | None = None,
) -> list[dict]:
    """Return candidates merged with machine annotation columns and priority flag."""
    machine_by_id: dict[str, dict] = {r["candidate_id"]: r for r in machine_rows}
    priority_ids = priority_ids or set()
    merged = []
    for row in candidates:
        cid = row.get("candidate_id", "")
        mc = machine_by_id.get(cid, {})
        merged_row = dict(row)
        merged_row["machine_claim_class"] = mc.get("machine_claim_class", "")
        # Prefer machine_suggested_labels; fall back to suggested_labels (older field name)
        machine_labels = mc.get("machine_suggested_labels", "").strip()
        if not machine_labels:
            machine_labels = mc.get("suggested_labels", "").strip()
        merged_row["machine_suggested_labels"] = machine_labels
        merged_row["machine_uncertainty_flag"] = mc.get("machine_uncertainty_flag", "")
        merged_row["machine_context_before"] = mc.get("machine_context_before", "")
        merged_row["machine_context_after"] = mc.get("machine_context_after", "")
        merged_row["machine_review_priority"] = mc.get("machine_review_priority", "")
        merged_row["machine_review_reasons"] = mc.get("machine_review_reasons", "")
        merged_row["machine_label_rationales"] = mc.get("machine_label_rationales", "")
        merged_row["is_priority"] = "yes" if cid in priority_ids else "no"
        merged_row["batch_class"] = get_batch_class(row, machine_by_id)
        merged.append(merged_row)
    return merged


PAGE_SIZE = 25


def sort_batch_candidates(
    rows: Sequence[MutableMapping[str, str] | dict[str, str]],
) -> list[dict[str, str]]:
    """Deterministically sort candidates for a batch by priority flag, priority score, and candidate_id."""
    def _key(r: MutableMapping[str, str] | dict[str, str]) -> tuple:
        is_pri = 0 if str(r.get("is_priority", "")).strip().lower() == "yes" else 1
        try:
            pri_score = float(r.get("machine_review_priority", 99))
        except (ValueError, TypeError):
            pri_score = 99.0
        cid = str(r.get("candidate_id", ""))
        return (is_pri, pri_score, cid)

    return sorted([dict(r) for r in rows], key=_key)


def calculate_batch_page_count(
    total_batch_count: int,
    page_size: int = PAGE_SIZE,
) -> int:
    """Return the total number of fixed pages for the batch."""
    if total_batch_count <= 0:
        return 1
    return max(1, math.ceil(total_batch_count / page_size))


def get_fixed_page_slice(
    sorted_batch_rows: Sequence[dict[str, str]],
    page: int,
    page_size: int = PAGE_SIZE,
) -> list[dict[str, str]]:
    """Return the fixed list of candidates assigned to page `page` (1-indexed)."""
    if page < 1:
        page = 1
    start = (page - 1) * page_size
    end = start + page_size
    return list(sorted_batch_rows[start:end])


def get_visible_page_candidates(
    assigned_page_rows: Sequence[dict[str, str]],
    candidate_by_id: MutableMapping[str, dict[str, str]],
    show_reviewed: bool = False,
) -> list[dict[str, str]]:
    """Return the candidates to display on this page.

    Preserves merged machine suggestion columns while reflecting live review status.
    By default (show_reviewed=False), reviewed candidates are removed immediately,
    leaving fewer visible candidates on that page without pulling items from other pages.
    """
    visible: list[dict[str, str]] = []
    for r in assigned_page_rows:
        cid = r.get("candidate_id", "")
        live = candidate_by_id.get(cid)
        current = live if live is not None else r
        if not show_reviewed and not is_pending_review(current):
            continue
        merged_row = dict(r)
        if live is not None:
            for k in (
                "annotation_status",
                "annotator",
                "source_text_checked",
                "source_verified",
                "verified_labels",
                "verified_sentence",
                "review_notes",
                "verification_method",
                "verified_at",
                "verified_by",
            ):
                if k in live and live[k] is not None:
                    merged_row[k] = live[k]
        visible.append(merged_row)
    return visible


def get_status_counts(candidates: list[dict]) -> dict[str, int]:
    """Return status counts derived consistently from the same status field."""
    counts: dict[str, int] = {
        "pending": 0,
        "approved": 0,
        "corrected": 0,
        "rejected": 0,
        "uncertain": 0,
        "other": 0,
    }
    for row in candidates:
        status = str(row.get("annotation_status", "pending")).strip().lower()
        # AI-batch rows still count as pending for review purposes
        if is_pending_review(row):
            counts["pending"] += 1
        elif status == "approved":
            counts["approved"] += 1
        elif status == "corrected":
            counts["corrected"] += 1
        elif status == "rejected":
            counts["rejected"] += 1
        elif status == "uncertain":
            counts["uncertain"] += 1
        else:
            counts["other"] += 1
    return counts


def save_review_decision_with_history(
    candidates: list[dict],
    candidate_id: str,
    action: str,
    reviewer: str,
    verified_labels: Sequence[str] = (),
    verified_sentence: str = "",
    source_text_checked: bool = False,
    review_note: str = "",
    batch_name: str = "",
    verification_method: str = "",
) -> dict:
    """Save a review decision and record it in annotation history."""
    row = next((r for r in candidates if r.get("candidate_id") == candidate_id), None)
    if row is None:
        raise KeyError(candidate_id)

    old_status = str(row.get("annotation_status", "")).strip()
    old_labels = str(row.get("verified_labels", "")).strip()

    updated = save_review_decision(
        candidates=candidates,
        candidate_id=candidate_id,
        action=action,
        reviewer=reviewer,
        verified_labels=verified_labels,
        verified_sentence=verified_sentence,
        source_text_checked=source_text_checked,
        review_note=review_note,
        verification_method=verification_method,
    )

    new_status = str(updated.get("annotation_status", "")).strip()
    new_labels = str(updated.get("verified_labels", "")).strip()

    append_history_row(
        candidate_id=candidate_id,
        reviewer=reviewer,
        action=action,
        old_status=old_status,
        old_labels=old_labels,
        new_status=new_status,
        new_labels=new_labels,
        source_checked=source_text_checked,
        batch_name=batch_name,
        review_note=review_note,
    )
    return updated


def apply_page_verification(
    page_rows: Sequence[MutableMapping[str, str] | dict[str, str]],
    reviewer: str,
    candidates: list[dict[str, str]],
    batch_name: str = "",
) -> list[dict[str, str]]:
    """Apply page attestation verification to decided, unverified rows on the current page.

    Requirements:
      - Only rows on the current page whose status is confirmed / corrected / rejected / uncertain
        and source_verified = false are affected.
      - Pending rows are NEVER touched or made verified/export-eligible.
      - Sets:
          source_verified = true
          source_text_checked = yes
          verified_at = timestamp
          verified_by = reviewer
          verification_method = "page_attestation"
      - Does NOT change labels, annotation_status, or sentence.
      - Records history for each updated row.
    """
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    updated_rows: list[dict[str, str]] = []

    for row in page_rows:
        status = str(row.get("annotation_status", "")).strip().casefold()
        if status not in DECIDED_STATUSES:
            continue
        notes = str(row.get("review_notes", "")).strip()
        if notes.startswith("AI-assisted annotation") and "human_decision=" not in notes:
            continue
        if is_source_verified(row):
            continue

        cid = row.get("candidate_id", "")
        old_status = status
        old_labels = str(row.get("verified_labels", "")).strip()

        row["source_verified"] = "true"
        row["source_text_checked"] = "yes"
        row["verified_at"] = ts
        row["verified_by"] = reviewer.strip()
        row["verification_method"] = "page_attestation"
        if not row.get("annotator"):
            row["annotator"] = reviewer.strip()

        meta_note = f"source_checked=yes; verified_at={ts}; verified_by={reviewer.strip()}; verification_method=page_attestation"
        if notes:
            row["review_notes"] = f"{notes}; {meta_note}"
        else:
            row["review_notes"] = f"human_decision={status}; reviewer={reviewer.strip()}; {meta_note}"

        append_history_row(
            candidate_id=cid,
            reviewer=reviewer.strip(),
            action="page_attestation_verify",
            old_status=old_status,
            old_labels=old_labels,
            new_status=old_status,
            new_labels=old_labels,
            source_checked=True,
            batch_name=batch_name,
            review_note=f"Page attestation verification applied by {reviewer.strip()}",
        )
        updated_rows.append(row)

    return updated_rows


def next_pending_candidate(
    batch_ids: Sequence[str],
    candidates_by_id: MutableMapping[str, dict[str, str]],
    selected_id: str | None,
) -> str | None:
    """Return the next pending item after selection, wrapping once through the batch."""
    pending = [
        candidate_id
        for candidate_id in batch_ids
        if candidate_id in candidates_by_id
        and is_pending_review(candidates_by_id[candidate_id])
    ]
    if not pending:
        return None
    if selected_id not in batch_ids:
        return pending[0]
    selected_index = batch_ids.index(selected_id)
    ordered = list(batch_ids[selected_index + 1:]) + list(batch_ids[:selected_index + 1])
    return next((candidate_id for candidate_id in ordered if candidate_id in pending), pending[0])


def apply_pending_picker_selection(
    session_state: MutableMapping,
    picker_key: str,
    pending_key: str,
    valid_ids: Sequence[str],
) -> None:
    """Apply a queued selection before the picker widget is instantiated."""
    pending_selection = session_state.pop(pending_key, None)
    if pending_selection in valid_ids:
        session_state[picker_key] = pending_selection
    elif session_state.get(picker_key) not in valid_ids and valid_ids:
        session_state[picker_key] = valid_ids[0]
