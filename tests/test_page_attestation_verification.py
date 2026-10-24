"""Tests for page-level attestation verification workflow and export eligibility.

Requirements verified:
  1. Nothing is verified by default; checkboxes start False and never default from config/script.
  2. Page checkbox resets on page change or batch change.
  3. "Apply verification to N decided rows on this page" affects ONLY decided rows
     on the current page (status in confirmed, corrected, rejected, uncertain) with source_verified = false.
  4. Pending rows (no human decision) NEVER become verified or export-eligible when page is ticked or applied.
  5. Labels, status, and sentences are NEVER modified by page attestation verification.
  6. Export-eligible = status in (confirmed, corrected, rejected) AND source_verified = true AND human action.
  7. Counter equals export row count across all conditions.
  8. Badges distinguish "verified (page)" vs "verified (row)" vs "not verified".
  9. New decisions while page checkbox is ticked are saved with verification_method="page_attestation".
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.claim_annotation_review import (
    BATCH_A,
    BATCH_B,
    BATCH_C,
    DECIDED_STATUSES,
    EXPORT_ELIGIBLE_STATUSES,
    apply_page_verification,
    get_verification_method,
    is_decided,
    is_eligible_human_annotation,
    is_source_verified,
    save_review_decision,
)


def _make_candidate(
    candidate_id: str = "cand_001",
    status: str = "pending",
    source_text_checked: str = "no",
    source_verified: str = "false",
    review_notes: str = "",
    verified_labels: str = "",
    annotator: str = "",
    verification_method: str = "",
) -> dict:
    return {
        "candidate_id": candidate_id,
        "paper_id": "paper_001",
        "title": "Test Paper",
        "doi": "",
        "url": "",
        "license": "CC BY",
        "license_url": "",
        "source_section": "Results",
        "sentence": "This treatment improved outcomes significantly.",
        "suggested_labels": "",
        "suggestion_scores": "",
        "verified_sentence": "",
        "source_text_checked": source_text_checked,
        "source_verified": source_verified,
        "verified_labels": verified_labels,
        "annotation_status": status,
        "annotator": annotator,
        "review_notes": review_notes,
        "extraction_notes": "",
        "verification_method": verification_method,
    }


def _page_src_key(batch_class: str, page: int) -> str:
    return f"page_src__{batch_class}__{page}"


def _row_verification_badge(row: dict) -> tuple[str, str]:
    if is_source_verified(row):
        vm = get_verification_method(row)
        if vm == "page_attestation":
            return ("🔒 verified (page)", "#10b981")
        return ("🔒 verified (row)", "#3b82f6")
    return ("🔓 not verified", "#f59e0b")


# ---------------------------------------------------------------------------
# 1. Nothing verified by default & key resets
# ---------------------------------------------------------------------------

class TestDefaultsAndReset:
    def test_nothing_verified_by_default(self):
        """No fresh candidate is verified by default."""
        row = _make_candidate("fresh_row")
        assert not is_source_verified(row)
        assert not is_eligible_human_annotation(row)
        assert _row_verification_badge(row)[0] == "🔓 not verified"

    def test_page_change_resets_checkbox_key(self):
        """Navigating from page 1 to page 2 changes the key, resetting state to False."""
        key_p1 = _page_src_key(BATCH_A, 1)
        key_p2 = _page_src_key(BATCH_A, 2)
        assert key_p1 != key_p2

        session_state = {key_p1: True}
        assert session_state.get(key_p2, False) is False

    def test_batch_change_resets_checkbox_key(self):
        """Switching from Batch A to Batch B produces a distinct key."""
        key_a = _page_src_key(BATCH_A, 1)
        key_b = _page_src_key(BATCH_B, 1)
        assert key_a != key_b


# ---------------------------------------------------------------------------
# 2. Apply affects only decided rows on the current page
# ---------------------------------------------------------------------------

class TestApplyVerificationScope:
    def test_apply_affects_only_decided_unverified_rows_on_page(self, tmp_path, monkeypatch):
        hist_file = tmp_path / "annotation_history.csv"
        monkeypatch.setattr("utils.claim_annotation_review.ANNOTATION_HISTORY_CSV", hist_file)

        # Current page has:
        # 1 approved unverified, 1 rejected unverified, 1 uncertain unverified,
        # 1 pending unverified, 1 already verified approved
        page_rows = [
            _make_candidate("c_app", status="approved", verified_labels="Efficacy", annotator="Mukesh"),
            _make_candidate("c_rej", status="rejected", annotator="Mukesh"),
            _make_candidate("c_unc", status="uncertain", annotator="Mukesh"),
            _make_candidate("c_pen", status="pending"),
            _make_candidate("c_alr", status="approved", source_text_checked="yes", source_verified="true",
                            verified_labels="Safety", annotator="Mukesh", verification_method="row_attestation"),
        ]
        # Row on another page
        other_page_row = _make_candidate("c_oth", status="approved", verified_labels="Novelty", annotator="Mukesh")
        candidates = page_rows + [other_page_row]

        updated = apply_page_verification(
            page_rows=page_rows,
            reviewer="Mukesh",
            candidates=candidates,
            batch_name="A: Likely Claim",
        )

        updated_ids = {r["candidate_id"] for r in updated}
        assert updated_ids == {"c_app", "c_rej", "c_unc"}

        # c_app, c_rej, c_unc are now verified
        assert is_source_verified(page_rows[0])
        assert page_rows[0]["verification_method"] == "page_attestation"
        assert page_rows[0]["verified_by"] == "Mukesh"

        assert is_source_verified(page_rows[1])
        assert page_rows[1]["verification_method"] == "page_attestation"

        assert is_source_verified(page_rows[2])
        assert page_rows[2]["verification_method"] == "page_attestation"

        # c_alr untouched (already verified)
        assert page_rows[4]["verification_method"] == "row_attestation"

        # other_page_row was NOT on page_rows, so untouched
        assert not is_source_verified(other_page_row)


# ---------------------------------------------------------------------------
# 3. Pending rows stay ineligible
# ---------------------------------------------------------------------------

class TestPendingRowsStayIneligible:
    def test_pending_rows_stay_ineligible_after_apply(self, tmp_path, monkeypatch):
        hist_file = tmp_path / "annotation_history.csv"
        monkeypatch.setattr("utils.claim_annotation_review.ANNOTATION_HISTORY_CSV", hist_file)

        pending_row = _make_candidate("c_pending_stay", status="pending")
        decided_row = _make_candidate("c_dec", status="approved", verified_labels="Efficacy", annotator="Mukesh")
        candidates = [pending_row, decided_row]

        apply_page_verification([pending_row, decided_row], "Mukesh", candidates)

        assert not is_source_verified(pending_row)
        assert not is_eligible_human_annotation(pending_row)
        assert pending_row["annotation_status"] == "pending"


# ---------------------------------------------------------------------------
# 4. Labels and status unchanged
# ---------------------------------------------------------------------------

class TestLabelsAndStatusUnchanged:
    def test_labels_and_status_unchanged_after_apply(self, tmp_path, monkeypatch):
        hist_file = tmp_path / "annotation_history.csv"
        monkeypatch.setattr("utils.claim_annotation_review.ANNOTATION_HISTORY_CSV", hist_file)

        row = _make_candidate(
            "c_preserve",
            status="corrected",
            verified_labels="Efficacy;Comparative",
            annotator="Mukesh",
        )
        candidates = [row]

        apply_page_verification([row], "Mukesh", candidates)

        assert row["annotation_status"] == "corrected"
        assert row["verified_labels"] == "Efficacy;Comparative"
        assert row["sentence"] == "This treatment improved outcomes significantly."
        assert is_source_verified(row)
        assert row["verification_method"] == "page_attestation"


# ---------------------------------------------------------------------------
# 5. Export-eligible rules & counter reconciliation
# ---------------------------------------------------------------------------

class TestExportEligibilityAndCounter:
    def test_counter_equals_export_row_count(self):
        """Sidebar counter rule and export rule must match exactly."""
        candidates = [
            # Confirmed + verified -> eligible
            _make_candidate("c1", status="approved", source_text_checked="yes",
                            verified_labels="Efficacy", annotator="Mukesh"),
            # Corrected + verified -> eligible
            _make_candidate("c2", status="corrected", source_text_checked="yes",
                            verified_labels="Safety", annotator="Mukesh"),
            # Rejected + verified -> eligible
            _make_candidate("c3", status="rejected", source_text_checked="yes",
                            annotator="Mukesh"),
            # Uncertain + verified -> NOT eligible
            _make_candidate("c4", status="uncertain", source_text_checked="yes",
                            annotator="Mukesh"),
            # Confirmed but UNVERIFIED -> NOT eligible
            _make_candidate("c5", status="approved", source_text_checked="no",
                            verified_labels="Efficacy", annotator="Mukesh"),
            # Pending -> NOT eligible
            _make_candidate("c6", status="pending"),
        ]

        sidebar_counter = sum(1 for r in candidates if is_eligible_human_annotation(r))
        exported_rows = [r for r in candidates if is_eligible_human_annotation(r)]

        assert sidebar_counter == len(exported_rows) == 3
        exported_ids = {r["candidate_id"] for r in exported_rows}
        assert exported_ids == {"c1", "c2", "c3"}

    def test_rejected_with_source_verification_is_export_eligible(self):
        row = _make_candidate("c_rej_v", status="rejected", source_text_checked="yes", annotator="Mukesh")
        assert is_eligible_human_annotation(row) is True

    def test_rejected_without_source_verification_is_ineligible(self):
        row = _make_candidate("c_rej_uv", status="rejected", source_text_checked="no", annotator="Mukesh")
        assert is_eligible_human_annotation(row) is False

    def test_uncertain_with_source_verification_is_ineligible(self):
        row = _make_candidate("c_unc_v", status="uncertain", source_text_checked="yes", annotator="Mukesh")
        assert is_eligible_human_annotation(row) is False


# ---------------------------------------------------------------------------
# 6. Badges & new actions while page is ticked
# ---------------------------------------------------------------------------

class TestBadgesAndNewActions:
    def test_row_badge_distinguishes_page_vs_row_vs_not_verified(self):
        row_page = _make_candidate("p1", status="approved", source_text_checked="yes",
                                   verification_method="page_attestation")
        row_row = _make_candidate("r1", status="approved", source_text_checked="yes",
                                  verification_method="row_attestation")
        row_unv = _make_candidate("u1", status="approved", source_text_checked="no")

        assert _row_verification_badge(row_page)[0] == "🔒 verified (page)"
        assert _row_verification_badge(row_row)[0] == "🔒 verified (row)"
        assert _row_verification_badge(row_unv)[0] == "🔓 not verified"

    def test_new_action_while_page_ticked_saves_page_attestation(self):
        candidates = [_make_candidate("c_new")]
        save_review_decision(
            candidates=candidates,
            candidate_id="c_new",
            action="Confirm",
            reviewer="Mukesh",
            verified_labels=["Efficacy"],
            source_text_checked=True,
            verification_method="page_attestation",
        )

        saved = candidates[0]
        assert is_source_verified(saved)
        assert saved["verification_method"] == "page_attestation"
        assert _row_verification_badge(saved)[0] == "🔒 verified (page)"
        assert is_eligible_human_annotation(saved) is True
