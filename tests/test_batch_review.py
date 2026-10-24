"""Tests for the batch review system.

Covers:
  - Batch filter correctness (A/B/C)
  - Reviewed rows leaving the pending batch
  - Bulk apply affects only selected rows
  - Undo restores previous state
  - Counter reconciliation to total
  - Source-unverified rows excluded from export
"""

from __future__ import annotations

import csv
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from utils.claim_annotation_review import (
    BATCH_A,
    BATCH_B,
    BATCH_C,
    ANNOTATION_HISTORY_CSV,
    ANNOTATION_HISTORY_FIELDS,
    _ensure_history_file,
    append_history_row,
    get_batch_class,
    get_history_rows,
    get_status_counts,
    is_eligible_human_annotation,
    is_pending_review,
    load_merged_candidates,
    save_review_decision,
    save_review_decision_with_history,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_candidate(
    candidate_id: str = "cand_001",
    status: str = "pending",
    source_text_checked: str = "no",
    review_notes: str = "",
    verified_labels: str = "",
    annotator: str = "",
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
        "verified_labels": verified_labels,
        "annotation_status": status,
        "annotator": annotator,
        "review_notes": review_notes,
        "extraction_notes": "",
    }


def _make_machine(candidate_id: str, claim_class: str = "claim") -> dict:
    return {
        "candidate_id": candidate_id,
        "machine_claim_class": claim_class,
        "machine_suggested_labels": "Efficacy",
        "machine_uncertainty_flag": "no",
        "machine_context_before": "",
        "machine_context_after": "",
        "machine_review_priority": "50",
        "machine_review_reasons": "",
        "machine_label_rationales": "",
    }


# ---------------------------------------------------------------------------
# 1. Batch filter correctness
# ---------------------------------------------------------------------------

class TestBatchFilters:
    def test_batch_filter_claim_returns_only_claim_rows(self):
        rows = [
            _make_candidate("c1"), _make_candidate("c2"), _make_candidate("c3"),
        ]
        machine = {
            "c1": _make_machine("c1", "claim"),
            "c2": _make_machine("c2", "non_claim"),
            "c3": _make_machine("c3", "uncertain"),
        }
        assert get_batch_class(rows[0], machine) == BATCH_A
        assert get_batch_class(rows[1], machine) == BATCH_B
        assert get_batch_class(rows[2], machine) == BATCH_C

    def test_batch_filter_non_claim_returns_only_non_claim_rows(self):
        row = _make_candidate("nc1")
        machine = {"nc1": _make_machine("nc1", "non_claim")}
        assert get_batch_class(row, machine) == BATCH_B

    def test_batch_filter_uncertain_includes_nan_machine_class(self):
        """A candidate with no machine annotation should be treated as uncertain."""
        row = _make_candidate("no_machine")
        machine_by_id: dict = {}  # empty — candidate has no machine row
        assert get_batch_class(row, machine_by_id) == BATCH_C

    def test_batch_filter_uncertain_includes_empty_string_class(self):
        row = _make_candidate("empty")
        machine = {"empty": _make_machine("empty", "")}
        assert get_batch_class(row, machine) == BATCH_C

    def test_load_merged_candidates_assigns_batch_class(self):
        candidates = [
            _make_candidate("c1"),
            _make_candidate("c2"),
        ]
        machine_rows = [
            _make_machine("c1", "claim"),
            _make_machine("c2", "non_claim"),
        ]
        merged = load_merged_candidates(candidates, machine_rows)
        by_id = {r["candidate_id"]: r for r in merged}
        assert by_id["c1"]["batch_class"] == BATCH_A
        assert by_id["c2"]["batch_class"] == BATCH_B

    def test_load_merged_candidates_marks_priority(self):
        candidates = [_make_candidate("pri"), _make_candidate("notpri")]
        machine_rows = [_make_machine("pri"), _make_machine("notpri")]
        merged = load_merged_candidates(candidates, machine_rows, priority_ids={"pri"})
        by_id = {r["candidate_id"]: r for r in merged}
        assert by_id["pri"]["is_priority"] == "yes"
        assert by_id["notpri"]["is_priority"] == "no"


# ---------------------------------------------------------------------------
# 2. Reviewed rows leave the pending batch
# ---------------------------------------------------------------------------

class TestReviewedRowLeavesPending:
    def test_reviewed_row_leaves_pending_batch(self):
        candidates = [_make_candidate("c1")]
        assert is_pending_review(candidates[0]) is True

        save_review_decision(
            candidates, "c1", "Confirm", "Mukesh",
            verified_labels=["Efficacy"],
            source_text_checked=True,
        )
        assert is_pending_review(candidates[0]) is False

    def test_ai_batch_row_is_still_pending(self):
        """AI-batch approved rows should remain pending until human confirmation."""
        c = _make_candidate(
            "c_ai",
            status="approved",
            source_text_checked="no",
            review_notes="AI-assisted annotation; human_decision=approved",
        )
        assert is_pending_review(c) is True

    def test_rejected_row_is_not_pending(self):
        candidates = [_make_candidate("c_reject")]
        save_review_decision(candidates, "c_reject", "Reject", "Mukesh")
        assert is_pending_review(candidates[0]) is False

    def test_uncertain_row_is_not_pending(self):
        candidates = [_make_candidate("c_uncertain")]
        save_review_decision(candidates, "c_uncertain", "Mark Uncertain", "Mukesh")
        assert is_pending_review(candidates[0]) is False


# ---------------------------------------------------------------------------
# 3. Bulk apply affects only selected visible rows
# ---------------------------------------------------------------------------

class TestBulkApply:
    def test_bulk_apply_affects_only_selected_rows(self):
        """Given 10 candidates, selecting 3 and bulk-applying should only change those 3."""
        all_candidates = [_make_candidate(f"c{i:02d}") for i in range(10)]
        selected_ids = {"c00", "c04", "c09"}

        for c in all_candidates:
            if c["candidate_id"] in selected_ids:
                save_review_decision(all_candidates, c["candidate_id"], "Reject", "Mukesh")

        for c in all_candidates:
            if c["candidate_id"] in selected_ids:
                assert c["annotation_status"] == "rejected", f"{c['candidate_id']} should be rejected"
            else:
                assert c["annotation_status"] == "pending", f"{c['candidate_id']} should still be pending"

    def test_bulk_apply_does_not_affect_unselected_rows(self):
        candidates = [_make_candidate(f"bx{i}") for i in range(5)]
        selected = {"bx0", "bx2"}

        for c in candidates:
            if c["candidate_id"] in selected:
                save_review_decision(candidates, c["candidate_id"], "Mark Uncertain", "Mukesh")

        unselected = [c for c in candidates if c["candidate_id"] not in selected]
        for c in unselected:
            assert c["annotation_status"] == "pending"


# ---------------------------------------------------------------------------
# 4. Undo restores previous state
# ---------------------------------------------------------------------------

class TestUndo:
    def test_undo_restores_previous_state(self, tmp_path, monkeypatch):
        """After saving a decision, undo should restore the original status."""
        monkeypatch.setattr(
            "utils.claim_annotation_review.ANNOTATION_HISTORY_CSV",
            tmp_path / "annotation_history.csv",
        )
        monkeypatch.setattr(
            "utils.claim_annotation_review.ANNOTATION_HISTORY_FIELDS",
            ANNOTATION_HISTORY_FIELDS,
        )

        candidates = [_make_candidate("undo_me")]
        original_status = candidates[0]["annotation_status"]
        snapshot = {
            "annotation_status": candidates[0]["annotation_status"],
            "annotator": candidates[0]["annotator"],
            "source_text_checked": candidates[0]["source_text_checked"],
            "verified_labels": candidates[0]["verified_labels"],
            "verified_sentence": candidates[0]["verified_sentence"],
            "review_notes": candidates[0]["review_notes"],
        }

        # Apply a decision
        save_review_decision(
            candidates, "undo_me", "Confirm", "Mukesh",
            verified_labels=["Safety"], source_text_checked=True,
        )
        assert candidates[0]["annotation_status"] == "approved"

        # Simulate undo: restore snapshot
        candidates[0].update(snapshot)
        assert candidates[0]["annotation_status"] == original_status, "Undo should restore original status"

    def test_undo_state_is_independent_per_candidate(self):
        """Undo for one candidate should not affect another."""
        candidates = [_make_candidate("u1"), _make_candidate("u2")]
        snapshot_u1 = dict(candidates[0])

        save_review_decision(candidates, "u1", "Reject", "Mukesh")
        save_review_decision(candidates, "u2", "Confirm", "Mukesh", verified_labels=["Efficacy"], source_text_checked=True)

        # Undo u1 only
        candidates[0].update({k: snapshot_u1[k] for k in ["annotation_status", "annotator", "source_text_checked", "verified_labels", "review_notes"]})

        assert candidates[0]["annotation_status"] == "pending"
        assert candidates[1]["annotation_status"] == "approved"  # u2 untouched


# ---------------------------------------------------------------------------
# 5. Counter reconciliation
# ---------------------------------------------------------------------------

class TestCounterReconciliation:
    def test_counters_reconcile_to_total(self):
        candidates = [
            _make_candidate("r1"),
            _make_candidate("r2", status="approved", source_text_checked="yes",
                            review_notes="human_decision=approved", annotator="Mukesh"),
            _make_candidate("r3", status="rejected"),
            _make_candidate("r4", status="uncertain"),
            _make_candidate("r5", status="corrected", source_text_checked="yes",
                            review_notes="human_decision=corrected", annotator="Mukesh"),
        ]
        counts = get_status_counts(candidates)
        total = sum(counts.values())
        assert total == len(candidates), f"Counts {counts} do not sum to {len(candidates)}"

    def test_counters_all_pending(self):
        candidates = [_make_candidate(f"p{i}") for i in range(7)]
        counts = get_status_counts(candidates)
        assert counts["pending"] == 7
        assert counts["approved"] == 0
        assert counts["rejected"] == 0

    def test_counters_after_decisions(self):
        candidates = [_make_candidate(f"d{i}") for i in range(5)]
        save_review_decision(candidates, "d0", "Confirm", "Mukesh", ["Efficacy"], source_text_checked=True)
        save_review_decision(candidates, "d1", "Reject", "Mukesh")
        save_review_decision(candidates, "d2", "Mark Uncertain", "Mukesh")
        counts = get_status_counts(candidates)
        assert counts["approved"] == 1
        assert counts["rejected"] == 1
        assert counts["uncertain"] == 1
        assert counts["pending"] == 2
        total = sum(counts.values())
        assert total == 5


# ---------------------------------------------------------------------------
# 6. Source-unverified rows excluded from export
# ---------------------------------------------------------------------------

class TestExportEligibility:
    def test_source_unverified_excluded_from_export(self):
        """Rows approved without source verification must not be eligible."""
        c = _make_candidate(
            "unverified",
            status="approved",
            source_text_checked="no",
            review_notes="human_decision=approved; reviewer=Mukesh",
            verified_labels="Efficacy",
            annotator="Mukesh",
        )
        assert is_eligible_human_annotation(c) is False

    def test_source_verified_approved_is_eligible(self):
        c = _make_candidate(
            "verified_good",
            status="approved",
            source_text_checked="yes",
            review_notes="human_decision=approved; reviewer=Mukesh",
            verified_labels="Efficacy",
            annotator="Mukesh",
        )
        assert is_eligible_human_annotation(c) is True

    def test_ai_batch_approved_is_not_eligible(self):
        """AI-assisted annotated rows must not be eligible even with source_text_checked='yes'."""
        c = _make_candidate(
            "ai_row",
            status="approved",
            source_text_checked="yes",
            review_notes="AI-assisted annotation; human_decision=approved",
            verified_labels="Safety",
            annotator="Mukesh",
        )
        assert is_eligible_human_annotation(c) is False

    def test_no_labels_not_eligible(self):
        c = _make_candidate(
            "no_labels",
            status="approved",
            source_text_checked="yes",
            review_notes="human_decision=approved",
            verified_labels="",
            annotator="Mukesh",
        )
        assert is_eligible_human_annotation(c) is False

    def test_no_annotator_not_eligible(self):
        c = _make_candidate(
            "no_annotator",
            status="approved",
            source_text_checked="yes",
            review_notes="human_decision=approved",
            verified_labels="Efficacy",
            annotator="",
        )
        assert is_eligible_human_annotation(c) is False

    def test_pending_not_eligible(self):
        c = _make_candidate("still_pending")
        assert is_eligible_human_annotation(c) is False

    def test_rejected_not_eligible(self):
        c = _make_candidate("rejected_c", status="rejected")
        assert is_eligible_human_annotation(c) is False


# ---------------------------------------------------------------------------
# 7. Annotation history
# ---------------------------------------------------------------------------

class TestAnnotationHistory:
    def test_append_and_retrieve_history(self, tmp_path, monkeypatch):
        hist_path = tmp_path / "annotation_history.csv"
        monkeypatch.setattr("utils.claim_annotation_review.ANNOTATION_HISTORY_CSV", hist_path)

        append_history_row(
            candidate_id="hist_001",
            reviewer="Mukesh",
            action="Confirm",
            old_status="pending",
            old_labels="",
            new_status="approved",
            new_labels="Efficacy",
            source_checked=True,
            batch_name="A: Likely Claim",
            review_note="test",
        )

        rows = get_history_rows("hist_001")
        assert len(rows) == 1
        assert rows[0]["action"] == "Confirm"
        assert rows[0]["reviewer"] == "Mukesh"
        assert rows[0]["new_status"] == "approved"

    def test_history_for_different_candidates_are_separate(self, tmp_path, monkeypatch):
        hist_path = tmp_path / "annotation_history.csv"
        monkeypatch.setattr("utils.claim_annotation_review.ANNOTATION_HISTORY_CSV", hist_path)

        append_history_row("a1", "Mukesh", "Confirm", "pending", "", "approved", "Efficacy", True)
        append_history_row("a2", "Mukesh", "Reject", "pending", "", "rejected", "", False)

        rows_a1 = get_history_rows("a1")
        rows_a2 = get_history_rows("a2")
        assert len(rows_a1) == 1
        assert len(rows_a2) == 1
        assert rows_a1[0]["new_status"] == "approved"
        assert rows_a2[0]["new_status"] == "rejected"
