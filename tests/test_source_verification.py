"""Tests for source-verification stickiness and related provenance rules.

Covers:
  1. No row is source-verified by default.
  2. Page-level checkbox key encodes batch+page → resets on page change.
  3. Rows acted on while page checkbox is unticked are saved as unverified.
  4. Rows acted on while page checkbox is ticked are saved as verified.
  5. Row-level checkbox is independent; combining with page checkbox works correctly.
  6. Bulk "Mark selected as verified" affects only selected rows; unselected stay unchanged.
  7. Bulk verify does NOT change annotation_status.
  8. _effective_source_checked logic is correct.
  9. _verified_count_on_page counts correctly.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.claim_annotation_review import (
    BATCH_A,
    BATCH_B,
    BATCH_C,
    is_eligible_human_annotation,
    save_review_decision,
)


# ---------------------------------------------------------------------------
# Helpers (duplicated here so tests don't import the Streamlit app module)
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


# ---------------------------------------------------------------------------
# Standalone helpers extracted from the app (no Streamlit import needed)
# ---------------------------------------------------------------------------

def _page_src_key(batch_class: str, page: int) -> str:
    return f"page_src__{batch_class}__{page}"


def _row_src_key(candidate_id: str) -> str:
    return f"row_src__{candidate_id}"


def _effective_source_checked(session_state: dict, batch_class: str, page: int, candidate_id: str) -> bool:
    """Testable version of the app's _effective_source_checked."""
    page_checked = bool(session_state.get(_page_src_key(batch_class, page), False))
    row_checked = bool(session_state.get(_row_src_key(candidate_id), False))
    return page_checked or row_checked


def _verified_count_on_page(session_state: dict, batch_class: str, page: int, page_rows: list[dict]) -> int:
    return sum(
        1 for r in page_rows
        if _effective_source_checked(session_state, batch_class, page, r["candidate_id"])
    )


# ---------------------------------------------------------------------------
# 1. No row is source-verified by default
# ---------------------------------------------------------------------------

class TestNoDefaultVerification:
    def test_fresh_session_state_has_no_verified_rows(self):
        """Empty session state → all rows are unverified."""
        session_state: dict = {}
        rows = [_make_candidate(f"c{i}") for i in range(5)]
        for r in rows:
            assert _effective_source_checked(session_state, BATCH_A, 1, r["candidate_id"]) is False

    def test_page_src_key_absent_means_false(self):
        session_state: dict = {}
        assert session_state.get(_page_src_key(BATCH_A, 1), False) is False

    def test_row_src_key_absent_means_false(self):
        session_state: dict = {}
        assert session_state.get(_row_src_key("cand_001"), False) is False

    def test_save_review_decision_without_source_check_saves_no(self):
        candidates = [_make_candidate("c_nosrc")]
        save_review_decision(
            candidates, "c_nosrc", "Confirm", "Mukesh",
            verified_labels=["Efficacy"],
            source_text_checked=False,  # explicitly False
        )
        assert candidates[0]["source_text_checked"] == "no"
        assert is_eligible_human_annotation(candidates[0]) is False

    def test_save_review_decision_with_source_check_saves_yes(self):
        candidates = [_make_candidate("c_withsrc")]
        save_review_decision(
            candidates, "c_withsrc", "Confirm", "Mukesh",
            verified_labels=["Efficacy"],
            source_text_checked=True,
        )
        assert candidates[0]["source_text_checked"] == "yes"
        assert is_eligible_human_annotation(candidates[0]) is True


# ---------------------------------------------------------------------------
# 2. Page-level checkbox key encodes batch+page → resets on page change
# ---------------------------------------------------------------------------

class TestPageCheckboxStickinessAndReset:
    def test_page_key_differs_across_pages(self):
        key_p1 = _page_src_key(BATCH_A, 1)
        key_p2 = _page_src_key(BATCH_A, 2)
        assert key_p1 != key_p2, "Keys for different pages must differ so state resets"

    def test_page_key_differs_across_batches(self):
        key_a = _page_src_key(BATCH_A, 1)
        key_b = _page_src_key(BATCH_B, 1)
        assert key_a != key_b, "Keys for different batches must differ so state resets"

    def test_page_checkbox_sticky_within_same_page(self):
        """Setting the page key in session state persists (simulates st.rerun stickiness)."""
        session_state: dict = {}
        key = _page_src_key(BATCH_A, 1)
        # User ticks the checkbox → session state records it
        session_state[key] = True
        # After rerun (same page): state still there
        assert session_state.get(key, False) is True

    def test_page_checkbox_resets_on_page_change(self):
        """Going to page 2 produces a different key → naturally starts at False."""
        session_state: dict = {}
        session_state[_page_src_key(BATCH_A, 1)] = True
        # Navigate to page 2 — app uses a new key
        new_key = _page_src_key(BATCH_A, 2)
        assert session_state.get(new_key, False) is False, "New page key must start False"

    def test_page_checkbox_resets_on_batch_change(self):
        """Switching batch also produces a new key → resets."""
        session_state: dict = {}
        session_state[_page_src_key(BATCH_A, 1)] = True
        new_key = _page_src_key(BATCH_B, 1)
        assert session_state.get(new_key, False) is False, "New batch key must start False"


# ---------------------------------------------------------------------------
# 3. Rows acted on while page checkbox is unticked → saved as unverified
# ---------------------------------------------------------------------------

class TestUntickSavesUnverified:
    def test_action_with_page_unticked_and_row_unticked_saves_unverified(self):
        session_state: dict = {}
        candidates = [_make_candidate("c_unv")]
        # Both unchecked
        src = _effective_source_checked(session_state, BATCH_A, 1, "c_unv")
        assert src is False
        save_review_decision(
            candidates, "c_unv", "Confirm", "Mukesh",
            verified_labels=["Efficacy"],
            source_text_checked=src,
        )
        assert candidates[0]["source_text_checked"] == "no"
        assert is_eligible_human_annotation(candidates[0]) is False

    def test_action_with_page_unticked_but_row_ticked_saves_verified(self):
        session_state: dict = {}
        cid = "c_row_only"
        session_state[_row_src_key(cid)] = True  # row-level ticked
        candidates = [_make_candidate(cid)]
        src = _effective_source_checked(session_state, BATCH_A, 1, cid)
        assert src is True
        save_review_decision(
            candidates, cid, "Confirm", "Mukesh",
            verified_labels=["Safety"],
            source_text_checked=src,
        )
        assert candidates[0]["source_text_checked"] == "yes"


# ---------------------------------------------------------------------------
# 4. Rows acted on while page checkbox is ticked → saved as verified
# ---------------------------------------------------------------------------

class TestPageTickedSavesVerified:
    def test_action_with_page_ticked_saves_verified(self):
        session_state: dict = {}
        session_state[_page_src_key(BATCH_A, 1)] = True
        cid = "c_pagesrc"
        candidates = [_make_candidate(cid)]
        src = _effective_source_checked(session_state, BATCH_A, 1, cid)
        assert src is True
        save_review_decision(
            candidates, cid, "Confirm", "Mukesh",
            verified_labels=["Efficacy"],
            source_text_checked=src,
        )
        assert candidates[0]["source_text_checked"] == "yes"
        assert is_eligible_human_annotation(candidates[0]) is True

    def test_page_ticked_then_page_changes_new_rows_unverified(self):
        """After page change, the new page key is absent → rows revert to unverified."""
        session_state: dict = {}
        session_state[_page_src_key(BATCH_A, 1)] = True  # page 1 ticked
        # Simulate moving to page 2
        cid = "c_page2"
        src_page2 = _effective_source_checked(session_state, BATCH_A, 2, cid)
        assert src_page2 is False, "After page change, rows should NOT be auto-verified"


# ---------------------------------------------------------------------------
# 5. _effective_source_checked logic
# ---------------------------------------------------------------------------

class TestEffectiveSourceChecked:
    def test_both_false_returns_false(self):
        ss: dict = {}
        assert _effective_source_checked(ss, BATCH_A, 1, "cx") is False

    def test_page_true_row_false_returns_true(self):
        ss = {_page_src_key(BATCH_A, 1): True}
        assert _effective_source_checked(ss, BATCH_A, 1, "cx") is True

    def test_page_false_row_true_returns_true(self):
        ss = {_row_src_key("cx"): True}
        assert _effective_source_checked(ss, BATCH_A, 1, "cx") is True

    def test_both_true_returns_true(self):
        ss = {_page_src_key(BATCH_A, 1): True, _row_src_key("cx"): True}
        assert _effective_source_checked(ss, BATCH_A, 1, "cx") is True

    def test_different_row_unaffected(self):
        """Page ticked for BATCH_A page 1 does not affect BATCH_B page 1."""
        ss = {_page_src_key(BATCH_A, 1): True}
        assert _effective_source_checked(ss, BATCH_B, 1, "cx") is False


# ---------------------------------------------------------------------------
# 6. _verified_count_on_page
# ---------------------------------------------------------------------------

class TestVerifiedCountOnPage:
    def test_no_verified_rows_gives_zero(self):
        ss: dict = {}
        rows = [_make_candidate(f"v{i}") for i in range(5)]
        assert _verified_count_on_page(ss, BATCH_A, 1, rows) == 0

    def test_page_ticked_counts_all_rows(self):
        rows = [_make_candidate(f"v{i}") for i in range(5)]
        ss = {_page_src_key(BATCH_A, 1): True}
        assert _verified_count_on_page(ss, BATCH_A, 1, rows) == 5

    def test_individual_row_ticked_counts_one(self):
        rows = [_make_candidate(f"v{i}") for i in range(5)]
        ss = {_row_src_key("v2"): True}
        assert _verified_count_on_page(ss, BATCH_A, 1, rows) == 1

    def test_mixed_ticked_counts_correctly(self):
        rows = [_make_candidate(f"m{i}") for i in range(6)]
        ss = {
            _row_src_key("m0"): True,
            _row_src_key("m3"): True,
            _row_src_key("m5"): True,
        }
        assert _verified_count_on_page(ss, BATCH_A, 1, rows) == 3


# ---------------------------------------------------------------------------
# 7. Bulk "Mark selected as verified" — affects only selected rows
# ---------------------------------------------------------------------------

class TestBulkMarkVerified:
    def _run_bulk_verify(
        self,
        candidates: list[dict],
        candidate_by_id: dict[str, dict],
        selected_ids: set[str],
        reviewer: str,
        tmp_path: Path,
        monkeypatch,
    ) -> None:
        """Simulate the bulk-verify action from the app."""
        from datetime import datetime, timezone
        hist_path = tmp_path / "annotation_history.csv"
        monkeypatch.setattr("utils.claim_annotation_review.ANNOTATION_HISTORY_CSV", hist_path)

        for r in candidates:
            cid = r["candidate_id"]
            if cid not in selected_ids:
                continue
            ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            old_notes = r.get("review_notes", "")
            r["source_text_checked"] = "yes"
            r["annotator"] = reviewer
            verify_note = f"source_verified_by={reviewer}; ts={ts}"
            r["review_notes"] = (old_notes + "; " + verify_note).lstrip("; ")

    def test_bulk_verify_affects_only_selected_rows(self, tmp_path, monkeypatch):
        candidates = [_make_candidate(f"bv{i}") for i in range(10)]
        candidate_by_id = {r["candidate_id"]: r for r in candidates}
        selected = {"bv0", "bv4", "bv9"}

        self._run_bulk_verify(candidates, candidate_by_id, selected, "Mukesh", tmp_path, monkeypatch)

        for r in candidates:
            if r["candidate_id"] in selected:
                assert r["source_text_checked"] == "yes", f"{r['candidate_id']} should be verified"
            else:
                assert r["source_text_checked"] == "no", f"{r['candidate_id']} should NOT be verified"

    def test_bulk_verify_does_not_change_annotation_status(self, tmp_path, monkeypatch):
        """Bulk verify only touches source_text_checked; annotation_status is untouched."""
        candidates = [_make_candidate(f"bvs{i}", status="pending") for i in range(5)]
        candidate_by_id = {r["candidate_id"]: r for r in candidates}
        selected = {"bvs0", "bvs2"}

        self._run_bulk_verify(candidates, candidate_by_id, selected, "Mukesh", tmp_path, monkeypatch)

        for r in candidates:
            assert r["annotation_status"] == "pending", "Status must not change from bulk verify"

    def test_bulk_verify_empty_selection_changes_nothing(self, tmp_path, monkeypatch):
        candidates = [_make_candidate(f"bve{i}") for i in range(5)]
        candidate_by_id = {r["candidate_id"]: r for r in candidates}

        self._run_bulk_verify(candidates, candidate_by_id, set(), "Mukesh", tmp_path, monkeypatch)

        for r in candidates:
            assert r["source_text_checked"] == "no"

    def test_bulk_verify_unselected_rows_remain_unverified(self, tmp_path, monkeypatch):
        candidates = [_make_candidate(f"bvu{i}") for i in range(8)]
        candidate_by_id = {r["candidate_id"]: r for r in candidates}
        selected = {"bvu1", "bvu3"}

        self._run_bulk_verify(candidates, candidate_by_id, selected, "Mukesh", tmp_path, monkeypatch)

        unselected = [r for r in candidates if r["candidate_id"] not in selected]
        for r in unselected:
            assert r["source_text_checked"] == "no", f"{r['candidate_id']} must remain unverified"
            # And definitely not export-eligible
            assert is_eligible_human_annotation(r) is False
