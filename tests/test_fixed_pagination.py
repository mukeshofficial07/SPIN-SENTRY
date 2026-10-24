"""Tests for fixed-group pagination and stable candidate page assignments.

Verifies:
  1. Reviewing one candidate on Page 1 leaves 24 visible candidates.
  2. The remaining 24 candidates retain their original IDs and order.
  3. Page 2 retains exactly its original candidate IDs.
  4. Reviewing all 25 original Page 1 candidates leaves Page 1 empty.
  5. Page 2 does not refill Page 1.
  6. Page count remains unchanged as candidates are reviewed.
  7. Refreshing or reopening the application preserves the same page assignments.
  8. Saved decisions and source-verification status remain attached to the correct candidate IDs.
  9. All three batches maintain independent page assignments.
  10. Pending and export-eligible counters remain accurate.
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
    PAGE_SIZE,
    calculate_batch_page_count,
    get_fixed_page_slice,
    get_status_counts,
    get_visible_page_candidates,
    is_eligible_human_annotation,
    is_pending_review,
    save_review_decision,
    sort_batch_candidates,
)


def _make_candidate(
    cid: str,
    status: str = "pending",
    source_text_checked: str = "no",
    source_verified: str = "false",
    verified_labels: str = "",
    annotator: str = "",
    pri_score: float = 50.0,
    is_priority: str = "no",
    batch_class: str = BATCH_A,
) -> dict:
    return {
        "candidate_id": cid,
        "paper_id": f"paper_{cid}",
        "title": f"Paper for {cid}",
        "doi": "",
        "url": "",
        "license": "CC BY",
        "license_url": "",
        "source_section": "Results",
        "sentence": f"Sentence for {cid}.",
        "suggested_labels": "Efficacy",
        "suggestion_scores": "",
        "verified_sentence": "",
        "source_text_checked": source_text_checked,
        "source_verified": source_verified,
        "verified_labels": verified_labels,
        "annotation_status": status,
        "annotator": annotator,
        "review_notes": "",
        "extraction_notes": "",
        "machine_review_priority": str(pri_score),
        "is_priority": is_priority,
        "batch_class": batch_class,
    }


# ---------------------------------------------------------------------------
# Test Suite: Fixed-group Pagination & Non-shifting Page Slices
# ---------------------------------------------------------------------------

class TestFixedPagination:
    @pytest.fixture
    def sample_batch_60(self):
        """A batch of 60 candidates (spans 3 pages: 25 + 25 + 10)."""
        # Assign varying priority scores to test deterministic ordering
        rows = [
            _make_candidate(f"cand_{i:03d}", pri_score=(100.0 - (i % 10)))
            for i in range(1, 61)
        ]
        return sort_batch_candidates(rows)

    def test_fixed_page_slices_initial_state(self, sample_batch_60):
        """Page 1 has 25 items, Page 2 has 25 items, Page 3 has 10 items."""
        sorted_batch = sample_batch_60
        total = len(sorted_batch)
        assert total == 60

        n_pages = calculate_batch_page_count(total, PAGE_SIZE)
        assert n_pages == 3

        p1_fixed = get_fixed_page_slice(sorted_batch, 1, PAGE_SIZE)
        p2_fixed = get_fixed_page_slice(sorted_batch, 2, PAGE_SIZE)
        p3_fixed = get_fixed_page_slice(sorted_batch, 3, PAGE_SIZE)

        assert len(p1_fixed) == 25
        assert len(p2_fixed) == 25
        assert len(p3_fixed) == 10

        # No overlap between pages
        p1_ids = {r["candidate_id"] for r in p1_fixed}
        p2_ids = {r["candidate_id"] for r in p2_fixed}
        p3_ids = {r["candidate_id"] for r in p3_fixed}
        assert len(p1_ids & p2_ids) == 0
        assert len(p2_ids & p3_ids) == 0

    def test_review_one_candidate_leaves_24_visible(self, sample_batch_60):
        """Requirement 1: Reviewing one candidate on Page 1 leaves 24 visible candidates."""
        sorted_batch = sample_batch_60
        candidate_by_id = {r["candidate_id"]: dict(r) for r in sorted_batch}

        p1_fixed = get_fixed_page_slice(sorted_batch, 1, PAGE_SIZE)
        target_cid = p1_fixed[0]["candidate_id"]

        # Review the first candidate
        candidate_list = list(candidate_by_id.values())
        save_review_decision(
            candidates=candidate_list,
            candidate_id=target_cid,
            action="Confirm",
            reviewer="Mukesh",
            verified_labels=["Efficacy"],
            source_text_checked=True,
        )
        candidate_by_id = {r["candidate_id"]: r for r in candidate_list}

        # Check visible candidates on Page 1
        p1_visible = get_visible_page_candidates(p1_fixed, candidate_by_id, show_reviewed=False)
        assert len(p1_visible) == 24
        assert target_cid not in [r["candidate_id"] for r in p1_visible]

    def test_remaining_24_retain_original_ids_and_order(self, sample_batch_60):
        """Requirement 2: The remaining 24 candidates retain their original IDs and order."""
        sorted_batch = sample_batch_60
        candidate_by_id = {r["candidate_id"]: dict(r) for r in sorted_batch}

        p1_fixed = get_fixed_page_slice(sorted_batch, 1, PAGE_SIZE)
        original_ids = [r["candidate_id"] for r in p1_fixed]

        # Review the first candidate
        candidate_list = list(candidate_by_id.values())
        save_review_decision(
            candidates=candidate_list,
            candidate_id=original_ids[0],
            action="Confirm",
            reviewer="Mukesh",
            verified_labels=["Efficacy"],
            source_text_checked=True,
        )
        candidate_by_id = {r["candidate_id"]: r for r in candidate_list}

        p1_visible = get_visible_page_candidates(p1_fixed, candidate_by_id, show_reviewed=False)
        visible_ids = [r["candidate_id"] for r in p1_visible]

        # Expected is exactly original_ids[1:] in identical order
        assert visible_ids == original_ids[1:]

    def test_page_2_retains_exactly_original_ids(self, sample_batch_60):
        """Requirement 3: Page 2 retains exactly its original candidate IDs."""
        sorted_batch = sample_batch_60
        candidate_by_id = {r["candidate_id"]: dict(r) for r in sorted_batch}

        p1_fixed = get_fixed_page_slice(sorted_batch, 1, PAGE_SIZE)
        p2_fixed = get_fixed_page_slice(sorted_batch, 2, PAGE_SIZE)
        p2_original_ids = [r["candidate_id"] for r in p2_fixed]

        # Review first candidate on Page 1
        candidate_list = list(candidate_by_id.values())
        save_review_decision(
            candidates=candidate_list,
            candidate_id=p1_fixed[0]["candidate_id"],
            action="Confirm",
            reviewer="Mukesh",
            verified_labels=["Efficacy"],
            source_text_checked=True,
        )
        candidate_by_id = {r["candidate_id"]: r for r in candidate_list}

        # Page 2 slice and visible candidates must be 100% identical to original
        p2_slice = get_fixed_page_slice(sorted_batch, 2, PAGE_SIZE)
        p2_visible = get_visible_page_candidates(p2_slice, candidate_by_id, show_reviewed=False)
        assert [r["candidate_id"] for r in p2_visible] == p2_original_ids

    def test_review_all_25_on_page_1_leaves_page_1_empty(self, sample_batch_60):
        """Requirement 4 & 5: Reviewing all 25 on Page 1 leaves Page 1 empty; Page 2 does NOT refill Page 1."""
        sorted_batch = sample_batch_60
        candidate_by_id = {r["candidate_id"]: dict(r) for r in sorted_batch}

        p1_fixed = get_fixed_page_slice(sorted_batch, 1, PAGE_SIZE)
        p2_fixed = get_fixed_page_slice(sorted_batch, 2, PAGE_SIZE)
        p2_original_ids = [r["candidate_id"] for r in p2_fixed]

        candidate_list = list(candidate_by_id.values())
        for r in p1_fixed:
            save_review_decision(
                candidates=candidate_list,
                candidate_id=r["candidate_id"],
                action="Confirm",
                reviewer="Mukesh",
                verified_labels=["Efficacy"],
                source_text_checked=True,
            )
        candidate_by_id = {r["candidate_id"]: r for r in candidate_list}

        # Page 1 now has 0 visible pending candidates
        p1_visible = get_visible_page_candidates(p1_fixed, candidate_by_id, show_reviewed=False)
        assert len(p1_visible) == 0

        # Page 2 candidates did NOT move into Page 1
        p2_visible = get_visible_page_candidates(p2_fixed, candidate_by_id, show_reviewed=False)
        assert len(p2_visible) == 25
        assert [r["candidate_id"] for r in p2_visible] == p2_original_ids

    def test_page_count_remains_unchanged(self, sample_batch_60):
        """Requirement 6: Page count remains unchanged as candidates are reviewed."""
        sorted_batch = sample_batch_60
        candidate_by_id = {r["candidate_id"]: dict(r) for r in sorted_batch}

        initial_page_count = calculate_batch_page_count(len(sorted_batch), PAGE_SIZE)
        assert initial_page_count == 3

        # Review 30 candidates (all of Page 1, plus 5 on Page 2)
        candidate_list = list(candidate_by_id.values())
        for r in sorted_batch[:30]:
            save_review_decision(
                candidates=candidate_list,
                candidate_id=r["candidate_id"],
                action="Reject",
                reviewer="Mukesh",
                source_text_checked=True,
            )
        candidate_by_id = {r["candidate_id"]: r for r in candidate_list}

        # Total page count is derived from the full batch size and remains 3
        page_count_after = calculate_batch_page_count(len(sorted_batch), PAGE_SIZE)
        assert page_count_after == initial_page_count == 3

    def test_reopening_preserves_same_page_assignments(self, sample_batch_60):
        """Requirement 7: Deterministic reconstruction preserves identical page assignments across sessions."""
        raw_rows = [dict(r) for r in sample_batch_60]

        # Re-sort from raw rows twice
        run1 = sort_batch_candidates(raw_rows)
        run2 = sort_batch_candidates(raw_rows)

        assert [r["candidate_id"] for r in run1] == [r["candidate_id"] for r in run2]

        for p in range(1, 4):
            slice1 = get_fixed_page_slice(run1, p, PAGE_SIZE)
            slice2 = get_fixed_page_slice(run2, p, PAGE_SIZE)
            assert [r["candidate_id"] for r in slice1] == [r["candidate_id"] for r in slice2]

    def test_saved_decisions_attached_to_correct_candidate_ids(self, sample_batch_60):
        """Requirement 8: Decisions and verification metadata remain attached to exact candidate IDs."""
        sorted_batch = sample_batch_60
        candidate_list = [dict(r) for r in sorted_batch]
        target = sorted_batch[12]
        target_id = target["candidate_id"]

        save_review_decision(
            candidates=candidate_list,
            candidate_id=target_id,
            action="Correct",
            reviewer="Mukesh",
            verified_labels=["Safety", "Recommendation"],
            source_text_checked=True,
            verification_method="page_attestation",
        )

        saved = next(r for r in candidate_list if r["candidate_id"] == target_id)
        assert saved["annotation_status"] == "corrected"
        assert saved["verified_labels"] == "Safety;Recommendation"
        assert saved["source_verified"] == "true"
        assert saved["verification_method"] == "page_attestation"
        assert saved["annotator"] == "Mukesh"

        # Other candidates remain pending and untouched
        other = next(r for r in candidate_list if r["candidate_id"] != target_id)
        assert other["annotation_status"] == "pending"
        assert is_pending_review(other) is True

    def test_all_three_batches_maintain_independent_assignments(self):
        """Requirement 9: All three batches maintain independent page assignments."""
        rows_a = [_make_candidate(f"a_{i:02d}", batch_class=BATCH_A) for i in range(30)]
        rows_b = [_make_candidate(f"b_{i:02d}", batch_class=BATCH_B) for i in range(15)]
        rows_c = [_make_candidate(f"c_{i:02d}", batch_class=BATCH_C) for i in range(50)]

        sorted_a = sort_batch_candidates(rows_a)
        sorted_b = sort_batch_candidates(rows_b)
        sorted_c = sort_batch_candidates(rows_c)

        assert calculate_batch_page_count(len(sorted_a), PAGE_SIZE) == 2
        assert calculate_batch_page_count(len(sorted_b), PAGE_SIZE) == 1
        assert calculate_batch_page_count(len(sorted_c), PAGE_SIZE) == 2

        # Page 1 of Batch A has only 'a_' IDs
        assert all(r["candidate_id"].startswith("a_") for r in get_fixed_page_slice(sorted_a, 1, PAGE_SIZE))
        # Page 1 of Batch B has only 'b_' IDs
        assert all(r["candidate_id"].startswith("b_") for r in get_fixed_page_slice(sorted_b, 1, PAGE_SIZE))
        # Page 1 of Batch C has only 'c_' IDs
        assert all(r["candidate_id"].startswith("c_") for r in get_fixed_page_slice(sorted_c, 1, PAGE_SIZE))

    def test_pending_and_export_eligible_counters_accurate(self, sample_batch_60):
        """Requirement 10: Pending and export-eligible counters remain accurate."""
        candidate_list = [dict(r) for r in sample_batch_60]
        counts_init = get_status_counts(candidate_list)
        assert counts_init["pending"] == 60
        assert sum(1 for r in candidate_list if is_eligible_human_annotation(r)) == 0

        # Review 1 confirmed (source verified) -> eligible
        save_review_decision(
            candidate_list, candidate_list[0]["candidate_id"],
            action="Confirm", reviewer="Mukesh", verified_labels=["Efficacy"], source_text_checked=True,
        )
        # Review 1 rejected (source verified) -> eligible
        save_review_decision(
            candidate_list, candidate_list[1]["candidate_id"],
            action="Reject", reviewer="Mukesh", source_text_checked=True,
        )
        # Review 1 confirmed (source NOT verified) -> NOT eligible
        save_review_decision(
            candidate_list, candidate_list[2]["candidate_id"],
            action="Confirm", reviewer="Mukesh", verified_labels=["Safety"], source_text_checked=False,
        )
        # Review 1 uncertain (source verified) -> NOT eligible
        save_review_decision(
            candidate_list, candidate_list[3]["candidate_id"],
            action="Mark Uncertain", reviewer="Mukesh", source_text_checked=True,
        )

        counts_after = get_status_counts(candidate_list)
        assert counts_after["pending"] == 56
        assert counts_after["approved"] == 2  # candidate 0 and candidate 2
        assert counts_after["rejected"] == 1  # candidate 1
        assert counts_after["uncertain"] == 1 # candidate 3
        assert sum(counts_after.values()) == 60

        eligible_count = sum(1 for r in candidate_list if is_eligible_human_annotation(r))
        assert eligible_count == 2  # candidate 0 and candidate 1

    def test_show_reviewed_toggle_preserves_original_positions(self, sample_batch_60):
        """Toggle 'Show reviewed' displays completed candidates in their original positions."""
        sorted_batch = sample_batch_60
        candidate_list = [dict(r) for r in sorted_batch]
        candidate_by_id = {r["candidate_id"]: r for r in candidate_list}

        p1_fixed = get_fixed_page_slice(sorted_batch, 1, PAGE_SIZE)
        p1_original_ids = [r["candidate_id"] for r in p1_fixed]

        # Review index 3 and 7
        save_review_decision(
            candidate_list, p1_original_ids[3],
            action="Confirm", reviewer="Mukesh", verified_labels=["Efficacy"], source_text_checked=True,
        )
        save_review_decision(
            candidate_list, p1_original_ids[7],
            action="Reject", reviewer="Mukesh", source_text_checked=True,
        )
        candidate_by_id = {r["candidate_id"]: r for r in candidate_list}

        # With show_reviewed=False: 23 visible
        visible_pending = get_visible_page_candidates(p1_fixed, candidate_by_id, show_reviewed=False)
        assert len(visible_pending) == 23

        # With show_reviewed=True: 25 items in exact original order
        visible_all = get_visible_page_candidates(p1_fixed, candidate_by_id, show_reviewed=True)
        assert len(visible_all) == 25
        assert [r["candidate_id"] for r in visible_all] == p1_original_ids
