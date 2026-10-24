"""
tests/test_human_review_integrity.py
------------------------------------
Regression tests verifying:
  1. Machine/AI labels are never falsely counted as human reviews even if annotator='Mukesh'.
  2. Source-unverified rows remain ineligible for human-reviewed training export.
  3. Unverified candidates remain visible and reviewable in the pending queue.
  4. Status counters are recalculated honestly from underlying records.
  5. Export eligibility filters properly and enforces duplicate consistency.
  6. Saving review decisions properly records timestamp, reviewer, decision, and provenance.
  7. Rejection and uncertainty decisions clear verified labels.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.export_claim_annotations import approved_rows, write_export
from utils.claim_annotation_review import (
    is_eligible_human_annotation,
    is_pending_review,
    next_pending_candidate,
    save_review_decision,
)


def _make_candidate(candidate_id: str, **kwargs) -> dict[str, str]:
    base = {
        "candidate_id": candidate_id,
        "paper_id": "PMC1000",
        "sentence": f"Clinical claim sentence for candidate {candidate_id}.",
        "doi": "10.1234/test",
        "url": "https://doi.org/10.1234/test",
        "license": "CC BY 4.0",
        "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "source_section": "Results",
        "suggested_labels": "Efficacy",
        "suggestion_scores": '{"Efficacy": 0.85}',
        "verified_sentence": "",
        "source_text_checked": "no",
        "verified_labels": "",
        "annotation_status": "pending",
        "annotator": "",
        "review_notes": "",
        "extraction_notes": "",
    }
    base.update(kwargs)
    return base


def test_machine_labels_not_counted_as_human_reviews():
    """AI-assisted batch rows stamped with annotator='Mukesh' must NOT be counted as human reviews."""
    ai_row = _make_candidate(
        "ai-1",
        annotation_status="approved",
        annotator="Mukesh",
        source_text_checked="no",
        verified_labels="Efficacy",
        review_notes="AI-assisted annotation (claim_class=claim); model=rules; ts=2026-10-20T07:43:00Z",
    )
    assert not is_eligible_human_annotation(ai_row), (
        "AI-assisted annotation stamped with Mukesh must be rejected as human review"
    )

    human_row = _make_candidate(
        "human-1",
        annotation_status="approved",
        annotator="Mukesh",
        source_text_checked="yes",
        verified_labels="Efficacy",
        review_notes="Verified against primary endpoint in Table 3",
    )
    assert is_eligible_human_annotation(human_row) is True


def test_source_unverified_rows_ineligible_for_export(tmp_path):
    """Rows with source_text_checked != 'yes' must remain ineligible for human-reviewed export."""
    unverified = _make_candidate(
        "unv-1",
        annotation_status="approved",
        annotator="Reviewer A",
        source_text_checked="no",
        verified_labels="Safety",
        review_notes="",
    )
    verified = _make_candidate(
        "ver-1",
        annotation_status="approved",
        annotator="Reviewer A",
        source_text_checked="yes",
        verified_labels="Safety",
        review_notes="Checked source",
    )

    # approved_rows must exclude unverified without crashing on verified
    rows = approved_rows([unverified, verified])
    assert len(rows) == 1
    assert rows[0]["labels"] == "Safety"
    assert "ver-1" in rows[0]["notes"]

    # When all rows are unverified, write_export must raise descriptive ValueError
    out_file = tmp_path / "export_test.csv"
    empty_rows = approved_rows([unverified])
    assert len(empty_rows) == 0
    with pytest.raises(ValueError, match="no approved/corrected human annotations"):
        write_export(out_file, empty_rows)


def test_queue_visibility_for_unverified_candidates():
    """Unverified candidates must remain visible in the pending queue even if previous automation marked them approved."""
    ai_stamped = _make_candidate(
        "ai-2",
        annotation_status="approved",
        source_text_checked="no",
        annotator="Mukesh",
        review_notes="AI-assisted annotation",
    )
    genuine_human = _make_candidate(
        "hum-2",
        annotation_status="approved",
        source_text_checked="yes",
        annotator="Mukesh",
        verified_labels="Novelty",
    )
    rejected = _make_candidate("rej-1", annotation_status="rejected")

    assert is_pending_review(ai_stamped) is True, "AI stamped row must be visible as pending"
    assert is_pending_review(genuine_human) is False, "Genuine human review must NOT be pending"
    assert is_pending_review(rejected) is False, "Rejected candidate must NOT be pending"


def test_status_counters_recalculated_honestly():
    """Summary status counters must accurately distinguish genuine human reviews from pending unverified rows."""
    candidates = [
        _make_candidate("c1", annotation_status="approved", source_text_checked="yes", verified_labels="Efficacy", annotator="Mukesh"),
        _make_candidate("c2", annotation_status="corrected", source_text_checked="yes", verified_labels="Safety", annotator="Mukesh"),
        _make_candidate("c3", annotation_status="rejected", annotator="Mukesh"),
        _make_candidate("c4", annotation_status="uncertain", annotator="Mukesh"),
        # AI-assisted row that previous automation marked approved with source_text_checked='no'
        _make_candidate("c5", annotation_status="approved", source_text_checked="no", annotator="Mukesh", review_notes="AI-assisted annotation"),
        # Standard pending row
        _make_candidate("c6", annotation_status="pending"),
    ]

    approved = [r for r in candidates if is_eligible_human_annotation(r) and r.get("annotation_status") == "approved"]
    corrected = [r for r in candidates if is_eligible_human_annotation(r) and r.get("annotation_status") == "corrected"]
    rejected = [r for r in candidates if r.get("annotation_status") == "rejected"]
    uncertain = [r for r in candidates if r.get("annotation_status") == "uncertain"]
    pending = [r for r in candidates if is_pending_review(r)]

    assert len(approved) == 1
    assert len(corrected) == 1
    assert len(rejected) == 1
    assert len(uncertain) == 1
    assert len(pending) == 2  # c5 (unverified) and c6 (pending)


def test_export_duplicate_detection():
    """Export must detect conflicting duplicate annotations on identical sentences within the same paper."""
    row1 = _make_candidate(
        "dup-1",
        paper_id="PMC999",
        sentence="Drug X significantly improved survival.",
        annotation_status="approved",
        source_text_checked="yes",
        verified_labels="Efficacy",
        annotator="Mukesh",
        source_section="Results",
    )
    row2_conflict = _make_candidate(
        "dup-2",
        paper_id="PMC999",
        sentence="Drug X significantly improved survival.",
        annotation_status="approved",
        source_text_checked="yes",
        verified_labels="Safety",  # Conflicting label
        annotator="Mukesh",
        source_section="Results",
    )
    row3_identical = _make_candidate(
        "dup-3",
        paper_id="PMC999",
        sentence="Drug X significantly improved survival.",
        annotation_status="approved",
        source_text_checked="yes",
        verified_labels="Efficacy",  # Identical label
        annotator="Mukesh",
        source_section="Results",
    )

    # Identical duplicates must deduplicate cleanly
    deduped = approved_rows([row1, row3_identical])
    assert len(deduped) == 1

    # Conflicting duplicates must raise ValueError
    with pytest.raises(ValueError, match="Duplicate sentence has conflicting approved annotations"):
        approved_rows([row1, row2_conflict])


def test_save_decision_records_timestamp_and_provenance():
    """Saving a decision must record reviewer, decision, timestamp, source verification, and optional note."""
    candidate = _make_candidate("save-1")
    candidates_list = [candidate]

    saved = save_review_decision(
        candidates_list,
        "save-1",
        "Confirm (Approve)",
        "Mukesh",
        verified_labels=["Efficacy", "Comparative"],
        verified_sentence="Original sentence text.",
        source_text_checked=True,
        review_note="Primary trial finding verified in Section 3.2",
    )

    assert saved["annotation_status"] == "approved"
    assert saved["annotator"] == "Mukesh"
    assert saved["source_text_checked"] == "yes"
    assert saved["verified_labels"] == "Efficacy;Comparative"
    assert "human_decision=approved" in saved["review_notes"]
    assert "reviewer=Mukesh" in saved["review_notes"]
    assert "source_checked=yes" in saved["review_notes"]
    assert "note=Primary trial finding verified in Section 3.2" in saved["review_notes"]
    assert "ts=" in saved["review_notes"]


def test_rejection_and_uncertain_clear_verified_labels():
    """Marking a candidate as rejected or uncertain must clear verified labels and sentence."""
    candidate_rej = _make_candidate("c-rej", verified_labels="Efficacy", verified_sentence="Some text")
    candidate_unc = _make_candidate("c-unc", verified_labels="Safety", verified_sentence="Some text")

    saved_rej = save_review_decision([candidate_rej], "c-rej", "Reject", "Mukesh", review_note="Not a result claim")
    assert saved_rej["annotation_status"] == "rejected"
    assert saved_rej["verified_labels"] == ""
    assert saved_rej["verified_sentence"] == ""

    saved_unc = save_review_decision([candidate_unc], "c-unc", "Mark Uncertain", "Mukesh", review_note="Needs adjudication")
    assert saved_unc["annotation_status"] == "uncertain"
    assert saved_unc["verified_labels"] == ""
    assert saved_unc["verified_sentence"] == ""


def test_next_pending_candidate_navigates_unverified_rows():
    """next_pending_candidate should advance through rows that still require human review."""
    candidates = {
        "c1": _make_candidate("c1", annotation_status="approved", source_text_checked="yes", verified_labels="Efficacy", annotator="M"),
        "c2": _make_candidate("c2", annotation_status="pending"),
        "c3": _make_candidate("c3", annotation_status="pending"),
    }
    batch = ["c1", "c2", "c3"]

    # c1 is resolved human review, next pending from c1 should be c2
    assert next_pending_candidate(batch, candidates, "c1") == "c2"

    # From c2, next pending should be c3
    assert next_pending_candidate(batch, candidates, "c2") == "c3"

    # Resolve c2 and c3
    candidates["c2"]["annotation_status"] = "rejected"
    candidates["c3"]["annotation_status"] = "approved"
    candidates["c3"]["source_text_checked"] = "yes"
    candidates["c3"]["verified_labels"] = "Safety"
    candidates["c3"]["annotator"] = "M"

    # All resolved
    assert next_pending_candidate(batch, candidates, "c3") is None
