from utils.claim_annotation_review import (
    apply_pending_picker_selection,
    next_pending_candidate,
    save_review_decision,
)


def _candidate(candidate_id, **overrides):
    row = {
        "candidate_id": candidate_id,
        "paper_id": "PMC-example",
        "sentence": f"Original sentence for {candidate_id}.",
        "source": "https://example.org/paper",
        "license": "CC BY 4.0",
        "machine_suggested_labels": "Efficacy",
        "annotation_status": "pending",
        "verified_labels": "",
        "verified_sentence": "",
        "source_text_checked": "",
        "annotator": "",
    }
    row.update(overrides)
    return row


def test_saving_decision_updates_human_fields_and_preserves_provenance_and_suggestions():
    candidate = _candidate("c1")
    saved = save_review_decision(
        [candidate], "c1", "Approve", "Reviewer A", ["Efficacy", "Safety"],
        candidate["sentence"], True,
    )

    assert saved["annotation_status"] == "approved"
    assert saved["verified_labels"] == "Efficacy;Safety"
    assert saved["source_text_checked"] == "yes"
    assert saved["annotator"] == "Reviewer A"
    assert saved["source"] == "https://example.org/paper"
    assert saved["license"] == "CC BY 4.0"
    assert saved["machine_suggested_labels"] == "Efficacy"


def test_after_saving_current_candidate_picker_advances_to_next_pending_candidate():
    batch = ["c1", "c2", "c3"]
    candidates = {cid: _candidate(cid) for cid in batch}
    save_review_decision([candidates["c2"]], "c2", "Reject", "Reviewer A")

    assert next_pending_candidate(batch, candidates, "c2") == "c3"


def test_completed_batch_has_no_next_pending_candidate():
    batch = ["c1", "c2"]
    candidates = {cid: _candidate(cid, annotation_status="approved") for cid in batch}

    assert next_pending_candidate(batch, candidates, "c2") is None


def test_queued_selection_is_applied_before_widget_and_stale_selection_falls_back():
    state = {"queued": "c2", "picker": "removed"}
    apply_pending_picker_selection(state, "picker", "queued", ["c1", "c2"])
    assert state == {"picker": "c2"}

    apply_pending_picker_selection(state, "picker", "queued", ["c1"])
    assert state["picker"] == "c1"
