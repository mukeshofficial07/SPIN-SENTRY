from scripts.claim_dataset import LABELS
from scripts.machine_annotate_claims import annotate_candidate, annotate_rows, build_review_batch, section_group, _near_duplicate


def test_finding_is_claim_with_contextual_efficacy_label():
    row = {"candidate_id": "a", "paper_id": "p", "source_section": "Results",
           "sentence": "The intervention improved pain scores among patients.",
           "extraction_notes": "", "suggested_labels": "", "annotation_status": "pending"}
    result = annotate_candidate(row)
    assert result["machine_claim_class"] == "claim"
    assert result["machine_suggested_labels"] == "Efficacy"
    assert "verified_labels" not in result


def test_procedural_objective_is_likely_non_claim():
    row = {"candidate_id": "b", "paper_id": "p", "source_section": "Methods",
           "sentence": "The objective was to evaluate the intervention.",
           "extraction_notes": "", "suggested_labels": "Efficacy", "annotation_status": "pending"}
    result = annotate_candidate(row)
    assert result["machine_claim_class"] == "non_claim"
    assert result["machine_suggested_labels"] == ""
    assert "existing_keyword_hints_not_contextually_confirmed" in result["machine_uncertainty_reasons"]


def test_context_can_support_anaphoric_finding_but_flags_uncertainty():
    row = {"candidate_id": "c", "paper_id": "p", "source_section": "Discussion",
           "sentence": "These findings support the intervention.",
           "extraction_notes": "", "suggested_labels": "", "annotation_status": "pending"}
    result = annotate_candidate(row, "Treatment reduced pain scores among participants.", "")
    assert "Efficacy" in result["machine_suggested_labels"].split(";")
    assert result["machine_uncertainty_flag"] == "yes"
    assert "label_depends_on_neighbor_context" in result["machine_uncertainty_reasons"]


def test_possible_extraction_damage_is_flagged_without_rewriting_sentence():
    sentence = "Participants were u to receive the assigned treatment."
    row = {"candidate_id": "d", "paper_id": "p", "source_section": "Methods",
           "sentence": sentence, "extraction_notes": "", "suggested_labels": "", "annotation_status": "pending"}
    result = annotate_candidate(row)
    assert result["machine_uncertainty_flag"] == "yes"
    assert "possible_extraction_damage" in result["machine_uncertainty_reasons"]
    assert result["machine_context_before"] == ""


def test_annotation_batch_leaves_human_fields_untouched_and_samples_each_label():
    rows = []
    for i, label in enumerate(LABELS):
        sentences = {
            "Efficacy": "The treatment improved pain outcomes among patients.",
            "Safety": "No serious adverse events were reported among participants.",
            "Generalizability": "The findings may not generalize beyond this single center.",
            "Comparative": "The active treatment was better than placebo for pain.",
            "Novelty": "This is the first trial to demonstrate the new approach.",
            "Recommendation": "The intervention should be considered in clinical practice.",
        }
        rows.append({"candidate_id": f"id-{i}", "paper_id": f"p-{i}", "title": "T", "doi": "d",
                     "url": "u", "license": "l", "license_url": "lu", "source_section": "Results",
                     "sentence": sentences[label], "suggested_labels": "", "suggestion_scores": "{}",
                     "verified_sentence": "", "source_text_checked": "no", "verified_labels": "",
                     "annotation_status": "pending", "annotator": "", "review_notes": "", "extraction_notes": ""})
    rows.append({**rows[0], "candidate_id": "already-rejected", "annotation_status": "rejected"})
    annotated, queue, sample = annotate_rows(rows, per_category=1)
    assert len(annotated) == len(queue) == 6
    assert len(sample) >= 6
    covered = {label for row in sample for label in row["representative_categories"].split(";") if label}
    assert covered == set(LABELS)
    assert all(row["verified_labels"] == "" and row["source_text_checked"] == "no" for row in annotated)
    assert "machine_confidence" not in annotated[0]


def test_review_batch_balances_categories_and_diversifies_papers():
    rows = []
    for i in range(48):
        label = LABELS[i % len(LABELS)]
        rows.append({"candidate_id": f"batch-{i}", "paper_id": f"paper-{i}",
                     "sentence": f"Distinct verified test sentence number {i} for {label} outcomes.",
                     "machine_suggested_labels": label, "machine_section_priority": str(1 + i % 5),
                     "machine_uncertainty_flag": "no", "machine_uncertainty_reasons": "",
                     "machine_claim_class": "claim", "machine_section_group": "Results/Outcomes"})
    batch = build_review_batch(rows, 20)
    counts = {label: sum(r["batch_review_stratum"] == f"category:{label}" for r in batch) for label in LABELS}
    assert len(batch) == 20
    assert set(counts.values()) == {2}
    assert len({r["paper_id"] for r in batch}) == len(batch)


def test_noisy_sections_are_prioritized_without_being_excluded():
    assert section_group("4. Safety and adverse events") == "Safety/Harms"
    assert section_group("Primary Outcomes") == "Results/Outcomes"
    assert section_group("Strengths and Limitations") == "Limitations"
    assert section_group("DISCUSSION") == "Discussion"
    assert section_group("1.") == "Other"


def test_near_duplicate_sentences_from_one_paper_are_skipped():
    first = {"paper_id": "same-paper", "sentence": "The intervention produced a meaningful reduction in pain scores among the participants."}
    second = {"paper_id": "same-paper", "sentence": "The intervention produced a meaningful reduction in pain scores among participants."}
    assert _near_duplicate(second, [first])
