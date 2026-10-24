from scripts.prepare_machine_pseudo_labels import LABEL_SOURCE, assess_row, prepare_rows
from scripts.train_claim_classifier import validate_pseudo_dataset


def _row(**overrides):
    base = {
        "candidate_id": "cid-1", "paper_id": "paper-1", "title": "Trial",
        "doi": "10.1/example", "url": "https://example.org/article",
        "license": "CC BY 4.0", "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "source_section": "Results", "sentence": "The intervention improved pain outcomes.",
        "machine_suggested_labels": "Efficacy;Safety", "machine_system": "rules",
        "machine_claim_class": "claim", "machine_claim_rationale": "finding cue",
        "machine_label_rationales": "{}", "machine_uncertainty_flag": "no",
        "machine_uncertainty_reasons": "", "machine_evidence_cues": "improved",
        "machine_context_before": "", "machine_context_after": "", "extraction_notes": "",
    }
    base.update(overrides)
    return base


def test_prepare_rows_keeps_multilabel_and_truthful_source():
    included, excluded, _, _ = prepare_rows([_row()])
    assert len(included) == 1 and not excluded
    assert included[0]["labels"] == "Efficacy;Safety"
    assert included[0]["label_source"] == LABEL_SOURCE
    assert "annotator" not in included[0]
    assert "verified_labels" not in included[0]


def test_pseudo_rows_with_uncertainty_nonclaim_or_damage_are_excluded():
    rows = [
        _row(candidate_id="ok"),
        _row(candidate_id="nonclaim", machine_claim_class="non_claim"),
        _row(candidate_id="uncertain", machine_uncertainty_flag="yes"),
        _row(candidate_id="damaged", machine_uncertainty_reasons="possible_extraction_damage"),
        _row(candidate_id="missing", machine_suggested_labels=""),
    ]
    included, excluded, primary, overlapping = prepare_rows(rows)
    assert [row["candidate_id"] for row in included] == ["ok"]
    assert len(excluded) == 4
    assert overlapping["likely_non_claim"] == 1
    assert overlapping["ambiguous_or_uncertain"] == 1
    assert overlapping["possible_extraction_damage"] == 1
    assert overlapping["missing_labels"] == 1


def test_validator_accepts_multilabel_machine_pseudo_rows(tmp_path):
    import csv
    from scripts.prepare_machine_pseudo_labels import PROVENANCE_FIELDS

    row = {
        "candidate_id": "cid-1", "paper_id": "paper-1", "sentence": "Exact sentence.",
        "labels": "Efficacy;Safety", "source": "10.1/example", "label_source": LABEL_SOURCE,
        "machine_claim_class": "claim", "machine_uncertainty_flag": "no",
        "machine_uncertainty_reasons": "", "extraction_notes": "",
    }
    path = tmp_path / "pseudo.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PROVENANCE_FIELDS)
        writer.writeheader()
        writer.writerow(row)
    errors, count, labels = validate_pseudo_dataset(path)
    assert errors == []
    assert count == 1
    assert labels["Efficacy"] == labels["Safety"] == 1
