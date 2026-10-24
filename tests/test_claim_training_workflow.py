import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.validate_claim_annotations import LABELS, validate
from scripts.claim_dataset import jats_sections, license_is_reusable, make_candidates, sentences
from scripts.export_claim_annotations import approved_rows, write_export


def test_empty_template_is_not_misreported_as_data():
    errors, count, label_counts = validate(ROOT / "data" / "claim_annotations_template.csv")
    assert count == 0
    assert "No annotation rows found; the template is intentionally empty." in errors
    assert sum(label_counts.values()) == 0


def test_validator_accepts_human_multilabel_rows(tmp_path):
    path = tmp_path / "annotations.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["paper_id", "sentence", "labels", "source", "annotator", "notes"])
        writer.writeheader()
        writer.writerow({"paper_id": "paper-1", "sentence": "Treatment reduced pain.", "labels": "Efficacy; Safety", "source": "DOI", "annotator": "Reviewer A", "notes": ""})
    errors, count, label_counts = validate(path)
    assert errors == []
    assert count == 1
    assert label_counts["Efficacy"] == 1
    assert label_counts["Safety"] == 1
    assert len(LABELS) == 6


def test_validator_rejects_unknown_label(tmp_path):
    path = tmp_path / "annotations.csv"
    path.write_text("paper_id,sentence,labels,source,annotator,notes\np1,Claim sentence,Positive,DOI,A,\n", encoding="utf-8")
    errors, _, _ = validate(path)
    assert any("unknown label" in error for error in errors)


def test_validator_requires_source_and_human_annotator(tmp_path):
    path = tmp_path / "annotations.csv"
    path.write_text("paper_id,sentence,labels,source,annotator,notes\np1,Claim sentence,Efficacy,,,\n", encoding="utf-8")
    errors, _, _ = validate(path)
    assert any("source DOI/URL is required" in error for error in errors)
    assert any("annotator is required" in error for error in errors)


def test_sentence_split_keeps_vs_measurements_together():
    assert sentences("Pain was lower (39.2 vs. 56.3). Tramadol use fell (50 mg vs. 100 mg).") == [
        "Pain was lower (39.2 vs. 56.3).", "Tramadol use fell (50 mg vs. 100 mg)."
    ]


def test_candidate_suggestions_are_separate_from_human_labels():
    paper = {key: "" for key in ("paper_id", "title", "doi", "url", "license", "license_url")}
    paper.update(paper_id="PMC1", title="A trial", doi="10.example/1", license="CC BY 4.0")
    candidate = make_candidates(paper, [("Abstract", "The treatment significantly reduced pain.")], {"Efficacy": ["reduced", "significantly"]})[0]
    assert candidate["suggested_labels"] == "Efficacy"
    assert candidate["verified_labels"] == ""
    assert candidate["annotation_status"] == "pending"
    assert candidate["source_section"] == "Abstract"


def test_jats_section_and_license_extraction():
    import xml.etree.ElementTree as ET
    article = ET.fromstring('''<article xmlns:xlink="http://www.w3.org/1999/xlink"><front><article-meta><title-group><article-title>Trial title</article-title></title-group><article-id pub-id-type="doi">10.1234/example</article-id><permissions><license xlink:href="https://creativecommons.org/licenses/by/4.0/"><license-p>CC BY 4.0</license-p></license></permissions><abstract><p>Abstract sentence.</p></abstract></article-meta></front><body><sec><title>Results</title><p>Outcome improved.</p></sec></body></article>''')
    title, doi, license_url, license_text, sections = jats_sections(article)
    assert title == "Trial title"
    assert doi == "10.1234/example"
    assert sections == [("Abstract", "Abstract sentence."), ("Results", "Outcome improved.")]
    assert license_is_reusable(license_url, license_text)
    assert not license_is_reusable("https://creativecommons.org/licenses/by-nd/4.0/")


def test_export_contains_only_source_checked_human_annotations(tmp_path):
    base = {"candidate_id": "abc", "paper_id": "PMC1", "sentence": "Pain improved.", "doi": "10.1234/trial", "url": "https://example.org", "source_section": "Abstract", "license": "CC BY 4.0", "license_url": "https://creativecommons.org/licenses/by/4.0/", "source_text_checked": "yes", "verified_sentence": "Pain improved.", "verified_labels": "Efficacy", "annotation_status": "approved", "annotator": "Reviewer", "review_notes": ""}
    uncertain = {**base, "candidate_id": "xyz", "annotation_status": "uncertain", "verified_labels": ""}
    rows = approved_rows([base, uncertain])
    assert len(rows) == 1
    assert rows[0]["labels"] == "Efficacy"
    assert rows[0]["source"].endswith("10.1234/trial")
    out = tmp_path / "approved.csv"
    write_export(out, rows)
    with out.open(encoding="utf-8", newline="") as handle:
        assert next(csv.reader(handle)) == ["paper_id", "sentence", "labels", "source", "annotator", "notes"]
