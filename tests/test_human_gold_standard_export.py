"""
tests/test_human_gold_standard_export.py
-----------------------------------------
Test suite validating:
  1. Human gold-standard export integrity (N=197 positive claims).
  2. Provenance rules: source_text_checked == 'yes', human annotator, non-AI.
  3. Manifest accounting and validity.
  4. Claim dataclass claim_types property access.
  5. Classifier prediction output on gold standard samples.
"""

import csv
import json
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "research_dataset"
GOLD_CSV = DATA_DIR / "human_gold_standard.csv"
MANIFEST_JSON = DATA_DIR / "human_gold_standard_manifest.json"
REVIEWED_CSV = ROOT / "data" / "claim_annotations_reviewed.csv"

LABELS = ["Efficacy", "Safety", "Generalizability", "Comparative", "Novelty", "Recommendation"]


class TestHumanGoldStandardExport:
    def test_gold_csv_exists_and_populated(self):
        assert GOLD_CSV.exists(), "human_gold_standard.csv must exist"
        with GOLD_CSV.open(encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) >= 180, f"Expected at least 180 gold standard rows, got {len(rows)}"
        assert len(rows) == 197, f"Expected exactly 197 gold standard rows, got {len(rows)}"

    def test_gold_provenance_rules(self):
        with GOLD_CSV.open(encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            # Must have source_text_checked == 'yes'
            assert r.get("source_text_checked") == "yes", f"Row {r.get('candidate_id')} not source checked"
            # Must have an annotator
            assert r.get("annotator") == "Mukesh", f"Row {r.get('candidate_id')} missing annotator"
            # Must have approved/corrected status
            assert r.get("annotation_status") in ("approved", "confirmed", "corrected")
            # Must not be an unreviewed AI batch note
            notes = r.get("review_notes", "")
            assert not notes.startswith("AI-assisted annotation"), f"Row {r.get('candidate_id')} has AI note"
            # Verified labels must be present and valid
            v_labels = [l.strip() for l in r.get("verified_labels", "").split(";") if l.strip()]
            assert len(v_labels) > 0, f"Row {r.get('candidate_id')} has empty verified_labels"
            for lbl in v_labels:
                assert lbl in LABELS, f"Invalid label {lbl} in row {r.get('candidate_id')}"

    def test_manifest_integrity(self):
        assert MANIFEST_JSON.exists(), "human_gold_standard_manifest.json must exist"
        with MANIFEST_JSON.open(encoding="utf-8") as f:
            manifest = json.load(f)
        assert manifest["manifest_name"] == "SPIN-SENTRY Human-Reviewed Gold Standard Dataset"
        accounting = manifest["corpus_accounting"]
        assert accounting["total_candidates_in_corpus"] == 3737
        assert accounting["sum_check"] == 3737
        assert accounting["positive_gold_standard_claims"] == 197
        stats = manifest["gold_standard_statistics"]
        assert stats["positive_claim_sentences"] == 197
        assert stats["papers_covered"] == 22

    def test_claim_dataclass_claim_types_property(self):
        from stages.stage1_claim_parser import Claim
        c = Claim(
            sentence="Test sentence.",
            types=["Efficacy", "Safety"],
            entities={},
            hedging=0.1,
            hedging_words=[],
            type_scores={"Efficacy": 0.9, "Safety": 0.8},
            zone_route="Results",
        )
        assert c.types == ["Efficacy", "Safety"]
        assert c.claim_types == ["Efficacy", "Safety"], "Claim.claim_types property must match types"

    def test_evaluate_classifier_loads_all_gold(self):
        from scripts.evaluate_classifier import load_gold_standard
        gold = load_gold_standard()
        assert len(gold) == 197, f"Expected 197 gold rows in evaluate_classifier, got {len(gold)}"

    def test_trained_classifier_prediction_on_gold_sample(self):
        from utils.trained_claim_classifier import predict_scores
        with GOLD_CSV.open(encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        sample_sentence = rows[0]["sentence"]
        scores = predict_scores(sample_sentence)
        assert isinstance(scores, dict)
        for lbl in LABELS:
            assert lbl in scores
            assert 0.0 <= scores[lbl] <= 1.0
