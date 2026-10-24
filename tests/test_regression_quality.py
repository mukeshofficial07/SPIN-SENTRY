"""
tests/test_regression_quality.py
--------------------------------
Regression and quality tests verifying:
- Non-drug entity exclusions (CABG, ANOVA, VAS, TUG, ICU, RCT)
- Structural role detection (Objective, Methods, Finding / Claim)
- Dehyphenation and PDF artifact cleaning
- Terminating header detection and table noise filtering in MZEM
- Cautious explanation generation and empirical vs protocol claim separation
"""

import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.ner_utils import extract_entities, _NON_DRUG_EXCLUSIONS
from utils.pdf_reader import clean_extracted_text, _split_abstract_body
from stages.stage1_claim_parser import _detect_sentence_role, parse_claims
from stages.stage2_zone_mapper import _classify_header, _is_clean_evidence_sentence, map_zones
from stages.stage3_nli import directional_nli, _token_overlap_score
from stages.stage4_eswe import _generate_explanation, generate_summary, ClaimReport, Claim


# ---------------------------------------------------------------------------
# Priority 2: Entity Typing & Exclusions
# ---------------------------------------------------------------------------

class TestEntityTypingExclusions:
    def test_cabg_and_anova_not_drugs(self):
        sentence = "Patients undergoing CABG were analyzed using two-way mixed ANOVA and VAS."
        entities = extract_entities(sentence)
        drugs_lower = [d.lower() for d in entities["drugs"]]
        assert "cabg" not in drugs_lower, "CABG is a surgical procedure, not a drug"
        assert "anova" not in drugs_lower, "ANOVA is a statistical test, not a drug"
        assert "vas" not in drugs_lower, "VAS is a pain measurement scale, not a drug"

    def test_icu_and_tug_not_drugs(self):
        sentence = "Patients in the ICU were tested on the TUG mobility scale and monitored for RCT criteria."
        entities = extract_entities(sentence)
        drugs_lower = [d.lower() for d in entities["drugs"]]
        assert "icu" not in drugs_lower
        assert "tug" not in drugs_lower
        assert "rct" not in drugs_lower

    def test_real_interventions_detected(self):
        sentence = "Elastic therapeutic taping reduced tramadol consumption compared to paracetamol and placebo."
        entities = extract_entities(sentence)
        drugs_lower = [d.lower() for d in entities["drugs"]]
        assert any("taping" in d for d in drugs_lower)
        assert any("tramadol" in d for d in drugs_lower)
        assert any("paracetamol" in d for d in drugs_lower)
        assert any("placebo" in d for d in drugs_lower)


# ---------------------------------------------------------------------------
# Priority 2: Structural Role Detection
# ---------------------------------------------------------------------------

class TestSentenceRoleDetection:
    def test_objective_detected(self):
        s = "To evaluate the effectiveness of elastic therapeutic taping on postoperative pain."
        role, is_empirical = _detect_sentence_role(s)
        assert role == "Objective"
        assert is_empirical is False

    def test_methods_recruitment_detected(self):
        s = "In this trial, 195 patients were recruited and randomly assigned to three groups."
        role, is_empirical = _detect_sentence_role(s)
        assert role == "Methods"
        assert is_empirical is False

    def test_methods_outcomes_definition_detected(self):
        s = "Secondary outcomes were Timed Up and Go (TUG), sleep quality, and total analgesic use."
        role, is_empirical = _detect_sentence_role(s)
        assert role == "Methods"
        assert is_empirical is False

    def test_methods_statistics_detected(self):
        s = "Data were analyzed using a prespecified two-way mixed ANOVA with Bonferroni correction."
        role, is_empirical = _detect_sentence_role(s)
        assert role == "Methods"
        assert is_empirical is False

    def test_empirical_finding_detected(self):
        s = "At 48 hours, resting pain was lower in the elastic therapeutic taping group (19.0 vs 41.2, p = 0.001)."
        role, is_empirical = _detect_sentence_role(s)
        assert role == "Finding / Claim"
        assert is_empirical is True

    def test_conclusion_detected(self):
        s = "These findings indicate that elastic therapeutic taping provides meaningful postoperative analgesia."
        role, is_empirical = _detect_sentence_role(s)
        assert role == "Conclusion"
        assert is_empirical is True


# ---------------------------------------------------------------------------
# Priority 2: PDF Cleaning & Dehyphenation
# ---------------------------------------------------------------------------

class TestPDFCleaning:
    def test_dehyphenation_across_lines(self):
        raw = "There were signifi-\n cant differ-\nences between groups."
        cleaned = clean_extracted_text(raw)
        assert "significant" in cleaned or "differ" in cleaned
        assert "signifi-\n cant" not in cleaned

    def test_dehyphenation_safe_with_conjunctions(self):
        raw = "placebo- and sham-controlled trial"
        cleaned = clean_extracted_text(raw)
        assert "placebo- and" in cleaned

    def test_doi_and_header_removal(self):
        raw = "PLOS ONE | https://doi.org/10.1371/journal.pone.0359282 September 28, 2026 1 / 16\nSome actual text."
        cleaned = clean_extracted_text(raw)
        assert "PLOS ONE | https" not in cleaned
        assert "Some actual text." in cleaned

    def test_abstract_tail_metadata_stripped(self):
        full = """Abstract
To evaluate pain after surgery.
Resting pain was significantly lower (p < 0.01).
Trial registration: ClinicalTrials.gov NCT06240299.
Introduction
Cardiac surgery is associated with pain."""
        abs_text, body_text = _split_abstract_body(full)
        assert "Resting pain was significantly lower" in abs_text
        assert "Trial registration" not in abs_text
        assert "Introduction" in body_text


# ---------------------------------------------------------------------------
# Priority 2 & 3: Zone Mapping & Evidence Cleaning
# ---------------------------------------------------------------------------

class TestEvidenceZoneQuality:
    def test_primary_outcome_fuzzy_maps_to_methods(self):
        # Header "Primary outcome" defines measurement endpoints in Methods
        result = _classify_header("Primary outcome")
        assert result is not None
        assert result[0] == "Methods"

    def test_secondary_outcomes_fuzzy_maps_to_methods(self):
        result = _classify_header("Secondary outcomes")
        assert result is not None
        assert result[0] == "Methods"

    def test_is_clean_evidence_sentence_filters_table_dumps(self):
        # Table dump with dense numbers and symbols
        table_row = "Group A 19.0 ± 10.2 41.2 ± 12.1 38.5 ± 11.0 45% 12% 8% | 0.001 | 0.04"
        assert not _is_clean_evidence_sentence(table_row)

    def test_is_clean_evidence_sentence_accepts_clean_prose(self):
        prose = "Resting pain scores were significantly lower in the active group compared with the control group at 48 hours."
        assert _is_clean_evidence_sentence(prose)


# ---------------------------------------------------------------------------
# Priority 3: Explanations and Cautious Labeling
# ---------------------------------------------------------------------------

class TestCautiousExplanations:
    def test_protocol_claim_explanation(self):
        c = Claim(
            sentence="To evaluate the effectiveness of taping on pain.",
            types=["Efficacy"],
            entities={},
            hedging=0.0,
            hedging_words=[],
            type_scores={"Efficacy": 1.0},
            zone_route="Methods",
            role="Objective",
            is_empirical_claim=False,
        )
        label, expl = _generate_explanation(
            claim=c, we_score=0.1, forward=0.15, reverse=0.05, gap=0.95,
            contradiction=0.01, spin_flag=True, zone_route="Methods"
        )
        assert "Protocol / Method" in label
        assert "Direct textual entailment in Methods was limited" in expl

    def test_supported_claim_explanation(self):
        c = Claim(
            sentence="Pain was significantly reduced.",
            types=["Efficacy"],
            entities={},
            hedging=0.0,
            hedging_words=[],
            type_scores={"Efficacy": 1.0},
            zone_route="Results",
            role="Finding / Claim",
            is_empirical_claim=True,
        )
        label, expl = _generate_explanation(
            claim=c, we_score=0.75, forward=0.92, reverse=0.10, gap=0.90,
            contradiction=0.01, spin_flag=False, zone_route="Results"
        )
        assert label == "Supported"
        assert "Strong forward entailment" in expl

    def test_flagged_claim_cautious_unverified_label(self):
        c = Claim(
            sentence="Tramadol use was 50 mg vs 100 mg (p = 0.001).",
            types=["Efficacy"],
            entities={},
            hedging=0.0,
            hedging_words=[],
            type_scores={"Efficacy": 1.0},
            zone_route="Results",
            role="Finding / Claim",
            is_empirical_claim=True,
        )
        label, expl = _generate_explanation(
            claim=c, we_score=0.15, forward=0.20, reverse=0.01, gap=0.99,
            contradiction=0.01, spin_flag=True, zone_route="Results"
        )
        assert "Flagged" in label
        assert "Generalizability Gap" in label or "Unverified" in label
        assert "low" in expl.lower() or "gap" in expl.lower()

    def test_summary_includes_empirical_breakdown(self):
        r1 = ClaimReport(
            claim="Finding 1", claim_types=["Efficacy"], zone_route="Results",
            evidence_items=[], esi=0.8, relevance=1.0, we_score=0.72,
            spin_flag=False, support_label="Supported", esi_metadata=None,
            top_evidence="ev", forward=0.9, reverse=0.2, gap=0.8,
            hedging=0.0, hedging_words=[], model_used="test",
            role="Finding / Claim", is_empirical_claim=True, explanation="Good"
        )
        r2 = ClaimReport(
            claim="Objective 1", claim_types=["Efficacy"], zone_route="Methods",
            evidence_items=[], esi=0.8, relevance=1.0, we_score=0.10,
            spin_flag=True, support_label="Protocol / Method (Low Direct Match)",
            esi_metadata=None, top_evidence="ev", forward=0.1, reverse=0.05, gap=0.95,
            hedging=0.0, hedging_words=[], model_used="test",
            role="Objective", is_empirical_claim=False, explanation="Protocol"
        )
        summary = generate_summary([r1, r2])
        assert summary["total_claims"] == 2
        assert summary["empirical_claims"] == 1
        assert summary["protocol_statements"] == 1
        assert summary["empirical_flagged"] == 0
        assert summary["empirical_avg_we"] == 0.72

    def test_claim_report_explanation_field_always_present(self):
        """Task 5 regression: verify explanation field is always available and non-empty."""
        # 1. Direct instantiation without providing explanation (triggers __post_init__)
        report_default = ClaimReport(
            claim="Taping reduced pain significantly.",
            claim_types=["Efficacy"],
            zone_route="Results",
            evidence_items=[],
            esi=0.8,
            relevance=1.0,
            we_score=0.75,
            spin_flag=False,
            support_label="Supported",
            esi_metadata=None,
            top_evidence="Evidence sentence",
            forward=0.95,
            reverse=0.20,
            gap=0.80,
            hedging=0.0,
            hedging_words=[],
            model_used="test_model",
        )
        assert hasattr(report_default, "explanation"), "ClaimReport must have 'explanation' attribute"
        assert isinstance(report_default.explanation, str)
        assert len(report_default.explanation) > 0, "Explanation should be non-empty via __post_init__ fallback"
        assert "Strong forward entailment" in report_default.explanation
        assert "explanation" in report_default.to_dict()

    def test_claim_report_from_pipeline_has_explanation(self):
        """Verify reports produced by compute_eswe contain the explanation field."""
        from stages.stage1_claim_parser import parse_claims
        from stages.stage2_zone_mapper import map_zones
        from stages.stage4_eswe import compute_eswe

        claims = parse_claims("Pain was reduced with taping at 48 hours.")
        zone_map = map_zones("Results\nPain was reduced with taping at 48 hours compared to control.")
        assert len(claims) > 0
        report = compute_eswe(claims[0], zone_map)

        assert hasattr(report, "explanation")
        assert isinstance(report.explanation, str)
        assert len(report.explanation) > 0
        # Verify serialization includes explanation
        d = report.to_dict()
        assert "explanation" in d
        assert d["explanation"] == report.explanation
