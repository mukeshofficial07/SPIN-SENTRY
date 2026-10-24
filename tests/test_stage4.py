"""
tests/test_stage4.py
---------------------
Tests for Stage 4: ESWE — Evidence-Strength Weighted Entailment.

Verifies paper Equation (4): WE(c, e) = NLI(c, e) × ESI(e) × relevance(t, z)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from stages.stage1_claim_parser import parse_claims
from stages.stage2_zone_mapper import map_zones
from stages.stage4_eswe import compute_eswe, run_eswe_pipeline, generate_summary, SPIN_FLAG_THRESHOLD
from utils.esi_table import (
    extract_study_metadata,
    get_zone_type_relevance,
    _lookup_esi,
    StudyMetadata,
)


# ----- Fixtures ----- #

CLAIM_TEXT = "Drug X was well-tolerated with a favorable safety profile."
BODY_TEXT = """
Methods
This was a randomized controlled trial with n=1240 participants. Intention-to-treat
analysis was the primary method. Eligibility required age 60-65.

Results
Primary endpoint: 34% reduction versus placebo (p < 0.001).
Secondary outcomes improved significantly.

Adverse Events
Serious adverse events occurred in 15% of the treatment arm.
Discontinuation rate was 8% due to adverse effects.
Grade 3 toxicity in 5% of patients.

Limitations
The study is limited to ages 60-65.
"""


# ----- ESI Table Tests ----- #

class TestESITable:
    def test_rct_large_itt_gives_1_0(self):
        """Paper Table I row 1: RCT >1000, ITT → ESI = 1.0"""
        meta = StudyMetadata(
            study_design="RCT",
            sample_size=1240,
            analysis_type="Intention-to-treat",
        )
        esi = _lookup_esi(meta)
        assert esi == 1.0, f"Expected ESI=1.0 for RCT+ITT+n>1000, got {esi}"

    def test_rct_small_pp_gives_0_7(self):
        """Paper Table I row 2: RCT <100, per-protocol → ESI = 0.7"""
        meta = StudyMetadata(
            study_design="RCT",
            sample_size=80,
            analysis_type="Per-protocol",
        )
        esi = _lookup_esi(meta)
        assert esi == 0.7, f"Expected ESI=0.7 for RCT+PP+n<100, got {esi}"

    def test_observational_large_gives_0_6(self):
        """Paper Table I row 3: Observational >5000, adjusted → ESI = 0.6"""
        meta = StudyMetadata(
            study_design="Observational",
            sample_size=6000,
            analysis_type="Adjusted",
        )
        esi = _lookup_esi(meta)
        assert esi == 0.6, f"Expected ESI=0.6 for Obs+n>5000, got {esi}"

    def test_case_series_small_gives_0_2(self):
        """Paper Table I row 4: Case series n<50 → ESI = 0.2"""
        meta = StudyMetadata(
            study_design="Case series",
            sample_size=30,
            analysis_type="Uncontrolled",
        )
        esi = _lookup_esi(meta)
        assert esi == 0.2, f"Expected ESI=0.2 for case series n<50, got {esi}"


class TestESIExtraction:
    def test_extract_rct_from_text(self):
        text = "This randomized controlled trial enrolled n=1240 participants using intention-to-treat analysis."
        meta = extract_study_metadata(text)
        assert meta.study_design == "RCT"
        assert meta.sample_size == 1240
        assert meta.analysis_type == "Intention-to-treat"
        assert meta.esi == 1.0

    def test_extract_observational(self):
        text = "This prospective cohort study included 6000 participants with adjusted analysis."
        meta = extract_study_metadata(text)
        assert meta.study_design == "Observational"

    def test_missing_metadata_uses_default(self):
        text = "The study examined patient outcomes."
        meta = extract_study_metadata(text)
        assert 0.0 <= meta.esi <= 1.0
        assert "Unknown" in meta.study_design or meta.esi == 0.5


class TestZoneTypeRelevance:
    def test_safety_harms_full_weight(self):
        """Safety claims should get full weight (1.0) from Harms zone."""
        rel = get_zone_type_relevance("Safety", "Harms")
        assert rel == 1.0

    def test_safety_results_reduced(self):
        """Safety claims from Results zone should have reduced weight."""
        rel = get_zone_type_relevance("Safety", "Results")
        assert rel < 1.0, "Safety claim from Results should have reduced relevance"

    def test_efficacy_results_full(self):
        """Efficacy claims should get full weight from Results zone."""
        rel = get_zone_type_relevance("Efficacy", "Results")
        assert rel == 1.0

    def test_generalizability_methods_full(self):
        """Generalizability claims scored primarily against Methods."""
        rel = get_zone_type_relevance("Generalizability", "Methods")
        assert rel == 1.0

    def test_all_combinations_in_range(self):
        """All (type, zone) pairs should return a value in (0, 1]."""
        types = ["Efficacy", "Safety", "Generalizability", "Comparative", "Novelty", "Recommendation"]
        zones = ["Results", "Harms", "Limitations", "Methods"]
        for t in types:
            for z in zones:
                rel = get_zone_type_relevance(t, z)
                assert 0.0 < rel <= 1.0, f"relevance({t},{z}) = {rel} out of range"


class TestESWEFormula:
    def test_we_score_equals_nli_times_esi_times_relevance(self):
        """
        Verify WE = NLI × ESI × relevance (paper Eq. 4) to 1e-3 tolerance.
        We compute each component separately and check the product.
        """
        claims = parse_claims(CLAIM_TEXT)
        zone_map = map_zones(BODY_TEXT)

        assert len(claims) > 0, "Should extract at least one claim"
        report = compute_eswe(claims[0], zone_map)

        # Verify formula: WE = forward × ESI × relevance
        expected_we = report.forward * report.esi * report.relevance
        assert abs(report.we_score - expected_we) < 1e-3, (
            f"WE formula mismatch: {report.we_score} ≠ {expected_we}"
        )

    def test_spin_flag_at_threshold(self):
        """WE < SPIN_FLAG_THRESHOLD → spin_flag = True."""
        claims = parse_claims(CLAIM_TEXT)
        zone_map = map_zones(BODY_TEXT)
        report = compute_eswe(claims[0], zone_map)

        # Verify flag logic
        assert report.spin_flag == (report.we_score < SPIN_FLAG_THRESHOLD)

    def test_report_has_required_fields(self):
        claims = parse_claims(CLAIM_TEXT)
        zone_map = map_zones(BODY_TEXT)
        report = compute_eswe(claims[0], zone_map)

        for field in ["claim", "claim_types", "zone_route", "esi", "relevance",
                      "we_score", "spin_flag", "support_label", "forward", "gap"]:
            val = getattr(report, field, None)
            assert val is not None or isinstance(val, bool), f"Missing field: {field}"

    def test_we_score_bounded(self):
        """WE must be in [0, 1]."""
        claims = parse_claims(CLAIM_TEXT)
        zone_map = map_zones(BODY_TEXT)
        report = compute_eswe(claims[0], zone_map)
        assert 0.0 <= report.we_score <= 1.0, f"WE out of range: {report.we_score}"

    def test_to_dict_serializable(self):
        claims = parse_claims(CLAIM_TEXT)
        zone_map = map_zones(BODY_TEXT)
        report = compute_eswe(claims[0], zone_map)
        d = report.to_dict()
        assert isinstance(d, dict)
        for key in ["claim", "we_score", "spin_flag", "support_label"]:
            assert key in d


class TestGenerateSummary:
    def test_summary_structure(self):
        claims = parse_claims(CLAIM_TEXT)
        zone_map = map_zones(BODY_TEXT)
        reports = run_eswe_pipeline(claims, zone_map)
        summary = generate_summary(reports)

        for key in ["total_claims", "flagged_claims", "supported_claims",
                    "weak_claims", "spin_risk", "avg_we_score"]:
            assert key in summary

    def test_empty_reports(self):
        summary = generate_summary([])
        assert summary["total_claims"] == 0

    def test_counts_consistent(self):
        claims = parse_claims(CLAIM_TEXT)
        zone_map = map_zones(BODY_TEXT)
        reports = run_eswe_pipeline(claims, zone_map)
        summary = generate_summary(reports)

        assert summary["total_claims"] == len(reports)
        assert summary["flagged_claims"] + summary["supported_claims"] + \
               summary["weak_claims"] == len(reports)
