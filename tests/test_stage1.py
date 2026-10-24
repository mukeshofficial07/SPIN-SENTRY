"""
tests/test_stage1.py
---------------------
Tests for Stage 1: HCP-BEA claim parser.

Uses the worked example from paper Table II:
  c1: "Drug X demonstrated a significant reduction in symptom severity"
  c2: "Drug X was well-tolerated with a favorable safety profile"
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from stages.stage1_claim_parser import parse_claims, Claim, _score_claim_types, _select_types


# ----- Fixtures ----- #

WORKED_EXAMPLE_ABSTRACT = (
    "Drug X demonstrated a significant reduction in symptom severity. "
    "Drug X was well-tolerated with a favorable safety profile."
)

MULTI_CLAIM_ABSTRACT = (
    "This randomized controlled trial shows that Drug Y is superior to placebo "
    "in reducing mortality. "
    "The drug was well-tolerated with no serious adverse events. "
    "These findings may be generalizable across diverse patient populations. "
    "We recommend Drug Y should be considered for clinical use in elderly patients."
)


# ----- Tests ----- #

class TestScoreClaimTypes:
    def test_efficacy_sentence(self):
        sent = "Drug X demonstrated a significant reduction in symptom severity."
        scores = _score_claim_types(sent)
        assert scores["Efficacy"] > 0, "Efficacy score should be > 0"

    def test_safety_sentence(self):
        sent = "Drug X was well-tolerated with a favorable safety profile."
        scores = _score_claim_types(sent)
        assert scores["Safety"] > 0, "Safety score should be > 0"

    def test_generalizability_sentence(self):
        sent = "The treatment is effective across diverse patient populations."
        scores = _score_claim_types(sent)
        assert scores["Generalizability"] > 0, "Generalizability score should be > 0"

    def test_comparative_sentence(self):
        sent = "Drug Y is superior to placebo and standard of care."
        scores = _score_claim_types(sent)
        assert scores["Comparative"] > 0, "Comparative score should be > 0"

    def test_novelty_sentence(self):
        sent = "This is the first demonstration of this novel mechanism."
        scores = _score_claim_types(sent)
        assert scores["Novelty"] > 0, "Novelty score should be > 0"

    def test_recommendation_sentence(self):
        sent = "Drug Z should be considered for clinical use in practice."
        scores = _score_claim_types(sent)
        assert scores["Recommendation"] > 0, "Recommendation score should be > 0"

    def test_scores_are_bounded(self):
        sent = "Some arbitrary sentence about the trial results."
        scores = _score_claim_types(sent)
        for t, s in scores.items():
            assert 0.0 <= s <= 1.0, f"Score for {t} out of [0,1]: {s}"


class TestSelectTypes:
    def test_returns_at_least_one(self):
        scores = {t: 0.0 for t in ["Efficacy", "Safety", "Generalizability",
                                     "Comparative", "Novelty", "Recommendation"]}
        result = _select_types(scores)
        assert len(result) >= 1

    def test_high_score_selected(self):
        scores = {"Efficacy": 0.8, "Safety": 0.1, "Generalizability": 0.0,
                  "Comparative": 0.0, "Novelty": 0.0, "Recommendation": 0.0}
        result = _select_types(scores)
        assert "Efficacy" in result


class TestParseClaims:
    def test_worked_example_c1(self):
        """Paper Table II c1: efficacy claim about Drug X."""
        claims = parse_claims(WORKED_EXAMPLE_ABSTRACT)
        assert len(claims) >= 1, "Should extract at least 1 claim"
        c1 = claims[0]
        assert "Efficacy" in c1.types or "Comparative" in c1.types, (
            f"c1 expected Efficacy or Comparative, got {c1.types}"
        )

    def test_worked_example_c2(self):
        """Paper Table II c2: safety claim about Drug X."""
        claims = parse_claims(WORKED_EXAMPLE_ABSTRACT)
        assert len(claims) >= 2, "Should extract at least 2 claims"
        c2 = claims[1]
        assert "Safety" in c2.types, f"c2 expected Safety, got {c2.types}"

    def test_zone_routing_safety_to_harms(self):
        """Safety claims must route to Harms zone (paper Section III-A)."""
        claims = parse_claims(WORKED_EXAMPLE_ABSTRACT)
        safety_claims = [c for c in claims if "Safety" in c.types]
        for c in safety_claims:
            assert c.zone_route == "Harms", (
                f"Safety claim should route to Harms, got {c.zone_route}"
            )

    def test_zone_routing_efficacy_to_results(self):
        """Efficacy claims must route to Results zone (paper Section III-A)."""
        claims = parse_claims(WORKED_EXAMPLE_ABSTRACT)
        efficacy_claims = [c for c in claims if "Efficacy" in c.types
                           and "Safety" not in c.types]
        for c in efficacy_claims:
            assert c.zone_route == "Results", (
                f"Efficacy claim should route to Results, got {c.zone_route}"
            )

    def test_hedging_score_bounded(self):
        """Hedging score must be in [0, 1]."""
        claims = parse_claims(MULTI_CLAIM_ABSTRACT)
        for c in claims:
            assert 0.0 <= c.hedging <= 1.0, f"Hedging score out of range: {c.hedging}"

    def test_entities_dict_structure(self):
        """Each claim must have entities dict with required keys."""
        claims = parse_claims(WORKED_EXAMPLE_ABSTRACT)
        for c in claims:
            assert "drugs" in c.entities
            assert "outcomes" in c.entities
            assert "populations" in c.entities

    def test_claim_to_dict(self):
        """to_dict() must return a serializable dict with required keys."""
        claims = parse_claims(WORKED_EXAMPLE_ABSTRACT)
        assert len(claims) > 0
        d = claims[0].to_dict()
        for key in ["sentence", "types", "entities", "hedging", "zone_route"]:
            assert key in d, f"Missing key in to_dict(): {key}"

    def test_multitype_multi_claim(self):
        """Multi-sentence abstract should yield multiple claims."""
        claims = parse_claims(MULTI_CLAIM_ABSTRACT)
        assert len(claims) >= 3, f"Expected ≥3 claims, got {len(claims)}"

    def test_empty_abstract(self):
        """Empty input should return empty list (no crash)."""
        claims = parse_claims("")
        assert isinstance(claims, list)

    def test_short_abstract(self):
        """Single sentence returns at least one claim."""
        claims = parse_claims("The drug improved outcomes significantly.")
        assert len(claims) >= 1
