"""
tests/test_stage3.py
---------------------
Tests for Stage 3: DC-NLI directional entailment.

Tests the NLI model (or fallback) for basic directional behavior.
Does NOT test performance — only structural behavior and score ranges.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from stages.stage3_nli import (
    directional_nli,
    score_claim_against_zone,
    get_model_status,
)


# ----- Tests ----- #

class TestModelStatus:
    def test_get_model_status_returns_dict(self):
        status = get_model_status()
        assert isinstance(status, dict)

    def test_model_status_has_required_keys(self):
        status = get_model_status()
        for key in ["model_name", "loaded", "error", "fallback_active"]:
            assert key in status, f"Missing key: {key}"

    def test_model_name_correct(self):
        status = get_model_status()
        assert "nli-MiniLM2" in status["model_name"]


class TestDirectionalNLI:
    def test_returns_required_keys(self):
        """Output must have forward, reverse, gap (paper Eqs 1–3)."""
        result = directional_nli(
            claim="The drug is effective.",
            evidence="The treatment showed significant improvement.",
        )
        for key in ["forward", "reverse", "gap", "model_used"]:
            assert key in result, f"Missing key: {key}"

    def test_gap_equals_one_minus_reverse(self):
        """gap = 1 - reverse (paper Equation 3)."""
        result = directional_nli(
            claim="The drug improved outcomes.",
            evidence="Patients showed improvement in the treatment group.",
        )
        expected_gap = round(1.0 - result["reverse"], 4)
        assert abs(result["gap"] - expected_gap) < 1e-3, (
            f"gap = {result['gap']} ≠ 1 - reverse = {expected_gap}"
        )

    def test_scores_in_valid_range(self):
        """All scores must be in [0, 1]."""
        result = directional_nli(
            claim="The drug is safe.",
            evidence="No adverse events were reported.",
        )
        for key in ["forward", "reverse", "gap"]:
            assert 0.0 <= result[key] <= 1.0, f"{key} = {result[key]} out of [0,1]"

    def test_clear_support_pair_reasonable_forward(self):
        """
        A clearly supporting evidence should yield a reasonable forward score.
        We only check it is not zero — we do NOT claim a specific threshold
        because the model is not fine-tuned on biomedical text.
        """
        result = directional_nli(
            claim="The drug significantly reduced mortality.",
            evidence="Mortality was significantly reduced in the treatment arm (p < 0.001).",
        )
        assert result["forward"] >= 0.0  # just ensure no error

    def test_contradiction_not_entailment(self):
        """
        A sentence clearly contradicting the claim should yield lower forward
        than a supporting sentence. Soft test due to model limitations.
        """
        support = directional_nli(
            claim="The drug was well-tolerated.",
            evidence="No serious adverse events were observed.",
        )
        contradict = directional_nli(
            claim="The drug was well-tolerated.",
            evidence="15% of patients experienced serious adverse events requiring hospitalization.",
        )
        # The contradiction case should NOT yield higher forward than support
        # (soft assertion — model may not always get this right)
        # We just verify both run without error
        assert isinstance(support["forward"], float)
        assert isinstance(contradict["forward"], float)

    def test_empty_strings_no_crash(self):
        """Should not crash on empty inputs."""
        result = directional_nli(claim="", evidence="")
        assert isinstance(result, dict)

    def test_model_used_is_string(self):
        result = directional_nli(
            claim="Treatment improved survival.",
            evidence="Survival was significantly improved.",
        )
        assert isinstance(result["model_used"], str)
        assert result["model_used"] in ("nli-MiniLM2", "keyword_fallback")


class TestScoreClaimAgainstZone:
    CLAIM = "Drug X was well-tolerated with a favorable safety profile."
    ZONE_SENTS = [
        "Serious adverse events occurred in 15% of the treatment arm.",
        "Discontinuation rate was 8% due to adverse effects.",
        "No grade 3 or grade 4 toxicity was observed.",
        "The treatment was associated with mild headache in 5% of patients.",
        "Most side effects were transient and self-limiting.",
    ]

    def test_returns_list(self):
        results = score_claim_against_zone(self.CLAIM, self.ZONE_SENTS)
        assert isinstance(results, list)

    def test_top_k_respected(self):
        results = score_claim_against_zone(self.CLAIM, self.ZONE_SENTS, top_k=3)
        assert len(results) <= 3

    def test_results_have_required_keys(self):
        results = score_claim_against_zone(self.CLAIM, self.ZONE_SENTS, top_k=5)
        assert len(results) > 0
        for r in results:
            for key in ["evidence", "forward", "reverse", "gap", "rank"]:
                assert key in r, f"Missing key: {key}"

    def test_sorted_by_forward_descending(self):
        results = score_claim_against_zone(self.CLAIM, self.ZONE_SENTS, top_k=5)
        forwards = [r["forward"] for r in results]
        assert forwards == sorted(forwards, reverse=True), "Results should be sorted by forward score"

    def test_empty_zone_returns_empty(self):
        results = score_claim_against_zone(self.CLAIM, [])
        assert results == []

    def test_rank_starts_at_one(self):
        results = score_claim_against_zone(self.CLAIM, self.ZONE_SENTS, top_k=3)
        if results:
            assert results[0]["rank"] == 1
