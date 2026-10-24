"""
tests/test_demo_papers_regression.py
------------------------------------
Regression tests verifying qualitative pipeline behavior on the three
synthetic demonstration PDFs:
  1. SPIN_SENTRY_Demo_Safe.pdf       -> Low Risk (internally consistent, cautious)
  2. SPIN_SENTRY_Demo_Risk.pdf       -> Moderate Risk (qualification / overstatement)
  3. SPIN_SENTRY_Demo_High_Risk.pdf  -> High Risk (direct contradictory signal)
"""

import os
from pathlib import Path
import pytest

from pipeline import run_pipeline

DEMO_DIR = Path(__file__).resolve().parent.parent / "data" / "demo_papers"


@pytest.mark.skipif(not DEMO_DIR.exists(), reason="Demo papers directory not found")
class TestDemoPapersRegression:

    def test_safe_demo_not_high_risk(self):
        safe_pdf = DEMO_DIR / "SPIN_SENTRY_Demo_Safe.pdf"
        assert safe_pdf.exists(), f"File missing: {safe_pdf}"

        result = run_pipeline(safe_pdf, top_k=3)
        summary = result.summary

        # Qualitative check: must not be High Risk
        assert summary["spin_risk"] == "🟢 Low", (
            f"Safe paper should be evaluated as Low Risk, got {summary['spin_risk']}"
        )
        assert summary["flagged_claims"] == 0, (
            f"Safe paper should have 0 flagged claims, got {summary['flagged_claims']}"
        )
        assert summary["empirical_flagged"] == 0

        # Check that the break increase finding is supported
        supported_reports = [r for r in result.reports if r.is_empirical_claim]
        assert len(supported_reports) == 1
        assert supported_reports[0].support_label == "Supported"
        assert supported_reports[0].spin_flag is False

        # Check that limitation sentence is recognized as cautious disclaimer
        lim_reports = [r for r in result.reports if r.role == "Limitation / Caution"]
        assert len(lim_reports) == 1
        assert "Cautious Limitation" in lim_reports[0].support_label
        assert lim_reports[0].spin_flag is False

    def test_moderate_risk_demo_overstatement(self):
        risk_pdf = DEMO_DIR / "SPIN_SENTRY_Demo_Risk.pdf"
        assert risk_pdf.exists(), f"File missing: {risk_pdf}"

        result = run_pipeline(risk_pdf, top_k=3)
        summary = result.summary

        # Qualitative check: Moderate Risk (overstatement, no statistical contradiction)
        assert summary["spin_risk"] == "🟡 Moderate", (
            f"FocusBoard should be evaluated as Moderate Risk, got {summary['spin_risk']}"
        )
        assert summary.get("contradiction_count", 0) == 0, (
            "FocusBoard should not have contradictory signal (it is an overstatement issue)"
        )
        assert summary["empirical_flagged"] >= 1, "Should flag overstatement"

        # Check that the study performance claim is flagged for outcome mismatch
        reports = result.reports
        claim3 = [r for r in reports if "effective for improving study" in r.claim.lower()]
        assert len(claim3) == 1
        assert claim3[0].spin_flag is True
        assert "Outcome Mismatch" in claim3[0].support_label or "Low Match" in claim3[0].support_label

    def test_high_risk_demo_contradiction(self):
        high_risk_pdf = DEMO_DIR / "SPIN_SENTRY_Demo_High_Risk.pdf"
        assert high_risk_pdf.exists(), f"File missing: {high_risk_pdf}"

        result = run_pipeline(high_risk_pdf, top_k=3)
        summary = result.summary

        # Qualitative check: High Risk (direct contradictory reporting)
        assert summary["spin_risk"] == "🔴 High", (
            f"StudyFlow should be evaluated as High Risk, got {summary['spin_risk']}"
        )
        assert summary.get("contradiction_count", 0) >= 1, (
            "StudyFlow must detect at least one contradictory signal between abstract claim and data"
        )

        # Conclusion claim must be flagged as Contradictory Signal
        conclusion_reports = [r for r in result.reports if "statistically significant improvement" in r.claim.lower()]
        assert len(conclusion_reports) >= 1
        assert "Contradictory" in conclusion_reports[0].support_label
        assert conclusion_reports[0].spin_flag is True
