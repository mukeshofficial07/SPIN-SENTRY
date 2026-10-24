"""
tests/test_stage2.py
---------------------
Tests for Stage 2: MZEM zone mapper.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from stages.stage2_zone_mapper import map_zones, ZoneMap, Zone, _classify_header


# ----- Fixtures ----- #

SYNTHETIC_BODY = """
Methods
Patients were randomly assigned in a 1:1 ratio. Eligibility criteria required age 18-75.
The study was a randomized controlled trial with n=1240 participants.
Intention-to-treat analysis was the primary analysis method.
Statistical analysis used Cox regression with adjustment for covariates.

Results
The primary endpoint showed a 34% reduction versus placebo (p < 0.001).
Secondary outcomes demonstrated improvement in quality of life scores.
Response rate was significantly higher in the treatment group.

Adverse Events
Serious adverse events occurred in 15% of the treatment arm.
Discontinuation rate was 8% due to adverse effects.
Grade 3 toxicity was observed in 5% of patients.

Limitations
The study was limited by the selected age range of 60-65 years.
The small sample size may limit generalizability.
Selection bias cannot be excluded in this observational component.
"""

NONSTANDARD_HEADERS_BODY = """
Patients and Methods
Participants were enrolled in this RCT with n=500.

Efficacy Results
Treatment reduced symptoms by 45% versus control.

Safety and Tolerability
Grade 3 adverse events were reported in 12% of patients.
Withdrawal due to side effects occurred in 9%.

Study Limitations
The trial was underpowered for subgroup analyses.
"""

MINIMAL_BODY = "The drug reduced symptoms significantly with p < 0.001."


# ----- Tests ----- #

class TestClassifyHeader:
    def test_results_exact(self):
        result = _classify_header("Results")
        assert result is not None
        assert result[0] == "Results"

    def test_methods_exact(self):
        result = _classify_header("Methods")
        assert result is not None
        assert result[0] == "Methods"

    def test_adverse_events(self):
        result = _classify_header("Adverse Events")
        assert result is not None
        assert result[0] == "Harms"

    def test_limitations_exact(self):
        result = _classify_header("Limitations")
        assert result is not None
        assert result[0] == "Limitations"

    def test_tolerability_fuzzy(self):
        """Non-standard 'Tolerability' header should fuzzy-match to Harms."""
        result = _classify_header("Tolerability")
        assert result is not None
        assert result[0] == "Harms", f"'Tolerability' should map to Harms, got {result}"

    def test_safety_and_tolerability_fuzzy(self):
        result = _classify_header("Safety and Tolerability")
        assert result is not None
        assert result[0] == "Harms"

    def test_short_line_rejected(self):
        result = _classify_header("AB")
        assert result is None

    def test_long_line_rejected(self):
        result = _classify_header("This is a very long sentence that is clearly not a section header at all.")
        assert result is None


class TestMapZones:
    def test_returns_zone_map(self):
        result = map_zones(SYNTHETIC_BODY)
        assert isinstance(result, ZoneMap)

    def test_four_zones_exist(self):
        result = map_zones(SYNTHETIC_BODY)
        for zone_name in ["Results", "Harms", "Limitations", "Methods"]:
            assert zone_name in result.zones

    def test_harms_not_empty(self):
        """Core requirement: Harms zone must be isolated."""
        result = map_zones(SYNTHETIC_BODY)
        harms = result.zones["Harms"]
        assert not harms.is_empty, "Harms zone should not be empty"

    def test_harms_and_results_distinct(self):
        """Core requirement: Harms zone sentences must differ from Results."""
        result = map_zones(SYNTHETIC_BODY)
        harms_sents = set(result.zones["Harms"].sentences)
        results_sents = set(result.zones["Results"].sentences)
        # They should not be identical (some separation)
        assert harms_sents != results_sents, "Harms and Results zones should not be identical"

    def test_methods_populated(self):
        result = map_zones(SYNTHETIC_BODY)
        methods = result.zones["Methods"]
        assert not methods.is_empty, "Methods zone should not be empty"

    def test_nonstandard_headers_handled(self):
        """Non-standard headers like 'Safety and Tolerability' should map to Harms."""
        result = map_zones(NONSTANDARD_HEADERS_BODY)
        harms = result.zones["Harms"]
        assert not harms.is_empty, (
            "Harms zone should be populated even with non-standard 'Safety and Tolerability' header"
        )

    def test_minimal_body_no_crash(self):
        """Should not crash on minimal body text."""
        result = map_zones(MINIMAL_BODY)
        assert isinstance(result, ZoneMap)

    def test_empty_body_no_crash(self):
        result = map_zones("")
        assert isinstance(result, ZoneMap)

    def test_zone_sentence_count_consistent(self):
        result = map_zones(SYNTHETIC_BODY)
        for name, zone in result.zones.items():
            assert zone.sentence_count == len(zone.sentences), (
                f"{name}: sentence_count mismatch"
            )

    def test_warnings_are_list(self):
        result = map_zones(SYNTHETIC_BODY)
        assert isinstance(result.warnings, list)

    def test_methods_text_available(self):
        result = map_zones(SYNTHETIC_BODY)
        assert isinstance(result.methods_text, str)
