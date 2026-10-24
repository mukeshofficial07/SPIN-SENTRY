"""
utils/esi_table.py
------------------
Evidence Strength Index (ESI) lookup and metadata extraction.

Implements Table I from the paper exactly:

    Study Design  | Sample Size | Analysis Type      | ESI
    --------------|-------------|---------------------|-----
    RCT           | > 1000      | Intention-to-treat  | 1.0
    RCT           | < 100       | Per-protocol        | 0.7
    Observational | > 5000      | Adjusted            | 0.6
    Case series   | N < 50      | Uncontrolled        | 0.2

For the prototype, study-design metadata is extracted from the evidence
sentence and surrounding zone text via regex. This is an approximation
of what the paper describes; a production system would parse structured
metadata from the paper's Methods section.
"""

import re
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# ESI table (exact from paper Table I)
# ---------------------------------------------------------------------------

@dataclass
class ESIEntry:
    study_design: str
    sample_size_qualifier: str   # "> 1000", "< 100", "> 5000", "N < 50"
    analysis_type: str
    esi: float


ESI_TABLE: list[ESIEntry] = [
    ESIEntry("RCT",          "> 1000", "Intention-to-treat", 1.0),
    ESIEntry("RCT",          "< 100",  "Per-protocol",       0.7),
    ESIEntry("Observational","> 5000", "Adjusted",           0.6),
    ESIEntry("Case series",  "N < 50", "Uncontrolled",       0.2),
]

# Default ESI when metadata is missing (paper Section V-C error analysis)
DEFAULT_ESI = 0.5


# ---------------------------------------------------------------------------
# Regex patterns for metadata extraction from text
# ---------------------------------------------------------------------------

_DESIGN_PATTERNS = {
    "RCT": re.compile(
        r"\b(randomized?\s*(controlled)?\s*trial|RCT|randomised?\s*clinical\s*trial"
        r"|double.?blind|placebo.?controlled|phase\s+[23])\b",
        re.IGNORECASE,
    ),
    "Observational": re.compile(
        r"\b(observational|cohort\s+study|prospective\s+study|retrospective"
        r"|cross.?sectional|registry|population.?based)\b",
        re.IGNORECASE,
    ),
    "Case series": re.compile(
        r"\b(case\s+series|case\s+report|case.?control|uncontrolled|open.?label)\b",
        re.IGNORECASE,
    ),
}

_SAMPLE_SIZE_RE = re.compile(
    r"\b(?:n\s*=\s*|N\s*=\s*|n\s*=\s*|enrolled\s+|recruited\s+|included\s+)"
    r"(\d[\d,]*)",
    re.IGNORECASE,
)

_ITT_RE = re.compile(r"\b(intention.?to.?treat|ITT|full\s+analysis\s+set|FAS)\b", re.IGNORECASE)
_PP_RE = re.compile(r"\b(per.?protocol|PP\s+analysis|per protocol)\b", re.IGNORECASE)
_ADJUSTED_RE = re.compile(r"\b(adjusted|multivariate|multivariable|Cox\s+regression|logistic\s+regression)\b", re.IGNORECASE)


@dataclass
class StudyMetadata:
    study_design: str = "Unknown"
    sample_size: Optional[int] = None
    analysis_type: str = "Unknown"
    esi: float = DEFAULT_ESI
    extraction_notes: list[str] = field(default_factory=list)


def extract_study_metadata(text: str) -> StudyMetadata:
    """
    Extract study design metadata from text (Methods zone preferred).

    Approximation: regex-based extraction from arbitrary text.
    May miss metadata in tables or figures. Falls back to DEFAULT_ESI.
    """
    meta = StudyMetadata()
    notes = []

    # --- Study design ---
    for design, pattern in _DESIGN_PATTERNS.items():
        if pattern.search(text):
            meta.study_design = design
            notes.append(f"Design detected: {design}")
            break

    # --- Sample size ---
    size_matches = _SAMPLE_SIZE_RE.findall(text)
    if size_matches:
        # Take the largest N found (most likely the total sample)
        sizes = []
        for m in size_matches:
            try:
                sizes.append(int(m.replace(",", "")))
            except ValueError:
                pass
        if sizes:
            meta.sample_size = max(sizes)
            notes.append(f"Sample size: n={meta.sample_size}")

    # --- Analysis type ---
    if _ITT_RE.search(text):
        meta.analysis_type = "Intention-to-treat"
        notes.append("Analysis: ITT")
    elif _PP_RE.search(text):
        meta.analysis_type = "Per-protocol"
        notes.append("Analysis: Per-protocol")
    elif _ADJUSTED_RE.search(text):
        meta.analysis_type = "Adjusted"
        notes.append("Analysis: Adjusted")
    else:
        meta.analysis_type = "Uncontrolled/Unknown"
        notes.append("Analysis type not detected, using default")

    # --- ESI lookup ---
    meta.esi = _lookup_esi(meta)
    notes.append(f"ESI assigned: {meta.esi}")
    meta.extraction_notes = notes

    return meta


def _lookup_esi(meta: StudyMetadata) -> float:
    """
    Lookup ESI from Table I using extracted metadata.
    Returns DEFAULT_ESI if no match found.
    """
    design = meta.study_design
    n = meta.sample_size
    analysis = meta.analysis_type

    if design == "RCT":
        if analysis == "Intention-to-treat" and n is not None and n > 1000:
            return 1.0
        elif analysis == "Per-protocol" and n is not None and n < 100:
            return 0.7
        elif n is not None and n > 1000:
            return 0.9   # RCT + large N, analysis type ambiguous
        elif n is not None and n < 100:
            return 0.65  # RCT + small N
        else:
            return 0.8   # RCT, unknown size

    elif design == "Observational":
        if n is not None and n > 5000:
            return 0.6
        elif n is not None and n > 1000:
            return 0.5
        else:
            return 0.45

    elif design == "Case series":
        if n is not None and n < 50:
            return 0.2
        else:
            return 0.3

    return DEFAULT_ESI


def get_zone_type_relevance(claim_type: str, zone: str) -> float:
    """
    Zone–type relevance weight as specified in the paper (Section III-D).

    Safety claims: full weight from Harms zone, reduced from Results.
    Generalizability claims: scored primarily against Methods.
    Others: Results zone is primary.

    Returns a multiplier in (0, 1].
    """
    relevance_table = {
        # (claim_type, zone) -> weight
        ("Safety",          "Harms"):       1.0,
        ("Safety",          "Results"):     0.4,
        ("Safety",          "Limitations"): 0.6,
        ("Safety",          "Methods"):     0.3,

        ("Efficacy",        "Results"):     1.0,
        ("Efficacy",        "Harms"):       0.5,
        ("Efficacy",        "Limitations"): 0.6,
        ("Efficacy",        "Methods"):     0.3,

        ("Generalizability","Methods"):     1.0,
        ("Generalizability","Results"):     0.5,
        ("Generalizability","Limitations"): 0.8,
        ("Generalizability","Harms"):       0.3,

        ("Comparative",     "Results"):     1.0,
        ("Comparative",     "Harms"):       0.6,
        ("Comparative",     "Methods"):     0.5,
        ("Comparative",     "Limitations"): 0.5,

        ("Novelty",         "Results"):     0.9,
        ("Novelty",         "Methods"):     0.7,
        ("Novelty",         "Harms"):       0.3,
        ("Novelty",         "Limitations"): 0.4,

        ("Recommendation",  "Results"):     0.9,
        ("Recommendation",  "Methods"):     0.6,
        ("Recommendation",  "Limitations"): 0.7,
        ("Recommendation",  "Harms"):       0.7,
    }

    key = (claim_type, zone)
    return relevance_table.get(key, 0.5)  # default: moderate relevance
