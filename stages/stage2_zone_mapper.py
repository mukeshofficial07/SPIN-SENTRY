"""
stages/stage2_zone_mapper.py
-----------------------------
Stage 2: Multi-Zone Evidence Mapper (MZEM)

Paper reference: Section III-B
"MZEM instead partitions the article into four evidence zones using a SciBERT
 section classifier augmented with discourse cues: (1) Results, (2) Harms,
 (3) Limitations, and (4) Methods."

MVP implementation:
  - Zone detection: regex pattern matching on section headers (zone_headers.json)
  - Augmented with: rapidfuzz fuzzy matching for non-standard headers
  - SciBERT section classifier: NOT used (would require fine-tuning; we use
    regex + fuzzy matching as a documented approximation)
  - Harms zone isolation: explicitly extracted and kept separate from Results
  - Primary vs. secondary outcome distinction: keyword-based within Results zone

Key design decision (paper Section V-C error analysis):
  Non-standard headers ("Tolerability", "Safety Outcomes") are handled via
  the fuzzy matching layer. Merged Results/Harms sections are flagged.

Output:
    dict with keys 'Results', 'Harms', 'Limitations', 'Methods'
    Each value is a Zone dataclass.
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from rapidfuzz import fuzz

_DATA_DIR = Path(__file__).parent.parent / "data"

ZONE_NAMES = ["Results", "Harms", "Limitations", "Methods", "Discussion", "Conclusion"]


# ---------------------------------------------------------------------------
# Load zone header patterns
# ---------------------------------------------------------------------------

def _load_zone_patterns() -> dict:
    path = _DATA_DIR / "zone_headers.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)

_ZONE_CONFIG = _load_zone_patterns()

# Pre-compile regex patterns
_COMPILED_PATTERNS: dict[str, list[re.Pattern]] = {}
for zone_name, config in _ZONE_CONFIG.items():
    if zone_name.startswith("_"):
        continue
    _COMPILED_PATTERNS[zone_name] = [
        re.compile(p, re.IGNORECASE) for p in config["patterns"]
    ]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Zone:
    name: str
    sentences: list[str] = field(default_factory=list)
    raw_text: str = ""
    header_found: str = ""             # the actual header text matched
    match_method: str = ""             # "regex", "fuzzy", or "keyword_fallback"
    confidence: float = 0.0
    sentence_count: int = 0
    is_empty: bool = True
    warning: Optional[str] = None

    def __post_init__(self):
        self.sentence_count = len(self.sentences)
        self.is_empty = len(self.sentences) == 0


@dataclass
class ZoneMap:
    zones: dict[str, Zone] = field(default_factory=dict)
    unmapped_sentences: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    methods_text: str = ""   # full Methods zone text for ESI extraction

    def get(self, zone_name: str) -> Zone:
        return self.zones.get(zone_name, Zone(name=zone_name))


# ---------------------------------------------------------------------------
# Header classification
# ---------------------------------------------------------------------------

_HEADER_RE = re.compile(
    r"^[\s\d\.\-–—]*([A-Z][A-Za-z\s\/\-]{1,80})\s*$",
    re.MULTILINE,
)

# Minimum characters in a line to consider it a section header candidate
_MIN_HEADER_LEN = 3
_MAX_HEADER_LEN = 80


def _classify_header(line: str) -> Optional[tuple[str, str, float]]:
    """
    Try to classify a line as a zone header.

    Returns (zone_name, method, confidence) or None.
    """
    line_stripped = line.strip()
    if len(line_stripped) < _MIN_HEADER_LEN or len(line_stripped) > _MAX_HEADER_LEN:
        return None

    # Remove leading numbers like "3." or "III."
    cleaned = re.sub(r"^[\d\s\.\-IVXivx]+\s*", "", line_stripped).strip()
    if not cleaned:
        cleaned = line_stripped

    # --- Regex match ---
    for zone_name, patterns in _COMPILED_PATTERNS.items():
        for pat in patterns:
            if pat.match(cleaned):
                return (zone_name, "regex", 1.0)

    # --- Fuzzy match for non-standard headers ---
    # Known alias → zone mapping for fuzzy matching
    _FUZZY_TARGETS = {
        "Results":      ["results", "findings", "efficacy results", "results and discussion",
                         "clinical outcomes", "clinical findings", "trial results"],
        "Harms":        ["adverse events", "safety", "tolerability", "side effects",
                         "toxicity", "harms", "safety and tolerability",
                         "tolerability and safety", "safety outcomes", "safety profile",
                         "adverse effects", "safety results", "aes", "saes"],
        "Limitations":  ["limitations", "study limitations", "strengths and limitations"],
        "Methods":      ["methods", "materials and methods", "patients and methods",
                         "methodology", "study design", "study population",
                         "primary outcome", "secondary outcomes", "outcome measures"],
        "Discussion":   ["discussion", "general discussion", "discussion and conclusions"],
        "Conclusion":   ["conclusions", "conclusion", "concluding remarks"],
    }

    best_zone, best_score = None, 0
    for zone_name, targets in _FUZZY_TARGETS.items():
        for target in targets:
            score = fuzz.ratio(cleaned.lower(), target.lower())
            if score > best_score:
                best_score = score
                best_zone = zone_name

    if best_score >= 80:
        return (best_zone, "fuzzy", best_score / 100)

    return None


# ---------------------------------------------------------------------------
# Main segmentation logic
# ---------------------------------------------------------------------------

_TERMINATING_HEADER_TARGETS = [
    "acknowledgments", "acknowledgements", "author contributions", "funding", "references",
    "supporting information", "supplementary material", "conflict of interest",
    "conflicts of interest", "data availability", "data availability statement"
]


def _is_clean_evidence_sentence(sent: str) -> bool:
    """Filter out non-sentence artefacts (table rows, dense numbers, captions)."""
    s = sent.strip()
    words = s.split()
    if len(words) < 4 or len(s) > 400:
        return False
    nums = len(re.findall(r"\b\d+(?:\.\d+)?\b", s))
    if nums > 6 and (s.count("±") >= 1 or s.count("%") >= 3):
        return False
    if s.lower().startswith(("table ", "fig ", "figure ", "variable ", "panel a", "panel b")):
        return False
    if s.count("|") >= 2:
        return False
    return True


def _split_into_blocks(text: str) -> list[tuple[str, list[str]]]:
    """
    Split body text into (header_or_None, [sentences]) blocks.

    A "block" is a section from one header to the next.
    Returns list of (header_text, sentence_list).
    """
    import nltk
    try:
        from nltk.tokenize import sent_tokenize
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)
        from nltk.tokenize import sent_tokenize

    lines = text.splitlines()
    blocks = []
    current_header = None
    current_lines = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        # Check if this line looks like a header
        is_potential_header = (
            len(stripped) <= _MAX_HEADER_LEN
            and not stripped.endswith(".")
            and not stripped.endswith(",")
            and len(stripped.split()) <= 8
        )

        if is_potential_header:
            cleaned = re.sub(r"^[\d\s\.\-IVXivx]+\s*", "", stripped).strip() or stripped
            classification = _classify_header(stripped)

            if classification:
                # Save previous block
                if current_lines:
                    block_text = " ".join(current_lines)
                    try:
                        sents = sent_tokenize(block_text)
                    except Exception:
                        sents = [s.strip() for s in block_text.split(".") if s.strip()]
                    blocks.append((current_header, sents))
                # Start new block
                current_header = (stripped, *classification)  # (text, zone, method, conf)
                current_lines = []
                continue

            # Check if this is a terminating non-zone header (e.g. Conclusions, References)
            is_terminating = any(
                fuzz.ratio(cleaned.lower(), th) >= 80 for th in _TERMINATING_HEADER_TARGETS
            )
            if is_terminating:
                if current_lines:
                    block_text = " ".join(current_lines)
                    try:
                        sents = sent_tokenize(block_text)
                    except Exception:
                        sents = [s.strip() for s in block_text.split(".") if s.strip()]
                    blocks.append((current_header, sents))
                current_header = None
                current_lines = []
                continue

        current_lines.append(stripped)

    # Final block
    if current_lines and current_header is not None:
        block_text = " ".join(current_lines)
        try:
            from nltk.tokenize import sent_tokenize
            sents = sent_tokenize(block_text)
        except Exception:
            sents = [s.strip() for s in block_text.split(".") if s.strip()]
        blocks.append((current_header, sents))

    return blocks


def map_zones(body_text: str) -> ZoneMap:
    """
    Partition the full-text body into four evidence zones.

    Parameters
    ----------
    body_text : str
        The body of the paper (everything after the abstract).

    Returns
    -------
    ZoneMap
        Zones with their extracted sentences and metadata.
    """
    zone_map = ZoneMap()
    warnings = []

    # Initialize empty zones
    for name in ZONE_NAMES:
        zone_map.zones[name] = Zone(name=name)

    # Split into blocks by headers
    blocks = _split_into_blocks(body_text)

    unmapped = []
    for header_info, sentences in blocks:
        if header_info is None:
            unmapped.extend(sentences)
            continue

        # header_info = (text, zone_name, method, confidence)
        header_text, zone_name, method, confidence = header_info

        clean_sents = [s for s in sentences if _is_clean_evidence_sentence(s)]
        if not clean_sents:
            clean_sents = sentences  # fallback if all were filtered

        zone = zone_map.zones[zone_name]
        zone.sentences.extend(clean_sents)
        zone.raw_text += " ".join(clean_sents) + " "
        if not zone.header_found:
            zone.header_found = header_text
        zone.match_method = method
        zone.confidence = max(zone.confidence, confidence)

    # Post-process
    for name, zone in zone_map.zones.items():
        zone.sentence_count = len(zone.sentences)
        zone.is_empty = zone.sentence_count == 0
        zone.raw_text = zone.raw_text.strip()

    # Keyword-based fallback for empty zones
    _apply_keyword_fallback(body_text, zone_map, warnings)
    # Check for merged Results/Harms (paper error analysis, Section V-C)
    if zone_map.zones["Harms"].is_empty and not zone_map.zones["Results"].is_empty:
        # Apply harm-specific keyword extraction from Results zone
        _extract_harms_from_results(zone_map, warnings)

    if zone_map.zones["Results"].is_empty:
        warnings.append("⚠️ Results zone not detected. Evidence retrieval may be limited.")

    zone_map.warnings = warnings
    zone_map.unmapped_sentences = unmapped
    zone_map.methods_text = zone_map.zones["Methods"].raw_text

    return zone_map


# ---------------------------------------------------------------------------
# Fallback strategies
# ---------------------------------------------------------------------------

# Keyword sets for fallback zone assignment when headers are missing
_FALLBACK_KEYWORDS = {
    "Results": [
        "reduction", "improvement", "response rate", "p <", "p=", "odds ratio",
        "confidence interval", "hazard ratio", "statistically significant",
        "primary endpoint", "secondary endpoint", "outcome", "efficacy"
    ],
    "Harms": [
        "adverse event", "adverse effect", "side effect", "toxicity",
        "discontinued", "discontinuation", "withdrawal", "serious adverse",
        "tolerability", "safety profile", "AE", "SAE", "grade 3", "grade 4",
        "mortality", "death", "complication"
    ],
    "Limitations": [
        "limitation", "weakness", "caveat", "underpowered", "selection bias",
        "small sample", "observational", "cannot generalize", "future study",
        "further research", "larger study"
    ],
    "Methods": [
        "randomized", "randomised", "eligible", "enrolled", "criteria",
        "inclusion", "exclusion", "intervention", "placebo", "intention-to-treat",
        "per-protocol", "statistical analysis", "sample size", "power calculation"
    ],
}


def _apply_keyword_fallback(body_text: str, zone_map: ZoneMap, warnings: list):
    """
    For empty zones: try to find sentences matching zone-specific keywords.
    This is a last-resort approximation when header detection fails.
    """
    import nltk
    try:
        all_sentences = nltk.sent_tokenize(body_text)
    except Exception:
        all_sentences = [s.strip() for s in body_text.split(".") if len(s.strip()) > 20]

    for zone_name, keywords in _FALLBACK_KEYWORDS.items():
        zone = zone_map.zones[zone_name]
        if not zone.is_empty:
            continue  # already populated

        fallback_sents = []
        for sent in all_sentences:
            sent_lower = sent.lower()
            if any(kw.lower() in sent_lower for kw in keywords):
                fallback_sents.append(sent)

        if fallback_sents:
            zone.sentences = fallback_sents[:50]  # cap to avoid noise
            zone.raw_text = " ".join(zone.sentences)
            zone.sentence_count = len(zone.sentences)
            zone.is_empty = False
            zone.match_method = "keyword_fallback"
            zone.confidence = 0.5
            zone.warning = f"Zone populated via keyword fallback (header not found)"
            warnings.append(
                f"ℹ️ {zone_name} zone populated via keyword fallback "
                f"({len(zone.sentences)} sentences). Header detection failed."
            )


_HARM_KEYWORDS = [
    "adverse event", "adverse effect", "side effect", "toxicity",
    "discontinued", "discontinuation", "withdrawal", "serious adverse",
    "tolerability", "SAE", "AE", "grade 3", "grade 4", "mortality",
    "complication", "safety"
]


def _extract_harms_from_results(zone_map: ZoneMap, warnings: list):
    """
    If Harms zone is empty, extract harm-relevant sentences from Results.
    This handles merged Results/Harms sections.
    """
    results_sents = zone_map.zones["Results"].sentences
    harm_sents = []
    for sent in results_sents:
        sent_lower = sent.lower()
        if any(kw.lower() in sent_lower for kw in _HARM_KEYWORDS):
            harm_sents.append(sent)

    if harm_sents:
        zone = zone_map.zones["Harms"]
        zone.sentences = harm_sents
        zone.raw_text = " ".join(harm_sents)
        zone.match_method = "isolated_from_merged_results"
        zone.confidence = 0.85
        zone.warning = "Harms zone isolated from Results/Discussion text (merged section per paper Section V-C)"
        warnings.append(
            f"ℹ️ Harms zone isolated from Results text ({len(harm_sents)} sentences) due to merged Results/Harms section."
        )
