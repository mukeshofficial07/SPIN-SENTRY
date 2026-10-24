"""
stages/stage1_claim_parser.py
------------------------------
Stage 1: Hybrid Claim Parser with Biomedical Entity Anchoring (HCP-BEA)

Paper reference: Section III-A
"Given an abstract A, HCP-BEA first applies BioBERT tokenization and sentence
 segmentation. Named-entity recognition then anchors each sentence to drug names,
 clinical outcomes, and population descriptors. A fine-tuned multi-label classifier
 assigns one or more of six claim types..."

MVP implementation:
  - Sentence segmentation: nltk.sent_tokenize (approximation for BioBERT tokenization)
  - NER anchoring: spaCy en_core_web_sm + regex overlays (approximation for BioBERT NER)
  - Claim classification: keyword/pattern lexicon scorer (approximation for fine-tuned
    multi-label classifier)
  - Hedging score: lexicon-based count scorer (see utils/hedging_scorer.py)

All approximations are labeled. The claim types and routing logic match the paper exactly.

Output format:
    List of Claim dicts, each with:
        {
            'sentence'   : str,           # original sentence
            'types'      : list[str],     # ≥1 of six claim types
            'entities'   : dict,          # NER output
            'hedging'    : float,         # 0–1 strength score
            'hedging_words': list[str],   # words found
            'type_scores': dict,          # per-type raw score
            'zone_route' : str,           # primary zone routing
        }
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import nltk

from utils.ner_utils import extract_entities
from utils.hedging_scorer import score_hedging, get_hedging_words_found

_DATA_DIR = Path(__file__).parent.parent / "data"

# Ensure NLTK punkt tokenizer is available
def _ensure_nltk():
    for resource in ["punkt", "punkt_tab"]:
        try:
            nltk.data.find(f"tokenizers/{resource}")
        except LookupError:
            try:
                nltk.download(resource, quiet=True)
            except Exception:
                pass

_ensure_nltk()


# ---------------------------------------------------------------------------
# Load keyword lexicon
# ---------------------------------------------------------------------------

def _load_keywords() -> dict:
    path = _DATA_DIR / "claim_type_keywords.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)

_KEYWORDS: dict = _load_keywords()


# ---------------------------------------------------------------------------
# Zone routing table (paper Section III-A)
# Safety → Harms; Efficacy → Results; Generalizability → Methods; others → Results
# ---------------------------------------------------------------------------

_ZONE_ROUTING: dict[str, str] = {
    "Safety":          "Harms",
    "Efficacy":        "Results",
    "Generalizability":"Methods",
    "Comparative":     "Results",
    "Novelty":         "Results",
    "Recommendation":  "Results",
}


# ---------------------------------------------------------------------------
# Claim dataclass
# ---------------------------------------------------------------------------

@dataclass
class Claim:
    sentence: str
    types: list[str]
    entities: dict
    hedging: float
    hedging_words: list[str]
    type_scores: dict
    zone_route: str
    role: str = "Finding / Claim"
    is_empirical_claim: bool = True

    @property
    def claim_types(self) -> list[str]:
        return self.types

    def to_dict(self) -> dict:
        return {
            "sentence":           self.sentence,
            "types":              self.types,
            "role":               self.role,
            "is_empirical_claim": self.is_empirical_claim,
            "entities":           self.entities,
            "hedging":            round(self.hedging, 3),
            "hedging_words":      self.hedging_words,
            "type_scores":        {k: round(v, 3) for k, v in self.type_scores.items()},
            "zone_route":         self.zone_route,
        }


# ---------------------------------------------------------------------------
# Core classifier
# ---------------------------------------------------------------------------

def _detect_sentence_role(sentence: str) -> tuple[str, bool]:
    """
    Detect the structural role of an abstract sentence:
    - 'Limitation / Caution': cautious interpretation, trial limitations, non-effect disclaimer
    - 'Objective': study objective / background intent statement
    - 'Methods': trial design / population recruitment / measurement protocol
    - 'Conclusion': concluding summary or recommendation
    - 'Finding / Claim': empirical findings / results claim
    """
    s = sentence.lower().strip()

    # 1. Limitation / Cautious non-effect disclaimers
    if re.search(
        r"\b(do\s+not\s+establish|does\s+not\s+establish|cannot\s+establish|did\s+not\s+establish|not\s+establish|"
        r"do\s+not\s+determine|does\s+not\s+determine|cannot\s+determine|"
        r"should\s+not\s+be\s+interpreted\s+as|no\s+claims?\s+about|"
        r"lacked\s+a\s+(?:randomized\s+|control\s+)*group|no\s+(?:randomized\s+)?control\s+group|"
        r"results?\s+(?:are|were)\s+descriptive|feasibility,?\s+not\s+efficacy|"
        r"does\s+not\s+prove|did\s+not\s+prove|cannot\s+prove|"
        r"may\s+not\s+generalize|limited\s+by|study\s+limitations?)\b",
        s,
    ):
        return ("Limitation / Caution", False)

    # 2. Objective / Study Intent / Description
    if re.match(r"^(to\s+(evaluate|assess|determine|investigate|compare|examine)|the\s+(aim|objective)|objective:|background:)", s):
        return ("Objective", False)
    if re.search(
        r"\b(this\s+(?:fictional\s+)?(?:feasibility\s+|pilot\s+|clinical\s+)?study\s+(?:describes|evaluates|investigates|assesses|examined|reports)|"
        r"we\s+(?:evaluated|investigated|assessed|described|examined)|"
        r"the\s+aim\s+of\s+this\s+study)\b",
        s,
    ):
        return ("Objective", False)

    # 3. Methods & Protocol
    if re.match(r"^(methods?|design|setting|participants?):", s):
        return ("Methods", False)
    if re.search(
        r"\b(randomly\s+assigned|were\s+recruited|were\s+enrolled|eligibility|was\s+applied|"
        r"was\s+assessed\s+using|were\s+analyzed\s+using|anova|kruskal|friedman|"
        r"trial\s+registration|protocol|study\s+design|primary\s+outcome|secondary\s+outcomes?|"
        r"outcomes?\s+were|volunteers\s+over\s+\w+\s+weeks)\b",
        s,
    ):
        return ("Methods", False)

    # 4. Conclusion
    if re.match(r"^(in\s+conclusion|these\s+findings\s+(suggest|indicate|support)|conclusions?:)", s):
        return ("Conclusion", True)

    return ("Finding / Claim", True)


def _score_claim_types(sentence: str, role: str = "Finding / Claim") -> dict[str, float]:
    """
    Score a sentence against each of the six claim types using keyword matching.

    APPROXIMATION: This is a keyword/TF-IDF rule scorer. The paper describes
    a fine-tuned multi-label classifier. Our scores are raw keyword hit counts
    normalized to [0, 1]; they are NOT probability estimates.
    """
    sentence_lower = sentence.lower()
    scores = {}

    # If the sentence is explicitly a limitation or cautious disclaimer, do not classify as positive efficacy/comparative
    if role == "Limitation / Caution":
        return {
            "Efficacy": 0.0,
            "Comparative": 0.0,
            "Safety": 0.0,
            "Generalizability": 0.4 if "generalize" in sentence_lower else 0.0,
            "Novelty": 0.0,
            "Recommendation": 0.0,
        }

    for claim_type, keywords in _KEYWORDS.items():
        hits = 0
        for kw in keywords:
            kw_clean = kw.lower()
            # Check for direct negation before the keyword (e.g. "not improve", "no significant difference")
            neg_pattern = r"(?:not|no|without|lacked\s+a)\s+(?:\w+\s+){0,3}" + re.escape(kw_clean)
            if re.search(neg_pattern, sentence_lower):
                continue
            pattern = r"\b" + re.escape(kw_clean) + r"\b"
            matches = re.findall(pattern, sentence_lower)
            hits += len(matches)
        # Normalize: 1+ hits → nonzero score, cap at 1.0
        scores[claim_type] = min(1.0, hits * 0.25) if hits > 0 else 0.0

    return scores


def _select_types(scores: dict[str, float], threshold: float | dict[str, float] = 0.15) -> list[str]:
    """
    Select claim types with score ≥ threshold. Return at least 1.
    If threshold is a dict, uses per-type thresholds.
    If no type meets threshold, return the top-scoring type.
    """
    if isinstance(threshold, dict):
        selected = [t for t, s in scores.items() if s >= threshold.get(t, 0.40)]
    else:
        selected = [t for t, s in scores.items() if s >= threshold]
    if not selected:
        # Fallback: pick highest scorer
        best = max(scores, key=scores.get)
        selected = [best]
    # Sort by score descending
    selected.sort(key=lambda t: scores[t], reverse=True)
    return selected


def _primary_zone_route(types: list[str], role: str = "Finding / Claim") -> str:
    """Determine primary zone routing from claim types and structural role."""
    if role == "Limitation / Caution":
        return "Limitations"
    if role in ("Methods", "Objective"):
        return "Methods"
    # Priority: Safety > Generalizability > others
    for priority_type in ["Safety", "Generalizability", "Efficacy", "Comparative",
                          "Novelty", "Recommendation"]:
        if priority_type in types:
            return _ZONE_ROUTING[priority_type]
    return "Results"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_claims(abstract_text: str, classifier_mode: str = "rules") -> list[Claim]:
    """
    Parse an abstract into a list of typed, entity-anchored claims.

    Parameters
    ----------
    abstract_text : str
        The abstract text to parse.
    classifier_mode : str
        ``rules`` preserves the baseline; ``trained`` uses the optional local model.

    Returns
    -------
    list[Claim]
        One Claim per sentence in the abstract.
    """
    # Sentence segmentation (NLTK punkt)
    try:
        sentences = nltk.sent_tokenize(abstract_text)
    except Exception:
        # Fallback: split on periods
        sentences = [s.strip() for s in abstract_text.split(".") if len(s.strip()) > 20]

    claims = []
    for sent in sentences:
        sent = sent.strip()
        if len(sent) < 15:  # skip very short fragments
            continue

        # Structural role detection
        role, is_empirical = _detect_sentence_role(sent)

        # NER anchoring
        entities = extract_entities(sent)

        # Claim type scoring
        if classifier_mode == "rules":
            type_scores = _score_claim_types(sent, role=role)
            selected_types = _select_types(type_scores, threshold=0.15)
        elif classifier_mode == "trained":
            from utils.trained_claim_classifier import get_thresholds, predict_scores
            if role == "Limitation / Caution":
                type_scores = {"Efficacy": 0.0, "Comparative": 0.0, "Safety": 0.0, "Generalizability": 0.3 if "generalize" in sent.lower() else 0.0, "Novelty": 0.0, "Recommendation": 0.0}
                selected_types = ["Generalizability"] if "generalize" in sent.lower() else ["Efficacy"]
            else:
                type_scores = predict_scores(sent)
                thresholds = get_thresholds()
                selected_types = _select_types(type_scores, threshold=thresholds)
        else:
            raise ValueError("classifier_mode must be 'rules' or 'trained'")

        # Hedging scoring
        hedging = score_hedging(sent)
        hedging_words = get_hedging_words_found(sent)

        # Zone routing (role-aware: Methods & Objectives route to Methods)
        zone_route = _primary_zone_route(selected_types, role=role)

        claim = Claim(
            sentence=sent,
            types=selected_types,
            entities=entities,
            hedging=hedging,
            hedging_words=hedging_words,
            type_scores=type_scores,
            zone_route=zone_route,
            role=role,
            is_empirical_claim=is_empirical,
        )
        claims.append(claim)

    return claims
