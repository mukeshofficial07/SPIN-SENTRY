"""
utils/hedging_scorer.py
-----------------------
Scores a sentence for hedging language to produce a continuous strength value
in [0, 1], where 1 = maximally certain, 0 = maximally hedged.

Approximation note: This is a lexicon-based, count-weighted scorer.
It is NOT a trained hedging classifier. It approximates the HCP-BEA
hedging-strength component described in the paper (Section III-A).
"""

import json
import re
from functools import lru_cache
from pathlib import Path

_DATA_DIR = Path(__file__).parent.parent / "data"


@lru_cache(maxsize=1)
def _load_lexicon() -> dict:
    """Load and cache the hedging lexicon."""
    path = _DATA_DIR / "hedging_lexicon.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def score_hedging(sentence: str) -> float:
    """
    Return a hedging strength score for a sentence.

    Returns
    -------
    float in [0.0, 1.0]
        1.0 = high certainty (no hedging)
        0.0 = maximally hedged

    The score starts at 1.0 and is reduced by hedging word hits,
    then increased by certainty marker hits. Clamped to [0, 1].
    """
    lexicon = _load_lexicon()
    sentence_lower = sentence.lower()

    score = 1.0  # baseline: assume certain

    for category, data in lexicon.items():
        if category.startswith("_"):
            continue
        weight = data["weight"]
        words = data["words"]
        for word in words:
            # Use word-boundary matching for multi-word phrases too
            pattern = r"\b" + re.escape(word.lower()) + r"\b"
            hits = len(re.findall(pattern, sentence_lower))
            if hits > 0:
                score -= weight * min(hits, 2)  # cap at 2 hits per word

    return max(0.0, min(1.0, score))


def get_hedging_words_found(sentence: str) -> list[str]:
    """Return list of hedging words found in the sentence (for UI display)."""
    lexicon = _load_lexicon()
    sentence_lower = sentence.lower()
    found = []

    for category, data in lexicon.items():
        if category.startswith("_"):
            continue
        for word in data["words"]:
            pattern = r"\b" + re.escape(word.lower()) + r"\b"
            if re.search(pattern, sentence_lower):
                found.append(word)

    return found
