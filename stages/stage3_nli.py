"""
stages/stage3_nli.py
---------------------
Stage 3: Directional Claim–Evidence NLI (DC-NLI)

Paper reference: Section III-C, Equations (1)–(3):
    forward = entailment(e, c)     — (1)
    reverse = entailment(c, e)     — (2)
    gap     = 1 − reverse          — (3)

The reverse score penalizes generalizability stretch: an abstract claim that
"works in elderly patients" fails the reverse check when the Methods section
reports enrolment limited to ages 60–65.

Model: cross-encoder/nli-MiniLM2-L6-H768 from HuggingFace (~90 MB).
  - This is a real NLI model producing entailment/neutral/contradiction probabilities.
  - NOT a cosine similarity scorer. Scores are softmax probabilities over NLI labels.
  - NOT fine-tuned on biomedical text. Performance on clinical trial language may
    differ from general-domain benchmarks.

Fallback: If model download fails or torch is unavailable, falls back to a
  simple keyword-overlap scorer with a clear warning in output.

Output per (claim, evidence) pair:
    {
        'forward'          : float,  # entailment(e → c), Eq. (1)
        'reverse'          : float,  # entailment(c → e), Eq. (2)
        'gap'              : float,  # 1 - reverse,       Eq. (3)
        'contradiction'    : float,  # contradiction prob (forward direction)
        'neutral'          : float,
        'model_used'       : str,    # 'nli-MiniLM2' or 'keyword_fallback'
    }
"""

import re
from functools import lru_cache
from typing import Optional

# Model name as specified in the implementation plan
_NLI_MODEL_NAME = "cross-encoder/nli-MiniLM2-L6-H768"
_LABEL_ORDER = ["contradiction", "entailment", "neutral"]


# ---------------------------------------------------------------------------
# Model loading with fallback
# ---------------------------------------------------------------------------

_model = None
_model_load_error: Optional[str] = None
_model_loaded: bool = False


def _load_model():
    """
    Lazy-load the NLI CrossEncoder model. Sets module-level flags.
    Thread-safe for single-threaded Streamlit use.
    """
    global _model, _model_load_error, _model_loaded
    if _model_loaded:
        return

    try:
        from sentence_transformers import CrossEncoder
        _model = CrossEncoder(_NLI_MODEL_NAME, num_labels=3)
        _model_loaded = True
        _model_load_error = None
    except Exception as exc:
        _model = None
        _model_load_error = str(exc)
        _model_loaded = True  # mark as attempted


def get_model_status() -> dict:
    """Return model loading status for UI display."""
    _load_model()
    return {
        "model_name": _NLI_MODEL_NAME,
        "loaded": _model is not None,
        "error": _model_load_error,
        "fallback_active": _model is None,
    }


# ---------------------------------------------------------------------------
# NLI scoring
# ---------------------------------------------------------------------------

def _nli_score_model(premise: str, hypothesis: str) -> dict:
    """
    Score using the NLI CrossEncoder model.
    Returns dict with entailment, neutral, contradiction probabilities.
    """
    import numpy as np

    scores = _model.predict([(premise, hypothesis)], apply_softmax=True)
    # scores shape: (1, 3), labels: contradiction=0, entailment=1, neutral=2
    s = scores[0]
    return {
        "contradiction": float(s[0]),
        "entailment":    float(s[1]),
        "neutral":       float(s[2]),
    }


def _nli_score_fallback(premise: str, hypothesis: str) -> dict:
    """
    Keyword-overlap fallback when NLI model is unavailable.

    FALLBACK NOTE: This is NOT NLI. It is a Jaccard token overlap score
    repurposed as an approximation. It cannot detect contradiction or
    generalizability stretch. Results will be labeled as 'keyword_fallback'
    in the output. Do not interpret these as entailment probabilities.
    """
    def _tokens(text: str) -> set:
        return set(re.findall(r"\b[a-zA-Z]{3,}\b", text.lower()))

    tok_p = _tokens(premise)
    tok_h = _tokens(hypothesis)

    if not tok_p or not tok_h:
        overlap = 0.0
    else:
        intersection = tok_p & tok_h
        union = tok_p | tok_h
        overlap = len(intersection) / len(union)

    # Very crude mapping: overlap → pseudo-entailment
    entailment = min(overlap * 1.5, 1.0)
    contradiction = max(0.0, 0.3 - overlap)
    neutral = max(0.0, 1.0 - entailment - contradiction)

    return {
        "contradiction": round(contradiction, 4),
        "entailment":    round(entailment, 4),
        "neutral":       round(neutral, 4),
    }


def score_pair(premise: str, hypothesis: str) -> dict:
    """
    Score a (premise, hypothesis) pair. Selects model or fallback.

    Returns dict:
        entailment, neutral, contradiction, model_used
    """
    _load_model()

    if _model is not None:
        raw = _nli_score_model(premise, hypothesis)
        model_used = "nli-MiniLM2"
    else:
        raw = _nli_score_fallback(premise, hypothesis)
        model_used = "keyword_fallback"

    raw["model_used"] = model_used
    return raw


# ---------------------------------------------------------------------------
# DC-NLI: Directional scoring (paper Equations 1–3)
# ---------------------------------------------------------------------------

def directional_nli(claim: str, evidence: str) -> dict:
    """
    Compute directional NLI scores for a (claim, evidence) pair.

    Implements paper Equations (1)–(3):
        forward = entailment(e, c)   — does evidence support the claim?
        reverse = entailment(c, e)   — does the claim entail the evidence?
        gap     = 1 − reverse        — measures generalizability stretch

    Parameters
    ----------
    claim    : str — the abstract claim sentence
    evidence : str — the evidence sentence from the body

    Returns
    -------
    dict:
        'forward'       : float  — Eq. (1): entailment(e→c)
        'reverse'       : float  — Eq. (2): entailment(c→e)
        'gap'           : float  — Eq. (3): 1 − reverse
        'fwd_contradiction': float
        'fwd_neutral'   : float
        'model_used'    : str
    """
    # Eq. (1): forward = entailment(evidence → claim)
    fwd = score_pair(premise=evidence, hypothesis=claim)

    # Eq. (2): reverse = entailment(claim → evidence)
    rev = score_pair(premise=claim, hypothesis=evidence)

    forward = fwd["entailment"]
    reverse = rev["entailment"]
    gap = 1.0 - reverse  # Eq. (3)

    return {
        "forward":          round(forward, 4),
        "reverse":          round(reverse, 4),
        "gap":              round(gap, 4),
        "fwd_contradiction":round(fwd["contradiction"], 4),
        "fwd_neutral":      round(fwd["neutral"], 4),
        "model_used":       fwd["model_used"],
    }


# ---------------------------------------------------------------------------
# Batch scoring: all claims × evidence sentences in a zone
# ---------------------------------------------------------------------------

_INFORMATIVE_STOPWORDS = {
    "the", "and", "was", "were", "with", "that", "from", "for", "between",
    "this", "these", "those", "have", "had", "has", "been", "are", "which",
    "than", "then", "into", "also", "both", "such", "there", "their", "they",
    "each", "when", "where", "while", "after", "before", "during", "through",
}


def _token_overlap_score(claim: str, sentence: str) -> float:
    """Compute informative lexical and numerical overlap between claim and evidence."""
    c_words = set(re.findall(r"\b[a-zA-Z]{3,}\b", claim.lower())) - _INFORMATIVE_STOPWORDS
    s_words = set(re.findall(r"\b[a-zA-Z]{3,}\b", sentence.lower())) - _INFORMATIVE_STOPWORDS
    word_overlap = len(c_words & s_words)

    c_nums = set(re.findall(r"\b\d+(?:\.\d+)?\b", claim))
    s_nums = set(re.findall(r"\b\d+(?:\.\d+)?\b", sentence))
    num_overlap = len(c_nums & s_nums)

    return float(word_overlap + (2.0 * num_overlap))


def score_claim_against_zone(
    claim: str,
    zone_sentences: list[str | tuple[str, str]],
    top_k: int = 5,
    max_candidates: int = 35,
    prioritize_contradiction: bool = False,
) -> list[dict]:
    """
    Score a claim against sentences in a zone, return top-k by forward score.

    Implements candidate pre-filtering and batched CrossEncoder inference
    for fast and scalable processing on CPU.

    Parameters
    ----------
    claim          : str — the abstract claim
    zone_sentences : list[str | tuple[str, str]] — sentences (or (text, section) tuples)
    top_k          : int — number of top evidence sentences to return
    max_candidates : int — maximum sentences to score with neural model
    prioritize_contradiction : bool — if True and forward is low with high contradiction,
                               prioritize conflicting evidence

    Returns
    -------
    list[dict] sorted by forward score descending, each with:
        'evidence', 'source_section', 'forward', 'reverse', 'gap',
        'fwd_contradiction', 'fwd_neutral', 'model_used', 'rank'
    """
    if not zone_sentences:
        return []

    valid_candidates: list[tuple[str, str]] = []
    for item in zone_sentences:
        if isinstance(item, tuple) and len(item) == 2:
            s_text, s_sec = str(item[0]).strip(), str(item[1]).strip()
        elif isinstance(item, dict) and "text" in item:
            s_text, s_sec = str(item["text"]).strip(), str(item.get("source_section", "Evidence"))
        else:
            s_text, s_sec = str(item).strip(), "Evidence"
        if len(s_text) >= 10:
            valid_candidates.append((s_text, s_sec))

    if not valid_candidates:
        return []

    # If zone has many sentences, prioritize by informative lexical & numerical overlap
    if len(valid_candidates) > max_candidates:
        scored_pre = [(_token_overlap_score(claim, s_text), s_text, s_sec) for s_text, s_sec in valid_candidates]
        scored_pre.sort(key=lambda x: x[0], reverse=True)
        candidates = [(s_text, s_sec) for _, s_text, s_sec in scored_pre[:max_candidates]]
    else:
        candidates = valid_candidates

    _load_model()
    results = []

    if _model is not None:
        # Batched neural NLI inference (vectorized tensor evaluation on CPU)
        cand_texts = [c[0] for c in candidates]
        fwd_pairs = [(sent, claim) for sent in cand_texts]
        rev_pairs = [(claim, sent) for sent in cand_texts]
        all_pairs = fwd_pairs + rev_pairs

        all_preds = _model.predict(all_pairs, apply_softmax=True, batch_size=32)

        n = len(candidates)
        fwd_preds = all_preds[:n]
        rev_preds = all_preds[n:]

        for i, (sent, sec) in enumerate(candidates):
            fwd_s = fwd_preds[i]
            rev_s = rev_preds[i]
            forward = float(fwd_s[1])
            reverse = float(rev_s[1])
            gap = 1.0 - reverse
            results.append({
                "evidence": sent,
                "source_section": sec,
                "forward": round(forward, 4),
                "reverse": round(reverse, 4),
                "gap": round(gap, 4),
                "fwd_contradiction": round(float(fwd_s[0]), 4),
                "fwd_neutral": round(float(fwd_s[2]), 4),
                "model_used": "nli-MiniLM2",
            })
    else:
        # Fallback when neural model is unavailable
        for sent, sec in candidates:
            scores = directional_nli(claim=claim, evidence=sent)
            scores["evidence"] = sent
            scores["source_section"] = sec
            results.append(scores)

    if prioritize_contradiction:
        has_strong_conflict = any(x.get("fwd_contradiction", 0.0) >= 0.35 for x in results)
        max_fwd = max(x["forward"] for x in results) if results else 0.0
        if has_strong_conflict and max_fwd < 0.50:
            results.sort(key=lambda x: (x.get("fwd_contradiction", 0.0), x["forward"]), reverse=True)
        else:
            results.sort(key=lambda x: (x["forward"], -x.get("fwd_contradiction", 0.0)), reverse=True)
    else:
        # Sort by forward entailment descending
        results.sort(key=lambda x: x["forward"], reverse=True)

    # Add rank
    for i, r in enumerate(results):
        r["rank"] = i + 1

    return results[:top_k]

