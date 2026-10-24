"""Load the optional locally trained multi-label Stage 1 classifier."""

from __future__ import annotations

import json
from pathlib import Path

LABELS = ["Efficacy", "Safety", "Generalizability", "Comparative", "Novelty", "Recommendation"]

_MODELS_DIR = Path(__file__).resolve().parents[1] / "models"

# Hierarchy: improved model > human_verified alias > machine pseudo-label model
MODEL_PATH = (
    _MODELS_DIR / "claim_type_model_improved.joblib"
    if (_MODELS_DIR / "claim_type_model_improved.joblib").exists()
    else (
        _MODELS_DIR / "claim_type_model_human_verified.joblib"
        if (_MODELS_DIR / "claim_type_model_human_verified.joblib").exists()
        else _MODELS_DIR / "claim_type_model_machine_pseudo.joblib"
    )
)

VECTORIZER_PATH = (
    _MODELS_DIR / "claim_type_vectorizer_improved.joblib"
    if (_MODELS_DIR / "claim_type_vectorizer_improved.joblib").exists()
    else (
        _MODELS_DIR / "claim_type_vectorizer_human_verified.joblib"
        if (_MODELS_DIR / "claim_type_vectorizer_human_verified.joblib").exists()
        else _MODELS_DIR / "claim_type_vectorizer_machine_pseudo.joblib"
    )
)

THRESHOLDS_PATH = _MODELS_DIR / "claim_type_thresholds_improved.json"


def get_thresholds() -> dict[str, float]:
    """Return per-label decision thresholds (defaulting to 0.40 if file not present)."""
    if THRESHOLDS_PATH.exists():
        try:
            with THRESHOLDS_PATH.open(encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {l: 0.40 for l in LABELS}


def predict_scores(sentence: str, model_path: Path = MODEL_PATH) -> dict[str, float]:
    """Predict calibrated per-label probabilities for a given sentence."""
    if not model_path.exists():
        raise FileNotFoundError(
            f"Classifier not found at {model_path}. Train with scripts/train_improved_classifier.py."
        )
    import joblib

    model = joblib.load(model_path)
    # Try predict_proba on OvR pipeline (returns list of 2-class probabilities)
    try:
        named_steps = model.named_steps
        vectorizer = named_steps["tfidf"]
        classifier = named_steps["classifier"]
        X = vectorizer.transform([sentence])
        scores = [est.predict_proba(X)[0][1] for est in classifier.estimators_]
        return {label: float(score) for label, score in zip(LABELS, scores)}
    except Exception:
        raw = model.predict_proba([sentence])[0]
        return {label: float(score) for label, score in zip(LABELS, raw)}
