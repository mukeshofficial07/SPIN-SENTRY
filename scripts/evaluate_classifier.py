"""
scripts/evaluate_classifier.py
--------------------------------
Honest evaluation of any saved SPIN-SENTRY classifier.

Produces a structured report that clearly distinguishes:
  - agreement_with_ai_generated_batch_labels (current)
  - human_gold_standard (requires ≥30 genuine human reviews per category)

Usage:
    python scripts/evaluate_classifier.py                    # evaluate default model
    python scripts/evaluate_classifier.py --model improved   # evaluate improved model
    python scripts/evaluate_classifier.py --gold-only        # evaluate only on human-reviewed rows
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

LABELS = ["Efficacy", "Safety", "Generalizability", "Comparative", "Novelty", "Recommendation"]

MODELS = {
    "ai_batch": ROOT / "models" / "claim_type_model_human_verified.joblib",
    "ai_batch_v1": ROOT / "models" / "claim_type_model_ai_batch_v1.joblib",
    "pseudo": ROOT / "models" / "claim_type_model_machine_pseudo.joblib",
    "improved": ROOT / "models" / "claim_type_model_improved.joblib",
}

CANDIDATES_CSV = ROOT / "data" / "research_dataset" / "candidates.csv"
REVIEWED_CSV   = ROOT / "data" / "claim_annotations_reviewed.csv"
GOLD_CSV       = ROOT / "data" / "research_dataset" / "human_gold_standard.csv"


def load_gold_standard() -> list[dict]:
    """Load only genuinely human-reviewed rows (not AI-batch)."""
    if not GOLD_CSV.exists():
        return []
    rows = []
    with GOLD_CSV.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            lbl = (row.get("verified_labels") or "").strip()
            status = (row.get("annotation_status") or "").strip().lower()
            if lbl and status in ("approved", "confirmed", "corrected"):
                rows.append(row)
    return rows


def get_model_thresholds(model_key: str) -> list[float]:
    """Return thresholds for a specific model key."""
    if model_key == "improved":
        thresh_path = ROOT / "models" / "claim_type_thresholds_improved.json"
        if thresh_path.exists():
            try:
                with thresh_path.open(encoding="utf-8") as f:
                    t_dict = json.load(f)
                    return [float(t_dict.get(lbl, 0.40)) for lbl in LABELS]
            except Exception:
                pass
    return [0.40] * len(LABELS)


def load_ai_batch_test_set(test_paper_ids: list[str]) -> list[dict]:
    """Load the standard holdout test papers from claim_annotations_reviewed.csv."""
    if not REVIEWED_CSV.exists():
        return []
    rows = []
    with REVIEWED_CSV.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("paper_id", "").strip() in test_paper_ids:
                lbl = (row.get("labels") or "").strip()
                if lbl:
                    rows.append(row)
    return rows


def predict_batch(model, sentences: list[str]) -> list[list[float]]:
    """Get per-label probability scores for a batch of sentences."""
    import numpy as np
    try:
        # Pipeline with named steps
        named = model.named_steps
        vectorizer = named["tfidf"]
        classifier = named["classifier"]
        X = vectorizer.transform(sentences)
        scores = np.column_stack([
            est.predict_proba(X)[:, 1] for est in classifier.estimators_
        ])
        return scores.tolist()
    except Exception:
        try:
            raw = model.predict_proba(sentences)
            if hasattr(raw, "tolist"):
                return raw.tolist()
        except Exception:
            pass
    return [[0.5] * len(LABELS)] * len(sentences)


def apply_thresholds(scores: list[list[float]], thresholds: list[float]) -> list[list[int]]:
    return [
        [1 if s >= t else 0 for s, t in zip(row, thresholds)]
        for row in scores
    ]


def compute_metrics(truth: list[list[int]], predicted: list[list[int]]) -> dict:
    """Compute multilabel metrics from binary matrices."""
    import numpy as np
    y_true = np.array(truth)
    y_pred = np.array(predicted)

    # Exact match
    exact_match = float((y_true == y_pred).all(axis=1).mean())

    # Hamming
    hamming = float((y_true != y_pred).mean())

    # Micro
    tp_micro = int((y_true & y_pred).sum())
    fp_micro = int(((1 - y_true) & y_pred).sum())
    fn_micro = int((y_true & (1 - y_pred)).sum())
    p_micro = tp_micro / (tp_micro + fp_micro) if (tp_micro + fp_micro) > 0 else 0.0
    r_micro = tp_micro / (tp_micro + fn_micro) if (tp_micro + fn_micro) > 0 else 0.0
    f1_micro = 2 * p_micro * r_micro / (p_micro + r_micro) if (p_micro + r_micro) > 0 else 0.0

    # Per-label
    per_label = {}
    f1_scores = []
    for i, lbl in enumerate(LABELS):
        col_true = y_true[:, i]
        col_pred = y_pred[:, i]
        tp = int((col_true & col_pred).sum())
        fp = int(((1 - col_true) & col_pred).sum())
        fn = int((col_true & (1 - col_pred)).sum())
        tn = int(((1 - col_true) & (1 - col_pred)).sum())
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        f1_scores.append(f1)
        per_label[lbl] = {
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f1, 4),
            "support": int(col_true.sum()),
            "confusion_matrix": [[tn, fp], [fn, tp]],
        }

    f1_macro = float(sum(f1_scores) / len(f1_scores))

    return {
        "exact_match": round(exact_match, 4),
        "hamming_loss": round(hamming, 4),
        "precision_micro": round(p_micro, 4),
        "recall_micro": round(r_micro, 4),
        "f1_micro": round(f1_micro, 4),
        "f1_macro": round(f1_macro, 4),
        "per_label": per_label,
    }


def rows_to_matrices(rows: list[dict], label_col: str = "labels") -> tuple[list[list[int]], list[str]]:
    """Convert rows with semicolon-separated labels to binary matrix."""
    sentences = []
    y_true = []
    for row in rows:
        sent = (row.get("sentence") or row.get("verified_sentence") or row.get("sentence", "")).strip()
        if not sent:
            continue
        lbl_raw = (row.get(label_col) or row.get("verified_labels") or "").strip()
        lbls = {l.strip() for l in lbl_raw.split(";") if l.strip() in LABELS}
        y_true.append([1 if l in lbls else 0 for l in LABELS])
        sentences.append(sent)
    return y_true, sentences


def evaluate_model(
    model_key: str,
    model_path: Path,
    rows: list[dict],
    evaluation_type: str,
    note: str,
    thresholds: list[float] | None = None,
    output_path: Path | None = None,
):
    import joblib
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_key} ({model_path.name})")
    print(f"Evaluation type: {evaluation_type}")
    print(f"Rows: {len(rows)}")
    print("="*60)

    if not model_path.exists():
        print(f"  MODEL NOT FOUND: {model_path}")
        return None

    y_true, sentences = rows_to_matrices(rows)
    if not sentences:
        print("  No valid rows to evaluate.")
        return None

    model = joblib.load(model_path)
    scores = predict_batch(model, sentences)
    thresh = thresholds or [0.40] * len(LABELS)
    y_pred = apply_thresholds(scores, thresh)

    metrics = compute_metrics(y_true, y_pred)

    result = {
        "model_key": model_key,
        "model_file": model_path.name,
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_type": evaluation_type,
        "evaluation_note": note,
        "n_examples": len(sentences),
        "thresholds_applied": {lbl: t for lbl, t in zip(LABELS, thresh)},
        **metrics,
    }

    # Print summary
    print(f"  Exact Match:  {metrics['exact_match']:.4f}")
    print(f"  Micro F1:     {metrics['f1_micro']:.4f}")
    print(f"  Macro F1:     {metrics['f1_macro']:.4f}")
    print(f"  Hamming Loss: {metrics['hamming_loss']:.4f}")
    print(f"\n  Per-label F1:")
    for lbl, v in metrics["per_label"].items():
        bar = "█" * int(v["f1"] * 20)
        print(f"    {lbl:<22} F1={v['f1']:.3f}  P={v['precision']:.3f}  R={v['recall']:.3f}  support={v['support']}  {bar}")

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"\n  Saved: {output_path}")

    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="ai_batch", choices=list(MODELS.keys()), help="Model key to evaluate")
    parser.add_argument("--gold-only", action="store_true", help="Only evaluate on genuinely human-reviewed rows")
    parser.add_argument("--all", action="store_true", help="Evaluate all available models")
    args = parser.parse_args()

    # Standard test paper IDs (same split as training)
    TEST_PAPER_IDS = ["PMC12381666", "PMC12446902", "PMC12662812", "PMC12743696", "PMC12815918"]

    print("\nSPIN-SENTRY — Honest Classifier Evaluation")
    print("=" * 60)

    # ── Gold standard evaluation ─────────────────────────────────
    gold_rows = load_gold_standard()
    print(f"\nGenuinely human-reviewed examples: {len(gold_rows)}")
    if len(gold_rows) < 30:
        print(f"  ⚠️  Too few human-reviewed examples ({len(gold_rows)}) for reliable gold-standard evaluation.")
        print(f"     Need at least 30 per category (≥180 total). Review more in Annotation Review Tool.")

    # Label distribution in gold rows
    gold_label_counts = Counter()
    for row in gold_rows:
        for lbl in (row.get("verified_labels") or "").split(";"):
            if lbl.strip() in LABELS:
                gold_label_counts[lbl.strip()] += 1
    print(f"\n  Gold standard label distribution:")
    for lbl in LABELS:
        cnt = gold_label_counts[lbl]
        bar = "█" * cnt
        status = "✅" if cnt >= 30 else "⚠️" if cnt >= 10 else "❌"
        print(f"    {lbl:<22} {cnt:>3}  {bar}  {status}")

    if args.gold_only:
        if len(gold_rows) >= 5:
            models_to_eval = [(args.model, MODELS[args.model])] if not args.all else [(k, v) for k, v in MODELS.items()]
            for key, path in models_to_eval:
                t = get_model_thresholds(key)
                evaluate_model(
                    key, path, gold_rows,
                    evaluation_type="human_gold_standard",
                    note=f"Evaluated on {len(gold_rows)} genuinely human-reviewed gold rows across 22 papers.",
                    thresholds=t,
                    output_path=ROOT / "reports" / f"eval_gold_{key}.json"
                )
        else:
            print("\n  Insufficient gold-standard examples for evaluation.")
        return

    # ── AI-batch evaluation on holdout test papers ────────────────
    ai_test_rows = load_ai_batch_test_set(TEST_PAPER_IDS)
    print(f"\nAI-batch holdout test rows: {len(ai_test_rows)} (papers: {', '.join(TEST_PAPER_IDS)})")

    models_to_eval = list(MODELS.items()) if args.all else [(args.model, MODELS[args.model])]

    results = []
    for key, path in models_to_eval:
        if not path.exists():
            print(f"\n  Skipping {key}: {path.name} not found.")
            continue
        t = get_model_thresholds(key)
        result = evaluate_model(
            key, path, ai_test_rows,
            evaluation_type="agreement_with_ai_generated_batch_labels",
            note=(
                "Holdout test: 5 papers (PMC12381666, PMC12446902, PMC12662812, PMC12743696, PMC12815918). "
                "552 total examples; 549 (99.5%) carry AI-batch heuristic labels. "
                "3 rows (0.5%) have genuine human annotations. "
                "Metrics reflect model agreement with AI heuristics, NOT human ground truth."
            ),
            thresholds=t,
            output_path=ROOT / "reports" / f"eval_ai_batch_{key}.json"
        )
        if result:
            results.append(result)

    # ── Comparison summary ───────────────────────────────────────
    if len(results) > 1:
        print(f"\n{'='*60}")
        print("MODEL COMPARISON SUMMARY (AI-batch evaluation)")
        print(f"{'='*60}")
        header = f"{'Model':<28} {'ExactMatch':>12} {'MicroF1':>9} {'MacroF1':>9} {'Hamming':>9}"
        print(header)
        print("-" * len(header))
        for r in results:
            print(f"{r['model_key']:<28} {r['exact_match']:>12.4f} {r['f1_micro']:>9.4f} {r['f1_macro']:>9.4f} {r['hamming_loss']:>9.4f}")

    # ── Gold standard evaluation ─────────────────────────────────
    if len(gold_rows) >= 5:
        print(f"\n{'='*60}")
        print(f"GENUINE HUMAN GOLD STANDARD EVALUATION ({len(gold_rows)} human-reviewed rows)")
        print(f"Independent test set from 22 papers (source_text_checked == 'yes')")
        print(f"{'='*60}")
        for key, path in models_to_eval:
            if path.exists():
                t = get_model_thresholds(key)
                evaluate_model(
                    key, path, gold_rows,
                    evaluation_type="human_gold_standard",
                    note=f"Evaluated on {len(gold_rows)} genuinely human-reviewed rows across 22 papers.",
                    thresholds=t,
                    output_path=ROOT / "reports" / f"eval_gold_{key}.json"
                )

        # Also evaluate specifically on the 5 holdout test papers (document-level unseen papers)
        gold_holdout = [r for r in gold_rows if r.get("paper_id") in TEST_PAPER_IDS]
        if gold_holdout:
            print(f"\n{'='*60}")
            print(f"HOLDOUT TEST PAPERS GOLD EVALUATION ({len(gold_holdout)} rows in 5 test papers)")
            print(f"Unseen papers during training: {', '.join(TEST_PAPER_IDS)}")
            print(f"{'='*60}")
            for key, path in models_to_eval:
                if path.exists():
                    t = get_model_thresholds(key)
                    evaluate_model(
                        f"{key}_holdout_papers", path, gold_holdout,
                        evaluation_type="human_gold_holdout_papers",
                        note=f"Evaluated on {len(gold_holdout)} human-reviewed rows from 5 holdout test papers only.",
                        thresholds=t,
                        output_path=ROOT / "reports" / f"eval_gold_holdout_{key}.json"
                    )


if __name__ == "__main__":
    main()
