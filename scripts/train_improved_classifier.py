"""
scripts/train_improved_classifier.py
------------------------------------
Train an improved paper-grouped TF-IDF + Logistic Regression multi-label classifier
for SPIN-SENTRY claim type classification.

Enhancements:
  1. Sublinear TF scaling (sublinear_tf=True) and unicode accent stripping.
  2. N-gram range (1, 2) with min_df=2 and max_features=60000.
  3. Class-weighted Logistic Regression with tuned regularisation C=1.5.
  4. 5-Fold GroupKFold cross-validation on the training set (grouped by paper).
  5. Per-category optimal decision thresholds learned strictly from out-of-fold
     validation probabilities (no test set leakage).
  6. Honest evaluation reporting distinguishing AI-batch heuristic agreement
     from human ground truth.

Usage:
    python scripts/train_improved_classifier.py
    python scripts/train_improved_classifier.py --validate-only
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.claim_dataset import LABELS

LABEL_SOURCE = "ai_batch_annotations"
EVALUATION_TARGET = "agreement_with_ai_generated_batch_labels"
EVALUATION_NOTE = (
    "Evaluated on 5 holdout test papers (PMC12381666, PMC12446902, PMC12662812, "
    "PMC12743696, PMC12815918) comprising 552 examples, of which 549 (99.5%) are "
    "AI-batch heuristic annotations and 3 (0.5%) are genuine human annotations. "
    "Metrics reflect model agreement with AI heuristics, NOT human ground truth."
)


def load_dataset(path: Path):
    """Load annotations CSV and extract texts, labels, and paper groups."""
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = [row for row in csv.DictReader(handle) if (row.get("sentence") or "").strip()]
    texts = [row["sentence"].strip() for row in rows]
    targets = [
        sorted({l.strip() for l in row["labels"].split(";") if l.strip() in LABELS}, key=LABELS.index)
        for row in rows
    ]
    groups = [row["paper_id"].strip() for row in rows]
    return rows, texts, targets, groups


def split_grouped_indices(targets, groups, test_size: float = 0.20, seed: int = 42):
    """Split indices by paper group ensuring all 6 labels appear in both splits."""
    from sklearn.model_selection import GroupShuffleSplit
    splitter = GroupShuffleSplit(n_splits=100, test_size=test_size, random_state=seed)
    for train_idx, test_idx in splitter.split(range(len(targets)), groups=groups):
        train_labels = {lbl for i in train_idx for lbl in targets[i]}
        test_labels = {lbl for i in test_idx for lbl in targets[i]}
        if set(LABELS).issubset(train_labels) and set(LABELS).issubset(test_labels):
            return train_idx, test_idx
    raise ValueError("Could not find a paper-grouped split with all 6 labels in train and test.")


def tune_thresholds_oof(train_texts, train_y, train_groups, tfidf_config, lr_config, seed: int = 42):
    """
    Run 5-fold GroupKFold cross-validation on the training set to compute out-of-fold
    probability predictions, and find per-label thresholds that maximize binary F1.
    """
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import precision_recall_fscore_support
    from sklearn.model_selection import GroupKFold
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.pipeline import Pipeline

    gkf = GroupKFold(n_splits=5)
    oof_probs = np.zeros(train_y.shape)
    fold_micro_scores = []
    fold_macro_scores = []

    print("\nRunning 5-fold GroupKFold cross-validation on training set...")
    for fold, (fold_train, fold_val) in enumerate(gkf.split(train_texts, train_y, train_groups), 1):
        fold_pipe = Pipeline([
            ("tfidf", TfidfVectorizer(**tfidf_config)),
            ("classifier", OneVsRestClassifier(LogisticRegression(**lr_config))),
        ])
        fold_pipe.fit([train_texts[i] for i in fold_train], train_y[fold_train])

        X_val = fold_pipe.named_steps["tfidf"].transform([train_texts[i] for i in fold_val])
        fold_probs = np.column_stack([
            est.predict_proba(X_val)[:, 1] for est in fold_pipe.named_steps["classifier"].estimators_
        ])
        oof_probs[fold_val] = fold_probs

        pred_50 = (fold_probs >= 0.50).astype(int)
        _, _, micro_f1, _ = precision_recall_fscore_support(train_y[fold_val], pred_50, average="micro", zero_division=0)
        _, _, macro_f1, _ = precision_recall_fscore_support(train_y[fold_val], pred_50, average="macro", zero_division=0)
        fold_micro_scores.append(float(micro_f1))
        fold_macro_scores.append(float(macro_f1))
        print(f"  Fold {fold}: Micro-F1={micro_f1:.4f}, Macro-F1={macro_f1:.4f}")

    cv_summary = {
        "cv_folds": 5,
        "cv_micro_f1_mean": float(np.mean(fold_micro_scores)),
        "cv_micro_f1_std": float(np.std(fold_micro_scores)),
        "cv_macro_f1_mean": float(np.mean(fold_macro_scores)),
        "cv_macro_f1_std": float(np.std(fold_macro_scores)),
    }
    print(f"  Mean CV Micro-F1: {cv_summary['cv_micro_f1_mean']:.4f} +/- {cv_summary['cv_micro_f1_std']:.4f}")
    print(f"  Mean CV Macro-F1: {cv_summary['cv_macro_f1_mean']:.4f} +/- {cv_summary['cv_macro_f1_std']:.4f}")

    print("\nSelecting per-label optimal thresholds on out-of-fold validation probabilities...")
    best_thresholds = {}
    oof_per_label = {}
    candidate_thresholds = np.arange(0.20, 0.75, 0.05)

    for j, label in enumerate(LABELS):
        best_t = 0.50
        best_f1 = 0.0
        best_p = 0.0
        best_r = 0.0
        for t in candidate_thresholds:
            col_pred = (oof_probs[:, j] >= t).astype(int)
            p, r, f1, _ = precision_recall_fscore_support(train_y[:, j], col_pred, average="binary", zero_division=0)
            if f1 > best_f1:
                best_f1 = float(f1)
                best_p = float(p)
                best_r = float(r)
                best_t = round(float(t), 2)
        best_thresholds[label] = best_t
        oof_per_label[label] = {
            "optimal_threshold": best_t,
            "oof_f1": round(best_f1, 4),
            "oof_precision": round(best_p, 4),
            "oof_recall": round(best_r, 4),
            "support": int(train_y[:, j].sum()),
        }
        print(f"  {label:<20}: threshold={best_t:.2f} (OOF F1={best_f1:.4f}, P={best_p:.4f}, R={best_r:.4f})")

    return best_thresholds, oof_per_label, cv_summary


def compute_metrics(truth, predicted, labels, matrices):
    """Compute exact match, micro, macro, and per-label metrics."""
    import numpy as np
    from sklearn.metrics import f1_score, hamming_loss, precision_recall_fscore_support

    exact_match = float((truth == predicted).all(axis=1).mean())
    hamming = float(hamming_loss(truth, predicted))
    p_micro, r_micro, f1_micro, _ = precision_recall_fscore_support(truth, predicted, average="micro", zero_division=0)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(truth, predicted, average="macro", zero_division=0)

    per_p, per_r, per_f1, per_sup = precision_recall_fscore_support(truth, predicted, average=None, zero_division=0)
    per_label = {}
    for i, lbl in enumerate(labels):
        per_label[lbl] = {
            "precision": float(round(per_p[i], 4)),
            "recall": float(round(per_r[i], 4)),
            "f1": float(round(per_f1[i], 4)),
            "support": int(per_sup[i]),
            "confusion_matrix": matrices[i].tolist(),
        }

    return {
        "exact_match_agreement": float(round(exact_match, 4)),
        "hamming_loss": float(round(hamming, 4)),
        "precision_micro": float(round(p_micro, 4)),
        "recall_micro": float(round(r_micro, 4)),
        "f1_micro": float(round(f1_micro, 4)),
        "f1_macro": float(round(f1_macro, 4)),
        "per_label": per_label,
    }


def write_markdown_report(path: Path, metrics: dict, thresholds: dict) -> None:
    """Write human-readable report detailing model performance and methodology."""
    lines = [
        "# Improved Classifier Evaluation Report",
        "",
        f"**Model:** `{metrics.get('model_name', 'claim_type_model_improved')}`",
        f"**Evaluation target:** `{metrics['evaluation_target']}`",
        f"**Trained at:** {metrics['trained_at_utc']}",
        "",
        "> [!IMPORTANT]",
        f"> {metrics['evaluation_note']}",
        "",
        "## Overall Metrics (Holdout Test Set)",
        "",
        f"- **Dataset Rows:** {metrics['dataset_rows']} ({metrics['train_rows']} train / {metrics['test_rows']} test)",
        f"- **Papers:** {metrics['paper_count']} total ({metrics['train_papers']} train / {metrics['test_papers']} test)",
        f"- **Exact-Match Agreement:** {metrics['exact_match_agreement']:.4f}",
        f"- **Micro F1:** {metrics['f1_micro']:.4f}",
        f"- **Macro F1:** {metrics['f1_macro']:.4f}",
        f"- **Precision (Micro):** {metrics['precision_micro']:.4f}",
        f"- **Recall (Micro):** {metrics['recall_micro']:.4f}",
        f"- **Hamming Loss:** {metrics['hamming_loss']:.4f}",
        "",
        "## 5-Fold GroupKFold Cross-Validation (Training Set Only)",
        "",
        f"- **CV Micro F1:** {metrics['cv']['cv_micro_f1_mean']:.4f} +/- {metrics['cv']['cv_micro_f1_std']:.4f}",
        f"- **CV Macro F1:** {metrics['cv']['cv_macro_f1_mean']:.4f} +/- {metrics['cv']['cv_macro_f1_std']:.4f}",
        "",
        "## Per-Category Results (Holdout Test with Tuned Thresholds)",
        "",
        "| Category | Optimal Threshold | Precision | Recall | F1 | Test Support | TN | FP | FN | TP |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for lbl in LABELS:
        item = metrics["per_label"][lbl]
        t = thresholds.get(lbl, 0.50)
        tn, fp = item["confusion_matrix"][0]
        fn, tp = item["confusion_matrix"][1]
        lines.append(
            f"| {lbl} | {t:.2f} | {item['precision']:.4f} | {item['recall']:.4f} | {item['f1']:.4f} | "
            f"{item['support']} | {tn} | {fp} | {fn} | {tp} |"
        )

    lines += [
        "",
        "## Reproduction Details",
        "",
        f"- **Split:** {metrics['split']}; random seed: `{metrics['seed']}`",
        f"- **Training Papers ({len(metrics['train_paper_ids'])}):** {', '.join(metrics['train_paper_ids'])}",
        f"- **Test Papers ({len(metrics['test_paper_ids'])}):** {', '.join(metrics['test_paper_ids'])}",
        f"- **Dataset SHA-256:** `{metrics['dataset_sha256']}`",
        f"- **Python Version:** {metrics['python_version']}",
        f"- **Scikit-Learn Version:** {metrics['scikit_learn_version']}",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "claim_annotations_reviewed.csv")
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-out", type=Path, default=ROOT / "models" / "claim_type_model_improved.joblib")
    parser.add_argument("--vectorizer-out", type=Path, default=ROOT / "models" / "claim_type_vectorizer_improved.joblib")
    parser.add_argument("--thresholds-out", type=Path, default=ROOT / "models" / "claim_type_thresholds_improved.json")
    parser.add_argument("--label-map-out", type=Path, default=ROOT / "models" / "claim_type_label_map_improved.json")
    parser.add_argument("--config-out", type=Path, default=ROOT / "models" / "claim_type_training_config_improved.json")
    parser.add_argument("--metrics-out", type=Path, default=ROOT / "reports" / "claim_type_metrics_improved.json")
    parser.add_argument("--report-out", type=Path, default=ROOT / "reports" / "claim_type_report_improved.md")
    parser.add_argument("--validate-only", action="store_true", help="Validate dataset and exit")
    args = parser.parse_args()

    if not args.data.exists():
        print(f"Dataset not found at {args.data}")
        return 1

    rows, texts, targets, groups = load_dataset(args.data)
    print(f"Loaded {len(rows)} rows across {len(set(groups))} papers from {args.data}")

    label_counts = {lbl: sum(lbl in t for t in targets) for lbl in LABELS}
    print("Label distribution in dataset:")
    for lbl, cnt in label_counts.items():
        print(f"  {lbl:<20} {cnt:>5}")

    if args.validate_only:
        print(json.dumps({"rows": len(rows), "papers": len(set(groups)), "label_counts": label_counts}, indent=2))
        return 0

    import joblib
    import numpy as np
    import sklearn
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import multilabel_confusion_matrix
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import MultiLabelBinarizer

    # Paper-grouped split (GroupShuffleSplit with seed 42)
    train_idx, test_idx = split_grouped_indices(targets, groups, test_size=args.test_size, seed=args.seed)
    train_papers = sorted({groups[i] for i in train_idx})
    test_papers = sorted({groups[i] for i in test_idx})

    overlap = set(train_papers) & set(test_papers)
    if overlap:
        raise RuntimeError(f"Paper leakage detected: {overlap}")

    print(f"\nPaper-grouped split:")
    print(f"  Training set: {len(train_idx)} rows across {len(train_papers)} papers")
    print(f"  Holdout test: {len(test_idx)} rows across {len(test_papers)} papers ({', '.join(test_papers)})")

    mlb = MultiLabelBinarizer(classes=LABELS)
    y = mlb.fit_transform(targets)
    train_y = y[train_idx]
    test_y = y[test_idx]

    train_texts = [texts[i] for i in train_idx]
    test_texts = [texts[i] for i in test_idx]
    train_groups = [groups[i] for i in train_idx]

    # Model configuration
    tfidf_config = {
        "ngram_range": (1, 2),
        "min_df": 2,
        "max_features": 60000,
        "sublinear_tf": True,
        "strip_accents": "unicode",
    }
    lr_config = {
        "C": 1.5,
        "max_iter": 2000,
        "class_weight": "balanced",
        "random_state": args.seed,
    }

    # Cross-validation and threshold tuning on training set only
    best_thresholds, oof_per_label, cv_summary = tune_thresholds_oof(
        train_texts, train_y, train_groups, tfidf_config, lr_config, seed=args.seed
    )

    # Train final model on full training set
    print("\nTraining final model on full training partition...")
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(**tfidf_config)),
        ("classifier", OneVsRestClassifier(LogisticRegression(**lr_config))),
    ])
    pipeline.fit(train_texts, train_y)

    # Evaluate on held-out test partition
    print("\nEvaluating on held-out test partition...")
    X_test = pipeline.named_steps["tfidf"].transform(test_texts)
    test_probs = np.column_stack([
        est.predict_proba(X_test)[:, 1] for est in pipeline.named_steps["classifier"].estimators_
    ])

    # Apply per-label tuned thresholds
    test_pred_thresh = np.column_stack([
        (test_probs[:, j] >= best_thresholds[lbl]).astype(int)
        for j, lbl in enumerate(LABELS)
    ])

    matrices = multilabel_confusion_matrix(test_y, test_pred_thresh)
    metrics = compute_metrics(test_y, test_pred_thresh, LABELS, matrices)

    metrics.update({
        "model_name": "claim_type_model_improved",
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "label_source": LABEL_SOURCE,
        "evaluation_target": EVALUATION_TARGET,
        "evaluation_note": EVALUATION_NOTE,
        "dataset_path": str(args.data.resolve()),
        "dataset_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
        "dataset_rows": len(rows),
        "train_rows": len(train_idx),
        "test_rows": len(test_idx),
        "test_size": args.test_size,
        "seed": args.seed,
        "paper_count": len(set(groups)),
        "split": "paper_grouped_GroupShuffleSplit",
        "train_papers": len(train_papers),
        "test_papers": len(test_papers),
        "train_paper_ids": train_papers,
        "test_paper_ids": test_papers,
        "labels": LABELS,
        "label_counts": label_counts,
        "cv": cv_summary,
        "oof_validation": oof_per_label,
        "thresholds_applied": best_thresholds,
        "python_version": sys.version,
        "scikit_learn_version": sklearn.__version__,
    })

    # Save artifacts
    print("\nSaving artifacts...")
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, args.model_out)
    print(f"  Saved model:      {args.model_out}")

    args.vectorizer_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline.named_steps["tfidf"], args.vectorizer_out)
    print(f"  Saved vectorizer: {args.vectorizer_out}")

    args.thresholds_out.parent.mkdir(parents=True, exist_ok=True)
    with args.thresholds_out.open("w", encoding="utf-8") as f:
        json.dump(best_thresholds, f, indent=2)
    print(f"  Saved thresholds: {args.thresholds_out}")

    label_map = {"classes": LABELS, "index_to_label": {str(i): lbl for i, lbl in enumerate(LABELS)}}
    args.label_map_out.parent.mkdir(parents=True, exist_ok=True)
    with args.label_map_out.open("w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=2)
    print(f"  Saved label map:  {args.label_map_out}")

    config = {
        "algorithm": "TF-IDF (sublinear, min_df=2, max_feat=60000) + OneVsRest(LogisticRegression(C=1.5, balanced))",
        "label_source": LABEL_SOURCE,
        "evaluation_target": EVALUATION_TARGET,
        "evaluation_note": EVALUATION_NOTE,
        "test_size": args.test_size,
        "seed": args.seed,
        "split": "GroupShuffleSplit by paper_id",
        "vectorizer": tfidf_config,
        "classifier": lr_config,
        "tuned_thresholds": best_thresholds,
        "labels_in_order": LABELS,
        "dataset_sha256": metrics["dataset_sha256"],
        "model_includes_vectorizer": True,
    }
    args.config_out.parent.mkdir(parents=True, exist_ok=True)
    with args.config_out.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(f"  Saved config:     {args.config_out}")

    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    with args.metrics_out.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Saved metrics:    {args.metrics_out}")

    write_markdown_report(args.report_out, metrics, best_thresholds)
    print(f"  Saved report:     {args.report_out}")

    print("\n" + "=" * 60)
    print("IMPROVED MODEL EVALUATION SUMMARY")
    print("=" * 60)
    print(f"  Exact Match:  {metrics['exact_match_agreement']:.4f}")
    print(f"  Micro F1:     {metrics['f1_micro']:.4f}")
    print(f"  Macro F1:     {metrics['f1_macro']:.4f}")
    print(f"  Hamming Loss: {metrics['hamming_loss']:.4f}")
    print("\nPer-Label Results on Test Set:")
    for lbl in LABELS:
        item = metrics["per_label"][lbl]
        t = best_thresholds[lbl]
        print(f"  {lbl:<20} (thr={t:.2f}): F1={item['f1']:.4f}  P={item['precision']:.4f}  R={item['recall']:.4f}  support={item['support']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
