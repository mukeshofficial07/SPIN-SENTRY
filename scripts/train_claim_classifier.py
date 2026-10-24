"""Train/evaluate a paper-grouped TF-IDF + logistic-regression multi-label classifier."""

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
from scripts.validate_claim_annotations import validate

PSEUDO_LABEL_SOURCE = "machine_generated_pseudo_label"
HUMAN_LABEL_SOURCE = "human_verified"


def load_rows(path: Path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = [row for row in csv.DictReader(handle) if (row.get("sentence") or "").strip()]
    texts = [row["sentence"].strip() for row in rows]
    targets = [sorted({label.strip() for label in row["labels"].split(";") if label.strip()}, key=LABELS.index) for row in rows]
    groups = [row["paper_id"].strip() for row in rows]
    return rows, texts, targets, groups


def validate_pseudo_dataset(path: Path) -> tuple[list[str], int, dict[str, int]]:
    errors: list[str] = []
    counts = {label: 0 for label in LABELS}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        required = {"candidate_id", "paper_id", "sentence", "labels", "source", "label_source", "machine_claim_class", "machine_uncertainty_flag"}
        missing = required - fields
        if missing:
            return ["Missing pseudo-label fields: " + ", ".join(sorted(missing))], 0, counts
        rows = list(reader)
    seen = set()
    for line, row in enumerate(rows, 2):
        if row.get("label_source") != PSEUDO_LABEL_SOURCE:
            errors.append(f"Row {line}: label_source must be {PSEUDO_LABEL_SOURCE!r}.")
        if not row.get("candidate_id", "").strip() or not row.get("paper_id", "").strip() or not row.get("sentence", "").strip() or not row.get("source", "").strip():
            errors.append(f"Row {line}: candidate_id, paper_id, sentence, and source are required.")
        if row.get("machine_claim_class") != "claim" or row.get("machine_uncertainty_flag") != "no":
            errors.append(f"Row {line}: only unambiguous machine claim rows may enter pseudo-label training.")
        if (row.get("extraction_notes") or "").strip() or "possible_extraction_damage" in (row.get("machine_uncertainty_reasons") or "").split(";"):
            errors.append(f"Row {line}: extraction-damaged rows must be excluded.")
        if (row.get("annotation_status") or "").strip() in {"approved", "corrected"} and (row.get("verified_labels") or "").strip():
            errors.append(f"Row {line}: do not mix human-verified fields into the pseudo-label dataset.")
        labels = [x.strip() for x in (row.get("labels") or "").split(";") if x.strip()]
        invalid = sorted(set(labels) - set(LABELS))
        if not labels:
            errors.append(f"Row {line}: at least one machine label is required.")
        if invalid:
            errors.append(f"Row {line}: unknown label(s): {', '.join(invalid)}")
        if len(labels) != len(set(labels)):
            errors.append(f"Row {line}: duplicate labels are not allowed.")
        key = (row.get("paper_id", "").strip(), row.get("sentence", "").strip().casefold())
        if key in seen:
            errors.append(f"Row {line}: duplicate sentence within the same paper.")
        seen.add(key)
        for label in set(labels) & set(LABELS):
            counts[label] += 1
    return errors, len(rows), counts


def split_indices(targets, groups, test_size, seed):
    from sklearn.model_selection import GroupShuffleSplit
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")
    if not all(groups):
        raise ValueError("paper_id is required in every row; fill it in to prevent paper-level leakage.")
    if len(set(groups)) <= 1:
        raise ValueError("At least two distinct paper_id values are required for leakage-safe evaluation.")
    splitter = GroupShuffleSplit(n_splits=100, test_size=test_size, random_state=seed)
    for train_idx, test_idx in splitter.split(range(len(targets)), groups=groups):
        train_labels = {label for i in train_idx for label in targets[i]}
        test_labels = {label for i in test_idx for label in targets[i]}
        if set(LABELS).issubset(train_labels) and set(LABELS).issubset(test_labels):
            return train_idx, test_idx, "paper_grouped", len(set(groups[i] for i in train_idx)), len(set(groups[i] for i in test_idx))
    raise ValueError("Could not create a paper-grouped split containing all six labels in both partitions. Add examples across more papers.")


def _backup_if_exists(path: Path) -> str | None:
    if not path.exists():
        return None
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    backup = path.with_name(f"{path.stem}.backup-{timestamp}{path.suffix}")
    shutil.copy2(path, backup)
    return str(backup.resolve())


def _write_json(path: Path, payload: dict) -> str | None:
    path.parent.mkdir(parents=True, exist_ok=True)
    backup = _backup_if_exists(path)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return backup


def _write_report(path: Path, metrics: dict) -> str | None:
    path.parent.mkdir(parents=True, exist_ok=True)
    backup = _backup_if_exists(path)
    lines = [
        "# Classifier evaluation report", "",
        f"**Evaluation target:** {metrics['evaluation_target']}.",
        "These figures measure agreement with the labels in the stated dataset; machine pseudo-label agreement is not human-label accuracy.", "",
        f"- Dataset examples: {metrics['dataset_rows']} ({metrics['train_rows']} train / {metrics['test_rows']} test)",
        f"- Paper groups: {metrics['paper_count']} total ({metrics['train_papers']} train / {metrics['test_papers']} test)",
        f"- Multi-label examples: {metrics['multi_label_rows']}",
        f"- Exact-match agreement: {metrics['exact_match_agreement']:.4f}",
        f"- Micro F1: {metrics['f1_micro']:.4f}; macro F1: {metrics['f1_macro']:.4f}; Hamming loss: {metrics['hamming_loss']:.4f}",
        "", "## Per-class results", "",
        "| Category | Precision | Recall | F1 | Test support | TN | FP | FN | TP |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label in LABELS:
        item = metrics["per_label"][label]
        tn, fp = item["confusion_matrix"][0]
        fn, tp = item["confusion_matrix"][1]
        lines.append(f"| {label} | {item['precision']:.4f} | {item['recall']:.4f} | {item['f1']:.4f} | {item['support']} | {tn} | {fp} | {fn} | {tp} |")
    lines += ["", "## Reproduction details", "", f"- Split: `{metrics['split']}`; random seed: `{metrics['seed']}`.", f"- Training paper IDs: {', '.join(metrics['train_paper_ids'])}", f"- Test paper IDs: {', '.join(metrics['test_paper_ids'])}", f"- Dataset SHA-256: `{metrics['dataset_sha256']}`", ""]
    path.write_text("\n".join(lines), encoding="utf-8")
    return backup


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "claim_annotations_template.csv")
    parser.add_argument("--label-source", choices=["auto", "human_verified", "machine_pseudo_label"], default="auto")
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-out", type=Path)
    parser.add_argument("--vectorizer-out", type=Path)
    parser.add_argument("--label-map-out", type=Path)
    parser.add_argument("--config-out", type=Path)
    parser.add_argument("--metrics-out", type=Path)
    parser.add_argument("--report-out", type=Path)
    parser.add_argument("--validate-only", action="store_true", help="Validate and report dataset counts without fitting a model.")
    args = parser.parse_args()

    with args.data.open(newline="", encoding="utf-8-sig") as handle:
        headers = set(csv.DictReader(handle).fieldnames or [])
    inferred_source = PSEUDO_LABEL_SOURCE if "label_source" in headers else HUMAN_LABEL_SOURCE
    if args.label_source == "machine_pseudo_label" and inferred_source != PSEUDO_LABEL_SOURCE:
        print("Training stopped: dataset does not identify machine-generated pseudo-labels.")
        return 2
    if args.label_source == "human_verified" and inferred_source != HUMAN_LABEL_SOURCE:
        print("Training stopped: pseudo-labeled data cannot be represented as human-verified.")
        return 2
    label_source = inferred_source

    if label_source == PSEUDO_LABEL_SOURCE:
        errors, row_count, counts = validate_pseudo_dataset(args.data)
    else:
        errors, row_count, counts = validate(args.data)
    if errors:
        print("Training stopped; annotations did not pass validation:")
        for error in errors:
            print(f"- {error}")
        print("No model or metrics were written.")
        return 2
    scarce = [label for label in LABELS if counts[label] < 2]
    if scarce:
        print("Training stopped; need at least two examples per category before evaluation: " + ", ".join(scarce))
        print("No model or metrics were written.")
        return 2

    if args.validate_only:
        with args.data.open(newline="", encoding="utf-8-sig") as handle:
            data_rows = list(csv.DictReader(handle))
        papers = {row.get("paper_id", "").strip() for row in data_rows if row.get("paper_id", "").strip()}
        print(json.dumps({"validated_rows": row_count, "papers": len(papers), "label_source": label_source, "category_counts": dict(counts)}, indent=2))
        return 0

    try:
        import joblib
        import sklearn
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import f1_score, hamming_loss, multilabel_confusion_matrix, precision_recall_fscore_support
        from sklearn.multiclass import OneVsRestClassifier
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import MultiLabelBinarizer
    except ImportError as exc:
        print(f"Training stopped: a compatible scikit-learn/joblib runtime is unavailable ({exc}).")
        print("No model or metrics were written. Run this script with the project's Python 3.13 virtual environment.")
        return 3

    rows, texts, targets, groups = load_rows(args.data)
    try:
        train_idx, test_idx, split_type, train_papers, test_papers = split_indices(targets, groups, args.test_size, args.seed)
    except ValueError as exc:
        print(f"Training stopped: {exc}\nNo model or metrics were written.")
        return 2
    binarizer = MultiLabelBinarizer(classes=LABELS)
    y = binarizer.fit_transform(targets)
    vectorizer_config = {"ngram_range": [1, 2], "min_df": 1, "max_features": 50000, "strip_accents": "unicode"}
    logistic_config = {"max_iter": 2000, "class_weight": "balanced", "random_state": args.seed, "threshold": 0.5}
    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=50000, strip_accents="unicode")),
        ("classifier", OneVsRestClassifier(LogisticRegression(max_iter=2000, class_weight="balanced", random_state=args.seed))),
    ])
    model.fit([texts[i] for i in train_idx], y[train_idx])
    predicted = model.predict([texts[i] for i in test_idx])
    truth = y[test_idx]
    per_precision, per_recall, per_f1, per_support = precision_recall_fscore_support(truth, predicted, average=None, zero_division=0)
    matrices = multilabel_confusion_matrix(truth, predicted)
    per_label = {}
    for i, label in enumerate(LABELS):
        per_label[label] = {
            "precision": float(per_precision[i]), "recall": float(per_recall[i]),
            "f1": float(per_f1[i]), "support": int(per_support[i]),
            "confusion_matrix": matrices[i].tolist(),
        }
    train_paper_ids = sorted({groups[i] for i in train_idx})
    test_paper_ids = sorted({groups[i] for i in test_idx})
    if set(train_paper_ids) & set(test_paper_ids):
        raise RuntimeError("Paper leakage detected after grouped split; refusing to write artifacts.")
    evaluation_target = "agreement_with_machine_generated_labels" if label_source == PSEUDO_LABEL_SOURCE else "agreement_with_human_verified_labels"
    metrics = {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "label_source": label_source,
        "evaluation_target": evaluation_target,
        "dataset_path": str(args.data.resolve()),
        "dataset_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
        "dataset_rows": len(rows), "train_rows": len(train_idx), "test_rows": len(test_idx),
        "test_size": args.test_size, "seed": args.seed,
        "paper_count": len(set(groups)), "split": split_type, "train_papers": train_papers, "test_papers": test_papers,
        "train_paper_ids": train_paper_ids, "test_paper_ids": test_paper_ids,
        "multi_label_rows": sum(len(target) > 1 for target in targets),
        "labels": LABELS, "label_counts": dict(counts),
        "exact_match_agreement": float((truth == predicted).all(axis=1).mean()),
        "f1_micro": float(f1_score(truth, predicted, average="micro", zero_division=0)),
        "f1_macro": float(f1_score(truth, predicted, average="macro", zero_division=0)),
        "precision_micro": float(precision_recall_fscore_support(truth, predicted, average="micro", zero_division=0)[0]),
        "recall_micro": float(precision_recall_fscore_support(truth, predicted, average="micro", zero_division=0)[1]),
        "hamming_loss": float(hamming_loss(truth, predicted)), "per_label": per_label,
        "python_version": sys.version, "scikit_learn_version": sklearn.__version__,
    }

    pseudo = label_source == PSEUDO_LABEL_SOURCE
    suffix = "_machine_pseudo" if pseudo else ""
    artifact_paths = {
        "model": args.model_out or ROOT / "models" / f"claim_type_model{suffix}.joblib",
        "vectorizer": args.vectorizer_out or ROOT / "models" / f"claim_type_vectorizer{suffix}.joblib",
        "label_map": args.label_map_out or ROOT / "models" / f"claim_type_label_map{suffix}.json",
        "config": args.config_out or ROOT / "models" / f"claim_type_training_config{suffix}.json",
        "metrics": args.metrics_out or ROOT / "reports" / f"claim_type_metrics{suffix}.json",
        "report": args.report_out or ROOT / "reports" / f"claim_type_report{suffix}.md",
    }
    metrics["artifacts"] = {key: str(path.resolve()) for key, path in artifact_paths.items()}
    config = {
        "algorithm": "TF-IDF + OneVsRest(LogisticRegression)",
        "label_source": label_source, "evaluation_target": evaluation_target,
        "test_size": args.test_size, "seed": args.seed, "split": "GroupShuffleSplit by paper_id",
        "vectorizer": vectorizer_config, "classifier": logistic_config,
        "labels_in_order": LABELS, "dataset_sha256": metrics["dataset_sha256"],
        "model_includes_vectorizer": True,
        "artifacts": metrics["artifacts"],
    }
    label_map = {"classes": LABELS, "index_to_label": {str(i): label for i, label in enumerate(LABELS)}}
    backups = {}
    for key in ["model", "vectorizer"]:
        path = artifact_paths[key]
        path.parent.mkdir(parents=True, exist_ok=True)
        backups[key] = _backup_if_exists(path)
        joblib.dump(model if key == "model" else model.named_steps["tfidf"], path)
    for key, payload in [("label_map", label_map), ("config", config), ("metrics", metrics)]:
        backups[key] = _write_json(artifact_paths[key], payload)
    backups["report"] = _write_report(artifact_paths["report"], metrics)
    print(json.dumps({"metrics": metrics, "artifacts": {k: str(v) for k, v in artifact_paths.items()}, "backups": backups}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
