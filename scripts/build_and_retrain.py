"""
Build human-verified training dataset from candidates.csv approved annotations
and retrain the TF-IDF + OvR Logistic Regression classifier.
"""
import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CANDIDATES_CSV = ROOT / "data" / "research_dataset" / "candidates.csv"
TRAINING_CSV   = ROOT / "data" / "claim_annotations_reviewed.csv"
PYTHON         = str(ROOT / "venv" / "Scripts" / "python.exe")

LABELS = ["Efficacy", "Safety", "Generalizability", "Comparative", "Novelty", "Recommendation"]

def build_training_csv():
    print("Building training dataset from approved candidates...")
    rows_out = []
    skipped_no_label = 0
    skipped_bad_labels = 0

    with CANDIDATES_CSV.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("annotation_status", "").strip() != "approved":
                continue

            labels_raw = row.get("verified_labels", "").strip()
            if not labels_raw:
                skipped_no_label += 1
                continue

            labels = [l.strip() for l in labels_raw.split(";") if l.strip() in LABELS]
            if not labels:
                skipped_bad_labels += 1
                continue

            rows_out.append({
                "candidate_id": row.get("candidate_id", "").strip(),
                "paper_id":     row.get("paper_id", "").strip(),
                "sentence":     row.get("sentence", "").strip(),
                "labels":       ";".join(labels),
                "source":       row.get("doi", "").strip() or row.get("url", "").strip() or row.get("source_section", "").strip() or "unknown",
                "annotator":    row.get("annotator", "Mukesh").strip() or "Mukesh",
                "notes":        row.get("review_notes", "").strip() or "",
            })

    TRAINING_CSV.parent.mkdir(parents=True, exist_ok=True)
    with TRAINING_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["candidate_id", "paper_id", "sentence", "labels", "source", "annotator", "notes"])
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"  Written:          {len(rows_out)} training rows")
    print(f"  Skipped (no lbl): {skipped_no_label}")
    print(f"  Skipped (bad lbl):{skipped_bad_labels}")
    print(f"  Output:           {TRAINING_CSV}")

    # Count per label
    label_counts = {l: 0 for l in LABELS}
    for row in rows_out:
        for l in row["labels"].split(";"):
            if l in label_counts:
                label_counts[l] += 1
    print("\n  Label distribution:")
    for lbl, cnt in label_counts.items():
        print(f"    {lbl:<20} {cnt:>5}")

    return len(rows_out), label_counts


def retrain():
    print("\nRunning train_claim_classifier.py ...")
    result = subprocess.run(
        [
            PYTHON, "scripts/train_claim_classifier.py",
            "--data",        str(TRAINING_CSV),
            "--label-source", "human_verified",
            "--test-size",   "0.20",
            "--seed",        "42",
            "--model-out",       "models/claim_type_model_human_verified.joblib",
            "--vectorizer-out",  "models/claim_type_vectorizer_human_verified.joblib",
            "--label-map-out",   "models/claim_type_label_map_human_verified.json",
            "--config-out",      "models/claim_type_training_config_human_verified.json",
            "--metrics-out",     "reports/claim_type_metrics_human_verified.json",
            "--report-out",      "reports/claim_type_report_human_verified.md",
        ],
        cwd=str(ROOT),
        capture_output=False,
        text=True,
    )
    return result.returncode


if __name__ == "__main__":
    print("=" * 70)
    print("SPIN-SENTRY — Build Training Dataset + Retrain Classifier")
    print("=" * 70)

    n_rows, label_counts = build_training_csv()

    if n_rows < 10:
        print(f"\nOnly {n_rows} rows — too few to train. Aborting.")
        sys.exit(1)

    scarce = [l for l, c in label_counts.items() if c < 2]
    if scarce:
        print(f"\nWarning: scarce labels (< 2 examples): {scarce}")
        print("Training may fail. Consider adding more annotations.")

    code = retrain()
    if code == 0:
        print("\n✅ Retraining complete!")
        print("  New model artifacts:")
        print("    models/claim_type_model_human_verified.joblib")
        print("    models/claim_type_vectorizer_human_verified.joblib")
        print("    models/claim_type_label_map_human_verified.json")
        print("    reports/claim_type_metrics_human_verified.json")
        print("    reports/claim_type_report_human_verified.md")
    else:
        print(f"\n❌ Training failed with exit code {code}")
        sys.exit(code)
