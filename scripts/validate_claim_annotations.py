"""Validate human annotations for the six Stage 1 claim categories."""

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

LABELS = {
    "Efficacy", "Safety", "Generalizability", "Comparative", "Novelty",
    "Recommendation",
}
REQUIRED = {"paper_id", "sentence", "labels", "source", "annotator", "notes"}


def validate(path: Path) -> tuple[list[str], int, Counter]:
    errors: list[str] = []
    counts: Counter = Counter()
    try:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                return ["CSV is missing its header row."], 0, counts
            missing = REQUIRED - set(reader.fieldnames)
            if missing:
                return [f"Missing required columns: {', '.join(sorted(missing))}"], 0, counts
            rows = list(reader)
    except (OSError, csv.Error) as exc:
        return [f"Could not read CSV: {exc}"], 0, counts

    seen: set[tuple[str, str]] = set()
    for line, row in enumerate(rows, start=2):
        paper = (row.get("paper_id") or "").strip()
        sentence = (row.get("sentence") or "").strip()
        labels_raw = (row.get("labels") or "").strip()
        if not any((paper, sentence, labels_raw, (row.get("source") or "").strip(), (row.get("annotator") or "").strip(), (row.get("notes") or "").strip())):
            continue
        if not paper:
            errors.append(f"Row {line}: paper_id is required for grouped evaluation.")
        if not sentence:
            errors.append(f"Row {line}: sentence is required.")
        if not labels_raw:
            errors.append(f"Row {line}: add at least one human-assigned label.")
        if not (row.get("source") or "").strip():
            errors.append(f"Row {line}: source DOI/URL is required.")
        if not (row.get("annotator") or "").strip():
            errors.append(f"Row {line}: annotator is required for human-verified data.")
        labels = [item.strip() for item in labels_raw.split(";") if item.strip()]
        invalid = sorted(set(labels) - LABELS)
        if invalid:
            errors.append(f"Row {line}: unknown label(s): {', '.join(invalid)}")
        if len(labels) != len(set(labels)):
            errors.append(f"Row {line}: duplicate labels are not allowed.")
        if sentence:
            key = (paper, sentence.casefold())
            if key in seen:
                errors.append(f"Row {line}: duplicate sentence for paper_id {paper!r}.")
            seen.add(key)
        for label in set(labels) & LABELS:
            counts[label] += 1

    if not any((row.get("sentence") or "").strip() for row in rows):
        errors.append("No annotation rows found; the template is intentionally empty.")
    return errors, len(rows), counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", nargs="?", type=Path, default=Path("data/claim_annotations_template.csv"))
    args = parser.parse_args()
    errors, row_count, counts = validate(args.csv_path)
    if errors:
        print(f"Validation failed for {args.csv_path} ({row_count} data rows):")
        for error in errors:
            print(f"- {error}")
        return 1
    print(f"Valid annotations: {row_count} rows")
    for label in sorted(LABELS):
        print(f"{label}: {counts[label]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
