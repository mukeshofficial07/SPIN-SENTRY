"""Export only human-reviewed approved/corrected candidates in training CSV format."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.claim_dataset import CANDIDATE_FIELDS, TRAINING_FIELDS, read_rows
from utils.claim_annotation_review import is_eligible_human_annotation


def approved_rows(candidates: list[dict[str, str]]) -> list[dict[str, str]]:
    """
    Filter and extract only genuine, source-verified human annotations.
    Unresolved or AI-generated suggestions are ineligible and excluded.
    """
    output = []
    seen: dict[tuple[str, str], tuple[str, str]] = {}
    for row in candidates:
        if not is_eligible_human_annotation(row):
            continue
        status = str(row.get("annotation_status", "")).strip().casefold()
        if status not in {"approved", "confirmed", "corrected"}:
            continue

        labels = ";".join(label.strip() for label in row.get("verified_labels", "").split(";") if label.strip())
        reviewed_sentence = row.get("verified_sentence", "").strip() or row.get("sentence", "")
        key = (row.get("paper_id", "").casefold(), reviewed_sentence.strip().casefold())
        note = row.get("review_notes", "").strip()
        label_key = (labels, row.get("source_section", ""))

        if key in seen:
            if seen[key] != label_key:
                raise ValueError(f"Duplicate sentence has conflicting approved annotations: {row.get('candidate_id')}")
            continue
        seen[key] = label_key

        doi = row.get("doi", "").strip()
        source = f"https://doi.org/{doi}" if doi else row.get("url", "")
        provenance = (
            f"section={row.get('source_section', '')}; "
            f"license={row.get('license', '')}; "
            f"license_url={row.get('license_url', '')}; "
            f"candidate_id={row.get('candidate_id', '')}"
        )
        output.append({
            "paper_id": row.get("paper_id", ""),
            "sentence": reviewed_sentence,
            "labels": labels,
            "source": source,
            "annotator": row.get("annotator", ""),
            "notes": f"{provenance}; review_note={note}".strip(),
        })
    return output


def write_export(path: Path, rows: list[dict[str, str]]) -> None:
    """Write verified human annotations to CSV."""
    if not rows:
        raise ValueError(
            "There are no approved/corrected human annotations with verified source text to export. "
            "Unverified suggestions, source-unverified rows, and uncertain rows are excluded."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=TRAINING_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data" / "research_dataset" / "candidates.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "claim_annotations_reviewed.csv")
    args = parser.parse_args()
    try:
        rows = approved_rows(read_rows(args.input, CANDIDATE_FIELDS))
        write_export(args.output, rows)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Exported {len(rows)} human-verified rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
