"""Build an explicitly machine-pseudo-labeled training set without altering human records."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.claim_dataset import LABELS

DEFAULT_INPUT = ROOT / "data" / "research_dataset" / "machine_suggested_annotations_current.csv"
LEGACY_INPUT = ROOT / "data" / "research_dataset" / "machine_suggested_annotations.csv"
DATASET_OUTPUT = ROOT / "data" / "research_dataset" / "machine_pseudo_labels.csv"
EXCLUSIONS_OUTPUT = ROOT / "data" / "research_dataset" / "machine_pseudo_label_exclusions.csv"
REPORT_OUTPUT = ROOT / "reports" / "machine_pseudo_label_preparation.json"
LABEL_SOURCE = "machine_generated_pseudo_label"

PROVENANCE_FIELDS = [
    "candidate_id", "paper_id", "title", "doi", "url", "license", "license_url",
    "source_section", "sentence", "labels", "source", "label_source", "machine_system",
    "machine_claim_class", "machine_claim_rationale", "machine_suggested_labels",
    "machine_label_rationales", "machine_uncertainty_flag", "machine_uncertainty_reasons",
    "machine_evidence_cues", "machine_context_before", "machine_context_after", "extraction_notes",
]
EXCLUSION_FIELDS = [
    "candidate_id", "paper_id", "title", "doi", "url", "license", "license_url",
    "source_section", "sentence", "machine_claim_class", "machine_suggested_labels",
    "machine_uncertainty_flag", "machine_uncertainty_reasons", "extraction_notes", "exclusion_reasons",
]


def assess_row(row: dict[str, str]) -> list[str]:
    """Return all transparent exclusion reasons; reasons may overlap."""
    reasons = []
    if "possible_extraction_damage" in (row.get("machine_uncertainty_reasons") or "").split(";") or (row.get("extraction_notes") or "").strip():
        reasons.append("possible_extraction_damage")
    claim_class = (row.get("machine_claim_class") or "").strip()
    if claim_class == "non_claim":
        reasons.append("likely_non_claim")
    elif claim_class != "claim":
        reasons.append("ambiguous_claim_status")
    if (row.get("machine_uncertainty_flag") or "").strip().casefold() != "no":
        reasons.append("ambiguous_or_uncertain")
    labels = [x.strip() for x in (row.get("machine_suggested_labels") or "").split(";") if x.strip()]
    if not labels:
        reasons.append("missing_labels")
    invalid = sorted(set(labels) - set(LABELS))
    if invalid:
        reasons.append("invalid_labels:" + ";".join(invalid))
    if not (row.get("paper_id") or "").strip():
        reasons.append("missing_paper_id")
    if not (row.get("sentence") or "").strip():
        reasons.append("missing_sentence")
    if not ((row.get("doi") or "").strip() or (row.get("url") or "").strip()):
        reasons.append("missing_source_provenance")
    return reasons


def prepare_rows(source_rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], list[dict[str, str]], Counter, Counter]:
    included, excluded = [], []
    primary_reasons: Counter = Counter()
    overlapping_reasons: Counter = Counter()
    seen: set[str] = set()
    for source in source_rows:
        reasons = assess_row(source)
        candidate_id = source.get("candidate_id") or ""
        if not candidate_id:
            reasons.append("missing_candidate_id")
        elif candidate_id in seen:
            reasons.append("duplicate_candidate_id")
        seen.add(candidate_id)
        if reasons:
            primary_reasons[reasons[0]] += 1
            overlapping_reasons.update(reasons)
            excluded.append({**source, "exclusion_reasons": ";".join(reasons)})
            continue
        labels = sorted({x.strip() for x in source["machine_suggested_labels"].split(";") if x.strip()}, key=LABELS.index)
        row = {
            "candidate_id": candidate_id,
            "paper_id": source["paper_id"],
            "title": source.get("title", ""),
            "doi": source.get("doi", ""),
            "url": source.get("url", ""),
            "license": source.get("license", ""),
            "license_url": source.get("license_url", ""),
            "source_section": source.get("source_section", ""),
            "sentence": source["sentence"],
            "labels": ";".join(labels),
            "source": (source.get("doi") or "").strip() or source.get("url", "").strip(),
            "label_source": LABEL_SOURCE,
        }
        for field in PROVENANCE_FIELDS:
            if field.startswith("machine_") or field == "extraction_notes":
                row[field] = source.get(field, "")
        included.append(row)
    return included, excluded, primary_reasons, overlapping_reasons


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_dataset(input_path: Path, dataset_path: Path, exclusions_path: Path, report_path: Path) -> dict:
    with input_path.open(newline="", encoding="utf-8-sig") as stream:
        source_rows = list(csv.DictReader(stream))
    included, excluded, primary, overlapping = prepare_rows(source_rows)
    papers = sorted({row["paper_id"] for row in included})
    category_counts = Counter(label for row in included for label in row["labels"].split(";") if label)
    report = {
        "source_file": str(input_path.resolve()),
        "source_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "label_source": LABEL_SOURCE,
        "source_rows": len(source_rows),
        "included_rows": len(included),
        "excluded_rows": len(excluded),
        "included_paper_count": len(papers),
        "included_paper_ids": papers,
        "multi_label_rows": sum(len(row["labels"].split(";")) > 1 for row in included),
        "category_counts": {label: category_counts[label] for label in LABELS},
        "primary_exclusion_counts_disjoint": dict(primary),
        "exclusion_flag_counts_overlapping": dict(overlapping),
        "human_verified_annotations_created": 0,
        "human_review_statuses_modified": 0,
        "interpretation": "Machine-generated pseudo-labels only; no human verification is asserted.",
    }
    _write_csv(dataset_path, PROVENANCE_FIELDS, included)
    _write_csv(exclusions_path, EXCLUSION_FIELDS, excluded)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    default_input = DEFAULT_INPUT if DEFAULT_INPUT.exists() else LEGACY_INPUT
    parser.add_argument("--input", type=Path, default=default_input)
    parser.add_argument("--output", type=Path, default=DATASET_OUTPUT)
    parser.add_argument("--exclusions", type=Path, default=EXCLUSIONS_OUTPUT)
    parser.add_argument("--report", type=Path, default=REPORT_OUTPUT)
    args = parser.parse_args()
    result = build_dataset(args.input, args.output, args.exclusions, args.report)
    print(json.dumps(result, indent=2))
    print(f"Dataset: {args.output}\nExclusions: {args.exclusions}\nReport: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
