"""Candidate extraction for a human-reviewed, provenance-tracked claim dataset.

Suggested labels are keyword hints only. This module never writes human labels.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET

try:
    from nltk.tokenize import sent_tokenize
except ImportError:  # Keep corpus preparation usable even without optional NLP packages.
    sent_tokenize = None

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
WORK_DIR = ROOT / "data" / "research_dataset"
PAPERS_CSV = WORK_DIR / "papers.csv"
CANDIDATES_CSV = WORK_DIR / "candidates.csv"
LABELS = ["Efficacy", "Safety", "Generalizability", "Comparative", "Novelty", "Recommendation"]
PAPER_FIELDS = ["paper_id", "title", "doi", "url", "license", "license_url", "source_file", "sha256", "source_type", "ingested_at_utc", "duplicate_of", "license_verified", "candidate_count"]
CANDIDATE_FIELDS = ["candidate_id", "paper_id", "title", "doi", "url", "license", "license_url", "source_section", "sentence", "suggested_labels", "suggestion_scores", "verified_sentence", "source_text_checked", "verified_labels", "annotation_status", "annotator", "review_notes", "extraction_notes"]
TRAINING_FIELDS = ["paper_id", "sentence", "labels", "source", "annotator", "notes"]
ALLOWED_LICENSES = ("creativecommons.org/publicdomain/zero/", "creativecommons.org/licenses/by/", "creativecommons.org/licenses/by-sa/")


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def text_content(element: ET.Element | None) -> str:
    return " ".join("".join(element.itertext()).split()) if element is not None else ""


def read_rows(path: Path, fields: list[str]) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def write_rows(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", newline="", encoding="utf-8", dir=path.parent, delete=False) as stream:
        temp_path = Path(stream.name)
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temp_path.replace(path)


def normalize_doi(doi: str) -> str:
    return re.sub(r"^https?://(dx\.)?doi\.org/", "", doi.strip(), flags=re.I).lower()


def suggest_labels(sentence: str, lexicon: dict[str, list[str]]) -> tuple[list[str], dict[str, int]]:
    lower = sentence.casefold()
    scores = {label: sum(lower.count(term.casefold()) for term in lexicon.get(label, [])) for label in LABELS}
    return [label for label in LABELS if scores[label] > 0], scores


def sentences(text: str) -> list[str]:
    # Keep each returned sentence verbatim from the extracted text, only trimming
    # whitespace at the edges. NLTK tokenization is a segmentation aid, not a label.
    output: list[str] = []
    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = " ".join(paragraph.split())
        if not paragraph:
            continue
        try:
            if sent_tokenize is None:
                raise LookupError
            parts = sent_tokenize(paragraph)
        except LookupError:
            protected = re.sub(r"\b(vs|e\.g|i\.e|fig|eq|dr|mr|mrs|prof)\.", lambda m: m.group(0)[:-1] + "<DOT>", paragraph, flags=re.I)
            parts = [part.replace("<DOT>", ".") for part in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\u201c\"(])", protected)]
        output.extend(part.strip() for part in parts if part.strip())
    return output


def jats_sections(root: ET.Element) -> tuple[str, str, str, str, list[tuple[str, str]]]:
    title = ""
    doi = ""
    license_url = ""
    license_text = ""
    abstract_parts: list[str] = []
    body_sections: list[tuple[str, str]] = []
    for element in root.iter():
        name = local_name(element.tag)
        if name == "article-title" and not title:
            title = text_content(element)
        elif name == "article-id" and element.attrib.get("pub-id-type", "").lower() == "doi":
            doi = text_content(element)
        elif name == "license":
            for key, value in element.attrib.items():
                if local_name(key) == "href":
                    license_url = value
            license_text = license_text or text_content(element)
    # Some Europe PMC OA articles express the license in JATS text and omit
    # xlink:href. Extract only the canonical license path: nearby words such as
    # "Open" or "This" can otherwise be mistaken for part of the URL.
    license_match = re.search(
        r"https?://creativecommons\.org/(licenses/(?:by-sa|by)/\d+(?:\.\d+)?|publicdomain/zero/\d+(?:\.\d+)?)/?",
        license_text, re.I,
    )
    if license_match:
        license_url = "https://creativecommons.org/" + license_match.group(1).rstrip("/") + "/"
    elif not license_url:
        license_name = license_text.casefold()
        version = re.search(r"\b(\d+(?:\.\d+)?)\b", license_name)
        if "creative commons" in license_name and version:
            if "attribution-sharealike" in license_name or "attribution share alike" in license_name:
                license_url = f"https://creativecommons.org/licenses/by-sa/{version.group(1)}/"
            elif "attribution" in license_name and "noncommercial" not in license_name and "no derivatives" not in license_name:
                license_url = f"https://creativecommons.org/licenses/by/{version.group(1)}/"
    for abstract in (e for e in root.iter() if local_name(e.tag) == "abstract"):
        abstract_parts.extend(text_content(p) for p in abstract.iter() if local_name(p.tag) == "p")
    body = next((e for e in root.iter() if local_name(e.tag) == "body"), None)
    if body is not None:
        def walk(section: ET.Element, heading: str = "Body"):
            title_el = next((e for e in section if local_name(e.tag) in {"title", "label"}), None)
            current = text_content(title_el) if title_el is not None else heading
            paragraphs = [text_content(e) for e in section if local_name(e.tag) == "p"]
            if paragraphs:
                body_sections.append((current, " ".join(p for p in paragraphs if p)))
            for child in section:
                if local_name(child.tag) == "sec":
                    walk(child, current)
        for child in body:
            if local_name(child.tag) == "sec":
                walk(child)
            elif local_name(child.tag) == "p":
                body_sections.append(("Body", text_content(child)))
    sections = [("Abstract", " ".join(p for p in abstract_parts if p)), *body_sections]
    return title, doi, license_url, license_text, sections


def license_is_reusable(license_url: str, license_text: str = "") -> bool:
    value = (license_url + " " + license_text).lower()
    if any(term in value for term in ("by-nd", "by-nc-nd", "all rights reserved")):
        return False
    if any(term in value for term in ALLOWED_LICENSES) or "cc0" in value:
        return True
    # Europe PMC contains articles whose JATS statement explicitly names CC BY
    # but omits both a canonical URL and the version number. Keep these usable,
    # while recording the missing version instead of guessing one.
    generic_cc_by = re.search(r"\bcc[\s-]*by\b|creative commons attribution(?: license)?", value)
    return bool(generic_cc_by and not any(term in value for term in ("noncommercial", "no derivatives", "by-nc", "by-nd")))


def make_candidates(paper: dict[str, str], sections: list[tuple[str, str]], lexicon: dict[str, list[str]], extraction_notes: str = "") -> list[dict[str, str]]:
    output = []
    for section, text in sections:
        for sentence in sentences(text):
            if len(sentence) < 25:
                continue
            suggested, scores = suggest_labels(sentence, lexicon)
            stable = "|".join([paper["paper_id"], section, sentence])
            output.append({
                "candidate_id": hashlib.sha256(stable.encode("utf-8")).hexdigest()[:20],
                "paper_id": paper["paper_id"], "title": paper["title"], "doi": paper["doi"],
                "url": paper["url"], "license": paper["license"], "license_url": paper["license_url"],
                "source_section": section, "sentence": sentence,
                "suggested_labels": ";".join(suggested), "suggestion_scores": json.dumps(scores, sort_keys=True),
                "verified_sentence": "", "source_text_checked": "no", "verified_labels": "", "annotation_status": "pending", "annotator": "", "review_notes": "",
                "extraction_notes": extraction_notes,
            })
    return output


def register_paper(paper: dict[str, str], sections: list[tuple[str, str]], lexicon: dict[str, list[str]], extraction_notes: str = "") -> tuple[int, str | None]:
    papers = read_rows(PAPERS_CSV, PAPER_FIELDS)
    existing = read_rows(CANDIDATES_CSV, CANDIDATE_FIELDS)
    key_doi = normalize_doi(paper.get("doi", ""))
    duplicate = next((row for row in papers if row["paper_id"].casefold() == paper["paper_id"].casefold() or (key_doi and normalize_doi(row.get("doi", "")) == key_doi) or (paper.get("sha256") and row.get("sha256") == paper["sha256"])), None)
    if duplicate:
        if paper["paper_id"].casefold() == duplicate["paper_id"].casefold():
            prior = [r for r in existing if r["paper_id"].casefold() == paper["paper_id"].casefold()]
            if any(r.get("annotation_status") != "pending" or r.get("verified_labels") for r in prior):
                return 0, duplicate["paper_id"]
            refreshed = make_candidates(paper, sections, lexicon, extraction_notes)
            existing = [r for r in existing if r["paper_id"].casefold() != paper["paper_id"].casefold()]
            for row in papers:
                if row["paper_id"].casefold() == paper["paper_id"].casefold():
                    row.update({"title": paper["title"], "doi": paper["doi"], "url": paper["url"], "license": paper["license"], "license_url": paper["license_url"], "source_file": paper["source_file"], "sha256": paper["sha256"], "candidate_count": str(len(refreshed))})
                    break
            write_rows(PAPERS_CSV, PAPER_FIELDS, papers)
            write_rows(CANDIDATES_CSV, CANDIDATE_FIELDS, existing + refreshed)
            return len(refreshed), None
        paper["duplicate_of"] = duplicate["paper_id"]
        paper["candidate_count"] = "0"
        papers.append(paper)
        write_rows(PAPERS_CSV, PAPER_FIELDS, papers)
        return 0, duplicate["paper_id"]
    candidates = make_candidates(paper, sections, lexicon, extraction_notes)
    seen_text = {(r["paper_id"], r["source_section"], r["sentence"]) for r in existing}
    new_candidates = [r for r in candidates if (r["paper_id"], r["source_section"], r["sentence"]) not in seen_text]
    paper["candidate_count"] = str(len(new_candidates))
    papers.append(paper)
    write_rows(PAPERS_CSV, PAPER_FIELDS, papers)
    write_rows(CANDIDATES_CSV, CANDIDATE_FIELDS, existing + new_candidates)
    return len(new_candidates), None


def ingest_pilot() -> tuple[int, str | None]:
    from utils.pdf_reader import extract_text_from_pdf
    pdf = ROOT / "data" / "sample_paper.pdf"
    credit = (ROOT / "data" / "SAMPLE_PAPER_CREDIT.md").read_text(encoding="utf-8")
    fields = {key: value.strip() for key, value in re.findall(r"^- \*\*(.+?)\*\*: (.+)$", credit, flags=re.M)}
    if not pdf.is_file() or fields.get("Licence") != "Creative Commons Attribution 4.0 International (CC BY 4.0)":
        raise ValueError("Pilot PDF/attribution is missing or its license could not be verified from SAMPLE_PAPER_CREDIT.md")
    extracted = extract_text_from_pdf(pdf)
    if not extracted["extraction_ok"] or not extracted["abstract_text"]:
        raise ValueError(f"Could not extract pilot PDF abstract: {extracted['error']}")
    lexicon = json.loads((ROOT / "data" / "claim_type_keywords.json").read_text(encoding="utf-8"))
    paper_id = fields.get("PMCID", "").removeprefix("PMC")
    paper_id = "PMC" + paper_id
    source_file = pdf.relative_to(ROOT).as_posix()
    sha = hashlib.sha256(pdf.read_bytes()).hexdigest()
    paper = {
        "paper_id": paper_id, "title": fields.get("Title", extracted["title"]),
        "doi": fields.get("DOI", ""), "url": fields.get("Europe PMC / PMC Record", fields.get("Source URL", "")),
        "license": "CC BY 4.0", "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "source_file": source_file, "sha256": sha, "source_type": "credited_local_pdf",
        "ingested_at_utc": datetime.now(timezone.utc).isoformat(), "duplicate_of": "",
        "license_verified": "yes_from_project_attribution", "candidate_count": "",
    }
    return register_paper(paper, [("Abstract", extracted["abstract_text"])], lexicon,
                          "PDF text extraction may contain line-break/column artifacts; compare with source PDF before approval.")


def ingest_pmcid(pmcid: str) -> tuple[int, str | None]:
    pmcid = pmcid.upper()
    if not pmcid.startswith("PMC") or not pmcid[3:].isdigit():
        raise ValueError("Enter a PMC ID in the form PMC123456.")
    endpoint = f"https://www.ebi.ac.uk/europepmc/webservices/rest/{quote(pmcid)}/fullTextXML"
    request = Request(endpoint, headers={"User-Agent": "SPIN-SENTRY-ResearchDataset/1.0"})
    with urlopen(request, timeout=45) as response:
        xml_bytes = response.read()
    root = ET.fromstring(xml_bytes)
    title, doi, license_url, license_text, sections = jats_sections(root)
    if not license_is_reusable(license_url, license_text):
        raise ValueError("Full text did not expose an accepted reusable license (CC0, CC BY, or CC BY-SA). No content was added.")
    if "/licenses/by-sa/" in license_url:
        license_name = "CC BY-SA " + license_url.rstrip("/").rsplit("/", 1)[-1]
    elif "/licenses/by/" in license_url:
        license_name = "CC BY " + license_url.rstrip("/").rsplit("/", 1)[-1]
    elif "/publicdomain/zero/" in license_url:
        license_name = "CC0 " + license_url.rstrip("/").rsplit("/", 1)[-1]
    elif re.search(r"\bcc[\s-]*by\b|creative commons attribution(?: license)?", license_text, re.I):
        license_name = "CC BY (version not stated in article XML)"
    else:
        license_name = license_text
    url = f"https://europepmc.org/articles/{pmcid}"
    lexicon = json.loads((ROOT / "data" / "claim_type_keywords.json").read_text(encoding="utf-8"))
    paper = {
        "paper_id": pmcid, "title": title, "doi": doi,
        # Keep a source record URL in `url`; DOI is stored independently.
        "url": url,
        "license": license_name or license_text, "license_url": license_url,
        "source_file": f"Europe PMC fullTextXML: {pmcid}",
        "sha256": hashlib.sha256(xml_bytes).hexdigest(), "source_type": "Europe PMC OA fullTextXML",
        "ingested_at_utc": datetime.now(timezone.utc).isoformat(), "duplicate_of": "",
        "license_verified": "yes_from_article_xml", "candidate_count": "",
    }
    return register_paper(paper, sections, lexicon)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pilot-sample", action="store_true", help="Extract only the credited sample PDF abstract.")
    group.add_argument("--pmcid", help="Fetch one open-access paper's full-text XML through Europe PMC and verify its article license.")
    args = parser.parse_args()
    try:
        count, duplicate = ingest_pilot() if args.pilot_sample else ingest_pmcid(args.pmcid)
    except Exception as exc:
        parser.error(str(exc))
    if duplicate:
        print(f"Skipped duplicate; already registered as {duplicate}.")
        return 0
    print(f"Added {count} pending sentence candidates. Suggested labels are not verified annotations.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
