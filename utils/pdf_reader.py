"""
utils/pdf_reader.py
-------------------
Extracts text from a PDF file using pdfplumber and splits it into:
  - abstract_text : str   (the abstract section)
  - body_text     : str   (everything after the abstract)
  - full_text     : str   (complete document text)
  - pages         : list[str] (per-page text)

Approach: rule-based heuristics on section headers. No model required.
Limitation: may misidentify abstract boundary on non-standard PDFs.
"""

import re
import pdfplumber
from pathlib import Path


def extract_text_from_pdf(pdf_path: str | Path) -> dict:
    """
    Extract text from a PDF file.

    Returns
    -------
    dict with keys:
        'full_text'     : complete document text (str)
        'pages'         : list of per-page strings
        'abstract_text' : heuristically identified abstract (str)
        'body_text'     : text after the abstract (str)
        'title'         : first non-empty line (str, best-effort)
        'extraction_ok' : bool
        'error'         : str or None
    """
    result = {
        "full_text": "",
        "pages": [],
        "abstract_text": "",
        "body_text": "",
        "title": "",
        "extraction_ok": False,
        "error": None,
    }

    try:
        with pdfplumber.open(pdf_path) as pdf:
            pages = []
            for i, page in enumerate(pdf.pages):
                # On page 1, check if there is a left sidebar (e.g. PLOS ONE / two-column)
                # where Abstract is shifted to x0 > 120. If so, crop to the main article column.
                if i == 0:
                    try:
                        words = page.extract_words()
                        abs_words = [w for w in words if w["text"].lower() == "abstract"]
                        if abs_words and abs_words[0]["x0"] > 120:
                            crop_x = max(0, abs_words[0]["x0"] - 15)
                            page = page.crop((crop_x, 0, page.width, page.height))
                    except Exception:
                        pass

                text = page.extract_text(x_tolerance=3, y_tolerance=3)
                if text:
                    pages.append(text)
                else:
                    pages.append("")
            result["pages"] = pages

        raw_full_text = "\n".join(pages)
        full_text = clean_extracted_text(raw_full_text)
        result["full_text"] = full_text

        # Best-effort title: first non-empty, non-short line
        for line in full_text.splitlines():
            stripped = line.strip()
            if len(stripped) > 10 and not stripped.lower().startswith("research article"):
                result["title"] = stripped
                break

        # Split abstract from body
        abstract_text, body_text = _split_abstract_body(full_text)
        result["abstract_text"] = clean_extracted_text(abstract_text)
        result["body_text"] = clean_extracted_text(body_text)
        result["extraction_ok"] = True

    except Exception as exc:
        result["error"] = str(exc)

    return result


_DEHYPHEN_STOPWORDS = {"and", "or", "in", "to", "for", "of", "with", "by", "as", "at", "on", "a", "an"}


def clean_extracted_text(text: str) -> str:
    """
    Clean extracted PDF text:
    - Remove running headers and footers (PLOS One, page numbers, etc.)
    - Dehyphenate words broken across lines or by layout (e.g. 'sternot- omy' -> 'sternotomy')
    - Normalize whitespace while preserving paragraph structure
    """
    if not text:
        return ""

    # 1. Remove running headers / footers and metadata
    text = re.sub(
        r"(?i)plos\s+one\s*\|\s*https?://\S+\s+[A-Za-z]+\s+\d{1,2},\s+\d{4}\s+\d+\s*/\s*\d+",
        "",
        text,
    )
    text = re.sub(r"(?i)\b\d{1,3}\s*/\s*\d{1,3}\b", "", text)
    text = re.sub(r"(?i)^\s*(?:plos\s+one|research\s+article)\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"https?://(?:dx\.)?doi\.org/\S+", "", text)
    text = re.sub(r"(?i)\b[a-z]\.pone\.\d+\s+[A-Za-z]+\s+\d{1,2},\s+\d{4}\b", "", text)

    # 2. Dehyphenate across line breaks: 'differ-\nences' -> 'differences'
    text = re.sub(r"(\b[A-Za-z]{2,})-\s*\n\s*([a-z]{2,}\b)", r"\1\2", text)

    # 3. Dehyphenate within lines with safe exclusion of conjunctions ('placebo- and')
    def _fix_hyphen_space(match):
        prefix = match.group(1)
        suffix = match.group(2)
        if suffix.lower() in _DEHYPHEN_STOPWORDS:
            return f"{prefix}- {suffix}"
        return f"{prefix}{suffix}"

    text = re.sub(r"(\b[A-Za-z]{2,})-\s+([a-z]{2,}\b)", _fix_hyphen_space, text)

    # 4. Normalize spaces (leave single newlines for block splitting)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# Patterns that mark the START of the abstract
_ABSTRACT_START_RE = re.compile(
    r"^\s*(abstract|summary|structured\s+abstract)\s*$",
    re.IGNORECASE | re.MULTILINE,
)

# Patterns that mark the END of the abstract / START of body
_BODY_START_RE = re.compile(
    r"(?:\n|^)[^\n]{0,80}\b(?:introduction|1[\.\s]+introduction|1[\.\s]+background|background(?!\s*[:—–\-]))\b",
    re.IGNORECASE,
)

# Footers / metadata at the end of the abstract to strip
_ABSTRACT_TAIL_RE = re.compile(
    r"\n\s*(?:(?:[a-z]\.)?pone\.\d+|trial\s+registration|clinicaltrials\.gov|data\s+availability|keywords?|funding:|and\s+reproduction\s+in\s+any\s+medium)[\s\S]*$",
    re.IGNORECASE,
)


def _split_abstract_body(full_text: str) -> tuple[str, str]:
    """
    Split full text into (abstract, body).

    Strategy (in priority order):
    1. Look for explicit 'Abstract' header followed by 'Introduction' header.
    2. Look for 'Abstract' inline label at start of a paragraph.
    3. Fallback: treat text after 'Abstract' or first 1500 chars as abstract.
    """
    # Strategy 1: explicit Abstract + Introduction
    abs_match = _ABSTRACT_START_RE.search(full_text)
    if abs_match:
        start_idx = abs_match.end()
        # Search starting at start_idx + 40 to avoid matching inline structured subheads (e.g. Background:)
        body_match = _BODY_START_RE.search(full_text, start_idx + 40)
        if body_match and body_match.start() > start_idx + 40:
            raw_abstract = full_text[start_idx: body_match.start()].strip()
            abstract = _ABSTRACT_TAIL_RE.sub("", raw_abstract).strip()
            body = full_text[body_match.start():].strip()
            return abstract, body
        else:
            # Fallback: take next 2500 chars after Abstract
            end_idx = min(len(full_text), start_idx + 2500)
            raw_abstract = full_text[start_idx:end_idx].strip()
            abstract = _ABSTRACT_TAIL_RE.sub("", raw_abstract).strip()
            body = full_text[end_idx:].strip()
            return abstract, body

    # Strategy 2: inline "Abstract—" or "Abstract:" label
    inline_re = re.search(
        r"Abstract[—:\-–]\s*", full_text, re.IGNORECASE
    )
    if inline_re:
        abs_start = inline_re.end()
        body_m = _BODY_START_RE.search(full_text, abs_start)
        if body_m and body_m.start() > abs_start:
            raw_abstract = full_text[abs_start: body_m.start()].strip()
            abstract = _ABSTRACT_TAIL_RE.sub("", raw_abstract).strip()
            body = full_text[body_m.start():].strip()
            return abstract, body

    # Strategy 3: heuristic — use first 1500 chars as abstract
    split_point = min(1500, len(full_text) // 5)
    return full_text[:split_point].strip(), full_text[split_point:].strip()

