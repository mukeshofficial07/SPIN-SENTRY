"""
pipeline.py
-----------
Orchestrator for the SPIN-SENTRY Forward Alignment Pipeline.

Calls Stages 1–4 in sequence as defined in the paper (Section III, Fig. 1):
    Stage 1 (HCP-BEA) → Stage 2 (MZEM) → Stage 3 (DC-NLI) → Stage 4 (ESWE)

Input:  PDF file path or pre-extracted text
Output: PipelineResult dataclass
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from utils.pdf_reader import extract_text_from_pdf
from stages.stage1_claim_parser import parse_claims, Claim
from stages.stage2_zone_mapper import map_zones, ZoneMap
from stages.stage4_eswe import run_eswe_pipeline, generate_summary, ClaimReport


@dataclass
class PipelineResult:
    """Complete output of the SPIN-SENTRY Forward Alignment Pipeline."""
    # Input metadata
    pdf_path: str
    title: str

    # Stage outputs
    claims: list[Claim]
    zone_map: ZoneMap
    reports: list[ClaimReport]
    summary: dict

    # Diagnostics
    elapsed_seconds: float
    warnings: list[str]
    model_status: dict
    extraction_ok: bool
    pdf_error: Optional[str]

    # Raw text (for debugging)
    abstract_text: str
    body_text: str

    def to_dict(self) -> dict:
        return {
            "pdf_path":          self.pdf_path,
            "title":             self.title,
            "screening_notice":  "Automated spin flags are heuristic screening signals for research exploration, not proof of scientific misconduct or clinical trial invalidity.",
            "summary":           self.summary,
            "claims":            [c.to_dict() for c in self.claims],
            "reports":           [r.to_dict() for r in self.reports],
            "warnings":          self.warnings,
            "elapsed_seconds":   round(self.elapsed_seconds, 2),
            "model_status":      self.model_status,
        }


def run_pipeline(
    pdf_path: str | Path,
    top_k: int = 5,
    progress_callback: Optional[Callable[[str, float], None]] = None,
    classifier_mode: str = "rules",
) -> PipelineResult:
    """
    Run the full SPIN-SENTRY pipeline on a PDF file.

    Parameters
    ----------
    pdf_path          : path to PDF
    top_k             : number of evidence sentences to retrieve per claim
    progress_callback : optional callable(message: str, fraction: float)
                        for Streamlit progress updates

    Returns
    -------
    PipelineResult
    """
    def _progress(msg: str, frac: float):
        if progress_callback:
            progress_callback(msg, frac)

    start = time.time()
    warnings: list[str] = []

    _progress("📄 Extracting text from PDF…", 0.05)

    # --- PDF extraction ---
    pdf_data = extract_text_from_pdf(pdf_path)

    if not pdf_data["extraction_ok"]:
        warnings.append(f"PDF extraction error: {pdf_data['error']}")

    abstract_text = pdf_data["abstract_text"]
    body_text = pdf_data["body_text"]
    title = pdf_data["title"]

    if not abstract_text:
        warnings.append("⚠️ Abstract not detected. Pipeline may produce poor results.")
    if not body_text:
        warnings.append("⚠️ Body text not detected. Evidence zones will be empty.")

    _progress("🔍 Stage 1: Parsing claims from abstract (HCP-BEA)…", 0.20)

    # --- Stage 1: HCP-BEA ---
    claims = parse_claims(abstract_text, classifier_mode=classifier_mode)

    if not claims:
        warnings.append("⚠️ No claims extracted from abstract.")

    _progress(f"🗺️ Stage 2: Mapping evidence zones (MZEM)…", 0.40)

    # --- Stage 2: MZEM ---
    zone_map = map_zones(body_text)
    warnings.extend(zone_map.warnings)

    _progress("🤖 Stage 3+4: NLI scoring and ESWE weighting…", 0.60)

    # --- Stages 3 + 4: DC-NLI + ESWE ---
    # (Stage 3 is called inside Stage 4 per claim)
    from stages.stage3_nli import get_model_status
    model_status = get_model_status()

    if not model_status["loaded"]:
        warnings.append(
            f"⚠️ NLI model could not be loaded: {model_status['error']}. "
            "Using keyword-overlap fallback. Scores are approximations only."
        )

    reports = run_eswe_pipeline(claims=claims, zone_map=zone_map, top_k=top_k)

    _progress("📊 Generating report…", 0.90)

    summary = generate_summary(reports)

    elapsed = time.time() - start
    _progress("✅ Pipeline complete.", 1.0)

    return PipelineResult(
        pdf_path=str(pdf_path),
        title=title,
        claims=claims,
        zone_map=zone_map,
        reports=reports,
        summary=summary,
        elapsed_seconds=elapsed,
        warnings=warnings,
        model_status=model_status,
        extraction_ok=pdf_data["extraction_ok"],
        pdf_error=pdf_data["error"],
        abstract_text=abstract_text,
        body_text=body_text,
    )
