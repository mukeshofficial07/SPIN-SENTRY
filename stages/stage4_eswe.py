"""
stages/stage4_eswe.py
----------------------
Stage 4: Evidence-Strength Weighted Entailment (ESWE)

Paper reference: Section III-D, Equation (4):
    WE(c, e) = NLI(c, e) × ESI(e) × relevance(t, z)

where:
    NLI(c, e) = forward entailment score from Stage 3
    ESI(e)    = Evidence Strength Index (Table I) — study design × sample size × analysis
    relevance(t, z) = zone–type relevance weight (paper Section III-D)

Spin flag threshold: WE < 0.5 → claim is flagged as potentially unsupported.
This threshold is an engineering choice for this prototype and is clearly labeled as such.

Output per claim:
    {
        'claim'           : str,
        'claim_types'     : list[str],
        'zone_route'      : str,
        'evidence_items'  : list[dict],     # top-k from Stage 3
        'esi'             : float,
        'relevance'       : float,
        'we_score'        : float,          # WE(c, e) — Eq. (4)
        'spin_flag'       : bool,           # WE < 0.5
        'support_label'   : str,            # "Supported" / "Weak" / "Flagged"
        'esi_metadata'    : StudyMetadata,
        'top_evidence'    : str,            # best evidence sentence
        'forward'         : float,
        'gap'             : float,
    }
"""

from dataclasses import dataclass, field
from typing import Optional

from stages.stage1_claim_parser import Claim
from stages.stage2_zone_mapper import ZoneMap
from stages.stage3_nli import score_claim_against_zone
from utils.esi_table import extract_study_metadata, get_zone_type_relevance, StudyMetadata

# Spin flag threshold (engineering choice, explicitly labeled in UI)
SPIN_FLAG_THRESHOLD = 0.5

# Support label thresholds
_SUPPORTED_THRESHOLD = 0.65
_WEAK_THRESHOLD = 0.50


@dataclass
class ClaimReport:
    """ESWE output for a single claim."""
    claim: str
    claim_types: list[str]
    zone_route: str
    evidence_items: list[dict]
    esi: float
    relevance: float
    we_score: float
    spin_flag: bool
    support_label: str
    esi_metadata: Optional[StudyMetadata]
    top_evidence: str
    forward: float
    reverse: float
    gap: float
    hedging: float
    hedging_words: list[str]
    model_used: str
    role: str = "Finding / Claim"
    is_empirical_claim: bool = True
    explanation: str = ""

    def __post_init__(self):
        if not getattr(self, "explanation", ""):
            if not getattr(self, "is_empirical_claim", True):
                self.explanation = f"Study protocol or design statement aligned with {self.zone_route} section."
            elif self.support_label == "Supported":
                self.explanation = f"Strong forward entailment ({self.forward:.2f}) from {self.zone_route} evidence; consistent with trial body."
            elif self.support_label == "Weak Support":
                self.explanation = f"Moderate forward entailment ({self.forward:.2f}); partial evidence identified in {self.zone_route}."
            else:
                self.explanation = f"Direct textual entailment in {self.zone_route} was low ({self.forward:.2f}); verify in {self.zone_route} evidence."

    def to_dict(self) -> dict:
        return {
            "claim":              self.claim,
            "claim_types":        self.claim_types,
            "role":               self.role,
            "is_empirical_claim": self.is_empirical_claim,
            "zone_route":         self.zone_route,
            "evidence_items":     self.evidence_items,
            "esi":                round(self.esi, 3),
            "relevance":          round(self.relevance, 3),
            "we_score":           round(self.we_score, 3),
            "spin_flag":          self.spin_flag,
            "support_label":      self.support_label,
            "explanation":        self.explanation,
            "top_evidence":       self.top_evidence,
            "forward":            round(self.forward, 3),
            "reverse":            round(self.reverse, 3),
            "gap":                round(self.gap, 3),
            "hedging":            round(self.hedging, 3),
            "hedging_words":      self.hedging_words,
            "model_used":         self.model_used,
            "esi_notes":          getattr(self.esi_metadata, "extraction_notes", []),
        }


def _generate_explanation(
    claim: Claim,
    we_score: float,
    forward: float,
    reverse: float,
    gap: float,
    contradiction: float,
    spin_flag: bool,
    zone_route: str,
    neutral: float = 0.0,
    top_section: str = "",
) -> tuple[str, str]:
    """
    Generate an appropriately cautious support label and evidence-grounded explanation.
    Distinguishes protocol/limitations, direct contradictions, empirical support, and unverified endpoints.
    """
    sec = top_section or zone_route
    role = getattr(claim, "role", "")
    is_empirical = getattr(claim, "is_empirical_claim", True)

    if not is_empirical:
        if spin_flag:
            return (
                "Protocol / Method (Low Direct Match)",
                f"Describes study protocol or objective. Direct textual entailment in {sec} was limited ({forward:.2f}); verify in Methods."
            )
        if role == "Limitation / Caution":
            return (
                "Cautious Limitation / Disclaimer (Aligned)",
                f"Trial limitation statement explicitly acknowledging constraints in {sec}; avoids overstatement."
            )
        return (
            "Protocol / Method (Consistent)",
            f"Study protocol or design statement aligned with {sec} section."
        )

    if forward >= 0.60 and contradiction < 0.25:
        return (
            "Supported",
            f"Strong forward entailment ({forward:.2f}) from {sec} evidence; consistent with trial body."
        )
    elif we_score >= _SUPPORTED_THRESHOLD and contradiction < 0.25:
        return (
            "Supported",
            f"Strong forward entailment ({forward:.2f}) from {sec} evidence; consistent with trial body."
        )
    elif contradiction >= 0.35 or (contradiction > forward + 0.15 and contradiction >= 0.25):
        return (
            "Flagged (Contradictory Signal)",
            f"Evidence in {sec} directly conflicts with abstract assertion (contradiction score: {contradiction:.2f})."
        )
    elif forward >= 0.40 and contradiction < 0.25:
        return (
            "Weak Support",
            f"Moderate forward entailment ({forward:.2f}); partial evidence identified in {sec}."
        )
    elif neutral >= 0.50 and contradiction < 0.25 and not ("Generalizability" in getattr(claim, "types", [])):
        return (
            "Flagged (Unverified / Outcome Mismatch)",
            f"Direct textual entailment in {sec} was low ({forward:.2f}). Specific numerical findings may be in unparsed tables or unverified by prose."
        )
    elif "Generalizability" in getattr(claim, "types", []) and gap >= 0.70 and reverse < 0.30:
        return (
            "Flagged (Generalizability Gap)",
            f"Evidence indicates limited reverse entailment (generalizability gap: {gap:.2f}), suggesting potential population or scope stretch."
        )
    else:
        return (
            "Flagged (Unverified / Low Direct Match)",
            f"Direct textual entailment in {sec} was low ({forward:.2f}). Specific numerical findings may be in unparsed tables or unverified by prose."
        )


def _support_label(we_score: float) -> str:
    """Map WE score to a human-readable support label."""
    if we_score >= _SUPPORTED_THRESHOLD:
        return "Supported"
    elif we_score >= _WEAK_THRESHOLD:
        return "Weak Support"
    else:
        return "Flagged (Low Support)"


def compute_eswe(
    claim: Claim,
    zone_map: ZoneMap,
    top_k: int = 5,
) -> ClaimReport:
    """
    Compute Evidence-Strength Weighted Entailment for a single claim.

    Implements paper Equation (4):
        WE(c, e) = NLI(c, e) × ESI(e) × relevance(t, z)

    Parameters
    ----------
    claim    : Claim — from Stage 1
    zone_map : ZoneMap — from Stage 2
    top_k    : int — top evidence sentences to retrieve

    Returns
    -------
    ClaimReport
    """
    zone_name = claim.zone_route
    target_zone = zone_map.get(zone_name)
    actual_zone = zone_name

    # Build candidate pool of (sentence, section_name) preserving exact traceability
    candidate_sentences: list[tuple[str, str]] = []
    for s in target_zone.sentences:
        candidate_sentences.append((s, zone_name))

    # Relevant cross-zone context
    if zone_name == "Results":
        for s in zone_map.get("Discussion").sentences:
            candidate_sentences.append((s, "Discussion"))
        if "Safety" in getattr(claim, "types", []):
            for s in zone_map.get("Harms").sentences:
                candidate_sentences.append((s, "Harms"))
    elif zone_name == "Harms":
        for s in zone_map.get("Results").sentences:
            candidate_sentences.append((s, "Results"))
    elif zone_name == "Limitations":
        for s in zone_map.get("Discussion").sentences:
            candidate_sentences.append((s, "Discussion"))
    elif zone_name == "Methods":
        for s in zone_map.get("Results").sentences:
            candidate_sentences.append((s, "Results"))

    # Fallback if primary pool is empty
    if not candidate_sentences:
        fallback_order = ["Results", "Harms", "Limitations", "Methods", "Discussion"]
        for fallback in fallback_order:
            fb_zone = zone_map.get(fallback)
            if fb_zone.sentences:
                for s in fb_zone.sentences:
                    candidate_sentences.append((s, fallback))
                actual_zone = fallback
                break

    # Stage 3: DC-NLI — get top-k evidence with conflict-aware prioritization
    evidence_items = score_claim_against_zone(
        claim=claim.sentence,
        zone_sentences=candidate_sentences,
        top_k=top_k,
        prioritize_contradiction=True,
    )

    if evidence_items:
        best = evidence_items[0]
        forward = best["forward"]
        reverse = best["reverse"]
        gap = best["gap"]
        top_evidence = best["evidence"]
        top_section = best.get("source_section", actual_zone)
        contradiction = best.get("fwd_contradiction", 0.0)
        neutral = best.get("fwd_neutral", 0.0)
        model_used = best["model_used"]
    else:
        forward = 0.0
        reverse = 0.0
        gap = 1.0
        top_evidence = ""
        top_section = actual_zone
        contradiction = 0.0
        neutral = 1.0
        model_used = "no_evidence"

    # ESI: extract from Methods zone text (best source) + full context
    methods_text = zone_map.methods_text or ""
    context_text = methods_text + " " + target_zone.raw_text
    esi_metadata = extract_study_metadata(context_text)
    esi = esi_metadata.esi

    # Zone–type relevance (paper Section III-D)
    primary_type = claim.types[0] if claim.types else "Efficacy"
    relevance = get_zone_type_relevance(primary_type, actual_zone)

    # Paper Equation (4): WE(c, e) = NLI(c, e) × ESI(e) × relevance(t, z)
    we_score = forward * esi * relevance

    # Determine spin_flag and label
    is_empirical = getattr(claim, "is_empirical_claim", True)
    if not is_empirical:
        spin_flag = False
    elif forward >= 0.60 and contradiction < 0.25:
        spin_flag = False
    elif we_score >= SPIN_FLAG_THRESHOLD and contradiction < 0.25:
        spin_flag = False
    elif forward >= 0.40 and contradiction < 0.25:
        spin_flag = False
    else:
        spin_flag = True

    label, explanation = _generate_explanation(
        claim=claim,
        we_score=we_score,
        forward=forward,
        reverse=reverse,
        gap=gap,
        contradiction=contradiction,
        spin_flag=spin_flag,
        zone_route=actual_zone,
        neutral=neutral,
        top_section=top_section,
    )

    return ClaimReport(
        claim=claim.sentence,
        claim_types=claim.types,
        zone_route=actual_zone,
        evidence_items=evidence_items,
        esi=esi,
        relevance=relevance,
        we_score=we_score,
        spin_flag=spin_flag,
        support_label=label,
        esi_metadata=esi_metadata,
        top_evidence=top_evidence,
        forward=forward,
        reverse=reverse,
        gap=gap,
        hedging=claim.hedging,
        hedging_words=claim.hedging_words,
        model_used=model_used,
        role=getattr(claim, "role", "Finding / Claim"),
        is_empirical_claim=is_empirical,
        explanation=explanation,
    )


def run_eswe_pipeline(
    claims: list[Claim],
    zone_map: ZoneMap,
    top_k: int = 5,
) -> list[ClaimReport]:
    """
    Run Stage 4 for all claims.

    Returns
    -------
    list[ClaimReport] — one per claim, in order.
    """
    reports = []
    for claim in claims:
        report = compute_eswe(claim=claim, zone_map=zone_map, top_k=top_k)
        reports.append(report)
    return reports


def generate_summary(reports: list[ClaimReport]) -> dict:
    """
    Generate a summary card from all ClaimReports.

    Returns dict with aggregate statistics for the UI.
    """
    if not reports:
        return {
            "total_claims": 0,
            "flagged_claims": 0,
            "supported_claims": 0,
            "weak_claims": 0,
            "spin_risk": "Unknown",
            "avg_we_score": 0.0,
        }

    flagged = sum(1 for r in reports if r.spin_flag)
    supported = sum(1 for r in reports if "Supported" in r.support_label and not r.spin_flag)
    weak = sum(1 for r in reports if "Weak" in r.support_label)
    avg_we = sum(r.we_score for r in reports) / len(reports)

    empirical = [r for r in reports if getattr(r, "is_empirical_claim", True)]
    emp_count = len(empirical)
    emp_flagged = sum(1 for r in empirical if r.spin_flag)
    emp_avg_we = (sum(r.we_score for r in empirical) / emp_count) if emp_count > 0 else 0.0
    protocol_count = len(reports) - emp_count

    emp_flag_rate = (emp_flagged / emp_count) if emp_count > 0 else 0.0
    has_contradiction = any("Contradictory" in r.support_label for r in reports)
    contradiction_count = sum(1 for r in reports if "Contradictory" in r.support_label)

    # Transparent & evidence-grounded risk aggregation:
    # 1. Direct statistical contradiction between abstract claim and trial data -> High Risk
    # 2. Overstatement, outcome mismatch, or unverified empirical claims (without contradiction) -> Moderate Risk
    # 3. Supported findings and cautious limitations -> Low Risk
    if has_contradiction:
        spin_risk = "🔴 High"
    elif emp_flagged >= 1 or emp_flag_rate >= 0.25:
        spin_risk = "🟡 Moderate"
    else:
        spin_risk = "🟢 Low"

    flag_rate = flagged / len(reports)

    return {
        "total_claims":        len(reports),
        "flagged_claims":      flagged,
        "supported_claims":    supported,
        "weak_claims":         weak,
        "spin_risk":           spin_risk,
        "avg_we_score":        round(avg_we, 3),
        "flag_rate":           round(flag_rate, 3),
        "empirical_claims":    emp_count,
        "empirical_flagged":   emp_flagged,
        "empirical_avg_we":    round(emp_avg_we, 3),
        "protocol_statements": protocol_count,
        "contradiction_count": contradiction_count,
    }
