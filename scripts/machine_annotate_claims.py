"""Create conservative, context-aware machine suggestions for pending claims.

This is a transparent rules system, not a trained/calibrated model. It writes
separate machine-output files and never changes human annotation fields.
"""

from __future__ import annotations

import argparse
import csv
import difflib
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.claim_dataset import CANDIDATE_FIELDS, LABELS, WORK_DIR, read_rows, write_rows

MACHINE_FIELDS = [
    "machine_system", "machine_claim_class", "machine_claim_rationale",
    "machine_suggested_labels", "machine_label_rationales",
    "machine_uncertainty_flag", "machine_uncertainty_reasons",
    "machine_evidence_cues", "machine_context_before", "machine_context_after",
    "machine_review_priority", "machine_review_reasons", "machine_section_group",
    "machine_section_priority", "representative_categories", "batch_review_stratum",
]
HUMAN_FIELDS = {"verified_sentence", "source_text_checked", "verified_labels", "annotation_status", "annotator", "review_notes"}
MACHINE_SOURCE_FIELDS = [field for field in CANDIDATE_FIELDS if field not in HUMAN_FIELDS]
LEGACY_OUTPUT_CSV = WORK_DIR / "machine_suggested_annotations.csv"
OUTPUT_CSV = WORK_DIR / "machine_suggested_annotations_current.csv"
QUEUE_CSV = WORK_DIR / "machine_review_queue_current.csv"
SAMPLE_CSV = WORK_DIR / "machine_category_sample_current.csv"
SUMMARY_MD = WORK_DIR / "machine_annotation_summary_current.md"
TRIAGE_MD = WORK_DIR / "machine_triage_report.md"
BATCH_CSV = WORK_DIR / "machine_review_batch.csv"
SYSTEM_NAME = "SPIN-SENTRY transparent section/context rules v1"
SECTION_ORDER = {"Safety/Harms": 1, "Results/Outcomes": 2, "Limitations": 3, "Discussion": 4, "Other": 5}
DEFAULT_BATCH_SIZE = 40
CATEGORY_BATCH_QUOTA = 5

PATTERNS = {
    "Efficacy": re.compile(
        r"\b(?:efficacy|effectiveness|effective|improv\w*|reduc\w*|decreas\w*|"
        r"increas\w*|lower\w*|higher\w*|better|worse|benefit\w*|response rate|"
        r"remission|cure|survival|recovery|therapeutic effect|no difference)\b", re.I,
    ),
    "Safety": re.compile(
        r"\b(?:safety|harm\w*|adverse event\w*|side effect\w*|toxicit\w*|"
        r"tolerab\w*|well[- ]tolerated|complication\w*|serious adverse|"
        r"mortality|mortalit\w*|death\w*|withdrawal due to|discontinu\w*)\b", re.I,
    ),
    "Generalizability": re.compile(
        r"\b(?:generaliz\w*|generalis\w*|external validity|applicab\w*|"
        r"representativeness|representative sample|single[- ]cent(?:er|re)|"
        r"only one (?:site|center|centre|setting)|may not (?:generaliz|apply)|"
        r"limited (?:generalizability|generalisability|applicability)|"
        r"across (?:multiple|different|diverse) (?:sites|centers|centres|settings|populations))\b", re.I,
    ),
    "Comparative": re.compile(
        r"\b(?:compared (?:with|to)|versus|vs\.?|greater than|less than|better than|worse than|higher than|"
        r"lower than|similar to|different from|no difference|difference between|"
        r"non[- ]?inferior|equivalent to|outperformed|relative to)\b", re.I,
    ),
    "Novelty": re.compile(
        r"\b(?:novel(?:ty)?|unprecedented|first[- ](?:in[- ]class|study|trial|report|"
        r"demonstration)|for the first time|to our knowledge|first to (?:show|demonstrate|"
        r"report|identify)|new approach)\b", re.I,
    ),
    "Recommendation": re.compile(
        r"\b(?:we recommend|recommend(?:ed|s|ation)?|should be considered|should receive|"
        r"should be used|should be adopted|may be considered|may be used|clinical use|"
        r"clinical practice|practice guideline|warranted|future (?:research|studies|trials) should|"
        r"further studies should|is indicated for|appropriate for)\b", re.I,
    ),
}

OUTCOME_CONTEXT = re.compile(
    r"\b(?:patient\w*|participant\w*|treatment|intervention|drug|therapy|symptom\w*|"
    r"outcome\w*|score\w*|pain|function\w*|disease|recovery|survival|hospitali[sz]\w*|"
    r"biomarker\w*|event\w*|group\w*|trial|clinical)\b", re.I,
)
EFFECT_CONTEXT = re.compile(
    r"\b(?:was|were|is|are|show\w*|find\w*|observ\w*|demonstrat\w*|report\w*|"
    r"result\w*|associated|effect\w*|difference|change\w*|improv\w*|reduc\w*|"
    r"increas\w*|decreas\w*|benefit\w*)\b", re.I,
)
ANAPHORA = re.compile(
    r"\b(?:these findings|these results|these data|this finding|this result|this effect|"
    r"this difference|the observed effect|such findings|such results|this suggests|"
    r"these observations)\b", re.I,
)
STRONG_NONCLAIM = re.compile(
    r"^(?:the (?:aim|objective|purpose|primary objective) was|we (?:aimed|sought|planned|"
    r"conducted|recruited|used|measured|assessed|collected|calculated|randomi[sz]ed)|"
    r"this study (?:aimed|sought|was designed)|patients? were (?:recruited|enrolled|randomi[sz]ed|"
    r"assigned|asked|instructed)|participants? were (?:recruited|enrolled|randomi[sz]ed|"
    r"assigned|asked|instructed)|written informed consent|the trial was registered|"
    r"ethics (?:approval|committee)|the datasets? (?:used|analysed|analyzed)|data (?:were|was) "
    r"(?:collected|analy[sz]ed)|sample size (?:was|were)|randomi[sz]ation was|"
    r"supplementary (?:material|figure|table)|funding statement|conflict of interest|"
    r"authors? (?:declare|report) no conflict|the mean\b|the median\b|"
    r"a total of .{0,80} (?:were|was) included|secondary outcomes? (?:were|was) "
    r"(?:assessed|measured|defined|recorded))", re.I,
)
ADMIN_SECTION = re.compile(
    r"\b(?:methods?|study design|randomi[sz]ation|statistical analysis|data availability|"
    r"funding|acknowledg|ethics|consent|eligibility|recruitment|procedures?|measurement|"
    r"physical function tests|outcome measures|intervention protocol|registration)\b", re.I,
)
FINDING_CUE = re.compile(
    r"\b(?:found|observed|demonstrated|showed|indicated|suggested|revealed|supports?|"
    r"associated with|resulted in|led to|was linked to|no difference|increased|decreased|"
    r"improved|reduced|higher|lower|similar|different|limited by|may not generaliz)\b", re.I,
)
ASSERTION_CUE = re.compile(
    r"\b(?:associated with|correlat\w* with|linked to|caus\w*|predict\w*|"
    r"more likely|less likely|increased risk|reduced risk|contribut\w* to|"
    r"consistent with|inconsistent with|supports? the hypothesis)\b", re.I,
)
RESULT_NUMBER = re.compile(r"\b(?:\d+(?:\.\d+)?%?|\d+\s*(?:patients?|participants?|events?|cases?))\b", re.I)
METHOD_CUE = re.compile(
    r"\b(?:randomi[sz]ed|recruited|enrolled|assigned|instructed|administered|measured|"
    r"assessed|collected|interviewed|calculated|analy[sz]ed|excluded|included in the analysis|"
    r"sample size|baseline characteristics|eligibility criteria|outcome measures|applied|"
    r"received|completed|underwent|delivered)\b", re.I,
)
FORM_OR_HEADING = re.compile(
    r"^(?:\d+(?:\.\d+)*\.?\s*)?(?:table|figure|fig\.?|safety:|primary outcome:|"
    r"secondary outcome:|data availability|funding|ethics|trial registration)\b|"
    r"\[(?:yes\|no|true\|false|n/?a)\]", re.I,
)
FRAGMENT = re.compile(r"(?:\[[^\]]{0,50}\]|\b(?:yes\|no|true\|false)\b|\.{3,})", re.I)
DAMAGED = re.compile(r"[\ufffd\u0000]|\bu\s+to\b|\b[bcdfghjklmnpqrstvwxyz]\s+(?:to|of|is|are|was|were|for|with|that)\b", re.I)
POSITIVE_DIRECTION = re.compile(r"\b(?:improv\w*|increas\w*|higher|better|benefit\w*|reduc\w* pain|effective)\b", re.I)
NEGATIVE_DIRECTION = re.compile(r"\b(?:worsen\w*|decreas\w*|lower|worse|harm\w*|ineffective|no benefit)\b", re.I)


def _cue_text(match: re.Match[str]) -> str:
    return match.group(0).strip()


def _matches(pattern: re.Pattern[str], text: str) -> list[str]:
    return list(dict.fromkeys(_cue_text(m) for m in pattern.finditer(text)))


def section_group(section: str) -> str:
    """Normalize noisy headings for prioritization, never for exclusion."""
    normalized = re.sub(r"[^a-z ]", " ", (section or "").casefold())
    normalized = re.sub(r"\s+", " ", normalized).strip()
    if re.search(r"\b(safety|harms?|adverse events?)\b", normalized):
        return "Safety/Harms"
    if re.search(r"\b(results?|outcomes?|primary outcome|secondary outcome|principal findings?)\b", normalized):
        return "Results/Outcomes"
    if re.search(r"\b(limitations?|strengths and limitations)\b", normalized):
        return "Limitations"
    if re.search(r"\bdiscussion\b", normalized):
        return "Discussion"
    return "Other"


def build_contexts(rows: list[dict[str, str]]) -> dict[str, tuple[str, str]]:
    """Adjacent sentence context from the same paper and source heading only."""
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        for section in (s.strip() for s in row.get("source_section", "").split(";")):
            if section:
                groups[(row.get("paper_id", ""), section)].append(i)
    neighbors: dict[int, tuple[list[str], list[str]]] = defaultdict(lambda: ([], []))
    for indices in groups.values():
        for pos, idx in enumerate(indices):
            before, after = neighbors[idx]
            if pos:
                s = rows[indices[pos - 1]].get("sentence", "")
                if s and s != rows[idx].get("sentence", "") and s not in before:
                    before.append(s)
            if pos + 1 < len(indices):
                s = rows[indices[pos + 1]].get("sentence", "")
                if s and s != rows[idx].get("sentence", "") and s not in after:
                    after.append(s)
    return {rows[i]["candidate_id"]: (" | ".join(v[0][:2]), " | ".join(v[1][:2])) for i, v in neighbors.items()}


def _category_evidence(text: str, context: str, section: str) -> tuple[list[str], dict[str, str], list[str]]:
    labels: list[str] = []
    rationales: dict[str, str] = {}
    all_cues: list[str] = []
    section_is_result = bool(re.search(r"\b(?:results?|discussion|conclusions?|findings|outcomes?)\b", section, re.I))
    contextual = bool(ANAPHORA.search(text))
    for label in LABELS:
        direct = _matches(PATTERNS[label], text)
        nearby = _matches(PATTERNS[label], context) if contextual else []
        use_context = not direct and bool(nearby)
        cues = direct or nearby
        if not cues:
            continue
        accepted = True
        extra = ""
        if label == "Efficacy":
            if re.search(r"\beffective (?:balance|blinding|randomi[sz]ation|masking)\b", text, re.I):
                accepted = False
            else:
                clinical = bool(OUTCOME_CONTEXT.search(text + " " + context))
                result_signal = bool(EFFECT_CONTEXT.search(text)) or section_is_result
                accepted = clinical and result_signal and not (
                    ADMIN_SECTION.search(section) and METHOD_CUE.search(text) and not FINDING_CUE.search(text)
                )
                extra = " clinical/outcome context and a finding or result-section cue"
        elif label == "Generalizability":
            strong = bool(re.search(r"generaliz|generalis|external validity|applicab|representat|single[- ]cent|only one|may not|limited", " ".join(cues), re.I))
            spread = bool(re.search(r"\b(?:across|multiple|different|diverse|settings|sites|population|center|centre)\b", text + " " + section, re.I))
            accepted = strong and (spread or bool(re.search(r"limit|strengths", section, re.I)))
            extra = " scope/applicability language" if accepted else ""
        elif label == "Comparative":
            comparator = bool(re.search(r"\b(?:compared|versus|vs\.?|than|difference|similar|different|non[- ]?inferior|equivalent|outperformed|relative to)\b", text, re.I))
            outcome = bool(OUTCOME_CONTEXT.search(text))
            result_support = section_is_result or bool(FINDING_CUE.search(text)) or bool(RESULT_NUMBER.search(text))
            method_only = bool(ADMIN_SECTION.search(section) and METHOD_CUE.search(text) and not FINDING_CUE.search(text))
            accepted = comparator and outcome and result_support and not method_only
            extra = " explicit comparator or contrast" if accepted else ""
        elif label == "Novelty":
            accepted = bool(re.search(r"\b(?:novel|novelty|unprecedented|to our knowledge|first[- ](?:study|trial|report|demonstration|in[- ]class)|for the first time|first to)\b", text, re.I))
            extra = " explicit novelty statement" if accepted else ""
        elif label == "Recommendation":
            accepted = bool(re.search(r"\b(?:recommend|should|may be considered|may be used|clinical use|clinical practice|guideline|warranted|future (?:research|studies|trials) should|appropriate for|indicated for)\b", text, re.I))
            extra = " prescriptive or practice implication" if accepted else ""
        elif label == "Safety":
            result_support = section_is_result or bool(FINDING_CUE.search(text)) or bool(re.search(r"\b(?:occurred|experienced|reported|withdrew|discontinued|led to|resulted in)\b", text, re.I))
            method_only = bool(ADMIN_SECTION.search(section) and METHOD_CUE.search(text) and not FINDING_CUE.search(text))
            accepted = result_support and not method_only
            extra = " explicit harm, safety, adverse-event, or tolerability language"
        if not accepted:
            continue
        labels.append(label)
        all_cues.extend(f"{label}: {cue}" for cue in cues)
        basis = "neighboring sentence context" if use_context else "the sentence text"
        rationales[label] = f"{label} suggested from {basis} cue(s) {', '.join(repr(c) for c in cues)}; {section or 'source section not recorded'} provides section context.{extra}."
        if use_context:
            rationales[label] += " This suggestion depends on an anaphoric reference and is flagged for review."
    return labels, rationales, list(dict.fromkeys(all_cues))


def annotate_candidate(row: dict[str, str], before: str = "", after: str = "") -> dict[str, str]:
    text = (row.get("sentence") or "").strip()
    section = row.get("source_section", "")
    context = " ".join(x for x in (before, text, after) if x)
    labels, label_rationales, cues = _category_evidence(text, context, section)
    uncertainty: list[str] = []
    text_damaged = bool(row.get("extraction_notes")) or bool(DAMAGED.search(text))
    if text_damaged:
        uncertainty.append("possible_extraction_damage")
    if FRAGMENT.search(text) or (len(text.split()) < 7 and not re.search(r"[.!?]$", text)):
        uncertainty.append("fragment_or_table_like_text")
    if any("neighboring sentence context" in reason for reason in label_rationales.values()):
        uncertainty.append("label_depends_on_neighbor_context")

    admin = bool(STRONG_NONCLAIM.search(text) or FORM_OR_HEADING.search(text))
    method_section = bool(ADMIN_SECTION.search(section))
    method_sentence = bool(METHOD_CUE.search(text))
    findings = bool(FINDING_CUE.search(text)) or bool(labels)
    substantive_predicate = bool(ASSERTION_CUE.search(text))
    result_section = bool(re.search(r"\b(?:results?|secondary outcomes?|primary outcomes?|discussion|conclusions?|findings)\b", section, re.I))
    reporting_result = bool(result_section and RESULT_NUMBER.search(text) and OUTCOME_CONTEXT.search(text))
    if (admin or method_sentence or method_section) and not findings and not reporting_result:
        claim_class = "non_claim"
        claim_reason = "The sentence is a study aim, method, administrative statement, form field, or table-like item, with no detected result or interpretation cue."
    elif findings or reporting_result or (substantive_predicate and len(text.split()) >= 8):
        claim_class = "claim"
        cue_text = ", ".join(cues[:5]) if cues else ", ".join(_matches(FINDING_CUE, text)[:5])
        basis = "a numeric result in a results-oriented section" if reporting_result else "a finding, interpretation, comparison, or scope cue"
        claim_reason = f"The sentence contains {basis}{(': ' + cue_text) if cue_text else ''}; section context is {section or 'not recorded'}."
    else:
        claim_class = "uncertain"
        claim_reason = "Available sentence and section cues do not safely distinguish a substantive research claim from background or procedural text."
        uncertainty.append("claim_status_unclear")

    if claim_class == "claim" and not labels:
        uncertainty.append("claim_outside_or_weakly_supported_by_six_categories")
    existing_hints = [x for x in (row.get("suggested_labels") or "").split(";") if x]
    if existing_hints and not labels:
        uncertainty.append("existing_keyword_hints_not_contextually_confirmed")
    if len(labels) >= 3:
        uncertainty.append("multiple_category_interpretation")
    if POSITIVE_DIRECTION.search(text) and NEGATIVE_DIRECTION.search(text):
        uncertainty.append("conflicting_directional_cues")
    if re.search(r"\b(?:no|not|without|failed to|did not)\b", text, re.I) and FINDING_CUE.search(text):
        uncertainty.append("negated_or_null_finding_requires_review")

    if label_rationales:
        general_rationale = " ".join(label_rationales.values())
    elif existing_hints:
        uncertainty.append("low_evidence_for_category_assignment")
        general_rationale = "No category rule had enough contextual support. The existing keyword hint(s) " + ", ".join(existing_hints) + " were not copied automatically."
    else:
        if claim_class != "non_claim":
            uncertainty.append("low_evidence_for_category_assignment")
        general_rationale = "No category-specific evidence was found by the contextual rules; this is not evidence that the sentence is scientifically irrelevant."
    unique_uncertainty = list(dict.fromkeys(uncertainty))
    return {
        "machine_system": SYSTEM_NAME,
        "machine_claim_class": claim_class,
        "machine_claim_rationale": claim_reason,
        "machine_suggested_labels": ";".join(label for label in LABELS if label in labels),
        "machine_label_rationales": json.dumps(label_rationales or {"_overall": general_rationale}, ensure_ascii=False, sort_keys=True),
        "machine_uncertainty_flag": "yes" if unique_uncertainty else "no",
        "machine_uncertainty_reasons": ";".join(unique_uncertainty),
        "machine_evidence_cues": ";".join(cues),
        "machine_context_before": before,
        "machine_context_after": after,
        "machine_section_group": section_group(section),
        "machine_section_priority": str(SECTION_ORDER[section_group(section)]),
        "batch_review_stratum": "",
    }


def _stable_key(row: dict[str, str], salt: str = "") -> str:
    return hashlib.sha256(f"{salt}|{row.get('candidate_id','')}".encode()).hexdigest()


def build_category_sample(rows: list[dict[str, str]], per_category: int = 10) -> list[dict[str, str]]:
    chosen: dict[str, dict[str, str]] = {}
    strata: dict[str, list[str]] = defaultdict(list)
    for label in LABELS:
        pool = [r for r in rows if label in (r.get("machine_suggested_labels") or "").split(";")]
        # Prefer directly supported, unflagged examples for a useful routine sample;
        # ambiguous and damaged rows enter the higher-priority queue separately.
        pool.sort(key=lambda r: (
            r.get("machine_uncertainty_flag") != "no",
            len((r.get("machine_suggested_labels") or "").split(";")),
            _stable_key(r, label),
        ))
        selected = 0
        used_papers: set[str] = set()
        for prefer_new_paper in (True, False):
            for row in pool:
                if selected >= per_category:
                    break
                cid = row["candidate_id"]
                if label in strata[cid]:
                    continue
                if _near_duplicate(row, list(chosen.values())):
                    continue
                is_new_paper = row.get("paper_id", "") not in used_papers
                if prefer_new_paper and not is_new_paper:
                    continue
                if not prefer_new_paper and is_new_paper:
                    continue
                chosen[cid] = row
                strata[cid].append(label)
                used_papers.add(row.get("paper_id", ""))
                selected += 1
            if selected >= per_category:
                break
    output = []
    for cid, row in chosen.items():
        item = dict(row)
        item["representative_categories"] = ";".join(strata[cid])
        item["machine_review_priority"] = "2"
        item["machine_review_reasons"] = "representative_category_sample"
        output.append(item)
    return output


def _normalized_sentence(sentence: str) -> str:
    return re.sub(r"\W+", " ", (sentence or "").casefold()).strip()


def _near_duplicate(row: dict[str, str], selected: list[dict[str, str]], threshold: float = 0.82) -> bool:
    text = _normalized_sentence(row.get("sentence", ""))
    if not text:
        return False
    for other in selected:
        if row.get("paper_id") != other.get("paper_id"):
            continue
        other_text = _normalized_sentence(other.get("sentence", ""))
        if text == other_text or (len(text) > 40 and len(other_text) > 40 and difflib.SequenceMatcher(None, text, other_text).ratio() >= threshold):
            return True
    return False


def build_review_batch(rows: list[dict[str, str]], batch_size: int = DEFAULT_BATCH_SIZE) -> list[dict[str, str]]:
    """Build a diverse human-review sample; the batch carries no verified labels."""
    if not 20 <= batch_size <= 50:
        raise ValueError("Review batch size must be between 20 and 50.")
    selected: list[dict[str, str]] = []
    paper_counts: Counter[str] = Counter()
    category_quota = max(1, int(batch_size * 0.75) // len(LABELS))

    def pick(pool: list[dict[str, str]], count: int, stratum: str, prefer_certain: bool = False) -> int:
        added = 0
        available = [r for r in pool if r["candidate_id"] not in {x["candidate_id"] for x in selected}]
        available.sort(key=lambda r: (
            paper_counts[r.get("paper_id", "")],
            int(r.get("machine_section_priority") or 5),
            int(r.get("machine_uncertainty_flag") == "yes") if prefer_certain else 0,
            _stable_key(r, stratum),
        ))
        for candidate in available:
            if added >= count:
                break
            if _near_duplicate(candidate, selected):
                continue
            item = dict(candidate)
            item["batch_review_stratum"] = stratum
            selected.append(item)
            paper_counts[item.get("paper_id", "")] += 1
            added += 1
        return added

    quota = category_quota
    # Scarce machine categories are selected first, while paper diversity is a global preference.
    label_order = sorted(LABELS, key=lambda label: sum(label in r.get("machine_suggested_labels", "").split(";") for r in rows))
    for label in label_order:
        pool = [r for r in rows if label in r.get("machine_suggested_labels", "").split(";")]
        pick(pool, quota, f"category:{label}", prefer_certain=True)

    remaining = batch_size - len(selected)
    audit_targets = [
        ("possible_extraction_damage", max(1, round(remaining * 0.35))),
        ("uncertain_priority_section", max(1, round(remaining * 0.30))),
        ("likely_non_claim", remaining),
    ]
    for stratum, target in audit_targets:
        if not remaining:
            break
        if stratum == "possible_extraction_damage":
            pool = [r for r in rows if "possible_extraction_damage" in r.get("machine_uncertainty_reasons", "").split(";")]
        elif stratum == "uncertain_priority_section":
            pool = [r for r in rows if r.get("machine_claim_class") == "uncertain" and r.get("machine_section_group") != "Other"]
        else:
            pool = [r for r in rows if r.get("machine_claim_class") == "non_claim"]
        got = pick(pool, min(target, remaining), stratum)
        remaining -= got
    if remaining:
        pick([r for r in rows if r["candidate_id"] not in {x["candidate_id"] for x in selected}], remaining, "additional_priority_review")
    return selected


def triage_report(rows: list[dict[str, str]], batch: list[dict[str, str]]) -> str:
    classes = Counter(r.get("machine_claim_class", "uncertain") for r in rows)
    categories = Counter(label for r in rows for label in r.get("machine_suggested_labels", "").split(";") if label)
    section_counts = Counter(r.get("machine_section_group", "Other") for r in rows)
    damaged = sum("possible_extraction_damage" in r.get("machine_uncertainty_reasons", "").split(";") for r in rows)
    batch_counts = Counter(r.get("batch_review_stratum", "") for r in batch)
    sample_papers = len({r.get("paper_id", "") for r in batch})
    lines = [
        "# Pending candidate triage report", "",
        f"- Candidates triaged: **{len(rows)}** pending rows.",
        f"- Likely substantive claims: **{classes['claim']}**; likely non-claims: **{classes['non_claim']}**; uncertain: **{classes['uncertain']}**.",
        f"- Possible extraction damage: **{damaged}** rows (a review flag, not a confirmed defect).",
        "- All candidates remain in the source CSV; section groups only affect order and sample selection.",
        "", "## Section priority coverage", "",
        "| Normalized source section | Pending candidates |", "|---|---:|",
    ]
    for group in ["Safety/Harms", "Results/Outcomes", "Limitations", "Discussion", "Other"]:
        lines.append(f"| {group} | {section_counts[group]} |")
    lines += ["", "## Machine category suggestions", "", "Counts overlap and remain unverified.", "", "| Category | Pending suggestions |", "|---|---:|"]
    for label in LABELS:
        lines.append(f"| {label} | {categories[label]} |")
    lines += ["", "## Proposed human-review batch", "", f"Batch size: **{len(batch)}**, spanning **{sample_papers} papers**; category rows use a global preference for different papers and skip near-duplicate sentences within a paper.", "", "| Batch stratum | Candidates |", "|---|---:|"]
    for stratum, count in sorted(batch_counts.items()):
        lines.append(f"| {stratum} | {count} |")
    lines += ["", "This batch is for human review only. Saving decisions updates the original annotation fields; machine suggestions do not become verified automatically.", ""]
    return "\n".join(lines)


def annotate_rows(source_rows: list[dict[str, str]], per_category: int = 10) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    pending = [r for r in source_rows if r.get("annotation_status", "pending").casefold() == "pending"]
    contexts = build_contexts(source_rows)
    annotated: list[dict[str, str]] = []
    for source in pending:
        before, after = contexts.get(source.get("candidate_id", ""), ("", ""))
        item = dict(source)
        item.update(annotate_candidate(source, before, after))
        item["representative_categories"] = ""
        annotated.append(item)
    samples = build_category_sample(annotated, per_category)
    sample_by_id = {r["candidate_id"]: r for r in samples}
    for row in annotated:
        reasons = []
        if row["machine_uncertainty_flag"] == "yes":
            row["machine_review_priority"] = "1"
            reasons.append("uncertain_or_possible_extraction_damage")
        elif row["candidate_id"] in sample_by_id:
            row["machine_review_priority"] = "2"
        else:
            row["machine_review_priority"] = "3"
        sample = sample_by_id.get(row["candidate_id"])
        if sample:
            row["representative_categories"] = sample["representative_categories"]
            reasons.append("representative_category_sample")
        row["machine_review_reasons"] = ";".join(reasons)
    queue = sorted(annotated, key=lambda r: (int(r["machine_section_priority"]), int(r["machine_review_priority"]), -int(r["machine_uncertainty_flag"] == "yes"), r["paper_id"], r["candidate_id"]))
    return annotated, queue, samples


def write_machine_outputs(source_path: Path = WORK_DIR / "candidates.csv", output_path: Path = OUTPUT_CSV,
                          queue_path: Path = QUEUE_CSV, sample_path: Path = SAMPLE_CSV,
                          summary_path: Path = SUMMARY_MD, per_category: int = 10,
                          batch_path: Path = BATCH_CSV, triage_path: Path = TRIAGE_MD,
                          batch_size: int = DEFAULT_BATCH_SIZE) -> dict:
    source_rows = read_rows(source_path, CANDIDATE_FIELDS)
    annotated, queue, samples = annotate_rows(source_rows, per_category)
    previous_path = output_path if output_path.exists() else LEGACY_OUTPUT_CSV
    previous = {r.get("candidate_id", ""): r for r in read_rows(previous_path, MACHINE_SOURCE_FIELDS + MACHINE_FIELDS)} if previous_path.exists() else {}
    pending_ids = {r["candidate_id"] for r in annotated}
    preserved = []
    for source in source_rows:
        if source["candidate_id"] in pending_ids or source.get("annotation_status", "pending").casefold() == "pending":
            continue
        old = previous.get(source["candidate_id"])
        if old:
            item = {key: source.get(key, "") for key in MACHINE_SOURCE_FIELDS}
            item.update({key: old.get(key, "") for key in MACHINE_FIELDS})
            preserved.append(item)
    all_machine = annotated + preserved
    samples = build_category_sample(all_machine, per_category)
    batch = build_review_batch(annotated, batch_size)
    fields = MACHINE_SOURCE_FIELDS + MACHINE_FIELDS
    write_rows(output_path, fields, all_machine)
    write_rows(queue_path, fields, queue)
    write_rows(sample_path, fields, samples)
    write_rows(batch_path, fields, batch)
    summary = summarize(source_rows, annotated, samples, per_category, len(all_machine), len(batch))
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(summary, encoding="utf-8")
    triage_path.parent.mkdir(parents=True, exist_ok=True)
    triage_path.write_text(triage_report(annotated, batch), encoding="utf-8")
    return {"source_rows": len(source_rows), "pending_processed": len(annotated), "review_queue": len(queue), "category_sample_rows": len(samples), "review_batch_rows": len(batch), "review_batch_papers": len({r.get('paper_id', '') for r in batch})}


def summarize(source_rows: list[dict[str, str]], rows: list[dict[str, str]], samples: list[dict[str, str]], per_category: int,
              machine_output_rows: int | None = None, batch_size: int = DEFAULT_BATCH_SIZE) -> str:
    class_counts = Counter(r["machine_claim_class"] for r in rows)
    label_counts = Counter(label for r in rows for label in r["machine_suggested_labels"].split(";") if label)
    uncertain = sum(r["machine_uncertainty_flag"] == "yes" for r in rows)
    damaged = sum("possible_extraction_damage" in r["machine_uncertainty_reasons"].split(";") for r in rows)
    pending_source = sum(r.get("annotation_status", "pending").casefold() == "pending" for r in source_rows)
    excluded = len(source_rows) - pending_source
    sample_counts = Counter(label for r in samples for label in r["representative_categories"].split(";") if label)
    lines = [
        "# Machine annotation run summary", "",
        f"- Source candidate rows: **{len(source_rows)}** ({pending_source} pending processed; {excluded} non-pending rows left untouched).",
        f"- Pending machine suggestions generated: **{len(rows)}**; machine-only output contains **{machine_output_rows if machine_output_rows is not None else len(rows)}** rows including preserved suggestions for previously reviewed candidates.",
        f"- Claim-status estimates: **{class_counts['claim']} claim**, **{class_counts['non_claim']} likely non-claim**, **{class_counts['uncertain']} uncertain**.",
        f"- Uncertainty flagged: **{uncertain}** rows; possible extraction damage flagged: **{damaged}** rows.",
        f"- Rows with at least one machine-suggested category: **{sum(bool(r['machine_suggested_labels']) for r in rows)}**.",
        "", "## Suggested category counts", "", "Counts are overlapping candidate suggestions, not verified labels or performance metrics.", "",
        "| Category | Rows suggested | Representative review sample |", "|---|---:|---:|",
    ]
    for label in LABELS:
        lines.append(f"| {label} | {label_counts[label]} | {sample_counts[label]} / {per_category} |")
    lines += [
        "", "## Review order", "",
        f"1. Start with `{BATCH_CSV.name}`; it contains the first {batch_size}-candidate mixed review batch.",
        f"2. Continue through `{QUEUE_CSV.name}`, which orders prioritized sections first and retains all remaining pending candidates.",
        "",
        f"`{SAMPLE_CSV.name}` contains the selected category-coverage sample, and `{BATCH_CSV.name}` contains the proposed human-review batch. Machine output files omit human-verification fields; the source candidate file was not changed by this run.",
        "",
        "## Human review steps",
        "",
        "1. Start the existing app from the project directory with `venv\\Scripts\\python.exe -m streamlit run scripts/review_claim_annotations.py`.",
        "2. Choose a batch size of 20, 30, 40, or 50. The first batch prioritizes six-category coverage and audit cases; later batches follow the prioritized queue. Saved decisions are persisted in `candidates.csv` and drop out of future sessions' pending queue.",
        "3. Check the original sentence against its source link. Machine labels stay unverified; use Approve or Correct only after checking the source and choosing human-verified labels. Mark uncertain or Reject when appropriate.",
        "4. The machine export is separate and labelled UNVERIFIED. The human-reviewed training export contains only approved/corrected rows and remains separate.",
        "",
        "## Method and limits", "",
        f"System: **{SYSTEM_NAME}**. This run uses transparent phrase/structure rules, existing source-section headings, and adjacent sentences from the same paper and section. Category labels require contextual patterns; existing keyword-only hints are not copied when the rules do not confirm them. The old keyword suggestions remain unchanged in their original columns.",
        "",
        "This is not a trained claim classifier or a language-model judgment. There are no confidence scores or calibrated probabilities. The uncertainty flag is a rule-triggered review signal, not model confidence. Rationale strings identify the matched cues and section/context used; they may be incomplete or wrong.",
        "",
        "Claim/non-claim and category counts are machine estimates, not scientific truth. Background assertions can be substantive claims; procedural sentences can contain claim-like words; section headings may be coarse. Every suggested label remains unverified. Do not train or evaluate on this file.",
        "",
        "Extraction damage flags are heuristic. They include existing extraction warnings, replacement characters, likely split-token artifacts, and table/form-like fragments. They do not prove that a sentence is damaged; inspect flagged text against its source.",
        "",
        "The only human annotations remain in the original candidate file. Suggested CSVs are for triage; record approval/correction through the existing review workflow after checking the paper source.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=WORK_DIR / "candidates.csv")
    parser.add_argument("--output", type=Path, default=OUTPUT_CSV)
    parser.add_argument("--queue", type=Path, default=QUEUE_CSV)
    parser.add_argument("--sample", type=Path, default=SAMPLE_CSV)
    parser.add_argument("--summary", type=Path, default=SUMMARY_MD)
    parser.add_argument("--batch", type=Path, default=BATCH_CSV)
    parser.add_argument("--triage", type=Path, default=TRIAGE_MD)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--per-category", type=int, default=10)
    args = parser.parse_args()
    if args.per_category < 1:
        parser.error("--per-category must be at least 1")
    if not 20 <= args.batch_size <= 50:
        parser.error("--batch-size must be between 20 and 50")
    counts = write_machine_outputs(args.input, args.output, args.queue, args.sample, args.summary, args.per_category, args.batch, args.triage, args.batch_size)
    print(json.dumps(counts, indent=2))
    print(f"Summary: {args.summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
