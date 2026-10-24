"""
utils/ner_utils.py
------------------
Named-entity recognition for biomedical claim anchoring.

Uses spaCy en_core_web_sm (general English NER) as the MVP implementation.
This is a deliberate approximation: a biomedical-specialized model (scispaCy,
BioBERT NER) would give better drug/outcome/population entity detection.

The paper (Section III-A) calls for BioBERT-based NER anchoring to
drug names, clinical outcomes, and population descriptors. We approximate
each category:

  - Drug/Intervention  → ORG, PRODUCT entities + drug pattern list
  - Clinical Outcome   → keyword heuristics on numeric patterns, outcome words
  - Population         → NORP, GPE entities + demographic keywords
"""

import re
from functools import lru_cache
from typing import Optional

# Lazy-load spaCy to avoid slow startup
_nlp = None


@lru_cache(maxsize=1)
def _get_nlp():
    """Load and cache spaCy model. Falls back gracefully if unavailable."""
    global _nlp
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        return nlp
    except OSError:
        return None
    except ImportError:
        return None


# Exclusions for terms frequently misclassified as drugs/interventions
_NON_DRUG_EXCLUSIONS = {
    # Surgical / anatomical / clinical procedures
    "cabg", "pci", "icu", "ccu", "surgery", "sternotomy", "bypass", "grafting",
    "intubation", "extubation", "biopsy", "reoperation", "sternot- omy",
    # Statistical tests, metrics, and trial design terms
    "anova", "ancova", "manova", "t-test", "chi-square", "kruskal-wallis", "friedman",
    "ci", "sd", "sem", "iqr", "mcid", "hr", "or", "rr", "p-value", "bonferroni",
    "rct", "fas", "itt", "pp", "demographic", "demographics",
    # Measurement scales and questionnaires
    "vas", "tug", "qol", "sf-36", "nrs", "bdi", "ham-d", "gcs", "spo2", "spo",
    # Publishing, institutions, journals
    "plos", "plos one", "pubmed", "medline", "zenodo", "clinicaltrials.gov",
    "university", "department", "hospital", "faculty", "center", "centre",
    "research article", "editor", "author", "peer review"
}

# Known pharmacological and trial interventions
_KNOWN_INTERVENTIONS = [
    "elastic therapeutic taping", "kinesio taping", "taping",
    "placebo", "sham", "tramadol", "opioid", "opioids",
    "morphine", "fentanyl", "paracetamol", "acetaminophen",
    "aspirin", "heparin", "warfarin", "metformin", "insulin",
    "vitamin d", "atorvastatin", "rosuvastatin", "amiodarone",
]

# Biomedical-specific pharmacological regex patterns
_DRUG_PATTERNS = re.compile(
    r"\b([A-Z][a-z]+-?[a-z]*(?:mab|nib|umab|tinib|ciclib|pril|sartan|statin|olol|azole|mycin|cillin|vir|zole|gliptin)\b"
    r"|drug\s+[A-Z0-9]+|Drug\s+[A-Z0-9]+|compound\s+[A-Z0-9]+)",
    re.MULTILINE,
)

_OUTCOME_KEYWORDS = [
    "mortality", "survival", "remission", "response", "relapse", "recurrence",
    "reduction", "improvement", "progression", "cure", "death", "hospitalization",
    "pain", "quality of life", "QoL", "HbA1c", "blood pressure", "LDL",
    "symptom", "endpoint", "outcome", "event", "mobility", "sleep quality",
    "analgesic consumption", "length of hospital stay", "hospital stay",
]

_POPULATION_KEYWORDS = [
    "patient", "patients", "participant", "participants", "subject", "subjects",
    "adult", "adults", "elderly", "children", "pediatric", "male", "female",
    "women", "men", "age", "aged", "population", "cohort", "group", "groups",
]


def extract_entities(sentence: str) -> dict:
    """
    Extract biomedical entities from a sentence.

    Returns
    -------
    dict with keys:
        'drugs'       : list[str]  — drug/intervention names found
        'outcomes'    : list[str]  — clinical outcome phrases found
        'populations' : list[str]  — population descriptors found
        'all_entities': list[dict] — raw spaCy entities (label, text)
        'nlp_available': bool
    """
    result = {
        "drugs": [],
        "outcomes": [],
        "populations": [],
        "all_entities": [],
        "nlp_available": False,
    }

    sent_lower = sentence.lower()

    # 1. Match known therapeutic interventions first
    for intervention in _KNOWN_INTERVENTIONS:
        if intervention in sent_lower:
            result["drugs"].append(intervention.title())

    # 2. spaCy pass (strict filtering to avoid false positives)
    nlp = _get_nlp()
    if nlp is not None:
        result["nlp_available"] = True
        doc = nlp(sentence)
        for ent in doc.ents:
            result["all_entities"].append({"text": ent.text, "label": ent.label_})
            ent_clean = ent.text.strip().lower()

            # Skip known non-drugs
            if ent_clean in _NON_DRUG_EXCLUSIONS:
                continue

            # Only PRODUCT entities that look pharmacological (not general ORG)
            if ent.label_ == "PRODUCT" and len(ent.text) > 3:
                result["drugs"].append(ent.text)
            elif ent.label_ in ("NORP", "CARDINAL", "QUANTITY"):
                if any(w in ent_clean for w in ["patient", "subject", "group", "men", "women", "year", "age"]):
                    result["populations"].append(ent.text)

    # 3. Pharmacological regex pattern matching
    for match in _DRUG_PATTERNS.finditer(sentence):
        candidate = match.group(0).strip()
        if candidate.lower() not in _NON_DRUG_EXCLUSIONS and candidate not in result["drugs"]:
            result["drugs"].append(candidate)

    # 4. Outcome keyword matching
    for kw in _OUTCOME_KEYWORDS:
        if kw.lower() in sent_lower:
            result["outcomes"].append(kw)

    # 5. Population keyword matching
    for kw in _POPULATION_KEYWORDS:
        pattern = r"\b" + re.escape(kw.lower()) + r"\b"
        if re.search(pattern, sent_lower):
            result["populations"].append(kw)

    # Filter out exclusions and deduplicate
    result["drugs"] = [d for d in dict.fromkeys(result["drugs"]) if d.lower() not in _NON_DRUG_EXCLUSIONS]
    result["outcomes"] = list(dict.fromkeys(result["outcomes"]))
    result["populations"] = list(dict.fromkeys(result["populations"]))

    return result


def entities_to_anchor_string(entities: dict) -> str:
    """Format entities into a display string for the UI."""
    parts = []
    if entities["drugs"]:
        parts.append("Drugs/Interventions: " + ", ".join(entities["drugs"][:5]))
    if entities["outcomes"]:
        parts.append("Outcomes: " + ", ".join(entities["outcomes"][:5]))
    if entities["populations"]:
        parts.append("Populations: " + ", ".join(entities["populations"][:5]))
    return " | ".join(parts) if parts else "No entities detected"
