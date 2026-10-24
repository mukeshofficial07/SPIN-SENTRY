"""
SPIN-SENTRY — AI Batch Annotation Script
=========================================
Processes all pending candidates in candidates.csv using:
  1. Trained TF-IDF + OvR Logistic Regression classifier (primary signal)
  2. Machine-generated suggestions from machine_suggested_annotations_current.csv (secondary)
  3. Rule-based claim/non-claim detection (fallback)

Saves decisions directly to candidates.csv with:
  annotator = "Mukesh"
  annotation_status = "approved"

Runs in batches of 100 and checkpoints after each batch.
"""

import sys
import os
import json
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime, timezone

# ── paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
CANDIDATES_CSV      = ROOT / "data" / "research_dataset" / "candidates.csv"
MACHINE_CSV         = ROOT / "data" / "research_dataset" / "machine_suggested_annotations_current.csv"
MODEL_PATH          = ROOT / "models" / "claim_type_model_machine_pseudo.joblib"
VECTORIZER_PATH     = ROOT / "models" / "claim_type_vectorizer_machine_pseudo.joblib"
LABEL_MAP_PATH      = ROOT / "models" / "claim_type_label_map_machine_pseudo.json"

CHECKPOINT_EVERY    = 100
ANNOTATOR           = "Mukesh"
CLAIM_THRESHOLD     = 0.35   # min prob to include a label
NON_CLAIM_SECTIONS  = {"references", "acknowledgements", "funding", "declarations",
                        "competing interests", "author contributions", "abbreviations"}

LABELS = ["Efficacy", "Safety", "Generalizability", "Comparative", "Novelty", "Recommendation"]

# ── non-claim keyword patterns ─────────────────────────────────────────────────
NON_CLAIM_PATTERNS = [
    "informed consent", "was obtained", "approved by", "ethics committee",
    "all authors", "the authors", "no conflict", "declare no", "funded by",
    "grant number", "supported by", "corresponding author", "email:",
    "received:", "accepted:", "published:", "doi:", "copyright",
    "table ", "figure ", "fig.", "supplementary", "appendix",
    "data availability", "code availability",
]

CLAIM_SIGNALS = [
    "significantly", "improved", "reduced", "increased", "decreased",
    "effective", "efficacious", "safe", "harmful", "adverse",
    "superior", "inferior", "comparable", "similar to", "better than",
    "generalizable", "applicable", "recommend", "suggest", "conclude",
    "novel", "first", "innovative", "demonstrated", "showed", "found",
    "associated with", "correlated with", "predicted", "risk of",
    "benefit", "outcome", "result", "treatment", "intervention",
    "p <", "p=", "95% ci", "odds ratio", "relative risk", "hazard ratio",
    "mean difference", "effect size", "statistically",
]

# ── helpers ───────────────────────────────────────────────────────────────────

def load_model():
    """Load the trained TF-IDF classifier and vectorizer."""
    try:
        model      = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
        with open(LABEL_MAP_PATH) as f:
            label_map = json.load(f)
        # Handle nested {classes: [...], index_to_label: {"0": "Efficacy", ...}} format
        if isinstance(label_map, dict) and "index_to_label" in label_map:
            idx_to_label = {int(k): v for k, v in label_map["index_to_label"].items()}
        elif isinstance(label_map, dict):
            first_key = next(iter(label_map))
            if first_key.isdigit():
                idx_to_label = {int(k): v for k, v in label_map.items()}
            else:
                idx_to_label = {int(v): k for k, v in label_map.items()}
        elif isinstance(label_map, list):
            idx_to_label = {i: l for i, l in enumerate(label_map)}
        else:
            idx_to_label = {i: l for i, l in enumerate(LABELS)}
        print(f"✅ Model loaded. Labels: {idx_to_label}")
        return model, vectorizer, idx_to_label
    except Exception as e:
        print(f"⚠️  Could not load model: {e}. Will use rule-based fallback.")
        return None, None, None


def is_claim_by_rules(sentence: str, source_section: str) -> bool:
    """Simple rule-based claim/non-claim triage."""
    if not isinstance(sentence, str) or len(sentence.strip()) < 20:
        return False
    sent_lower = sentence.lower()
    sec_lower  = str(source_section).lower() if pd.notna(source_section) else ""

    # Reject by section
    if any(s in sec_lower for s in NON_CLAIM_SECTIONS):
        return False
    # Reject by pattern
    if any(p in sent_lower for p in NON_CLAIM_PATTERNS):
        return False
    # Accept if has claim signal
    if any(s in sent_lower for s in CLAIM_SIGNALS):
        return True
    # Accept from high-yield sections
    if any(s in sec_lower for s in ["abstract", "result", "conclusion", "discussion", "finding"]):
        return True
    return False


def classify_labels_rules(sentence: str, source_section: str) -> list:
    """Rule-based label assignment."""
    sent_lower = sentence.lower()
    sec_lower  = str(source_section).lower() if pd.notna(source_section) else ""
    labels_found = []

    efficacy_kw   = ["effective", "efficacy", "improved", "benefit", "outcome", "success", "response rate", "remission", "cure", "healed", "recovery", "performance", "significantly better"]
    safety_kw     = ["adverse", "side effect", "harm", "toxic", "safety", "tolerab", "risk", "complication", "mortality", "death", "serious", "event", "reaction"]
    general_kw    = ["generaliz", "applicab", "population", "external valid", "diverse", "broad", "all patients", "regardless", "irrespective", "heterogeneous", "representative"]
    comparative_kw= ["compared", "versus", "vs.", "superior", "inferior", "better than", "worse than", "similar to", "equivalent", "non-inferior", "no difference", "control group", "placebo", "standard of care"]
    novelty_kw    = ["novel", "first", "innovative", "new approach", "previously unreported", "pioneer", "unique", "original", "breakthrough", "unprecedented"]
    recomm_kw     = ["recommend", "suggest", "should be", "ought to", "guideline", "clinical practice", "standard", "advise", "propose", "consider", "preferred"]

    if any(k in sent_lower for k in efficacy_kw):   labels_found.append("Efficacy")
    if any(k in sent_lower for k in safety_kw):     labels_found.append("Safety")
    if any(k in sent_lower for k in general_kw):    labels_found.append("Generalizability")
    if any(k in sent_lower for k in comparative_kw):labels_found.append("Comparative")
    if any(k in sent_lower for k in novelty_kw):    labels_found.append("Novelty")
    if any(k in sent_lower for k in recomm_kw):     labels_found.append("Recommendation")

    # fallback by section
    if not labels_found:
        if "result" in sec_lower or "finding" in sec_lower:
            labels_found.append("Efficacy")
        elif "conclusion" in sec_lower or "discussion" in sec_lower:
            labels_found.append("Efficacy")

    return labels_found


def classify_with_model(sentences: list, model, vectorizer, idx_to_label) -> list:
    """Run trained model on a batch of sentences. Returns list of label lists."""
    try:
        X = vectorizer.transform(sentences)
        # predict_proba returns list of arrays (one per estimator in OvR)
        results = []
        # Try decision_function first
        try:
            scores_matrix = np.column_stack([
                est.predict_proba(X)[:, 1]
                for est in model.estimators_
            ])
            # scores_matrix shape: (n_sentences, n_labels)
            for row in scores_matrix:
                labels = []
                for idx, prob in enumerate(row):
                    if prob >= CLAIM_THRESHOLD and idx in idx_to_label:
                        labels.append(idx_to_label[idx])
                results.append(labels)
        except AttributeError:
            # OvR with predict
            preds = model.predict(X)
            for row in preds:
                labels = [idx_to_label[i] for i, v in enumerate(row) if v == 1 and i in idx_to_label]
                results.append(labels)
        return results
    except Exception as e:
        print(f"  ⚠️  Model predict error: {e}")
        return [[] for _ in sentences]


def merge_labels(model_labels: list, machine_labels_str: str, rule_labels: list) -> list:
    """Merge model, machine, and rule labels — union with deduplication."""
    final = set(model_labels)
    # Add machine suggestions
    if isinstance(machine_labels_str, str) and machine_labels_str.strip():
        for lbl in machine_labels_str.split(";"):
            lbl = lbl.strip()
            if lbl in LABELS:
                final.add(lbl)
    # Add rule-based only if model gave nothing
    if not final:
        final.update(rule_labels)
    # Filter to valid labels only
    return [l for l in LABELS if l in final]  # preserve canonical order


def decide_claim_class(machine_class: str, sentence: str, source_section: str) -> str:
    """Decide claim/non_claim/uncertain using machine class + rules."""
    mc = str(machine_class).lower() if pd.notna(machine_class) else "uncertain"
    if mc == "claim":
        return "claim"
    if mc == "non_claim":
        # double-check with rules; machine might be wrong on high-signal sentences
        if is_claim_by_rules(sentence, source_section):
            return "claim"
        return "non_claim"
    # uncertain — use rules
    if is_claim_by_rules(sentence, source_section):
        return "claim"
    return "non_claim"


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("SPIN-SENTRY — AI Batch Annotation")
    print("=" * 70)

    # Load data
    print("\n📂 Loading data...")
    candidates = pd.read_csv(CANDIDATES_CSV, dtype=str)
    machine    = pd.read_csv(MACHINE_CSV, dtype=str)

    # Merge machine suggestions into candidates on candidate_id
    machine_cols = ["candidate_id", "machine_claim_class", "machine_suggested_labels",
                    "machine_uncertainty_flag", "machine_label_rationales"]
    machine_sub  = machine[machine_cols].drop_duplicates("candidate_id")
    df = candidates.merge(machine_sub, on="candidate_id", how="left")

    # Filter pending only
    pending_mask = df["annotation_status"] == "pending"
    pending_idx  = df.index[pending_mask].tolist()
    total        = len(pending_idx)
    print(f"✅ Loaded {len(candidates)} total candidates, {total} pending.")

    # Load model
    model, vectorizer, idx_to_label = load_model()
    has_model = model is not None

    # Prepare tracking
    now_str  = datetime.now(timezone.utc).isoformat()
    counts   = {"claim": 0, "non_claim": 0, "uncertain": 0, "label_dist": {l: 0 for l in LABELS}}
    processed = 0

    print(f"\n🔄 Starting annotation of {total} candidates (checkpoint every {CHECKPOINT_EVERY})...")
    print("-" * 70)

    # Process in batches for checkpointing
    batch_sentences  = []
    batch_indices    = []

    def flush_batch(batch_indices, batch_sentences):
        nonlocal processed
        if not batch_indices:
            return

        # Get model predictions for whole batch at once
        if has_model and batch_sentences:
            model_label_lists = classify_with_model(batch_sentences, model, vectorizer, idx_to_label)
        else:
            model_label_lists = [[] for _ in batch_sentences]

        for i, (row_idx, sentence) in enumerate(zip(batch_indices, batch_sentences)):
            row = df.loc[row_idx]
            source_section    = row.get("source_section", "")
            machine_class     = row.get("machine_claim_class", "uncertain")
            machine_lbl_str   = row.get("machine_suggested_labels", "")

            # Decide claim class
            claim_class = decide_claim_class(machine_class, sentence, source_section)

            if claim_class == "non_claim":
                final_labels = []
                counts["non_claim"] += 1
            else:
                rule_labels  = classify_labels_rules(sentence, source_section)
                final_labels = merge_labels(model_label_lists[i], machine_lbl_str, rule_labels)
                if not final_labels:
                    # If model/machine/rules yield nothing for a claim, use suggested_labels fallback
                    existing_suggested = str(row.get("suggested_labels", "") or "")
                    for lbl in existing_suggested.split(";"):
                        lbl = lbl.strip()
                        if lbl in LABELS:
                            final_labels.append(lbl)
                if not final_labels:
                    final_labels = ["Efficacy"]  # last resort default
                counts["claim"] += 1
                for lbl in final_labels:
                    counts["label_dist"][lbl] = counts["label_dist"].get(lbl, 0) + 1

            # Write back to candidates DataFrame
            candidates.at[row_idx, "verified_labels"]   = ";".join(final_labels) if final_labels else ""
            candidates.at[row_idx, "verified_sentence"]  = sentence
            candidates.at[row_idx, "annotation_status"]  = "approved"
            candidates.at[row_idx, "annotator"]          = ANNOTATOR
            candidates.at[row_idx, "review_notes"]       = (
                f"AI-assisted annotation (claim_class={claim_class}); "
                f"model={'TF-IDF+OvR' if has_model else 'rules'}; "
                f"machine_class={machine_class}; ts={now_str}"
            )

            processed += 1

        # Checkpoint save
        candidates.to_csv(CANDIDATES_CSV, index=False)
        print(f"  💾 Checkpoint saved — {processed}/{total} done "
              f"(claims={counts['claim']}, non_claims={counts['non_claim']})")

    # Iterate over pending rows
    for row_idx in pending_idx:
        row      = df.loc[row_idx]
        sentence = str(row.get("sentence", "")).strip()

        batch_sentences.append(sentence)
        batch_indices.append(row_idx)

        if len(batch_indices) >= CHECKPOINT_EVERY:
            flush_batch(batch_indices, batch_sentences)
            batch_sentences = []
            batch_indices   = []

    # Flush remaining
    if batch_indices:
        flush_batch(batch_indices, batch_sentences)

    # Final save
    candidates.to_csv(CANDIDATES_CSV, index=False)

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("✅ ANNOTATION COMPLETE")
    print("=" * 70)
    print(f"  Total processed  : {processed}")
    print(f"  Claims           : {counts['claim']}")
    print(f"  Non-claims       : {counts['non_claim']}")
    print(f"\n  Label distribution:")
    for lbl, cnt in counts["label_dist"].items():
        bar = "█" * (cnt // 10)
        print(f"    {lbl:<20} {cnt:>5}  {bar}")
    print(f"\n  Annotator        : {ANNOTATOR}")
    print(f"  Saved to         : {CANDIDATES_CSV}")

    # Verify final state
    final_df = pd.read_csv(CANDIDATES_CSV)
    print(f"\n  Final status counts:")
    print(final_df["annotation_status"].value_counts().to_string())
    print("=" * 70)


if __name__ == "__main__":
    main()
