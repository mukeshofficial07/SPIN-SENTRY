"""
build_review_queue.py
---------------------
Builds a prioritised human-review queue from candidates.csv and
machine_suggested_annotations_current.csv.

Goal: Help Mukesh efficiently review 200 high-value candidates to build a
genuine gold-standard test set for SPIN-SENTRY.

Outputs
-------
  data/research_dataset/priority_review_queue.csv   -- 200 prioritised rows
  data/research_dataset/human_gold_standard.csv     -- 13 genuine gold rows
"""

from __future__ import annotations

import re
import sys
import math
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

# Force UTF-8 output so the print summary never crashes on Windows consoles
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# --- Paths -------------------------------------------------------------------

BASE_DIR   = Path(__file__).resolve().parent.parent          # SPIN-SENTRY root
DATA_DIR   = BASE_DIR / "data" / "research_dataset"
MODEL_DIR  = BASE_DIR / "models"

CANDIDATES_CSV   = DATA_DIR / "candidates.csv"
ANNOTATIONS_CSV  = DATA_DIR / "machine_suggested_annotations_current.csv"
MODEL_PATH       = MODEL_DIR / "claim_type_model_human_verified.joblib"

OUTPUT_QUEUE     = DATA_DIR / "priority_review_queue.csv"
OUTPUT_GOLD      = DATA_DIR / "human_gold_standard.csv"

QUEUE_SIZE       = 200
MIN_PER_PAPER    = 5
SCORE_THRESHOLD  = 0.40

LABEL_ORDER      = ["Efficacy", "Safety", "Generalizability",
                    "Comparative", "Novelty", "Recommendation"]
LABEL_SCORE_COLS = [f"model_{lbl.lower()}_score" for lbl in LABEL_ORDER]

# --- Step 1 & 2: Load data ---------------------------------------------------

print("=" * 65)
print("SPIN-SENTRY  -  build_review_queue.py")
print("=" * 65)

print("\n[1/7] Loading candidates.csv ...")
candidates = pd.read_csv(CANDIDATES_CSV, dtype=str)
print(f"      {len(candidates):,} rows loaded.")

print("[2/7] Loading machine_suggested_annotations_current.csv ...")
annotations = pd.read_csv(ANNOTATIONS_CSV, dtype=str)
print(f"      {len(annotations):,} rows loaded.")

# --- Step 3: Load model ------------------------------------------------------

print("[3/7] Loading trained model ...")
model_loaded = False
model = None
try:
    model = joblib.load(MODEL_PATH)
    # Quick sanity check
    _test = model.predict_proba(["test sentence"])
    model_loaded = True
    print(f"      Model loaded OK  - {MODEL_PATH.name}")
except Exception as exc:
    print(f"      WARNING: Could not load model - {exc}")
    print("      Model scores will be filled with NaN.")

# --- Step 4: Identify AI-batch rows ------------------------------------------

print("[4/7] Identifying AI-batch rows ...")

ai_mask     = candidates["review_notes"].fillna("").str.startswith("AI-assisted annotation")
ai_batch_df = candidates[ai_mask].copy()
human_df    = candidates[~ai_mask].copy()

print(f"      AI-batch rows : {len(ai_batch_df):,}")
print(f"      Human rows    : {len(human_df):,}")

# --- Save human gold standard ------------------------------------------------

print("[5/7] Saving human_gold_standard.csv ...")
genuine_gold = human_df[
    human_df["annotation_status"].isin(["approved", "corrected"])
].copy()
genuine_gold.to_csv(OUTPUT_GOLD, index=False)
print(f"      {len(genuine_gold)} genuine gold rows saved to: {OUTPUT_GOLD.name}")

# --- Step 5: Merge AI-batch rows with machine annotations --------------------

print("[6/7] Building priority queue ...")

# Merge on candidate_id (left join so all 3720 AI-batch candidates are kept)
merged = ai_batch_df.merge(
    annotations[[
        "candidate_id",
        "machine_claim_class",
        "machine_suggested_labels",
        "machine_uncertainty_flag",
        "machine_context_before",
        "machine_context_after",
    ]],
    on="candidate_id",
    how="left",
)

# -- Parse AI-batch claim_class from review_notes ----------------------------
def parse_ai_claim_class(note: str) -> str:
    """Extract claim_class=VALUE from AI review_notes string."""
    m = re.search(r"claim_class=(\w+)", str(note))
    return m.group(1) if m else "unknown"

merged["ai_batch_labels"]  = merged["suggested_labels"].fillna("")
merged["_ai_claim_class"]  = merged["review_notes"].apply(parse_ai_claim_class)

# -- Priority score computation -----------------------------------------------

def compute_priority(row: pd.Series) -> tuple[int, list[str]]:
    """Return (score, [reason_strings]) for an AI-batch candidate."""
    score   = 0
    reasons: list[str] = []

    msl = str(row.get("machine_suggested_labels", "") or "")

    # RARE_LABEL_BOOST: +3 if machine labels contain Generalizability or Recommendation
    if "Generalizability" in msl or "Recommendation" in msl:
        score += 3
        reasons.append("RARE_LABEL(+3)")

    # SAFETY_BOOST: +2 if labels contain Safety
    if "Safety" in msl:
        score += 2
        reasons.append("SAFETY(+2)")

    # NOVELTY_BOOST: +1 if labels contain Novelty
    if "Novelty" in msl:
        score += 1
        reasons.append("NOVELTY(+1)")

    # DISAGREEMENT_BOOST: +2 if AI batch says claim but machine says non_claim, or vice versa
    ai_cls  = str(row.get("_ai_claim_class", "")).lower()
    mac_cls = str(row.get("machine_claim_class", "")).lower()
    disagreement = (
        (ai_cls == "claim"     and mac_cls == "non_claim") or
        (ai_cls == "non_claim" and mac_cls == "claim")
    )
    if disagreement:
        score += 2
        reasons.append("DISAGREEMENT(+2)")

    # UNCERTAIN_BOOST: +2 if machine_uncertainty_flag == 'yes'
    if str(row.get("machine_uncertainty_flag", "")).lower() == "yes":
        score += 2
        reasons.append("UNCERTAIN(+2)")

    # SHORT_SENTENCE_PENALTY: -1 if sentence < 50 chars
    sentence = str(row.get("sentence", "") or "")
    if len(sentence) < 50:
        score -= 1
        reasons.append("SHORT_SENT(-1)")

    return score, reasons


scores_and_reasons = merged.apply(compute_priority, axis=1)
merged["priority_score"]   = [x[0] for x in scores_and_reasons]
merged["priority_reasons"] = [
    "; ".join(x[1]) if x[1] else "baseline"
    for x in scores_and_reasons
]

# --- Step 6: Proportional per-paper selection (top 200) ----------------------

papers = merged["paper_id"].unique()
n_papers = len(papers)
print(f"      Papers in AI-batch : {n_papers}")

paper_counts = merged["paper_id"].value_counts()

# Quota (Largest Remainder) method with floor = MIN_PER_PAPER
exact   = {p: QUEUE_SIZE * paper_counts[p] / len(merged) for p in papers}
floor_a = {p: max(MIN_PER_PAPER, int(exact[p])) for p in papers}

remainder = QUEUE_SIZE - sum(floor_a.values())

# Sort by fractional part descending; only give extras to papers with room
fracs = sorted(
    [(exact[p] - int(exact[p]), p) for p in papers if floor_a[p] < paper_counts[p]],
    reverse=True,
)
for _, p in fracs:
    if remainder <= 0:
        break
    floor_a[p] += 1
    remainder -= 1

# Fallback if still short (e.g. some papers have fewer rows than MIN_PER_PAPER)
if remainder > 0:
    for p in sorted(papers, key=lambda x: paper_counts[x], reverse=True):
        if floor_a[p] < paper_counts[p] and remainder > 0:
            floor_a[p] += 1
            remainder -= 1

alloc = {p: min(floor_a[p], paper_counts[p]) for p in papers}

# Redistribute any still-unallocated quota caused by paper-size caps
total_alloc = sum(alloc.values())
if total_alloc < QUEUE_SIZE:
    deficit = QUEUE_SIZE - total_alloc
    for p in sorted(papers, key=lambda x: paper_counts[x], reverse=True):
        room = paper_counts[p] - alloc[p]
        take = min(room, deficit)
        alloc[p] += take
        deficit -= take
        if deficit <= 0:
            break

# Select top-N rows per paper sorted by priority_score descending
queue_parts: list[pd.DataFrame] = []
for p in papers:
    paper_rows = (
        merged[merged["paper_id"] == p]
        .sort_values("priority_score", ascending=False)
    )
    queue_parts.append(paper_rows.head(alloc.get(p, MIN_PER_PAPER)))

queue_df = (
    pd.concat(queue_parts, ignore_index=True)
    .sort_values("priority_score", ascending=False)
    .reset_index(drop=True)
)

# --- Step 6b: Compute model scores -------------------------------------------

print("      Computing model scores ...")
if model_loaded and model is not None:
    sentences = queue_df["sentence"].fillna("").tolist()
    try:
        proba_raw = model.predict_proba(sentences)

        # Handle two possible sklearn OvR output formats:
        # 1) np.ndarray shape (n_samples, n_labels)  <- modern sklearn
        # 2) list of n_labels arrays each shape (n_samples, 2)  <- legacy OvR
        if isinstance(proba_raw, list):
            proba = np.column_stack([arr[:, 1] for arr in proba_raw])
        else:
            proba = np.array(proba_raw)
            if proba.ndim == 1:
                proba = proba.reshape(-1, 1)

        for i, col in enumerate(LABEL_SCORE_COLS):
            if i < proba.shape[1]:
                queue_df[col] = np.round(proba[:, i], 4)
            else:
                queue_df[col] = np.nan

    except Exception as exc:
        print(f"      WARNING: predict_proba failed - {exc}")
        for col in LABEL_SCORE_COLS:
            queue_df[col] = np.nan
else:
    for col in LABEL_SCORE_COLS:
        queue_df[col] = np.nan

# --- Step 7: Assemble output columns and save --------------------------------

OUTPUT_COLS = [
    "candidate_id",
    "paper_id",
    "title",
    "source_section",
    "sentence",
    "machine_claim_class",
    "machine_suggested_labels",
    "ai_batch_labels",
    "priority_score",
    "priority_reasons",
    "model_efficacy_score",
    "model_safety_score",
    "model_generalizability_score",
    "model_comparative_score",
    "model_novelty_score",
    "model_recommendation_score",
    "machine_context_before",
    "machine_context_after",
    "source_url",
]

# source_url comes from the 'url' column in candidates
if "url" in queue_df.columns and "source_url" not in queue_df.columns:
    queue_df["source_url"] = queue_df["url"]

# Ensure all output columns exist (fill missing with empty string)
for col in OUTPUT_COLS:
    if col not in queue_df.columns:
        queue_df[col] = ""

output_queue = queue_df[OUTPUT_COLS].copy()
output_queue.to_csv(OUTPUT_QUEUE, index=False)
print(f"      Priority queue saved: {OUTPUT_QUEUE.name}  ({len(output_queue)} rows)")

# --- Summary report ----------------------------------------------------------

print("\n" + "=" * 65)
print("SUMMARY REPORT")
print("=" * 65)

total_candidates = len(candidates)
ai_batch_count   = len(ai_batch_df)
queue_size_out   = len(output_queue)

print(f"\n  Total candidates    : {total_candidates:,}")
print(f"  AI-batch rows       : {ai_batch_count:,}")
print(f"  Human gold rows     : {len(genuine_gold)}")
print(f"  Priority queue size : {queue_size_out}")

print("\n  Per-paper counts in priority queue:")
per_paper = output_queue["paper_id"].value_counts().sort_values(ascending=False)
for paper_id, cnt in per_paper.items():
    print(f"    {paper_id:20s}  {cnt:3d}")

print("\n  Priority reasons distribution:")
all_reasons: dict[str, int] = {}
for reasons_str in output_queue["priority_reasons"]:
    for r in str(reasons_str).split("; "):
        all_reasons[r] = all_reasons.get(r, 0) + 1
for reason, cnt in sorted(all_reasons.items(), key=lambda x: -x[1]):
    print(f"    {reason:30s}  {cnt:3d}")

print("\n  Rare label distribution in queue:")
gen_count = (
    output_queue["machine_suggested_labels"].fillna("")
    .str.contains("Generalizability").sum()
)
rec_count = (
    output_queue["machine_suggested_labels"].fillna("")
    .str.contains("Recommendation").sum()
)
print(f"    Generalizability : {gen_count}")
print(f"    Recommendation   : {rec_count}")

print("\n" + "=" * 65)
print("WHAT MUKESH SHOULD REVIEW AND WHY")
print("=" * 65)
print(f"""
  PRIORITY REVIEW QUEUE  ({queue_size_out} rows)
  Output file: {OUTPUT_QUEUE.name}
  -----------------------------------------------------------------
  These {queue_size_out} rows were selected from {ai_batch_count:,} AI-annotated candidates
  across all {n_papers} papers. They are ranked by a priority score that
  rewards:

    DISAGREEMENT (+2) -- AI batch and machine model disagree on whether
                         the sentence is a claim vs. non_claim. These are
                         the most valuable rows for calibrating the model.

    UNCERTAINTY  (+2) -- Machine flagged the row as uncertain. Genuine
                         human judgment is needed to resolve.

    RARE LABELS  (+3) -- Sentence may be a Generalizability or
                         Recommendation claim (very underrepresented in
                         training data -- critical for model balance).

    SAFETY       (+2) -- Safety claims need careful verification given
                         clinical stakes.

    NOVELTY      (+1) -- Novelty claims add diversity to the test set.

  Each paper contributes at least {MIN_PER_PAPER} rows so the test set spans all
  {n_papers} RCTs uniformly.

  HUMAN GOLD STANDARD  ({len(genuine_gold)} rows)
  Output file: {OUTPUT_GOLD.name}
  -----------------------------------------------------------------
  These {len(genuine_gold)} rows were reviewed and approved/corrected by a human
  annotator. They form the foundation of the genuine gold standard
  test set and should NOT be modified without strong cause.

  RECOMMENDED WORKFLOW
  -----------------------------------------------------------------
  1. Open priority_review_queue.csv (sorted by priority_score desc).
  2. For each row, read: sentence, machine_context_before/after,
     source_url, and machine_suggested_labels.
  3. Decide:
       - Is this a claim? (yes / no)
       - Which label(s) apply?
           Efficacy | Safety | Generalizability
           Comparative | Novelty | Recommendation
  4. Update annotation_status and verified_labels in candidates.csv.
  5. After all {queue_size_out} are reviewed, re-run model training to incorporate
     the new gold examples.
""")
print("=" * 65)
print("Done.")
