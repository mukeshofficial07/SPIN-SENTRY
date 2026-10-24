# Machine annotation run summary

- Source candidate rows: **3737** (3736 pending processed; 1 non-pending rows left untouched).
- Machine-suggestion rows written: **3736**. Original text/provenance columns were copied unchanged.
- Claim-status estimates: **1344 claim**, **890 likely non-claim**, **1502 uncertain**.
- Uncertainty flagged: **2536** rows; possible extraction damage flagged: **31** rows.
- Rows with at least one machine-suggested category: **948**.

## Suggested category counts

Counts are overlapping candidate suggestions, not verified labels or performance metrics.

| Category | Rows suggested | Representative review sample |
|---|---:|---:|
| Efficacy | 744 | 10 / 10 |
| Safety | 146 | 10 / 10 |
| Generalizability | 20 | 10 / 10 |
| Comparative | 196 | 10 / 10 |
| Novelty | 16 | 10 / 10 |
| Recommendation | 35 | 10 / 10 |

## Review order

1. `machine_review_queue.csv` priority 1: uncertain classification, context-dependent category assignment, conflicting cues, or possible extraction damage.
2. Priority 2: deterministic representative sample, targeting the requested number from each machine-suggested category and spreading across papers where possible.
3. Priority 3: remaining pending candidates.

`machine_category_sample.csv` contains the selected category-coverage sample. Rows may represent more than one category. Machine output is stored separately; source `verified_labels`, `verified_sentence`, and human review status are never changed.

## Human review steps

1. Start the existing app with `venv\Scripts\python.exe -m streamlit run scripts/review_claim_annotations.py`.
2. Leave Review status on `Needs review`; choose `Uncertain or extraction-damaged` first, then `Category coverage sample` to inspect a balanced set across all six suggested categories. `Priority order` places flagged candidates first. Filter by paper when checking the source.
3. Compare each exact sentence with the cited paper. Use the adjacent same-section context, DOI/source link, and license shown with the candidate. Treat the machine rationale and labels as suggestions only. Correct the sentence if extraction differs from the source; choose Approve, Correct, Uncertain, or Reject and add your reviewer name.
4. To approve/correct, mark source text checked and select the labels you verified. The optional checkbox can copy machine suggestions into the editable human-label selector; nothing is verified until you save a decision. Download the approved annotations CSV from the bottom of the page after review.

## Method and limits

System: **SPIN-SENTRY transparent section/context rules v1**. This run uses transparent phrase/structure rules, existing source-section headings, and adjacent sentences from the same paper and section. Category labels require contextual patterns; existing keyword-only hints are not copied when the rules do not confirm them. The old keyword suggestions remain unchanged in their original columns.

This is not a trained claim classifier or a language-model judgment. There are no confidence scores or calibrated probabilities. The uncertainty flag is a rule-triggered review signal, not model confidence. Rationale strings identify the matched cues and section/context used; they may be incomplete or wrong.

Claim/non-claim and category counts are machine estimates, not scientific truth. Background assertions can be substantive claims; procedural sentences can contain claim-like words; section headings may be coarse. Every suggested label remains unverified. Do not train or evaluate on this file.

Extraction damage flags are heuristic. They include existing extraction warnings, replacement characters, likely split-token artifacts, and table/form-like fragments. They do not prove that a sentence is damaged; inspect flagged text against its source.

The only human annotations remain in the original candidate file. Suggested CSVs are for triage; record approval/correction through the existing review workflow after checking the paper source.
