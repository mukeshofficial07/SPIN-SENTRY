# Machine annotation run summary

- Source candidate rows: **3737** (3733 pending processed; 4 non-pending rows left untouched).
- Pending machine suggestions generated: **3733**; machine-only output contains **3736** rows including preserved suggestions for previously reviewed candidates.
- Claim-status estimates: **1342 claim**, **890 likely non-claim**, **1501 uncertain**.
- Uncertainty flagged: **2534** rows; possible extraction damage flagged: **31** rows.
- Rows with at least one machine-suggested category: **946**.

## Suggested category counts

Counts are overlapping candidate suggestions, not verified labels or performance metrics.

| Category | Rows suggested | Representative review sample |
|---|---:|---:|
| Efficacy | 744 | 10 / 10 |
| Safety | 146 | 10 / 10 |
| Generalizability | 19 | 10 / 10 |
| Comparative | 196 | 10 / 10 |
| Novelty | 15 | 10 / 10 |
| Recommendation | 35 | 10 / 10 |

## Review order

1. Start with `machine_review_batch.csv`; it contains the first 40-candidate mixed review batch.
2. Continue through `machine_review_queue_current.csv`, which orders prioritized sections first and retains all remaining pending candidates.

`machine_category_sample_current.csv` contains the selected category-coverage sample, and `machine_review_batch.csv` contains the proposed human-review batch. Machine output files omit human-verification fields; the source candidate file was not changed by this run.

## Human review steps

1. Start the existing app from the project directory with `venv\Scripts\python.exe -m streamlit run scripts/review_claim_annotations.py`.
2. Choose a batch size of 20, 30, 40, or 50. The first batch prioritizes six-category coverage and audit cases; later batches follow the prioritized queue. Saved decisions are persisted in `candidates.csv` and drop out of future sessions' pending queue.
3. Check the original sentence against its source link. Machine labels stay unverified; use Approve or Correct only after checking the source and choosing human-verified labels. Mark uncertain or Reject when appropriate.
4. The machine export is separate and labelled UNVERIFIED. The human-reviewed training export contains only approved/corrected rows and remains separate.

## Method and limits

System: **SPIN-SENTRY transparent section/context rules v1**. This run uses transparent phrase/structure rules, existing source-section headings, and adjacent sentences from the same paper and section. Category labels require contextual patterns; existing keyword-only hints are not copied when the rules do not confirm them. The old keyword suggestions remain unchanged in their original columns.

This is not a trained claim classifier or a language-model judgment. There are no confidence scores or calibrated probabilities. The uncertainty flag is a rule-triggered review signal, not model confidence. Rationale strings identify the matched cues and section/context used; they may be incomplete or wrong.

Claim/non-claim and category counts are machine estimates, not scientific truth. Background assertions can be substantive claims; procedural sentences can contain claim-like words; section headings may be coarse. Every suggested label remains unverified. Do not train or evaluate on this file.

Extraction damage flags are heuristic. They include existing extraction warnings, replacement characters, likely split-token artifacts, and table/form-like fragments. They do not prove that a sentence is damaged; inspect flagged text against its source.

The only human annotations remain in the original candidate file. Suggested CSVs are for triage; record approval/correction through the existing review workflow after checking the paper source.
