# SPIN-SENTRY
### The Tri-Review Architecture: Forward Alignment Pipeline (Stages 1–4)

A Python + Streamlit research prototype for **intra-document spin detection** in
clinical-trial papers. Detects abstract claims that may overstate findings relative
to the paper's own Results, Harms, Limitations, and Methods sections.

---

## Quick Start

### 1. Launch Applications

```bash
# Main Clinical Research Intelligence App (Port 8501)
.\venv\Scripts\python.exe -m streamlit run app.py --server.port 8501

# Interactive Human Annotation Review Tool (Port 8503)
.\venv\Scripts\python.exe -m streamlit run scripts/review_claim_annotations.py --server.port 8503
```

On first run, the NLI model (`cross-encoder/nli-MiniLM2-L6-H768`, ~90 MB) will be downloaded automatically from HuggingFace Hub.

---

## Run Tests

The repository maintains a full regression test suite (222 tests passing):

```bash
# Run entire test suite
.\venv\Scripts\python.exe -m pytest tests/ -v

# Individual test suites
.\venv\Scripts\python.exe -m pytest tests/test_stage1.py -v                # Stage 1: HCP-BEA Claim Parser
.\venv\Scripts\python.exe -m pytest tests/test_stage2.py -v                # Stage 2: MZEM Zone Mapper
.\venv\Scripts\python.exe -m pytest tests/test_stage3.py -v                # Stage 3: DC-NLI Directional Entailment
.\venv\Scripts\python.exe -m pytest tests/test_stage4.py -v                # Stage 4: ESWE Weighted Entailment
.\venv\Scripts\python.exe -m pytest tests/test_pipeline_integration.py -v  # End-to-End Pipeline
.\venv\Scripts\python.exe -m pytest tests/test_fixed_pagination.py -v      # Deterministic Review Pagination
.\venv\Scripts\python.exe -m pytest tests/test_human_gold_standard_export.py -v # Gold Standard Integrity
```

---

## Human-Reviewed Gold Standard & Provenance

The project maintains strict provenance standards: **AI heuristics are never auto-converted to human ground truth.**

### Corpus Statistics:
- **Total Sentence Candidates**: 3,737 across 22 PubMed Central open-access clinical trial papers.
- **Genuine Human Reviews**: 765 candidate decisions recorded by human reviewer Mukesh.
- **Source-Verified Positive Claims (Human Gold Standard)**: **197 unique claims** (`human_gold_standard.csv`).
- **Verified Non-Claims (Rejected Candidates)**: **336 sentences**.
- **Pending Candidates**: 2,972 unreviewed sentences.
- **Audit Manifest**: `data/research_dataset/human_gold_standard_manifest.json`.

### Gold-Standard Claim Label Distribution:
- **Efficacy**: 150 sentences (✅ Sufficient, target ≥ 30)
- **Comparative**: 63 sentences (✅ Sufficient, target ≥ 30)
- **Safety**: 33 sentences (✅ Sufficient, target ≥ 30)
- **Generalizability**: 16 sentences (⚠️ Provisional, target ≥ 30)
- **Recommendation**: 15 sentences (⚠️ Provisional, target ≥ 30)
- **Novelty**: 8 sentences (❌ Scarce, target ≥ 30)

---

## Stage 1 Multi-Label Claim Classification

The pipeline supports both rule-based keyword lexicon classification and trained TF-IDF multi-label models (`claim_type_model_improved.joblib` with tuned per-category decision thresholds).

### Evaluation against Genuine Human Gold Standard (N=197):
- **Macro-F1**: `0.7303` across all 6 claim categories
- **Micro-F1**: `0.7826`
- **Exact Match Ratio**: `46.7%`
- **Hamming Loss**: `0.1066`

### Generalization on 5 Unseen Holdout Papers (N=49):
- **Macro-F1**: `0.7173`
- **Micro-F1**: `0.7857`
- **Exact Match**: `48.98%`
- **Hamming Loss**: `0.1020`

To re-run honest model evaluation:
```powershell
# Evaluate on human gold standard
.\venv\Scripts\python.exe scripts/evaluate_classifier.py --model improved --all
```

---

## Train an optional Stage 1 claim classifier

The default classifier remains the existing keyword/rule baseline. To train a
real TF-IDF + Logistic Regression multi-label model, first have people annotate
sentences in `data/claim_annotations_template.csv`. Each row represents one
sentence; `paper_id` identifies its source paper, and `labels` contains one or
more exact category names separated by semicolons. The six allowed labels are
Efficacy, Safety, Generalizability, Comparative, Novelty, and Recommendation.
Annotators should label from the sentence and its context, record the source and
their name, and resolve disagreements before treating labels as gold data. The
template starts empty on purpose; the project does not ship fabricated labels.

Validate the filled CSV before training:

```powershell
.\venv\Scripts\python.exe scripts/validate_claim_annotations.py data/claim_annotations_template.csv
```

Then train and evaluate:

```powershell
.\venv\Scripts\python.exe scripts/train_claim_classifier.py --data data/claim_annotations_template.csv
```

Evaluation uses a paper-grouped split, so sentences from one paper cannot land
in both train and test sets. The script requires all six labels in both groups;
collect examples for every category from several papers before training. It
reports exact-match accuracy, micro/macro F1, precision, recall, Hamming loss,
and per-label F1. Only a successful fit writes
`models/claim_type_model.joblib` and `reports/claim_type_metrics.json`. In the
app's Stage 1 classifier setting, select **Trained TF-IDF model** after training;
the rule-based baseline stays available by default. Training infrastructure is
included. The separately named experimental model below was trained on
filtered machine-generated pseudo-labels; it is not a human-annotated model,
and its evaluation measures agreement with those machine suggestions. The
human-verified training workflow remains separate.

### Train the experimental machine-pseudo-label model

This fast-start path uses existing machine suggestions without relabelling them
as human verified. Preparation creates a separate dataset and exclusion ledger,
preserves source provenance, and excludes likely non-claims, uncertain or
ambiguous rows, damaged extraction, and missing or invalid labels. It does not
change review decisions or the original suggestions file.

From the project directory, run:

```powershell
.\venv\Scripts\python.exe scripts\prepare_machine_pseudo_labels.py
.\venv\Scripts\python.exe scripts\train_claim_classifier.py --data data\research_dataset\machine_pseudo_labels.csv --label-source machine_pseudo_label
.\venv\Scripts\python.exe -m streamlit run app.py
```

Prepared rows carry `label_source=machine_generated_pseudo_label`. Multiple
categories are retained using One-vs-Rest Logistic Regression. Evaluation uses
a paper-grouped split and reports **agreement with machine-generated labels**;
it is not accuracy against human judgments. The current run used 649 examples
from 21 papers (457 train examples across 15 papers; 192 test examples across
6 papers). Exact-match agreement was 0.6250, micro F1 0.7294, macro F1 0.5147,
and Hamming loss 0.0998. Per-class metrics and confusion matrices are in
`reports/claim_type_report_machine_pseudo.md` and
`reports/claim_type_metrics_machine_pseudo.json`.

The model and companion artifacts are saved under `models/` with the
`_machine_pseudo` suffix. Existing artifacts at those paths are copied to
timestamped backups before a new training run overwrites them. The rule-based
baseline remains the app default; choose **Experimental TF-IDF model (machine
pseudo-labels)** in the Stage 1 classifier control to use this model. Launch
the annotation review app separately with:

```powershell
.\venv\Scripts\python.exe -m streamlit run scripts\review_claim_annotations.py
```

No machine suggestion is marked human verified by this workflow. For a
human-verified model, review sentences in the annotation app, export the
reviewed annotations, validate that export, and use the human-verified training
command above. Do not describe pseudo-label agreement as human accuracy.

## Build a manually verified research dataset

The sentence-review workflow is separate from classifier training:

1. Build the sample-paper pilot from the credited PDF (abstract sentences only):

   ```powershell
   .\venv\Scripts\python.exe scripts/claim_dataset.py --pilot-sample
   ```

   This writes source/license records to `data/research_dataset/papers.csv`
   and extracted candidates to `data/research_dataset/candidates.csv`. The
   paper record includes its DOI, URL, license, local file, and SHA-256. Each
   candidate preserves the paper ID, source section, extracted sentence,
   suggested labels and raw keyword-hit counts, and a separate blank human
   label/status field. Re-running the pilot refreshes pending rows, but will not
   replace any row that has been reviewed.

2. Add another article by its **PMC ID**, for example
   `PMC12345678` (choose a clinical research article whose reuse terms allow
   reuse):

   ```powershell
   .\venv\Scripts\python.exe scripts/claim_dataset.py --pmcid PMC12345678
   ```

   This fetches only that article's full-text XML through the Europe PMC REST
   fullTextXML endpoint. It requires an explicit CC0, CC BY, or CC BY-SA license
   in the article XML and stops without importing text if it cannot verify an
   accepted license. The script tracks DOI/ID/content-hash duplicates. It does
   not bulk-download or scrape the PMC website. License terms vary by article;
   the article's own license remains the authority.

3. Open the annotation screen:

   ```powershell
   .\venv\Scripts\python.exe -m streamlit run scripts/review_claim_annotations.py
   ```

   Review one sentence at a time. Compare it with the cited PDF/XML, fix any
   extraction error in **Verified sentence text**, choose **Approve** or
   **Correct**, and select the labels you confirm. **Uncertain** and **Reject**
   remain recorded in the candidate file but never enter the model-training
   export. Suggested labels are displayed separately and are never treated as
   human labels unless you explicitly start from them and save an approval.

4. Export only approved/corrected, source-checked annotations:

   ```powershell
   .\venv\Scripts\python.exe scripts/export_claim_annotations.py
   ```

   This creates `data/claim_annotations_reviewed.csv` with the existing exact
   six-column training schema: `paper_id,sentence,labels,source,annotator,notes`.
   The DOI/URL is in `source`; `notes` retains section and license provenance.
   You can also download the same CSV from the review screen. The exporter
   refuses to run if there are no approved human annotations.

5. Validate, then train only on that reviewed export:

   ```powershell
   .\venv\Scripts\python.exe scripts/validate_claim_annotations.py data/claim_annotations_reviewed.csv
   .\venv\Scripts\python.exe scripts/train_claim_classifier.py --data data/claim_annotations_reviewed.csv
   ```

   Training still needs enough examples from multiple papers to place all six
   labels in both paper-grouped partitions. The single-paper pilot cannot meet
   that requirement, and is not a training/evaluation set yet.

The sample extraction currently has a few visible PDF layout artifacts (for
example, an extra `u` in one sentence). These candidates are flagged for source
comparison; correct the text in the review screen before approving them. The
unrelated `extracted_paper.txt` is not used as a source. Synthetic examples are
not mixed into the real-paper candidate or reviewed-annotation files.

---

## Project Structure

```
SPIN-SENTRY/
├── app.py                          # Streamlit UI
├── pipeline.py                     # Orchestrator (Stages 1–4)
├── stages/
│   ├── stage1_claim_parser.py      # HCP-BEA
│   ├── stage2_zone_mapper.py       # MZEM
│   ├── stage3_nli.py               # DC-NLI
│   └── stage4_eswe.py              # ESWE
├── utils/
│   ├── pdf_reader.py               # PDF → text extraction
│   ├── ner_utils.py                # Entity anchoring
│   ├── hedging_scorer.py           # Hedging language scorer
│   └── esi_table.py                # ESI lookup + zone-type relevance
├── data/
│   ├── claim_type_keywords.json    # Keyword seeds per claim type
│   ├── hedging_lexicon.json        # Hedging word weights
│   ├── zone_headers.json           # Section header patterns
│   └── sample_paper.pdf            # Demo PDF (see below)
├── tests/                          # pytest test suite
├── requirements.txt
└── README.md
```

---

## Four-Stage Architecture

Based exactly on paper Section III and Figure 1.

### Stage 1 — HCP-BEA: Hybrid Claim Parser with Biomedical Entity Anchoring
- Segments abstract into sentences
- NER: anchors each sentence to drugs, outcomes, populations
- Classifies each sentence into ≥1 of **6 claim types**:
  Efficacy · Safety · Generalizability · Comparative · Novelty · Recommendation
- Scores **hedging language** (continuous strength value)
- Routes claims to target zone: Safety → Harms; Efficacy → Results; Generalizability → Methods

### Stage 2 — MZEM: Multi-Zone Evidence Mapper
- Partitions full text into **4 evidence zones**:
  Results · Harms · Limitations · Methods
- **Harms zone isolated explicitly** (paper's key design decision)
- Handles non-standard headers ("Tolerability", "Safety Outcomes") via fuzzy matching
- Keyword fallback for papers with no detectable headers

### Stage 3 — DC-NLI: Directional Claim–Evidence NLI
Implements paper Equations (1)–(3):
```
forward = entailment(e, c)   # evidence supports claim
reverse = entailment(c, e)   # claim entails evidence (generalizability check)
gap     = 1 − reverse        # measures generalizability stretch
```

### Stage 4 — ESWE: Evidence-Strength Weighted Entailment
Implements paper Equation (4):
```
WE(c, e) = NLI(c, e) × ESI(e) × relevance(type, zone)
```

ESI from paper Table I:

| Study Design | Sample Size | Analysis Type | ESI |
|---|---|---|---|
| RCT | >1000 | Intention-to-treat | 1.0 |
| RCT | <100 | Per-protocol | 0.7 |
| Observational | >5000 | Adjusted | 0.6 |
| Case series | N<50 | Uncontrolled | 0.2 |

Spin flag: **WE < 0.5** → claim flagged (engineering threshold, not validated).

---

## Sample Paper

A CC-licensed clinical trial PDF from PubMed Central Open Access is used for demo.
See `data/SAMPLE_PAPER_CREDIT.md` for full citation, PMCID, DOI, and licence.

---

## Honest Implementation Notes

| Paper Feature | MVP Implementation | Label in UI |
|---|---|---|
| BioBERT tokenization | `nltk.sent_tokenize` | "rule-based" |
| Fine-tuned claim classifier | Keyword lexicon scorer | "approximation" |
| SciBERT section classifier | Regex + fuzzy matching | "approximation" |
| BioBERT NER | spaCy `en_core_web_sm` | "general NER" |
| NLI (forward + reverse) | `cross-encoder/nli-MiniLM2-L6-H768` | **actual NLI** |
| ESI weighting | Rule-based Table I lookup | Exact from paper |

**Paper metrics (89% accuracy, F1=0.93, 81% Precision@5) are from the authors'
evaluation on their annotated corpus. They are NOT reproduced or claimed by this
implementation.**

---

## Known Limitations

1. **PDF extraction**: `pdfplumber` works on most text-based PDFs; scanned PDFs will fail.
2. **Abstract detection**: Heuristic only; unusual layouts may split incorrectly.
3. **Zone detection**: Papers with merged sections or no standard headers degrade accuracy.
4. **NLI model**: General-domain model; may not reliably detect contradictions in biomedical text.
5. **Evaluation corpus size**: The human gold standard contains 197 positive claim sentences across 22 papers. While Efficacy, Comparative, and Safety meet statistical thresholds (≥30), Generalizability (16), Recommendation (15), and Novelty (8) remain provisional.
6. **Screening tool, not clinical verdict**: Automated spin flags (WE < 0.50) indicate potential discrepancies between abstract claims and trial evidence for human review; they do not constitute proof of research spin or scientific misconduct.

---

## Environment

- Python 3.13.14 (Windows)
- torch (CPU-only)
- sentence-transformers 3.x
- spaCy 3.x + en_core_web_sm
- streamlit, pdfplumber, nltk, rapidfuzz, pandas, pytest
