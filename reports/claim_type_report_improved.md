# Improved Classifier Evaluation Report

**Model:** `claim_type_model_improved`
**Evaluation target:** `agreement_with_ai_generated_batch_labels`
**Trained at:** 2026-10-20T09:03:38.367353+00:00

> [!IMPORTANT]
> Evaluated on 5 holdout test papers (PMC12381666, PMC12446902, PMC12662812, PMC12743696, PMC12815918) comprising 552 examples, of which 549 (99.5%) are AI-batch heuristic annotations and 3 (0.5%) are genuine human annotations. Metrics reflect model agreement with AI heuristics, NOT human ground truth.

## Overall Metrics (Holdout Test Set)

- **Dataset Rows:** 2375 (1823 train / 552 test)
- **Papers:** 22 total (17 train / 5 test)
- **Exact-Match Agreement:** 0.6105
- **Micro F1:** 0.7838
- **Macro F1:** 0.6627
- **Precision (Micro):** 0.7618
- **Recall (Micro):** 0.8070
- **Hamming Loss:** 0.0885

## 5-Fold GroupKFold Cross-Validation (Training Set Only)

- **CV Micro F1:** 0.7382 +/- 0.0392
- **CV Macro F1:** 0.5805 +/- 0.0171

## Per-Category Results (Holdout Test with Tuned Thresholds)

| Category | Optimal Threshold | Precision | Recall | F1 | Test Support | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Efficacy | 0.45 | 0.7940 | 0.9147 | 0.8501 | 375 | 88 | 89 | 32 | 343 |
| Safety | 0.45 | 0.7945 | 0.5979 | 0.6824 | 97 | 440 | 15 | 39 | 58 |
| Generalizability | 0.45 | 0.4545 | 0.2174 | 0.2941 | 23 | 523 | 6 | 18 | 5 |
| Comparative | 0.40 | 0.6697 | 0.8202 | 0.7374 | 89 | 427 | 36 | 16 | 73 |
| Novelty | 0.45 | 0.8378 | 0.7561 | 0.7949 | 41 | 505 | 6 | 10 | 31 |
| Recommendation | 0.40 | 0.6000 | 0.6364 | 0.6176 | 33 | 505 | 14 | 12 | 21 |

## Reproduction Details

- **Split:** paper_grouped_GroupShuffleSplit; random seed: `42`
- **Training Papers (17):** PMC12486974, PMC12543108, PMC12550449, PMC12551264, PMC12581794, PMC12647125, PMC12676765, PMC12678040, PMC12690432, PMC12725555, PMC12783969, PMC12825039, PMC13054461, PMC13065175, PMC13220170, PMC13618946, PMC7618564
- **Test Papers (5):** PMC12381666, PMC12446902, PMC12662812, PMC12743696, PMC12815918
- **Dataset SHA-256:** `684d92afc88f853afde7f03717fd078f23945186cbc273a72239bd4c5102a969`
- **Python Version:** 3.13.14 (tags/v3.13.14:fd17997, Jun 10 2026, 13:03:48) [MSC v.1944 64 bit (AMD64)]
- **Scikit-Learn Version:** 1.9.1
