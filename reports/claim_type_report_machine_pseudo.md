# Classifier evaluation report

**Evaluation target:** agreement_with_machine_generated_labels.
These figures measure agreement with the labels in the stated dataset; machine pseudo-label agreement is not human-label accuracy.

- Dataset examples: 649 (457 train / 192 test)
- Paper groups: 21 total (15 train / 6 test)
- Multi-label examples: 107
- Exact-match agreement: 0.6250
- Micro F1: 0.7294; macro F1: 0.5147; Hamming loss: 0.0998

## Per-class results

| Category | Precision | Recall | F1 | Test support | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Efficacy | 0.7574 | 0.9552 | 0.8449 | 134 | 17 | 41 | 6 | 128 |
| Safety | 0.8571 | 0.1500 | 0.2553 | 40 | 151 | 1 | 34 | 6 |
| Generalizability | 1.0000 | 0.5000 | 0.6667 | 2 | 190 | 0 | 1 | 1 |
| Comparative | 0.8182 | 0.5455 | 0.6545 | 33 | 155 | 4 | 15 | 18 |
| Novelty | 0.0000 | 0.0000 | 0.0000 | 10 | 181 | 1 | 10 | 0 |
| Recommendation | 1.0000 | 0.5000 | 0.6667 | 4 | 188 | 0 | 2 | 2 |

## Reproduction details

- Split: `paper_grouped`; random seed: `42`.
- Training paper IDs: PMC12486974, PMC12543108, PMC12550449, PMC12581794, PMC12647125, PMC12676765, PMC12678040, PMC12690432, PMC12725555, PMC12743696, PMC12783969, PMC12825039, PMC13065175, PMC13220170, PMC7618564
- Test paper IDs: PMC12381666, PMC12446902, PMC12551264, PMC12662812, PMC12815918, PMC13054461
- Dataset SHA-256: `5d7e9001298cade309acc53cc11dc3cc42f0822d1faa08ca1a34a1bc69e16866`
