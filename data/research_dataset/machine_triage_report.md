# Pending candidate triage report

- Candidates triaged: **3733** pending rows.
- Likely substantive claims: **1342**; likely non-claims: **890**; uncertain: **1501**.
- Possible extraction damage: **31** rows (a review flag, not a confirmed defect).
- All candidates remain in the source CSV; section groups only affect order and sample selection.

## Section priority coverage

| Normalized source section | Pending candidates |
|---|---:|
| Safety/Harms | 61 |
| Results/Outcomes | 361 |
| Limitations | 97 |
| Discussion | 492 |
| Other | 2722 |

## Machine category suggestions

Counts overlap and remain unverified.

| Category | Pending suggestions |
|---|---:|
| Efficacy | 744 |
| Safety | 146 |
| Generalizability | 19 |
| Comparative | 196 |
| Novelty | 15 |
| Recommendation | 35 |

## Proposed human-review batch

Batch size: **40**, spanning **22 papers**; category rows use a global preference for different papers and skip near-duplicate sentences within a paper.

| Batch stratum | Candidates |
|---|---:|
| category:Comparative | 5 |
| category:Efficacy | 5 |
| category:Generalizability | 5 |
| category:Novelty | 5 |
| category:Recommendation | 5 |
| category:Safety | 5 |
| likely_non_claim | 3 |
| possible_extraction_damage | 4 |
| uncertain_priority_section | 3 |

This batch is for human review only. Saving decisions updates the original annotation fields; machine suggestions do not become verified automatically.
