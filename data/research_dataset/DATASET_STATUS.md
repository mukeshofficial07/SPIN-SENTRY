# SPIN-SENTRY research candidate dataset status

Generated from `papers.csv` and `candidates.csv`. This is a candidate corpus, not a human-verified training dataset.

## Current counts

- Registered, license-checked papers: **22** (21 retrieved from the Europe PMC fullTextXML API and 1 credited local pilot paper).
- Unique sentence candidates after within-paper deduplication: **3737**.
- Duplicate sentence rows removed: **25**. Repeated source section headings were merged into the retained candidate row; no sentence text was manually rewritten.
- Annotation status: **3737 pending**, 0 other statuses.
- Human-verified labels: **0**. No classifier training or evaluation was performed; no model metrics exist.
- Candidate suggestions are present on 1708 rows; 2029 rows have no keyword suggestion. 523 rows have more than one suggested category.

## License and source records

License evidence was read from each article’s Europe PMC JATS full text. No restricted full text was downloaded. License summary: **20 CC BY 4.0**; **2 CC BY (version not stated in article XML)**.
For the two records without a license link or version in their JATS statement, the license is recorded as **CC BY (version not stated in article XML)**, and `license_url` remains blank rather than inventing a canonical versioned URL. Their article statements are:
- PMC12690432: “This is an open access article distributed under the terms of the CC-BY License.”
- PMC12725555: “This is an open access article distributed under the terms of the Creative Commons Attribution License, which permits unrestricted use, distribution, and reproduction in any medium, provided the original author and source are credited.”
The local pilot license is backed by `data/SAMPLE_PAPER_CREDIT.md`. `paper_id` preserves the PMCID; `doi` stores DOI separately; `url` points to the Europe PMC article record.

| Paper ID | Title | DOI | License | Source URL | Candidates |
|---|---|---|---|---|---:|
| PMC13618946 | Effectiveness of elastic therapeutic taping in pain management after median sternotomy: A randomized controlled trial | 10.1371/journal.pone.0359282 | CC BY 4.0 | https://europepmc.org/articles/PMC13618946 | 14 |
| PMC12676765 | Effectiveness of implementing family involvement on patient outcomes in individuals with psychotic disorders: a pragmatic cluster randomised controlled trial | 10.1186/s12888-025-07501-z | CC BY 4.0 | https://europepmc.org/articles/PMC12676765 | 327 |
| PMC12743696 | Transauricular Vagus Nerve Stimulation in Acute Ischaemic Stroke Requiring Mechanical Thrombectomy: Sham-Controlled, Randomised Device Trial | 10.1007/s12975-025-01404-7 | CC BY 4.0 | https://europepmc.org/articles/PMC12743696 | 179 |
| PMC12551264 | Safety and efficacy of inhaled PEG-ADM in ARDS patients: a randomised controlled trial | 10.1186/s13054-025-05617-y | CC BY 4.0 | https://europepmc.org/articles/PMC12551264 | 152 |
| PMC12815918 | The efficacy of genetic counselling for familial colorectal cancer. A randomised clinical trial | 10.1038/s41431-025-01921-x | CC BY 4.0 | https://europepmc.org/articles/PMC12815918 | 172 |
| PMC13054461 | A Randomized, Placebo‐Controlled Trial of Hydroxychloroquine in Incomplete Lupus | 10.1002/art.43391 | CC BY 4.0 | https://europepmc.org/articles/PMC13054461 | 199 |
| PMC12783969 | Inhaled Ciclesonide for Community‐Based COVID‐19: A Placebo‐Controlled Randomised Trial | 10.1111/resp.70125 | CC BY 4.0 | https://europepmc.org/articles/PMC12783969 | 180 |
| PMC12446902 | Adverse drug reaction reporting with the Med Safety app in Uganda: a cluster-randomised, controlled trial | 10.1016/S2214-109X(25)00299-2 | CC BY 4.0 | https://europepmc.org/articles/PMC12446902 | 190 |
| PMC13065175 | Immunosuppression in Older Kidney Transplant Recipients | 10.1681/ASN.0000000924 | CC BY 4.0 | https://europepmc.org/articles/PMC13065175 | 153 |
| PMC12381666 | Poststroke Cardiorespiratory Exercise for Brain Volume and Cognition | 10.1001/jamanetworkopen.2025.28907 | CC BY 4.0 | https://europepmc.org/articles/PMC12381666 | 151 |
| PMC7618564 | Randomized Trial of High Dose Oral Rifampicin in Adult Tuberculous Meningitis | 10.1056/NEJMoa2502866 | CC BY 4.0 | https://europepmc.org/articles/PMC7618564 | 111 |
| PMC12678040 | Exercise in Pregnancy and Risk of Postpartum Depression: A Randomised Controlled Trial | 10.1111/1471-0528.70010 | CC BY 4.0 | https://europepmc.org/articles/PMC12678040 | 122 |
| PMC12581794 | Communication About Sexuality (COSY) Encourages Physical Activity in COPD: A Randomised Trial | 10.2147/COPD.S539514 | CC BY 4.0 | https://europepmc.org/articles/PMC12581794 | 159 |
| PMC13220170 | Behavioral Interventions Improve Mailed Colorectal Cancer Screening Among Overdue Patients in a Randomized Trial | 10.1016/j.cgh.2025.09.015 | CC BY 4.0 | https://europepmc.org/articles/PMC13220170 | 116 |
| PMC12662812 | Effect of Health Model‐Based Menopausal Self‐Care Training on Quality of Life and Health: A Randomised Controlled Trial | 10.1111/hex.70511 | CC BY 4.0 | https://europepmc.org/articles/PMC12662812 | 195 |
| PMC12550449 | A Multifaceted Digital Intervention for the Prevention of Type 2 Diabetes Mellitus in Primary Care (PREDIABETEXT): Cluster Randomized Trial | 10.2196/70981 | CC BY 4.0 | https://europepmc.org/articles/PMC12550449 | 301 |
| PMC12543108 | Pragmatic, quasi-experimental, pseudo-randomized clinical trial to assess the impact of patient safety monitors on clinical and patient safety outcomes: The Akershus Clinical Trial (ACT) 1 | 10.1371/journal.pone.0335052.r005 | CC BY 4.0 | https://europepmc.org/articles/PMC12543108 | 150 |
| PMC12486974 | The SUGAR handshake intervention to prevent hypoglycaemia in elderly people with type 2 diabetes: process evaluation within a pragmatic randomised controlled trial | 10.1186/s12877-025-06361-2 | CC BY 4.0 | https://europepmc.org/articles/PMC12486974 | 231 |
| PMC12825039 | Investigating the effectiveness of mobilisation alarms to prevent hospital falls using disinvestment: A randomised clinical trial | 10.1016/j.ijnurstu.2025.105320 | CC BY 4.0 | https://europepmc.org/articles/PMC12825039 | 147 |
| PMC12647125 | Strength training intervention for hybrid workers: a randomised pilot feasibility trial | 10.1038/s41598-025-27567-9 | CC BY 4.0 | https://europepmc.org/articles/PMC12647125 | 174 |
| PMC12690432 | Combined Gastric Electrical Stimulation and Pyloroplasty in Gastroparesis | 10.1001/jamanetworkopen.2025.46332 | CC BY (version not stated in article XML) | https://europepmc.org/articles/PMC12690432 | 131 |
| PMC12725555 | The use of carbogen for interruption of febrile seizures - the randomized controlled CARDIF trial | 10.1371/journal.pone.0324422.r006 | CC BY (version not stated in article XML) | https://europepmc.org/articles/PMC12725555 | 183 |

## Section coverage

Coverage is grouped by the original source heading; the exact source heading remains in `candidates.csv`.

| Section group | Candidate sentences | Papers represented |
|---|---:|---:|
| Abstract | 366 | 22 |
| Results / outcomes | 340 | 17 |
| Discussion | 492 | 16 |
| Safety / harms | 49 | 6 |
| Limitations | 99 | 8 |
| Other source sections | 2391 | 21 |

## Automatically suggested categories

Lexicon-generated hints only. These are not human annotations or ground truth.

| Suggested category | Candidate rows |
|---|---:|---:|
| Efficacy | 740 |
| Safety | 491 |
| Generalizability | 263 |
| Comparative | 554 |
| Novelty | 143 |
| Recommendation | 133 |

## Import and review notes

- Two full-text license statements initially lacked versioned links. They were included only after confirming their article XML explicitly names CC BY; the missing version remains explicit in the paper registry.
- Three requests (`PMC12486974`, `PMC12825039`, `PMC12647125`) first returned transient connection/DNS errors and succeeded on sequential retry.
- All API requests used Europe PMC endpoints sequentially. No scraping or restricted full-text retrieval was used.
- Each candidate retains paper ID, title, DOI, source URL, license and license URL (blank only where no URL is stated), source section, extracted sentence text, suggestion labels and scores, and review fields.
- All candidates remain pending and source-text-unchecked. Automatic suggestions remain separate from `verified_labels`.
- Synthetic examples and `extracted_paper.txt` are not included.
