# Results

## Scoring contract

The primary score is the unweighted mean of 28 source-by-task cells: four MMAD
sources by seven tasks. Every Anomaly Detection cell uses balanced accuracy
over normal and anomalous labels; the other 24 cells use accuracy. A prediction
is valid only when its completion contains exactly one case-sensitive
`<answer>X</answer>` tag whose label is present in the options. Invalid outputs
are incorrect under intention-to-treat scoring.

## Full MMAD-compatible census

Eight disjoint shards cover exactly 39,670 frozen IDs. All result rows share
one adapter identity and normalized inference contract. Aggregation was repeated
and reproduced the same analysis SHA-256.

| Task | Archived FREB strict | ECHO-CAVER | Delta (points) |
|---|---:|---:|---:|
| Anomaly Detection | 65.397576 | 65.823142 | +0.425566 |
| Defect Analysis | 89.716549 | 89.609353 | -0.107196 |
| Defect Classification | 75.236682 | 76.769823 | +1.533142 |
| Defect Description | 85.084414 | 86.155581 | +1.071167 |
| Defect Localization | 74.118318 | 75.353123 | +1.234805 |
| Object Analysis | 88.209034 | 88.052137 | -0.156897 |
| Object Classification | 95.529390 | 95.154770 | -0.374620 |
| **Overall** | **81.898852** | **82.416847** | **+0.517995** |

Additional ECHO metrics:

| Metric | Value |
|---|---:|
| non-AD six-task macro | 85.182465% |
| pooled AD balanced accuracy | 65.647530% |
| pooled AD TNR / TPR | 48.586355% / 82.708705% |
| raw ITT accuracy | 81.651122% (32,391/39,670) |
| valid-only accuracy | 81.743849% |
| strict answer validity | 99.886564% (45 invalid) |

The archived FREB analysis uses the same 39,670-row population and score
formula, but its serialized manifest SHA differs and its row-level predictions
are unavailable. Task and Overall deltas are therefore descriptive.

## Source-by-task ECHO scores

| Source | AD | DA | DC | DD | DL | OA | OC |
|---|---:|---:|---:|---:|---:|---:|---:|
| GoodsAD | 58.8812 | 91.9178 | 72.2349 | 85.0713 | 76.2284 | 85.4767 | 92.9810 |
| MVTec-AD | 81.0014 | 94.6058 | 81.4108 | 92.1682 | 88.0134 | 93.4018 | 99.5699 |
| MVTec-LOCO | 51.1748 | 84.3816 | 76.3747 | 82.3409 | 64.1548 | 86.4449 | 94.4853 |
| VisA | 72.2351 | 87.5322 | 77.0588 | 85.0420 | 73.0159 | 86.8852 | 93.5829 |

## Clean paired panel

The confirmatory panel contains 200 rows in each of 28 source-by-task cells. It
excludes every query/reference connected component overlapping CAPRI/PACE
training assets and completed a near-duplicate component audit. Inference was
paired on identical rows.

| Metric | FREB-CAVER | ECHO-CAVER | Delta |
|---|---:|---:|---:|
| Overall | 80.8060 | 81.0758 | +0.2698 |
| non-AD macro | 83.7292 | 84.0417 | +0.3125 |
| AD balanced accuracy | 62.9112 | 62.8450 | -0.0662 |
| ITT correct | 4,556 | 4,572 | +16 |

Paired transitions comprise 58 ECHO-only corrections and 42 FREB-only
corrections. Exact McNemar `p=0.1332106`. A 50,000-replicate query-reference
component bootstrap gives Overall delta 95% CI `[-0.093689, +0.631711]` points
and `P(delta>0)=0.92716`. The result is directionally positive but not
conventionally significant and does not pass the preregistered absolute
`Overall > 82%` clean-panel promotion rule.

## Compute

Training took 183.07 seconds on one NVIDIA H200. Full inference used at most
four H200s across two pods and eight workers. The measured inference window was
3:14:59. The conservative two-pod lifetime compute meter was $71.84; that total
includes earlier ECHO work on the primary pod.

Machine-readable evidence is under `results/` and `artifacts/training/`.

