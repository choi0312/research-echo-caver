# Results

## Scoring contract

The primary score is the unweighted mean of 28 source-by-task cells: four MMAD
sources by seven tasks. Every Anomaly Detection cell uses balanced accuracy
over normal and anomalous labels; the other 24 cells use accuracy. A prediction
is valid only when its completion contains exactly one case-sensitive
`<answer>X</answer>` tag whose label is present in the options. Invalid outputs
are incorrect under intention-to-treat scoring.

## Full MMAD paper comparison

Eight disjoint shards cover exactly 39,670 frozen IDs. All result rows share
one adapter identity and normalized inference contract. Aggregation was repeated
and reproduced the same analysis SHA-256.

| Model | AD | DC | DL | DD | DA | OC | OA | Overall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| JUDO, Table 1 | 65.04 | 74.74 | 73.01 | 84.56 | 89.41 | 94.04 | 87.58 | 81.20 |
| AD-Copilot, Table II | 73.64 | 67.89 | 64.08 | 80.60 | 85.91 | 91.06 | 87.78 | 78.71 |
| AD-Copilot-Thinking, Table II | **73.95** | 74.29 | **76.40** | 84.92 | 86.93 | 91.86 | 87.67 | 82.29 |
| **ECHO-CAVER v1** | 65.8231 | **76.7698** | 75.3531 | **86.1556** | **89.6094** | **95.1548** | **88.0521** | **82.4168** |

| Descriptive Overall comparison | Delta (points) |
|---|---:|
| ECHO minus JUDO | **+1.216847** |
| ECHO minus AD-Copilot-Thinking | **+0.126847** |

Additional ECHO metrics:

| Metric | Value |
|---|---:|
| non-AD six-task macro | 85.182465% |
| pooled AD balanced accuracy | 65.647530% |
| pooled AD TNR / TPR | 48.586355% / 82.708705% |
| raw ITT accuracy | 81.651122% (32,391/39,670) |
| valid-only accuracy | 81.743849% |
| strict answer validity | 99.886564% (45 invalid) |

JUDO and AD-Copilot scores are the standard 1-shot values reported in their
primary paper tables. They were not re-scored from row-level predictions with
this archive's strict exact-tag parser. ECHO uses a 39,670-row JUDO-compatible
census and an unweighted mean over 28 source-by-task cells. Task semantics and
the 1-shot setting align, but parser, checkpoint, preprocessing, manifest
cardinality, and aggregation details may differ. Cross-paper deltas are
therefore descriptive and do not support paired significance claims.

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
