# ECHO-CAVER experiment ledger (2026-08-24)

## Objective and frozen promotion rule

Derive one end-to-end FREB-CAVER extension that can approach 83% MMAD Overall.
Candidate selection is output-blind and uses exactly the same free-generation
contract for FREB-CAVER and every candidate: batch size 4, maximum 1,024 new
tokens, image size 512, grid size 16, SDPA, and JUDO's released prompt typo.
A method is retained only if sampled seven-task MMAD Overall is **strictly
greater than 82%**.  Accuracy is not inspected while a panel is incomplete.

The confirmatory panel was fixed before candidate inference.  It contains
5,600 rows: 4 sources x 7 tasks x 200 rows.  It excludes every query/reference
connected component that overlaps the CAPRI/PACE training assets and records a
completed near-duplicate component audit.  The manifest SHA-256 is
`cedf23061dc8d95d255fbadd8975de9a7a45a597946b525d83a423f431ee5b33`.

## Method: ECHO-CAVER

**Evidence-tangent Causal Hidden-state Orthogonal residual-reader LoRA** keeps
the released FREB-CAVER checkpoint frozen.  FREB writes visual-comparison
evidence after decoder layers 18, 20, 22, 24, and 26.  ECHO places rank-16
LoRA readers on the `o_proj` and `down_proj` modules in the immediately
downstream layers 19, 21, 23, 25, and 27.

For each site, ECHO computes the leading rank-32 left-singular basis of the
frozen CAVER `belief_up` writer.  Both LoRA-B gradients and post-Adam weights
are projected into that evidence span.  Thus the new reader can amplify or
reinterpret directions that the visual-evidence carrier actually writes, but
cannot create an unconstrained language-only residual direction.  This differs
from continual-learning orthogonal LoRA methods that place a new task outside
previous task subspaces.

Training uses 963 non-AD pairs from the existing CAPRI-CAVER panel, with
answer cross-entropy, frozen-CAVER retention KL, wrong-answer repair margin,
and semantic option-permutation equivariance.  It uses 100 optimizer steps at
learning rate 2e-5.  The adapter has 2,375,680 trainable parameters (0.0285% of
8.328B total) and took 183.1 seconds on one H200.  At inference it is one
FREB+ECHO checkpoint/runtime: no router, threshold, ensemble, crop, mask, or
post-hoc correction.

Checkpoint identities:

- ECHO adapter weights: `21623bfafc3d329128e2753a5600a9197088ef527deeb44687ea0eacd89f4bb9`
- ECHO scientific identity: `b538d55b4aaa2d88f0f62f89342741fe1ec807a195db35a2730559c3e5e97bf0`
- released FREB weights: `ac3233dbee2de9dd1aaea4acede275ddc2d7a322427795717d9232291e03b0ed`

## Frozen 320-row screening result

| Arm | Overall | Non-AD macro | AD balanced accuracy | Decision |
|---|---:|---:|---:|---|
| FREB-CAVER | 80.4688 | 83.8542 | 60.1562 | baseline |
| matched trajectory LoRA, no projection | 81.3616 | 84.8958 | 60.1562 | reject |
| ECHO with full trajectory supervision | 80.9152 | 84.3750 | 60.1562 | reject |
| **ECHO answer-repair v1** | **82.0312** | **85.4167** | **61.7188** | **promote** |

ECHO v1 changes nine paired outcomes relative to FREB: seven corrections and
two regressions.  Its Overall delta is +1.5625 points; the component-cluster
95% interval on this small screen is [-0.4464, +3.7220] points and exact
McNemar p=0.1797.  Therefore the 320-row result satisfies the user's absolute
screen gate but is not itself confirmatory evidence.

## 5,600-row clean confirmatory evaluation

FREB and ECHO were evaluated in parallel on two NVIDIA H200 GPUs.  The final
analysis is paired by sample ID and uses exact McNemar plus 50,000
component-cluster bootstrap replicates.

| Arm | Overall | Non-AD macro | AD balanced accuracy | ITT correct |
|---|---:|---:|---:|---:|
| FREB-CAVER | 80.8060 | 83.7292 | 62.9112 | 4,556/5,600 |
| ECHO-CAVER v1 | 81.0758 | 84.0417 | 62.8450 | 4,572/5,600 |

ECHO changes 100 paired outcomes: 58 corrections and 42 regressions.  Its
Overall delta is +0.2698 points with component-cluster 95% CI
[-0.0937, +0.6317] points, `P(delta > 0)=0.92716`, and exact McNemar
`p=0.13321`.  It therefore has a weak positive signal but fails both the
strictly-greater-than-82% user gate and conventional paired significance.
Under that pre-registered clean-panel promotion rule, **ECHO-CAVER v1 was not
promoted at this stage**.  The user subsequently overrode disposal and
requested a complete Full-MMAD census; that later result is recorded below.

The causal failure analysis is informative.  ECHO changed the generated
thinking on 1,349/5,600 rows even though its intended role was to read CAVER's
answer-state evidence.  Among discordant decisions, 35/58 corrections and
30/42 regressions also changed the thinking trace.  The released FREB carrier
already exposes an exact answer-prefix decision mask, but the downstream PEFT
LoRA was active at every token.  The next prospective candidate therefore
shares FREB's causal decision mask with the ECHO readers so that all
segmentation/reasoning tokens remain exactly the frozen FREB computation.

Collected run:
`research/runs/h200x2_echo_caver_screen5600_v1-20260824T081025Z-19744`.

## Full-MMAD census after retention override

ECHO-CAVER v1 was retained and inferred over all 39,670 MMAD questions.  The
manifest was split into eight exact, disjoint shards and evaluated with a
maximum of four NVIDIA H200 GPUs across two pods.  Shards 0--3 came from the
primary 2xH200 pod and shards 4--7 from the secondary 2xH200 pod.  The joined
result covers all 39,670 frozen IDs exactly once.  The measured inference
window was 3:14:59 (10:12:51--13:27:50 UTC); remaining ETA is zero.

The strict score is the mean of the 28 source-by-task cells.  Every AD cell is
label-balanced, invalid exact-answer generations count as incorrect, and no
stored prediction field is trusted over the generated completion.

| Metric | Full-MMAD result |
|---|---:|
| **Overall** | **82.416847%** |
| non-AD six-task macro | 85.182465% |
| pooled AD balanced accuracy | 65.647530% |
| raw ITT accuracy | 81.651122% (32,391/39,670) |
| strict-format validity | 99.886564% (45 invalid) |

| Task | ECHO-CAVER v1 | Archived FREB strict result | Descriptive delta |
|---|---:|---:|---:|
| Anomaly Detection | 65.823142 | 65.397576 | +0.425566 |
| Defect Analysis | 89.609353 | 89.716549 | -0.107196 |
| Defect Classification | 76.769823 | 75.236682 | +1.533142 |
| Defect Description | 86.155581 | 85.084414 | +1.071167 |
| Defect Localization | 75.353123 | 74.118318 | +1.234805 |
| Object Analysis | 88.052137 | 88.209034 | -0.156897 |
| Object Classification | 95.154770 | 95.529390 | -0.374620 |
| **Overall** | **82.416847** | **81.898852** | **+0.517995** |

The FREB comparison uses the same 39,670-row population and the same strict
cell-normalized score formula, but is descriptive rather than a paired test:
the archived FREB result's manifest serialization has a different SHA-256 and
its row-level predictions are not redistributed.  The Full-MMAD score also is
not a clean-generalization estimate because benchmark-targeted/training-overlap
components are present.  The component-disjoint 5,600-row panel above remains
the relevant clean check and did not cross 82% or conventional significance.

Final analysis:
`research/runs/echo_caver_full_mmad_v1-20260824/analyses/echo_full_mmad.json`.
Cross-pod ledger:
`research/runs/echo_caver_full_mmad_v1-20260824/RUN_LEDGER.json`.
