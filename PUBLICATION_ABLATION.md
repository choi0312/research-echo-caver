# ECHO-CAVER publication ablation

## Contract

Nine arms were evaluated on the same 5,600 ordered sample IDs under one prompt,
parser, decoding configuration, and inference-contract hash. Uncertainty uses
50,000 shared query/reference-component bootstrap draws over 2,280 components,
seed `20260829`. Invalid outputs are incorrect under intention-to-treat scoring.

| Arm | Overall | non-AD | AD balanced | Role |
|---|---:|---:|---:|---|
| JUDO | 80.5223 | 83.4167 | 62.6392 | no writer, no reader |
| FREB | 80.9815 | 83.8750 | 63.1831 | writer only |
| Reader-only | 80.7588 | 83.7083 | 62.5364 | reader without writer |
| ECHO full | 81.0957 | 83.9792 | 63.3129 | writer and constrained reader |
| Unconstrained | 81.0390 | 83.9583 | 63.1268 | ordinary reader subspace |
| Random subspace | 81.2044 | 84.1042 | **63.3987** | capacity-matched random basis |
| Answer-only | 81.2915 | 84.2500 | 63.0042 | token-phase control |
| CE + KL | 81.0926 | 84.0000 | 63.1733 | objective control |
| CE-only | **81.5821** | **84.6250** | 62.6098 | objective control |

## Main paired findings

- FREB versus JUDO: `+0.4593` Overall points, 95% CI
  `[+0.1055,+0.8248]`, exact McNemar `p=0.02785`.
- ECHO-full versus JUDO: `+0.5734` points, 95% CI
  `[+0.1100,+1.0398]`, exact McNemar `p=0.01592`.
- ECHO-full versus FREB: `+0.1141` points, 95% CI
  `[-0.2660,+0.4955]`, exact McNemar `p=0.45969`.
- CE-only versus JUDO: `+1.0598` points, 95% CI
  `[+0.5243,+1.5947]`, exact McNemar `p=0.00009597`.
- The 2x2 Overall writer-reader interaction is `-0.1224` points with 95%
  interval `[-0.5685,+0.3252]`.

## Interpretation

The replication supports FREB's writer as an Overall-improving intervention and
shows that adding the full ECHO reader to JUDO+FREB remains directionally
positive. It does not establish that the writer-derived SVD subspace uniquely
causes the gain: CE-only and random-subspace controls are competitive, and the
writer-reader interaction interval includes zero. The most defensible ECHO
contribution is therefore a structured reader design with a clearly specified
evidence geometry, not a proven exclusive causal mechanism.

CE-only is an ablation finding, not a promoted final model: it has no Full-MMAD
evaluation, and selecting it on this confirmatory panel would require a new
disjoint evaluation before a final claim.

Machine-readable values are in
`results/publication_ablation_5600_summary.json`. Row-level JSONLs and the frozen
manifest are retained only in the private Drive bundle.
