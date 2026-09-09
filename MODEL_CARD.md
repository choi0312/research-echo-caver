# ECHO-CAVER model card

## Model summary

ECHO-CAVER is a compact residual-reader extension for a frozen JUDO plus
FREB-CAVER runtime. FREB writes query/reference comparison evidence into five
decoder layers; ECHO adapts the immediately downstream layers while constraining
LoRA output directions to the singular span of the corresponding FREB writer.

| Property | Released ECHO-v1 value |
|---|---|
| Base model | `woodavid31/JUDO` |
| Required parent | FREB-CAVER `v0.1.0` |
| FREB weight SHA-256 | `ac3233dbee2de9dd1aaea4acede275ddc2d7a322427795717d9232291e03b0ed` |
| ECHO weight SHA-256 | `21623bfafc3d329128e2753a5600a9197088ef527deeb44687ea0eacd89f4bb9` |
| Writer layers | 18, 20, 22, 24, 26 |
| Reader layers | 19, 21, 23, 25, 27 |
| Target modules | `o_proj`, `down_proj` |
| LoRA rank / alpha | 16 / 32 |
| Writer-basis rank | 32 |
| Trainable parameters | 2,375,680 |
| Trainable fraction | 0.02853% |
| Checkpoint size | approximately 9.1MB |

JUDO and FREB remain frozen. The ECHO adapter and both parents are loaded into
one autoregressive runtime. No router, threshold, ensemble, or answer selector
produces the final prediction.

## Geometric constraint

For each frozen FREB writer matrix `W_i`, ECHO computes the leading left
singular basis `U_i`. The LoRA-B gradient and weight are projected through
`U_i U_i^T`, both before and after the optimizer update. The reader therefore
adapts inside the established evidence carrier instead of opening an arbitrary
language residual direction. See [METHOD.md](METHOD.md).

## Training contract

The released adapter was trained for 100 AdamW steps, batch size two semantic
option pairs, learning rate `2e-5`, zero weight decay, and seed `20260824`.
Its objective combines answer cross-entropy, option-permutation equivariance,
frozen-FREB KL retention, and an incorrect-to-correct repair margin. Training
used 963 non-AD pairs and validation used 184 pairs.

## Evaluation

The released checkpoint records a 39,670-row Full-MMAD-compatible strict
28-cell Overall of 82.416847%. This is a benchmark-targeted census, not an
independent generalization estimate. A separate 5,600-row component-disjoint
paired panel records 81.0758% versus FREB's 80.8060% (`+0.2698` points; 95% CI
crosses zero; exact McNemar `p=0.13321`).

A later publication ablation retrained fresh controls and should be treated as
a separate replication. Its ECHO-full arm is not byte-identical to the released
Full-MMAD checkpoint. See [PUBLICATION_ABLATION.md](PUBLICATION_ABLATION.md).

## Intended use and limitations

Intended use is research on evidence-aware MLLM adaptation and reproducible
industrial anomaly reasoning. It is not validated for production quality
control or safety-critical decisions.

- Independent improvement over FREB is not statistically established.
- ECHO does not materially solve anomaly detection and may alter generated
  reasoning trajectories.
- Full reproduction requires separately licensed JUDO, FREB, and MMAD assets.
- Cross-paper comparisons with JUDO and AD-Copilot are descriptive unless all
  models are rerun under one harmonized row-level protocol.
