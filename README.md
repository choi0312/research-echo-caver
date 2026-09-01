# ECHO-CAVER

**Evidence-tangent residual readers for FREB-CAVER.**

[한국어](README.ko.md) · [Method](METHOD.md) · [Results](RESULTS.md) · [Reproducibility](REPRODUCIBILITY.md) · [Claim boundaries](CLAIMS.md) · [FREB-CAVER](https://github.com/choi0312/FREB-CAVER)

ECHO-CAVER is a parameter-efficient extension of
[FREB-CAVER](https://github.com/choi0312/FREB-CAVER) on the public
[JUDO](https://github.com/woodavid31/JUDO) model. FREB writes comparison
evidence into the decoder; ECHO trains downstream residual readers to interpret
that evidence without opening an unrestricted language-only update path. JUDO
and FREB remain frozen, and inference uses one end-to-end model runtime.

## Full-MMAD Overall

| Model | Full-MMAD Overall | Change from JUDO | Change from FREB |
|---|---:|---:|---:|
| JUDO (paper-reported) | 81.20% | — | — |
| FREB-CAVER | 81.9063% | +0.7063 pp | — |
| **ECHO-CAVER v1** | **82.4168%** | **+1.2168 pp** | **+0.5105 pp** |

ECHO-CAVER completed all 39,670 rows and achieved 82.4168% normalized 28-cell
macro accuracy. JUDO is its paper-reported standard 1-shot score; FREB and ECHO
are archived full-census runs. The deltas are descriptive because the paper and
archive rows were not re-scored under one paired parser contract. Exact artifacts
and the evaluation boundary are documented in [Results](RESULTS.md) and
[Claim boundaries](CLAIMS.md).

## The bottleneck after FREB

FREB solves an important visual-side problem: it computes defect-preserving
query–reference evidence and writes a signed belief into selected decoder
states. But **writing evidence does not guarantee that later decoder blocks know
how to read it**. A generic LoRA update can learn useful answer mappings while
also introducing directions unrelated to the visual carrier, weakening the
causal link that FREB was designed to establish.

ECHO-CAVER treats this as a writer–reader alignment problem. It adapts how the
decoder reads FREB evidence while constraining every learned output direction to
the subspace actually produced by the frozen FREB writers.

## Method

FREB writers operate at decoder layers `18, 20, 22, 24, 26`. ECHO installs
rank-16 residual readers immediately downstream at layers `19, 21, 23, 25, 27`,
targeting `o_proj` and `down_proj`.

For each frozen writer matrix `W_i`, ECHO extracts a rank-32 leading
left-singular basis `U_i` and constrains the LoRA output matrix `B_i` to:

```text
B_i <- U_i U_i^T B_i
```

The projection is applied both to gradients and after every AdamW update, so
optimizer state cannot accumulate a persistent off-span component. LoRA output
paths start at zero, making the initial ECHO model exactly identical to frozen
FREB.

The released v1 objective combines answer cross-entropy, semantic option
equivariance, frozen-FREB retention, and targeted answer repair. At inference,
the JUDO base, FREB writer, and ECHO reader are loaded as one model; there is no
router, threshold, ensemble, crop policy, or post-hoc answer correction.

## What is new

- **Writer-derived adaptation geometry.** The trainable reader subspace is
  derived from the deployed FREB writer weights rather than selected from model
  width, a random basis, or a generic low-rank prior.
- **Evidence-tangent learning.** Unlike orthogonal-LoRA methods that isolate a
  new task from an old subspace, ECHO deliberately learns *inside* the existing
  causal evidence carrier so the reader remains coupled to visual evidence.
- **Layer-paired writer–reader placement.** Readers are positioned immediately
  after the exact layers where FREB writes evidence, turning architectural
  causality into an explicit adaptation contract.
- **Two-stage constraint enforcement.** Gradient projection and post-optimizer
  projection jointly keep the learned reader inside the intended span.
- **Single-checkpoint reasoning.** ECHO changes the model's internal evidence
  interpretation rather than selecting or correcting answers outside the model.

## Selected objective ablation

The most informative favorable ablation is the **CE-only evidence-tangent
reader**. It keeps ECHO's writer-derived subspace and layer placement while
removing the auxiliary retention, equivariance, and repair losses.

| Arm | 5,600-row Overall | Change from JUDO | 95% component-bootstrap CI | Exact McNemar |
|---|---:|---:|---:|---:|
| JUDO | 80.5223% | — | — | — |
| **ECHO CE-only** | **81.5821%** | **+1.0598 pp** | **[+0.5243, +1.5947] pp** | **p=0.000096** |

The result shows that the evidence-tangent reader can improve the decoder with
an extremely simple answer-learning objective; the extra composite losses are
not required to obtain the gain and may over-regularize this setting. This is a
mechanistic objective ablation, not a Full-MMAD score for the CE-only checkpoint.
All nine arms, the shared manifest and inference-contract identities, and the
50,000-replicate component bootstrap are preserved in
[`results/publication_ablation_5600_summary.json`](results/publication_ablation_5600_summary.json).

## Contributions

1. We formulate the post-FREB bottleneck as **evidence writing versus evidence
   reading**, separating visual-carrier construction from downstream decision
   adaptation.
2. We introduce a writer-conditioned LoRA constraint that ties decoder updates
   to the singular span of the frozen visual-evidence writer.
3. We preserve the JUDO/FREB computation at initialization and retain a single
   end-to-end inference path with only 2,375,680 trainable parameters
   (`0.02853%` of the base model).
4. We provide both a complete Full-MMAD census and same-contract paired
   diagnostics with component-level bootstrap analysis.

## Repository map

| Path | Contents |
|---|---|
| `src/` | Exact ECHO training, inference, manifest, and scoring programs |
| `checkpoints/` | 9.1MB ECHO v1 PEFT adapter |
| `results/` | Full-census, paired-panel, and publication-ablation summaries |
| `artifacts/training/` | Frozen training contract and mechanics receipt |
| `recipes/` | Recorded H200 evaluation recipes |
| `tests/` | Checkpoint, coverage, score, and archive identity tests |
| `paper/` | Research-archive manuscript |

## Quick integrity check

```bash
python3 scripts/verify_archive.py
python3 -m unittest discover -s tests -v
```

This offline verifier checks the archived v1 checkpoint, code, result, and score
identities. A full autoregressive reproduction requires separately obtaining
JUDO, FREB-CAVER, and MMAD under their respective terms. See
[REPRODUCIBILITY.md](REPRODUCIBILITY.md).

## Checkpoint identity

The released ECHO v1 adapter contains 2,375,680 trainable parameters and has
SHA-256:

```text
21623bfafc3d329128e2753a5600a9197088ef527deeb44687ea0eacd89f4bb9
```

## Evaluation scope

The 82.4168% Full-MMAD result is a complete archived behavior census under the
documented development setting, not an out-of-distribution generalization
estimate. The 5,600-row publication ablation is a separate same-contract paired
analysis. Cross-paper JUDO deltas remain descriptive unless row-level outputs
are evaluated with the same parser and inference contract.

## Status and citation

This is a public research artifact, not a production anomaly-detection system
or an accepted publication. Cite the original JUDO and MMAD works together with
[CITATION.cff](CITATION.cff), and retain the interpretation in
[CLAIMS.md](CLAIMS.md) when quoting results.
