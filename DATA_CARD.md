# ECHO-CAVER data card

## Released-checkpoint training data

ECHO-v1 was trained on 963 non-AD semantic-option pairs from the CAPRI-CAVER
panel and selected on 184 validation pairs. Each item contains the original
multiple-choice ordering and a semantic option permutation. The exact manifest
contents are not redistributed because they contain benchmark-derived question
content and restricted asset paths.

| Artifact | SHA-256 |
|---|---|
| Training manifest | `24f05f2a8cfdcb9784c79b90aac5e405cce2bb997d0ef87953272b10c9981ba5` |
| Validation manifest | `240f519d973779752f071cfe3b057445911b6f04f6a4c654ba1d4d714da18be2` |
| Frozen FREB weights | `ac3233dbee2de9dd1aaea4acede275ddc2d7a322427795717d9232291e03b0ed` |

The training summary in `artifacts/training/echo_training_summary.json` records
all optimizer arguments, reader/writer layers, basis identities, gradient
checks, validation metrics, and historical paths.

## Evaluation populations

### Full-MMAD-compatible census

- 39,670 unique rows in eight disjoint shards
- manifest SHA-256:
  `61a266302585230610ad58876e76e073ed04f7dccb31c9c61daeaf44a81d4693`
- strict exact-answer parsing with invalid generations counted incorrect
- unweighted mean over 28 source-by-task cells; label-balanced AD cells

This population contains benchmark-targeted/training-overlap components and is
not an independent generalization estimate.

### Component-disjoint panels

The released checkpoint's 5,600-row paired panel excludes connected query and
reference components overlapping CAPRI/PACE training assets. The later
publication ablation uses a separately sealed 5,600-row manifest with SHA-256
`cedf23061dc8d95d255fbadd8975de9a7a45a597946b525d83a423f431ee5b33`.
These two analyses must be identified separately.

## Redistribution policy

The public repository contains aggregate analyses, hashes, code, and the small
adapter. It does not contain MMAD images, questions, row-level manifests, or raw
model generations. The private cold-storage bundle retains row outputs needed
for internal audit but does not grant redistribution rights.

Reproducers must obtain all upstream data independently, map by canonical asset
and sample identity rather than path order, and confirm content hashes before
training or evaluation.
