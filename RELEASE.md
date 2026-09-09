# ECHO-CAVER reproducibility release

## Authoritative identities

| Artifact | Identity |
|---|---|
| Source | Git tag `v0.2.0-repro` and its commit |
| ECHO-v1 weights | `21623bfafc3d329128e2753a5600a9197088ef527deeb44687ea0eacd89f4bb9` |
| FREB parent weights | `ac3233dbee2de9dd1aaea4acede275ddc2d7a322427795717d9232291e03b0ed` |
| Exact inference runner | `f160385a1004749485f2dff060e121ac1bf280770d92914171d7613289b8d6a9` |
| Full analysis | `68195e2e0fe5e49108b0a48b8b602a1b6e1617611bad2c0b11330ebb6eb8e547` |
| Scientific identity | `b538d55b4aaa2d88f0f62f89342741fe1ec807a195db35a2730559c3e5e97bf0` |

`ARCHIVE_MANIFEST.sha256` seals the committed archive, excluding the manifest
itself and Git metadata.

## Verification tiers

### Offline archive verification

```bash
python3 scripts/verify_archive.py
python3 -m unittest discover -s tests -v
```

This validates the adapter, exact inference source, aggregate results,
Full-MMAD coverage ledger, clean-panel statistics, and archive manifest without
network, GPU, JUDO, or MMAD.

### Full execution

1. Check out the source tag.
2. Obtain JUDO, FREB-CAVER, and MMAD under their upstream terms.
3. Verify the JUDO revision, FREB weight, ECHO weight, manifest, and inference
   contract hashes.
4. Install the pinned environment from [REPRODUCIBILITY.md](REPRODUCIBILITY.md).
5. Run a small frozen-manifest smoke test.
6. Run every full shard and fail closed on duplicate/missing IDs or contract
   drift.
7. Recompute the strict 28-cell score from all completed shards.

## Relationship between checkpoints

The released Full-MMAD checkpoint has ECHO weight SHA `21623b...`. The later
publication ablation retrained fresh factorial and control adapters. Those
weights and scores are a replication/ablation family and must not be described
as byte-identical to the Full-MMAD checkpoint.

## Required reporting

- identify the source tag and all three model checkpoints;
- distinguish the Full-MMAD-compatible census from component-disjoint panels;
- report strict invalid-output handling and 28-cell aggregation;
- attach training-data and evaluation-exposure statements;
- call external paper comparisons descriptive unless rerun under one protocol;
- report unfavorable and favorable ablations together;
- do not promote an ablation selected on a confirmatory panel without a new
  disjoint evaluation.
