#!/usr/bin/env python3
"""Fail-closed single-arm MMAD analysis for independently inferred shards.

Each shard is validated against its own manifest and inference contract before
the predictions are joined in memory.  The union must be an exact, disjoint
cover of the frozen full manifest.  Completions remain the sole prediction
source; no result row is rewritten during aggregation.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from analyze_paired_adapter_eval import (
    MMAD_TASKS,
    build_records,
    component_clusters,
    index_unique,
    loaded_adapter_parameter_count,
    read_jsonl,
    require_result_metadata,
    score_run,
    validate_manifest_row,
    write_json,
)
from paired_eval_contract import canonical_sha256, manifest_row_sha256, sha256_file


def _validate_shard_contract(
    manifest_rows: Sequence[Mapping[str, Any]],
    manifest_digest: str,
    results: Mapping[str, Mapping[str, Any]],
    *,
    name: str,
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    expected_ids = [str(row["sample_id"]) for row in manifest_rows]
    if set(expected_ids) != set(results) or len(expected_ids) != len(results):
        raise ValueError(
            f"{name} is not an exact shard cover: manifest={len(expected_ids)} "
            f"results={len(results)}"
        )

    digests = {row.get("inference_contract_sha256") for row in results.values()}
    if None in digests or len(digests) != 1:
        raise ValueError(f"{name} has missing or mixed inference-contract digests")
    digest = str(next(iter(digests)))
    contracts = [row.get("inference_contract") for row in results.values()]
    if any(not isinstance(value, dict) for value in contracts):
        raise ValueError(f"{name} has a missing/non-object inference contract")
    if {canonical_sha256(value) for value in contracts} != {digest}:
        raise ValueError(f"{name} inference-contract payload/digest mismatch")
    contract = dict(contracts[0])
    if contract.get("manifest_sha256") != manifest_digest:
        raise ValueError(f"{name} inference-contract manifest hash mismatch")
    if contract.get("selected_ids_sha256") != canonical_sha256(expected_ids):
        raise ValueError(f"{name} inference-contract selected-ID hash mismatch")
    if int(contract.get("selected_rows", -1)) != len(expected_ids):
        raise ValueError(f"{name} inference-contract row count mismatch")

    model = contract.get("base_model", {}).get("model_reference")
    typo = contract.get("prompt", {}).get("official_leading_n_typo")
    identities = {row.get("adapter_identity_sha256") for row in results.values()}
    paths = {row.get("adapter") for row in results.values()}
    counts = {loaded_adapter_parameter_count(row) for row in results.values()}
    if None in identities or len(identities) != 1:
        raise ValueError(f"{name} must contain one non-null adapter identity")
    if None in paths or "" in paths or len(paths) != 1:
        raise ValueError(f"{name} has a missing or mixed adapter path")
    if len(counts) != 1 or not isinstance(next(iter(counts)), int) or next(iter(counts)) <= 0:
        raise ValueError(f"{name} has an invalid/mixed loaded adapter parameter count")
    identity = str(next(iter(identities)))
    for sample_id, row in results.items():
        if row.get("model") != model or row.get("official_prompt_typo") != typo:
            raise ValueError(f"{name} {sample_id} flat provenance disagrees with contract")
        provenance = row.get("adapter_provenance")
        if not isinstance(provenance, dict):
            raise ValueError(f"{name} {sample_id} lacks adapter provenance")
        if provenance.get("adapter_identity_sha256") != identity:
            raise ValueError(f"{name} {sample_id} adapter identity mismatch")

    normalized = deepcopy(contract)
    for field in ("manifest_sha256", "selected_ids_sha256", "selected_rows"):
        normalized.pop(field, None)
    return contract, digest, {
        "adapter_identity_sha256": identity,
        "adapter_path": str(next(iter(paths))),
        "loaded_adapter_parameter_count": int(next(iter(counts))),
        "normalized_inference_contract_sha256": canonical_sha256(normalized),
    }


def analyze(
    full_manifest_path: Path,
    shard_manifest_paths: Sequence[Path],
    result_paths: Sequence[Path],
) -> dict[str, Any]:
    if not shard_manifest_paths or len(shard_manifest_paths) != len(result_paths):
        raise ValueError("provide the same positive number of shard manifests and results")

    full_rows = read_jsonl(full_manifest_path)
    full = index_unique(full_rows, "full manifest")
    for row in full_rows:
        validate_manifest_row(row)
    tasks = {str(row["question_type"]) for row in full_rows}
    if tasks != set(MMAD_TASKS):
        raise ValueError(f"full manifest is not canonical seven-task MMAD: {sorted(tasks)}")

    joined: dict[str, dict[str, Any]] = {}
    shard_receipts: list[dict[str, Any]] = []
    global_adapter: dict[str, Any] | None = None
    for index, (manifest_path, result_path) in enumerate(
        zip(shard_manifest_paths, result_paths), 1
    ):
        shard_rows = read_jsonl(manifest_path)
        shard = index_unique(shard_rows, f"shard manifest {index}")
        result_rows = read_jsonl(result_path)
        results = index_unique(result_rows, f"shard result {index}")
        manifest_digest = sha256_file(manifest_path)

        for row in shard_rows:
            validate_manifest_row(row)
            sample_id = str(row["sample_id"])
            if sample_id not in full:
                raise ValueError(f"shard {index} contains ID absent from full manifest: {sample_id}")
            if manifest_row_sha256(row) != manifest_row_sha256(full[sample_id]):
                raise ValueError(f"shard {index} row drift from full manifest: {sample_id}")
            require_result_metadata(results[sample_id], row, manifest_digest, f"shard {index}")

        _, contract_digest, adapter = _validate_shard_contract(
            shard_rows, manifest_digest, results, name=f"shard {index}"
        )
        if global_adapter is None:
            global_adapter = adapter
        elif adapter != global_adapter:
            raise ValueError(f"shard {index} model/adapter/generation contract drift")
        overlap = set(joined).intersection(results)
        if overlap:
            raise ValueError(f"shards overlap; first duplicate IDs: {sorted(overlap)[:10]}")
        joined.update(results)

        sidecar = result_path.with_suffix(result_path.suffix + ".native_adapter_runtime.json")
        if not sidecar.is_file():
            raise ValueError(f"native runtime attestation absent: {sidecar}")
        shard_receipts.append(
            {
                "index": index,
                "manifest": str(manifest_path.resolve()),
                "manifest_sha256": manifest_digest,
                "result": str(result_path.resolve()),
                "result_sha256": sha256_file(result_path),
                "rows": len(results),
                "inference_contract_sha256": contract_digest,
                "native_runtime_attestation": str(sidecar.resolve()),
                "native_runtime_attestation_sha256": sha256_file(sidecar),
            }
        )

    if set(joined) != set(full):
        missing = sorted(set(full) - set(joined))[:10]
        extra = sorted(set(joined) - set(full))[:10]
        raise ValueError(
            f"shard union is not full census: expected={len(full)} got={len(joined)} "
            f"missing={missing} extra={extra}"
        )

    clusters, cluster_reason = component_clusters(full_rows)
    records = build_records(set(full), full, joined, joined, clusters)
    metrics = score_run(records, "candidate")
    by_source: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        by_source[str(record["source"])].append(record)

    assert global_adapter is not None
    return {
        "analysis_schema_version": "strict-sharded-single-adapter-analysis-v1",
        "scope": "full-7-task",
        "rows": len(records),
        "coverage": {
            "full_manifest_ids": len(full),
            "joined_result_ids": len(joined),
            "disjoint_exact_cover": True,
            "shards": len(shard_receipts),
        },
        "full_manifest": str(full_manifest_path.resolve()),
        "full_manifest_sha256": sha256_file(full_manifest_path),
        "adapter": global_adapter,
        "answer_policy": {
            "parser": "exactly one case-sensitive <answer>X</answer> tag; X must be in options",
            "invalid_scoring": "incorrect under ITT; semantic AD decision is null, never normal",
            "prediction_source": "completion only; stored prediction fields ignored",
        },
        "metrics": metrics,
        "source_metrics": {
            source: score_run(group, "candidate") for source, group in sorted(by_source.items())
        },
        "component_clusters": None if clusters is None else len(set(clusters.values())),
        "component_cluster_reason": cluster_reason,
        "shard_receipts": shard_receipts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--shard-manifest", type=Path, action="append", required=True)
    parser.add_argument("--result", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.manifest, args.shard_manifest, args.result)
    write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
