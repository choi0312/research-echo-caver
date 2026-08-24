#!/usr/bin/env python3
"""Build the fixed, output-blind 5,600-row ECHO-CAVER screening panel.

The panel contains 200 rows from each of MMAD's 28 source x task cells.  Rows
sharing a measured query/reference component with any supplied training asset
are excluded before sampling.  Within each cell, label quotas follow the
eligible Full-MMAD population by deterministic largest-remainder allocation.

No model output is an input to this program.  The resulting panel is suitable
for paired candidate/control evaluation; repeated visual components remain
explicit so downstream uncertainty must use the query-reference component
graph rather than an IID row bootstrap.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = "echo-caver-screen-manifest-v1"
SUMMARY_SCHEMA_VERSION = "echo-caver-screen-summary-v1"
DEFAULT_SEED = 20260824
DEFAULT_ROWS_PER_CELL = 200


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            yield value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_rank(seed: int, *parts: object) -> str:
    material = "\0".join([str(seed), *(str(part) for part in parts)])
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def require_text(row: dict[str, Any], field: str, where: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{where}: missing {field}")
    return value


def collect_training_assets(paths: list[Path]) -> set[str]:
    assets: set[str] = set()
    for path in paths:
        for line_number, row in enumerate(iter_jsonl(path), 1):
            where = f"{path}:{line_number}"
            query = row.get("image", row.get("query_path"))
            reference = row.get("template_image", row.get("reference_path"))
            if not isinstance(query, str) or not isinstance(reference, str):
                raise ValueError(f"{where}: training row lacks a query/reference pair")
            assets.update((query, reference))
    return assets


def largest_remainder_quotas(
    label_counts: Counter[str], total: int
) -> dict[str, int]:
    population = sum(label_counts.values())
    if population < total or not label_counts:
        raise ValueError("label population cannot fill requested quota")
    exact = {label: total * count / population for label, count in label_counts.items()}
    quotas = {label: int(value) for label, value in exact.items()}
    remaining = total - sum(quotas.values())
    order = sorted(
        label_counts,
        key=lambda label: (-(exact[label] - quotas[label]), label),
    )
    for label in order[:remaining]:
        quotas[label] += 1
    if sum(quotas.values()) != total:
        raise AssertionError("largest-remainder allocation drift")
    return quotas


def canonical_row(
    row: dict[str, Any], *, seed: int, screen_rows: int
) -> dict[str, Any]:
    value = dict(row)
    value["parent_schema_version"] = row.get("schema_version")
    value["schema_version"] = SCHEMA_VERSION
    value["screen_seed"] = seed
    value["screen_split"] = f"fixed-screen{screen_rows}"
    value["training_component_excluded"] = True
    value["selection_used_model_outputs"] = False
    return value


def build(
    *,
    full_manifest: Path,
    training_manifests: list[Path],
    output: Path,
    summary_path: Path,
    rows_per_cell: int,
    seed: int,
    shards: int = 2,
) -> dict[str, Any]:
    if rows_per_cell <= 0:
        raise ValueError("rows_per_cell must be positive")
    if shards <= 0 or rows_per_cell % shards:
        raise ValueError("shards must be positive and divide rows_per_cell")
    full_rows = list(iter_jsonl(full_manifest))
    if len({require_text(row, "sample_id", "full manifest") for row in full_rows}) != len(full_rows):
        raise ValueError("Full-MMAD sample IDs are not unique")

    asset_components: dict[str, set[str]] = defaultdict(set)
    for row in full_rows:
        query = require_text(row, "image", "full manifest")
        reference = require_text(row, "template_image", "full manifest")
        query_component = require_text(row, "query_component_id", "full manifest")
        reference_component = require_text(row, "reference_component_id", "full manifest")
        asset_components[query].add(query_component)
        asset_components[reference].add(reference_component)

    training_assets = collect_training_assets(training_manifests)
    mapped_training_assets = sorted(asset for asset in training_assets if asset in asset_components)
    unmapped_training_assets = sorted(training_assets - set(mapped_training_assets))
    forbidden_components = {
        component
        for asset in mapped_training_assets
        for component in asset_components[asset]
    }

    eligible_by_cell: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    exclusion_counts = Counter()
    for row in full_rows:
        query_component = str(row["query_component_id"])
        reference_component = str(row["reference_component_id"])
        if query_component in forbidden_components or reference_component in forbidden_components:
            exclusion_counts[(str(row["source"]), str(row["question_type"]))] += 1
            continue
        eligible_by_cell[(str(row["source"]), str(row["question_type"]))].append(row)

    if len(eligible_by_cell) != 28:
        raise ValueError(f"expected 28 source x task cells, got {len(eligible_by_cell)}")

    selected: list[dict[str, Any]] = []
    cell_receipts: dict[str, Any] = {}
    for (source, task), rows in sorted(eligible_by_cell.items()):
        labels = Counter(str(row["label"]) for row in rows)
        quotas = largest_remainder_quotas(labels, rows_per_cell)
        selected_cell: list[dict[str, Any]] = []
        for label, quota in sorted(quotas.items()):
            pool = [row for row in rows if str(row["label"]) == label]
            ranked = sorted(
                pool,
                key=lambda row: stable_rank(
                    seed,
                    source,
                    task,
                    label,
                    row["sample_id"],
                    row["query_component_id"],
                    row["reference_component_id"],
                ),
            )
            if len(ranked) < quota:
                raise ValueError(f"cell {(source, task, label)} has {len(ranked)} < {quota}")
            selected_cell.extend(ranked[:quota])
        if len(selected_cell) != rows_per_cell:
            raise AssertionError("cell quota drift")
        selected_cell.sort(
            key=lambda row: stable_rank(seed, "output-order", row["sample_id"])
        )
        selected.extend(
            canonical_row(row, seed=seed, screen_rows=28 * rows_per_cell)
            for row in selected_cell
        )
        cell_key = f"{source}|{task}"
        cell_receipts[cell_key] = {
            "full_rows": sum(
                1
                for row in full_rows
                if str(row["source"]) == source and str(row["question_type"]) == task
            ),
            "excluded_rows": exclusion_counts[(source, task)],
            "eligible_rows": len(rows),
            "eligible_label_counts": dict(sorted(labels.items())),
            "selected_rows": len(selected_cell),
            "selected_label_quotas": dict(sorted(quotas.items())),
        }

    expected_rows = 28 * rows_per_cell
    if len(selected) != expected_rows or len({row["sample_id"] for row in selected}) != expected_rows:
        raise AssertionError("screen row/sample-ID count drift")
    if any(
        row["query_component_id"] in forbidden_components
        or row["reference_component_id"] in forbidden_components
        for row in selected
    ):
        raise AssertionError("training component reached the screen panel")

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    shard_receipts = {}
    shard_root = output.parent / "shards"
    shard_root.mkdir(parents=True, exist_ok=True)
    for shard_index in range(shards):
        shard_rows = [row for index, row in enumerate(selected) if index % shards == shard_index]
        shard_path = shard_root / f"part-{shard_index:02d}-of-{shards:02d}.jsonl"
        with shard_path.open("w", encoding="utf-8", newline="\n") as handle:
            for row in shard_rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        shard_cells = Counter(
            (str(row["source"]), str(row["question_type"])) for row in shard_rows
        )
        if len(shard_cells) != 28 or set(shard_cells.values()) != {rows_per_cell // shards}:
            raise AssertionError("round-robin shard lost exact source x task balance")
        shard_receipts[shard_path.name] = {
            "path": str(shard_path.resolve()),
            "sha256": sha256_file(shard_path),
            "rows": len(shard_rows),
            "rows_per_source_task_cell": rows_per_cell // shards,
        }

    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "status": "fixed-before-candidate-inference",
        "seed": seed,
        "selection_used_model_outputs": False,
        "rows_per_source_task_cell": rows_per_cell,
        "source_task_cells": len(eligible_by_cell),
        "rows": len(selected),
        "full_manifest": str(full_manifest.resolve()),
        "full_manifest_sha256": sha256_file(full_manifest),
        "training_manifests": {
            str(path.resolve()): sha256_file(path) for path in training_manifests
        },
        "training_assets": len(training_assets),
        "mapped_training_assets": len(mapped_training_assets),
        "unmapped_training_assets": len(unmapped_training_assets),
        "unmapped_training_asset_examples": unmapped_training_assets[:20],
        "forbidden_components": len(forbidden_components),
        "selected_unique_query_components": len(
            {row["query_component_id"] for row in selected}
        ),
        "selected_unique_reference_components": len(
            {row["reference_component_id"] for row in selected}
        ),
        "component_exclusion": (
            "exclude a row when either measured query/reference component is joined "
            "to an exact query/reference asset in any supplied training manifest"
        ),
        "within_cell_allocation": "eligible-population label ratio by largest remainder",
        "within_label_selection": "ascending seeded SHA-256 rank",
        "uncertainty_unit": "query-reference connected component",
        "shards": shard_receipts,
        "cells": cell_receipts,
        "output": str(output.resolve()),
        "output_sha256": sha256_file(output),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return summary


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full-manifest",
        type=Path,
        default=root / "research/manifests/m1_full_eval_v1/full.jsonl",
    )
    parser.add_argument(
        "--training-manifest",
        action="append",
        type=Path,
        default=[],
        help="May be repeated; defaults to the PACE and CAPRI training manifests",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "research/manifests/echo_caver_v1/screen5600.jsonl",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=root / "research/manifests/echo_caver_v1/screen5600_summary.json",
    )
    parser.add_argument("--rows-per-cell", type=int, default=DEFAULT_ROWS_PER_CELL)
    parser.add_argument("--shards", type=int, default=2)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    if not args.training_manifest:
        args.training_manifest = [
            root / "research/data_sources/pace_multitask_panel_v1/train.jsonl",
            root / "research/data_sources/capri_caver_panel_v1/train.jsonl",
        ]
    return args


def main() -> None:
    args = parse_args()
    summary = build(
        full_manifest=args.full_manifest.resolve(),
        training_manifests=[path.resolve() for path in args.training_manifest],
        output=args.output.resolve(),
        summary_path=args.summary.resolve(),
        rows_per_cell=args.rows_per_cell,
        seed=args.seed,
        shards=args.shards,
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
