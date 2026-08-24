#!/usr/bin/env python3
"""Build a deterministic, shardable 39,670-row MMAD evaluation manifest.

This freezes one normal reference per query before expanding its questions,
normalizes MMAD's Object Details/Structure aliases to Object Analysis, and
binds query/reference graph component IDs from the audited native-parent and
pHash<=4 maps.  It performs no model inference and never edits source data.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any


EXPECTED_MMAD_SHA256 = "95dc8159340f21cd9d7d77acb6575fdf855d5c40a0eed7048a5a669d6e0b4b35"
EXPECTED_NATIVE_SHA256 = "08e130e8284fb672b92011e4614aa29cb3d3ef3a8e5341a47727f0c5fc2cd693"
EXPECTED_NEAR_SHA256 = "5501e8c46596b02aeebe04cfefc23095689026f67c4bab0c51321c0004da3ba1"
EXPECTED_ASSET_INDEX_SHA256 = "5ca2e575f0132f4ceeefd07f0aec668c21143984cb342b4497b5dc72ef3eb212"
EXPECTED_ROWS = 39_670
EXPECTED_QUERIES = 8_366
SEED = 20260821


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, value: str) -> str:
        self.parent.setdefault(value, value)
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            if right_root < left_root:
                left_root, right_root = right_root, left_root
            self.parent[right_root] = left_root


def component_assignments(native: dict[str, str], near: dict[str, str]) -> dict[str, str]:
    assets = sorted(set(native) | set(near))
    if set(native) != set(near):
        raise ValueError("native-parent and near-duplicate assignment scopes differ")
    union = UnionFind()
    first_by_native: dict[str, str] = {}
    first_by_near: dict[str, str] = {}
    for asset in assets:
        union.find(asset)
        for group, firsts in ((native[asset], first_by_native), (near[asset], first_by_near)):
            if group in firsts:
                union.union(asset, firsts[group])
            else:
                firsts[group] = asset
    members: dict[str, list[str]] = defaultdict(list)
    for asset in assets:
        members[union.find(asset)].append(asset)
    component_id: dict[str, str] = {}
    for group_members in members.values():
        digest = hashlib.sha256("\n".join(sorted(group_members)).encode("utf-8")).hexdigest()[:24]
        for asset in group_members:
            component_id[asset] = f"component:{digest}"
    return component_id


def choose_reference(query: str, candidates: list[str], seed: int) -> str:
    if not candidates:
        raise ValueError(f"query has no random normal reference: {query}")
    payload = f"{seed}\0{query}".encode("utf-8")
    return candidates[int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % len(candidates)]


def normalized_source(query: str) -> tuple[str, str]:
    raw = query.split("/", 1)[0]
    return ("MVTec-AD" if raw == "DS-MVTec" else raw), raw


def normalized_task(task: str) -> str:
    return "Object Analysis" if task in {"Object Details", "Object Structure"} else task


def semantic_ad_label(query: str, conversation: list[dict[str, Any]]) -> str:
    """Recover the query label from its AD answer, with a path-only fallback.

    Four published MMAD paths disagree with a simple ``/good/`` heuristic.  The
    benchmark answer is authoritative for scoring, so Full evaluation must use
    the semantic Yes/No target whenever an AD question is present.  A small
    number of queries contain only non-AD questions; their label is descriptive
    only and retains the legacy path fallback.
    """

    for question in conversation:
        if question.get("type") != "Anomaly Detection":
            continue
        answer = str(question.get("Answer"))
        option = str(question.get("Options", {}).get(answer, "")).strip().lower().rstrip(".")
        if option == "yes":
            return "anomaly"
        if option == "no":
            return "normal"
        raise ValueError(f"non-binary AD answer semantics for {query}: {option!r}")
    return "normal" if "/good/" in query else "anomaly"


def build(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pinned = {
        "mmad": (args.mmad_json, EXPECTED_MMAD_SHA256),
        "native_parent_map": (args.native_parent_map, EXPECTED_NATIVE_SHA256),
        "near_duplicate_map": (args.near_duplicate_map, EXPECTED_NEAR_SHA256),
        "asset_index": (args.asset_index, EXPECTED_ASSET_INDEX_SHA256),
    }
    observed: dict[str, str] = {}
    for name, (path, expected) in pinned.items():
        digest = sha256_file(path)
        if digest != expected:
            raise ValueError(f"{name} SHA-256 mismatch: {digest}")
        observed[name] = digest
    corpus = json.loads(args.mmad_json.read_text(encoding="utf-8"))
    if not isinstance(corpus, dict) or len(corpus) != EXPECTED_QUERIES:
        raise ValueError(f"expected {EXPECTED_QUERIES} MMAD queries")
    native_payload = json.loads(args.native_parent_map.read_text(encoding="utf-8"))
    near_payload = json.loads(args.near_duplicate_map.read_text(encoding="utf-8"))
    components = component_assignments(native_payload["assignments"], near_payload["assignments"])

    rows: list[dict[str, Any]] = []
    for query, metadata in corpus.items():
        reference = choose_reference(query, list(metadata.get("random_templates", [])), args.seed)
        if query not in components or reference not in components:
            raise ValueError(f"component assignment missing for {query} or {reference}")
        source, source_raw = normalized_source(query)
        category = query.split("/")[1]
        query_label = semantic_ad_label(query, list(metadata["conversation"]))
        for question_index, question in enumerate(metadata["conversation"]):
            options = {str(k): str(v) for k, v in question["Options"].items()}
            task = normalized_task(str(question["type"]))
            identity = {
                "seed": args.seed,
                "query": query,
                "reference": reference,
                "question_index": question_index,
                "question": question["Question"],
                "options": options,
            }
            sample_id = hashlib.sha256(stable_json(identity).encode("utf-8")).hexdigest()[:24]
            rows.append(
                {
                    "sample_id": sample_id,
                    "schema_version": "mmad-m1-full-eval-manifest-v1",
                    "split": "full",
                    "seed": args.seed,
                    "source": source,
                    "source_raw": source_raw,
                    "category": category,
                    "image": query,
                    "canonical_image": query,
                    "template_image": reference,
                    "canonical_template_image": reference,
                    "query_component_id": components[query],
                    "reference_component_id": components[reference],
                    "question_index": question_index,
                    "question": str(question["Question"]),
                    "question_type": task,
                    "question_type_raw": str(question["type"]),
                    "correct_answer": str(question["Answer"]),
                    "options": options,
                    "options_text": "".join(f"{key}. {value}\n" for key, value in options.items()),
                    "label": query_label,
                    "component_near_duplicate_audit_complete": True,
                    "dataset_native_physical_parent_audit_complete": True,
                    "latent_physical_parent_audit_complete": False,
                    "fully_sealed": False,
                }
            )
    if len(rows) != EXPECTED_ROWS or len({row["sample_id"] for row in rows}) != EXPECTED_ROWS:
        raise ValueError("Full manifest row/sample-ID contract failed")
    selected_assets = sorted({row[field] for row in rows for field in ("image", "template_image")})
    asset_index: dict[str, dict[str, Any]] = {}
    with args.asset_index.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"blank asset-index row at line {line_number}")
            record = json.loads(line)
            asset = str(record["canonical_asset"])
            if asset in asset_index:
                raise ValueError(f"duplicate asset-index row: {asset}")
            asset_index[asset] = record
    missing_index = sorted(set(selected_assets) - set(asset_index))
    if missing_index:
        raise ValueError(f"{len(missing_index)} Full assets missing from index; first={missing_index[:3]}")
    if args.data_root:
        missing = [value for value in selected_assets if not (args.data_root / value).is_file()]
        if missing:
            raise ValueError(f"{len(missing)} Full assets missing; first={missing[:3]}")
        if args.verify_asset_bytes:
            mismatched = [
                value
                for value in selected_assets
                if sha256_file(args.data_root / value) != asset_index[value]["raw_sha256"]
            ]
            if mismatched:
                raise ValueError(f"{len(mismatched)} Full asset SHA mismatches; first={mismatched[:3]}")
    counts = Counter((row["source"], row["question_type"]) for row in rows)
    receipt = {
        "schema_version": "mmad-m1-full-eval-receipt-v1",
        "rows": len(rows),
        "queries": len(corpus),
        "seed": args.seed,
        "shards": args.shards,
        "inputs_sha256": observed,
        "selected_assets": len(selected_assets),
        "selected_asset_bytes": sum(int(asset_index[value]["raw_bytes"]) for value in selected_assets),
        "source_task_counts": {f"{source}::{task}": count for (source, task), count in sorted(counts.items())},
        "ad_semantic_label_counts": dict(
            Counter(row["label"] for row in rows if row["question_type"] == "Anomaly Detection")
        ),
        "claim_boundary": "Matched deterministic Full census; latent same-manufactured-unit identity remains unobservable.",
    }
    receipt["_asset_inventory"] = asset_index
    return rows, receipt


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(stable_json(row) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[2]
    parser.add_argument("--mmad-json", type=Path, default=root / "external/MMAD/dataset/MMAD/mmad.json")
    parser.add_argument("--native-parent-map", type=Path, default=root / "research/data_sources/mmad_full_asset_seal_v1/dataset_native_parent_map.json")
    parser.add_argument("--near-duplicate-map", type=Path, default=root / "research/data_sources/mmad_full_asset_seal_v1/near_duplicate_map.json")
    parser.add_argument("--asset-index", type=Path, default=root / "research/data_sources/mmad_full_asset_seal_v1/asset_index.jsonl")
    parser.add_argument("--data-root", type=Path, default=root / "data/mmad-full-assets")
    parser.add_argument("--verify-asset-bytes", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--shard-dir", type=Path, required=True)
    parser.add_argument("--shards", type=int, default=4)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    if args.shards < 1:
        raise ValueError("--shards must be positive")
    rows, receipt = build(args)
    asset_index = receipt.pop("_asset_inventory")
    write_jsonl(args.output, rows)
    shard_sizes: list[int] = []
    shard_asset_counts: dict[str, int] = {}
    shard_asset_bytes: dict[str, int] = {}
    shard_asset_sha256: dict[str, str] = {}
    for shard in range(args.shards):
        start = len(rows) * shard // args.shards
        stop = len(rows) * (shard + 1) // args.shards
        subset = rows[start:stop]
        shard_name = f"part-{shard:02d}-of-{args.shards:02d}"
        write_jsonl(args.shard_dir / f"{shard_name}.jsonl", subset)
        selected = sorted({row[field] for row in subset for field in ("image", "template_image")})
        inventory_path = args.shard_dir / f"{shard_name}.assets.jsonl"
        inventory_rows = [
            {
                "asset": asset,
                "bytes": int(asset_index[asset]["raw_bytes"]),
                "raw_sha256": str(asset_index[asset]["raw_sha256"]),
            }
            for asset in selected
        ]
        write_jsonl(inventory_path, inventory_rows)
        shard_sizes.append(len(subset))
        shard_asset_counts[shard_name] = len(selected)
        shard_asset_bytes[shard_name] = sum(row["bytes"] for row in inventory_rows)
        shard_asset_sha256[inventory_path.name] = sha256_file(inventory_path)
    receipt["shard_rows"] = shard_sizes
    receipt["manifest_sha256"] = sha256_file(args.output)
    receipt["shard_sha256"] = {
        path.name: sha256_file(path)
        for path in sorted(args.shard_dir.glob("part-*.jsonl"))
        if not path.name.endswith(".assets.jsonl")
    }
    receipt["shard_asset_counts"] = shard_asset_counts
    receipt["shard_asset_bytes"] = shard_asset_bytes
    receipt["shard_asset_inventory_sha256"] = shard_asset_sha256
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
