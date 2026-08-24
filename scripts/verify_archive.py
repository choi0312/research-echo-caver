#!/usr/bin/env python3
"""Verify checkpoint/code identities and recompute archived ECHO metrics."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

EXPECTED = {
    "checkpoints/echo-caver-v1/adapter_model.safetensors":
        "21623bfafc3d329128e2753a5600a9197088ef527deeb44687ea0eacd89f4bb9",
    "src/run_judo_manifest.py":
        "f160385a1004749485f2dff060e121ac1bf280770d92914171d7613289b8d6a9",
    "results/full_mmad_analysis.json":
        "68195e2e0fe5e49108b0a48b8b602a1b6e1617611bad2c0b11330ebb6eb8e547",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"archive verification failed: {message}")


def main() -> None:
    for relative, expected in EXPECTED.items():
        path = ROOT / relative
        require(path.is_file(), f"missing {relative}")
        require(sha256_file(path) == expected, f"SHA-256 drift in {relative}")

    full = json.loads((ROOT / "results/full_mmad_analysis.json").read_text())
    ledger = json.loads((ROOT / "results/full_run_ledger.json").read_text())
    clean = json.loads((ROOT / "results/clean_panel_analysis.json").read_text())
    metadata = json.loads((ROOT / "metadata/archive.json").read_text())
    comparison = json.loads((ROOT / "results/paper_reported_comparison.json").read_text())

    coverage = full["coverage"]
    require(coverage["disjoint_exact_cover"] is True, "full cover is not exact")
    require(coverage["full_manifest_ids"] == 39670, "unexpected full manifest size")
    require(coverage["joined_result_ids"] == 39670, "unexpected joined size")
    require(coverage["shards"] == 8, "unexpected shard count")

    metrics = full["metrics"]
    task_values = list(metrics["per_task_macro"].values())
    recomputed = sum(task_values) / len(task_values)
    require(abs(recomputed - metrics["mmad_overall_macro"]) < 1e-15,
            "Overall is not the mean of seven task/source macros")
    require(metrics["rows"] == 39670, "full metric row count drift")
    require(metrics["correct"] == 32391, "full correct count drift")
    require(metrics["invalid"] == 45, "invalid count drift")

    require(ledger["status"] == "COMPLETE", "run ledger is not complete")
    require(ledger["coverage"]["disjoint_exact_cover"] is True,
            "run ledger exact-cover flag is false")
    require(abs(ledger["metrics"]["mmad_overall_percent"]
                - 100 * metrics["mmad_overall_macro"]) < 1e-12,
            "ledger/analysis score mismatch")

    require(clean["rows"] == 5600, "clean panel row count drift")
    delta = clean["paired_delta_treatment_minus_control"]["mmad_overall_macro"]
    require(abs(delta - 0.0026977664813337965) < 1e-15,
            "clean paired delta drift")
    require(clean["mcnemar_exact"]["overall"]["two_sided_exact_p"]
            == 0.13321061920721333, "clean McNemar p drift")

    require(metadata["title"].startswith("[Research]"),
            "archive title must begin with [Research]")
    require(abs(metadata["full_result"]["overall_percent"]
                - 100 * metrics["mmad_overall_macro"]) < 1e-12,
            "metadata score drift")

    echo_scores = comparison["models"]["ECHO-CAVER-v1"]["scores_percent"]
    require(abs(echo_scores[-1] - 100 * metrics["mmad_overall_macro"]) < 1e-6,
            "paper comparison ECHO Overall drift")
    require(comparison["models"]["JUDO"]["scores_percent"][-1] == 81.20,
            "JUDO paper score drift")
    require(comparison["models"]["AD-Copilot-Thinking"]["scores_percent"][-1] == 82.29,
            "AD-Copilot-Thinking paper score drift")

    manifest_path = ROOT / "ARCHIVE_MANIFEST.sha256"
    require(manifest_path.is_file(), "missing ARCHIVE_MANIFEST.sha256")
    manifest_entries = 0
    for line in manifest_path.read_text().splitlines():
        if not line.strip():
            continue
        expected, relative = line.split("  ", 1)
        path = ROOT / relative
        require(path.is_file(), f"manifest entry is missing: {relative}")
        require(sha256_file(path) == expected,
                f"archive manifest drift in {relative}")
        manifest_entries += 1
    require(manifest_entries >= 25, "archive manifest is unexpectedly small")

    print("ECHO-CAVER archive verification: PASS")
    print(f"  Full rows: {metrics['rows']:,}")
    print(f"  Full Overall: {100 * metrics['mmad_overall_macro']:.6f}%")
    print(f"  Manifest entries: {manifest_entries}")
    print(f"  Checkpoint: {EXPECTED['checkpoints/echo-caver-v1/adapter_model.safetensors']}")


if __name__ == "__main__":
    main()
