#!/usr/bin/env python3
"""Fail-closed publication analysis for the nine-arm ECHO-CAVER ablation.

The input is the frozen component-disjoint 5,600-row ECHO screen and one
result JSONL per preregistered arm.  The analyzer reuses the strict MMAD
manifest/result, exact-answer, ITT, source/task macro, and exact-McNemar
contracts from :mod:`analyze_paired_adapter_eval`.

All confidence intervals are obtained from one common bootstrap draw matrix.
Each draw resamples query/reference *connected components*, and every arm is
evaluated on that same draw.  This both avoids pseudoreplication and preserves
the covariance required by the writer-reader interaction

    (W1R1 - W1R0) - (W0R1 - W0R0).

The script writes a machine-readable publication summary plus individual
paired reports.  It never changes checkpoints or inference artifacts.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from analyze_paired_adapter_eval import (
    AD_TASK,
    HEADLINE_METRICS,
    MMAD_TASKS,
    component_clusters,
    evaluate_one,
    index_unique,
    infer_scope,
    mcnemar_strata,
    metric_delta,
    paired_transitions,
    percentile,
    read_jsonl,
    require_result_metadata,
    safe_rate,
    score_run,
    source_strata,
    validate_manifest_row,
    validate_run_contracts,
    write_json,
)
from paired_eval_contract import canonical_sha256, sha256_file


ANALYSIS_SCHEMA_VERSION = "echo-caver-publication-ablation-analysis-v1"
PAIR_SCHEMA_VERSION = "echo-caver-publication-paired-report-v1"
BOOTSTRAP_SCHEMA_VERSION = "echo-caver-common-component-bootstrap-v1"
EXPECTED_MANIFEST_SHA256 = (
    "cedf23061dc8d95d255fbadd8975de9a7a45a597946b525d83a423f431ee5b33"
)
EXPECTED_ROWS = 5_600
DEFAULT_REPLICATES = 50_000
DEFAULT_SEED = 20260829

ARMS = (
    "judo",
    "freb",
    "echo_full",
    "reader_only",
    "unconstrained",
    "random_subspace",
    "answer_only",
    "ce_only",
    "ce_kl",
)
TREATMENTS_VS_JUDO = tuple(arm for arm in ARMS if arm != "judo")
ECHO_CONTROLS = (
    "freb",
    "reader_only",
    "unconstrained",
    "random_subspace",
    "answer_only",
    "ce_only",
    "ce_kl",
)
FACTORIAL = {
    "W0R0": "judo",
    "W1R0": "freb",
    "W0R1": "reader_only",
    "W1R1": "echo_full",
}


class AnalysisContractError(ValueError):
    """Raised when an input cannot support the preregistered publication claim."""


def _rename_pair(value: Any, control: str, treatment: str) -> Any:
    """Replace inherited baseline/candidate labels without changing values."""

    if isinstance(value, dict):
        renamed: Dict[str, Any] = {}
        for key, child in value.items():
            new_key = str(key).replace("baseline", control).replace(
                "candidate", treatment
            )
            renamed[new_key] = _rename_pair(child, control, treatment)
        return renamed
    if isinstance(value, list):
        return [_rename_pair(child, control, treatment) for child in value]
    return value


def _pair_records(
    sample_ids: Sequence[str],
    manifest: Mapping[str, Mapping[str, Any]],
    evaluated: Mapping[str, Mapping[str, Mapping[str, Any]]],
    clusters: Mapping[str, str],
    control: str,
    treatment: str,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for sample_id in sample_ids:
        row = manifest[sample_id]
        records.append(
            {
                "sample_id": sample_id,
                "source": str(row["source"]),
                "question_type": str(row["question_type"]),
                "label": str(row["label"]),
                "cluster_id": clusters[sample_id],
                "baseline": evaluated[control][sample_id],
                "candidate": evaluated[treatment][sample_id],
            }
        )
    return records


def _atomic_cell(row: Mapping[str, Any]) -> str:
    source = str(row["source"])
    task = str(row["question_type"])
    if task == AD_TASK:
        label = str(row["label"])
        if label not in {"normal", "anomaly"}:
            raise AnalysisContractError(f"invalid AD label in {row['sample_id']}: {label}")
        return f"{source}::{task}::{label}"
    return f"{source}::{task}"


def _score_sufficient(
    cell_names: Sequence[str],
    counts: Sequence[float],
    correct: Sequence[float],
) -> Dict[str, Optional[float]]:
    """Compute inherited headline metrics from bootstrap sufficient statistics."""

    if len(cell_names) != len(counts) or len(counts) != len(correct):
        raise AnalysisContractError("bootstrap sufficient-statistic shape mismatch")
    count_by_cell = {name: float(value) for name, value in zip(cell_names, counts)}
    correct_by_cell = {name: float(value) for name, value in zip(cell_names, correct)}

    non_ad_cells: List[float] = []
    overall_cells: List[float] = []
    ad_normal_count = ad_normal_correct = 0.0
    ad_anomaly_count = ad_anomaly_correct = 0.0
    sources = sorted({name.split("::", 1)[0] for name in cell_names})
    for source in sources:
        for task in MMAD_TASKS:
            if task == AD_TASK:
                normal_key = f"{source}::{task}::normal"
                anomaly_key = f"{source}::{task}::anomaly"
                normal_count = count_by_cell.get(normal_key, 0.0)
                anomaly_count = count_by_cell.get(anomaly_key, 0.0)
                if normal_count > 0 and anomaly_count > 0:
                    overall_cells.append(
                        0.5
                        * (
                            correct_by_cell.get(normal_key, 0.0) / normal_count
                            + correct_by_cell.get(anomaly_key, 0.0) / anomaly_count
                        )
                    )
                ad_normal_count += normal_count
                ad_normal_correct += correct_by_cell.get(normal_key, 0.0)
                ad_anomaly_count += anomaly_count
                ad_anomaly_correct += correct_by_cell.get(anomaly_key, 0.0)
            else:
                key = f"{source}::{task}"
                count = count_by_cell.get(key, 0.0)
                if count > 0:
                    value = correct_by_cell.get(key, 0.0) / count
                    non_ad_cells.append(value)
                    overall_cells.append(value)

    total_count = sum(float(value) for value in counts)
    total_correct = sum(float(value) for value in correct)
    tnr = ad_normal_correct / ad_normal_count if ad_normal_count else None
    tpr = ad_anomaly_correct / ad_anomaly_count if ad_anomaly_count else None
    return {
        "itt_accuracy": total_correct / total_count if total_count else None,
        "mmad_overall_macro": (
            sum(overall_cells) / len(overall_cells) if overall_cells else None
        ),
        "non_ad_six_task_macro": (
            sum(non_ad_cells) / len(non_ad_cells) if non_ad_cells else None
        ),
        "ad_balanced_accuracy": (
            0.5 * (tnr + tpr) if tnr is not None and tpr is not None else None
        ),
        "ad_tnr": tnr,
        "ad_tpr": tpr,
    }


def _bootstrap_sufficient_statistics(
    records: Sequence[Mapping[str, Any]],
    arms: Sequence[str],
) -> Tuple[List[str], List[str], List[List[float]], Dict[str, List[List[float]]]]:
    cluster_names = sorted({str(row["cluster_id"]) for row in records})
    if len(cluster_names) < 2:
        raise AnalysisContractError("at least two query/reference components are required")
    cell_names = sorted({_atomic_cell(row) for row in records})
    cluster_index = {name: index for index, name in enumerate(cluster_names)}
    cell_index = {name: index for index, name in enumerate(cell_names)}
    counts = [[0.0 for _ in cell_names] for _ in cluster_names]
    correct = {
        arm: [[0.0 for _ in cell_names] for _ in cluster_names] for arm in arms
    }
    for row in records:
        cluster = cluster_index[str(row["cluster_id"])]
        cell = cell_index[_atomic_cell(row)]
        counts[cluster][cell] += 1.0
        for arm in arms:
            correct[arm][cluster][cell] += float(bool(row[arm]["correct"]))
    return cluster_names, cell_names, counts, correct


def _common_bootstrap_python(
    cell_names: Sequence[str],
    counts: Sequence[Sequence[float]],
    correct: Mapping[str, Sequence[Sequence[float]]],
    arms: Sequence[str],
    replicates: int,
    seed: int,
) -> Dict[str, Dict[str, List[float]]]:
    """Dependency-free reference implementation used for tests/fallback."""

    cluster_count = len(counts)
    cell_count = len(cell_names)
    rng = random.Random(seed)
    draws = {
        arm: {metric: [] for metric in HEADLINE_METRICS} for arm in arms
    }
    for _ in range(replicates):
        multiplicities: Dict[int, int] = defaultdict(int)
        for cluster in rng.choices(range(cluster_count), k=cluster_count):
            multiplicities[int(cluster)] += 1
        sampled_counts = [0.0] * cell_count
        sampled_correct = {arm: [0.0] * cell_count for arm in arms}
        for cluster, multiplicity in multiplicities.items():
            weight = float(multiplicity)
            cluster_counts = counts[cluster]
            for cell in range(cell_count):
                sampled_counts[cell] += weight * float(cluster_counts[cell])
            for arm in arms:
                cluster_correct = correct[arm][cluster]
                target = sampled_correct[arm]
                for cell in range(cell_count):
                    target[cell] += weight * float(cluster_correct[cell])
        for arm in arms:
            metrics = _score_sufficient(cell_names, sampled_counts, sampled_correct[arm])
            for metric in HEADLINE_METRICS:
                value = metrics[metric]
                if value is None:
                    raise AnalysisContractError(
                        f"undefined {metric} in component bootstrap replicate"
                    )
                draws[arm][metric].append(float(value))
    return draws


def _common_bootstrap_numpy(
    cell_names: Sequence[str],
    counts: Sequence[Sequence[float]],
    correct: Mapping[str, Sequence[Sequence[float]]],
    arms: Sequence[str],
    replicates: int,
    seed: int,
    chunk_size: int,
) -> Tuple[Dict[str, Dict[str, List[float]]], str]:
    """Batched exact multinomial component bootstrap for publication runs."""

    import numpy as np

    count_matrix = np.asarray(counts, dtype=np.float64)
    correct_matrix = np.concatenate(
        [np.asarray(correct[arm], dtype=np.float64) for arm in arms], axis=1
    )
    cluster_count, cell_count = count_matrix.shape
    probabilities = np.full(cluster_count, 1.0 / cluster_count, dtype=np.float64)
    rng = np.random.Generator(np.random.PCG64(seed))
    draws = {
        arm: {metric: [] for metric in HEADLINE_METRICS} for arm in arms
    }
    completed = 0
    while completed < replicates:
        current = min(chunk_size, replicates - completed)
        weights = rng.multinomial(cluster_count, probabilities, size=current).astype(
            np.float64, copy=False
        )
        sampled_counts = weights @ count_matrix
        sampled_correct_flat = weights @ correct_matrix
        for row_index in range(current):
            counts_row = sampled_counts[row_index].tolist()
            for arm_index, arm in enumerate(arms):
                start = arm_index * cell_count
                correct_row = sampled_correct_flat[
                    row_index, start : start + cell_count
                ].tolist()
                metrics = _score_sufficient(cell_names, counts_row, correct_row)
                for metric in HEADLINE_METRICS:
                    value = metrics[metric]
                    if value is None:
                        raise AnalysisContractError(
                            f"undefined {metric} in component bootstrap replicate"
                        )
                    draws[arm][metric].append(float(value))
        completed += current
    return draws, str(np.__version__)


def common_component_bootstrap(
    records: Sequence[Mapping[str, Any]],
    arms: Sequence[str],
    *,
    replicates: int,
    seed: int,
    chunk_size: int = 128,
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, List[float]]]]:
    if replicates <= 0:
        raise AnalysisContractError("bootstrap replicates must be positive")
    if chunk_size <= 0:
        raise AnalysisContractError("bootstrap chunk size must be positive")
    if not records or any(row.get("cluster_id") is None for row in records):
        raise AnalysisContractError(
            "complete query/reference connected-component IDs are required"
        )
    cluster_names, cell_names, counts, correct = _bootstrap_sufficient_statistics(
        records, arms
    )
    aggregate_counts = [sum(row[cell] for row in counts) for cell in range(len(cell_names))]
    for arm in arms:
        aggregate_correct = [
            sum(row[cell] for row in correct[arm]) for cell in range(len(cell_names))
        ]
        sufficient = _score_sufficient(cell_names, aggregate_counts, aggregate_correct)
        strict_records = [
            {
                "source": row["source"],
                "question_type": row["question_type"],
                "label": row["label"],
                "candidate": row[arm],
            }
            for row in records
        ]
        strict = score_run(strict_records, "candidate")
        for metric in HEADLINE_METRICS:
            left, right = sufficient[metric], strict[metric]
            if left is None or right is None or abs(float(left) - float(right)) > 1e-12:
                raise AnalysisContractError(
                    f"bootstrap sufficient statistics drift from strict scorer: "
                    f"arm={arm} metric={metric} sufficient={left} strict={right}"
                )
    try:
        draws, numpy_version = _common_bootstrap_numpy(
            cell_names,
            counts,
            correct,
            arms,
            replicates,
            seed,
            chunk_size,
        )
        engine = "numpy.random.PCG64 multinomial"
    except ImportError:
        draws = _common_bootstrap_python(
            cell_names, counts, correct, arms, replicates, seed
        )
        numpy_version = None
        engine = "python.random.Random.choices reference fallback"
    return (
        {
            "schema_version": BOOTSTRAP_SCHEMA_VERSION,
            "available": True,
            "resampling_unit": "query-reference connected component",
            "clusters": len(cluster_names),
            "atomic_source_task_label_cells": len(cell_names),
            "replicates": replicates,
            "seed": seed,
            "chunk_size": chunk_size,
            "engine": engine,
            "numpy_version": numpy_version,
            "same_component_draws_shared_by_all_arms": True,
            "strict_point_estimate_parity": True,
        },
        draws,
    )


def _bootstrap_metric_summary(
    control_values: Sequence[float], treatment_values: Sequence[float]
) -> Dict[str, Any]:
    if len(control_values) != len(treatment_values) or not control_values:
        raise AnalysisContractError("paired bootstrap draw count mismatch")
    deltas = [
        float(treatment) - float(control)
        for control, treatment in zip(control_values, treatment_values)
    ]
    return {
        "control_ci95": [
            percentile(control_values, 0.025),
            percentile(control_values, 0.975),
        ],
        "treatment_ci95": [
            percentile(treatment_values, 0.025),
            percentile(treatment_values, 0.975),
        ],
        "delta_ci95": [percentile(deltas, 0.025), percentile(deltas, 0.975)],
        "probability_delta_gt_zero": safe_rate(
            sum(value > 0.0 for value in deltas), len(deltas)
        ),
        "defined_replicates": len(deltas),
    }


def _paired_report(
    records: Sequence[Mapping[str, Any]],
    control: str,
    treatment: str,
    draws: Mapping[str, Mapping[str, Sequence[float]]],
    bootstrap_contract: Mapping[str, Any],
    artifact_provenance: Mapping[str, Any],
) -> Dict[str, Any]:
    control_metrics = score_run(records, "baseline")
    treatment_metrics = score_run(records, "candidate")
    bootstrap_metrics: Dict[str, Any] = {}
    for metric in HEADLINE_METRICS:
        summary = _bootstrap_metric_summary(
            draws[control][metric], draws[treatment][metric]
        )
        summary.update(
            {
                "control_point": control_metrics[metric],
                "treatment_point": treatment_metrics[metric],
                "delta_point": metric_delta(
                    control_metrics, treatment_metrics
                )[metric],
            }
        )
        bootstrap_metrics[metric] = summary
    return {
        "analysis_schema_version": PAIR_SCHEMA_VERSION,
        "control_arm": control,
        "treatment_arm": treatment,
        "delta_direction": f"{treatment} minus {control}",
        "rows": len(records),
        "provenance": dict(artifact_provenance),
        "control_metrics": control_metrics,
        "treatment_metrics": treatment_metrics,
        "paired_delta_treatment_minus_control": metric_delta(
            control_metrics, treatment_metrics
        ),
        "source_strata": _rename_pair(source_strata(records), control, treatment),
        "paired_transitions": _rename_pair(
            paired_transitions(records), control, treatment
        ),
        "mcnemar_exact": _rename_pair(mcnemar_strata(records), control, treatment),
        "mcnemar_scope_note": (
            "Exact two-sided row-paired McNemar is reported for continuity; "
            "the connected-component bootstrap is the dependence-aware CI."
        ),
        "cluster_bootstrap": {
            **bootstrap_contract,
            "metrics": bootstrap_metrics,
        },
    }


def _interaction_report(
    arm_metrics: Mapping[str, Mapping[str, Any]],
    draws: Mapping[str, Mapping[str, Sequence[float]]],
    bootstrap_contract: Mapping[str, Any],
) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}
    for metric in HEADLINE_METRICS:
        point = (
            float(arm_metrics["echo_full"][metric])
            - float(arm_metrics["freb"][metric])
            - float(arm_metrics["reader_only"][metric])
            + float(arm_metrics["judo"][metric])
        )
        interaction_draws = [
            echo - freb - reader + judo
            for echo, freb, reader, judo in zip(
                draws["echo_full"][metric],
                draws["freb"][metric],
                draws["reader_only"][metric],
                draws["judo"][metric],
            )
        ]
        metrics[metric] = {
            "interaction_point": point,
            "interaction_ci95": [
                percentile(interaction_draws, 0.025),
                percentile(interaction_draws, 0.975),
            ],
            "probability_interaction_gt_zero": safe_rate(
                sum(value > 0.0 for value in interaction_draws),
                len(interaction_draws),
            ),
            "defined_replicates": len(interaction_draws),
        }
    return {
        "analysis_schema_version": "echo-caver-writer-reader-interaction-v1",
        "factorial_arms": FACTORIAL,
        "contrast": "(W1R1 - W1R0) - (W0R1 - W0R0)",
        "expanded_contrast": "(echo_full - freb) - (reader_only - judo)",
        "interpretation": (
            "A positive interaction supports a reader benefit conditional on the "
            "FREB writer; it does not by itself establish visual causality."
        ),
        "cluster_bootstrap": {**bootstrap_contract, "metrics": metrics},
    }


def _validate_inputs(
    manifest_path: Path, results_dir: Path
) -> Tuple[
    List[Dict[str, Any]],
    Dict[str, Dict[str, Any]],
    Dict[str, Dict[str, Dict[str, Any]]],
    Dict[str, Any],
    Dict[str, Dict[str, Any]],
]:
    manifest_path = manifest_path.resolve(strict=True)
    results_dir = results_dir.resolve(strict=True)
    if not results_dir.is_dir():
        raise AnalysisContractError(f"results path is not a directory: {results_dir}")
    manifest_digest = sha256_file(manifest_path)
    if manifest_digest != EXPECTED_MANIFEST_SHA256:
        raise AnalysisContractError(
            "publication analyzer requires the frozen clean5600 manifest: "
            f"expected={EXPECTED_MANIFEST_SHA256} got={manifest_digest}"
        )
    manifest_rows = read_jsonl(manifest_path)
    if len(manifest_rows) != EXPECTED_ROWS:
        raise AnalysisContractError(
            f"clean5600 row count drift: expected={EXPECTED_ROWS} got={len(manifest_rows)}"
        )
    manifest = index_unique(manifest_rows, "manifest")
    for row in manifest_rows:
        validate_manifest_row(row)

    expected_files = {f"{arm}.jsonl" for arm in ARMS}
    present_files = {path.name for path in results_dir.glob("*.jsonl")}
    if present_files != expected_files:
        raise AnalysisContractError(
            "nine-arm result file set mismatch: "
            f"missing={sorted(expected_files - present_files)} "
            f"unexpected={sorted(present_files - expected_files)}"
        )
    runs = {
        arm: index_unique(read_jsonl(results_dir / f"{arm}.jsonl"), arm)
        for arm in ARMS
    }
    expected_ids = set(manifest)
    for arm, run in runs.items():
        if set(run) != expected_ids:
            raise AnalysisContractError(
                f"{arm} ID coverage mismatch: expected={len(expected_ids)} got={len(run)}"
            )
        infer_scope("full-7-task", manifest, set(run))
        for sample_id in sorted(run):
            require_result_metadata(
                run[sample_id], manifest[sample_id], manifest_digest, arm
            )

    provenance: Dict[str, Dict[str, Any]] = {
        "judo": {
            "adapter_identity_sha256": None,
            "adapter_path": None,
            "loaded_adapter_parameter_count": 0,
        }
    }
    shared_contract: Optional[Dict[str, Any]] = None
    for arm in TREATMENTS_VS_JUDO:
        validated = validate_run_contracts(
            manifest_rows, manifest_digest, runs["judo"], runs[arm]
        )
        contract = validated["inference_contract"]
        if shared_contract is None:
            shared_contract = contract
        elif contract != shared_contract:
            raise AnalysisContractError(f"{arm} inference contract differs")
        first = next(iter(runs[arm].values()))
        provenance[arm] = {
            "adapter_identity_sha256": validated[
                "candidate_adapter_identity_sha256"
            ],
            "adapter_path": validated["candidate_adapter_path"],
            "loaded_adapter_parameter_count": (
                first.get("loaded_adapter_parameter_count")
                if "loaded_adapter_parameter_count" in first
                else first.get("loaded_lora_parameter_count")
            ),
        }
    assert shared_contract is not None
    candidate_identities = [
        provenance[arm]["adapter_identity_sha256"] for arm in TREATMENTS_VS_JUDO
    ]
    if len(set(candidate_identities)) != len(candidate_identities):
        raise AnalysisContractError(
            "two treatment arms have the same adapter/runtime artifact identity"
        )
    contract_digests = {
        str(next(iter(run.values()))["inference_contract_sha256"])
        for run in runs.values()
    }
    if len(contract_digests) != 1:
        raise AnalysisContractError("nine arms do not share one inference contract hash")
    return manifest_rows, manifest, runs, shared_contract, provenance


def analyze(
    manifest_path: Path,
    results_dir: Path,
    *,
    bootstrap_replicates: int = DEFAULT_REPLICATES,
    seed: int = DEFAULT_SEED,
    bootstrap_chunk_size: int = 128,
) -> Dict[str, Any]:
    manifest_rows, manifest, runs, inference_contract, provenance = _validate_inputs(
        manifest_path, results_dir
    )
    selected_ids = [str(row["sample_id"]) for row in manifest_rows]
    clusters, cluster_reason = component_clusters(manifest_rows)
    if clusters is None:
        raise AnalysisContractError(str(cluster_reason))
    if not all(
        row.get("component_near_duplicate_audit_complete") is True
        for row in manifest_rows
    ):
        raise AnalysisContractError(
            "clean5600 near-duplicate component audit is not complete"
        )

    evaluated = {
        arm: {
            sample_id: evaluate_one(runs[arm][sample_id], manifest[sample_id])
            for sample_id in selected_ids
        }
        for arm in ARMS
    }
    multi_records: List[Dict[str, Any]] = []
    for sample_id in selected_ids:
        row = manifest[sample_id]
        record: Dict[str, Any] = {
            "sample_id": sample_id,
            "source": str(row["source"]),
            "question_type": str(row["question_type"]),
            "label": str(row["label"]),
            "cluster_id": clusters[sample_id],
        }
        for arm in ARMS:
            record[arm] = evaluated[arm][sample_id]
        multi_records.append(record)

    arm_metrics: Dict[str, Dict[str, Any]] = {}
    for arm in ARMS:
        arm_records = [
            {
                **{
                    key: row[key]
                    for key in (
                        "sample_id",
                        "source",
                        "question_type",
                        "label",
                        "cluster_id",
                    )
                },
                "candidate": row[arm],
            }
            for row in multi_records
        ]
        arm_metrics[arm] = score_run(arm_records, "candidate")

    bootstrap_contract, draws = common_component_bootstrap(
        multi_records,
        ARMS,
        replicates=bootstrap_replicates,
        seed=seed,
        chunk_size=bootstrap_chunk_size,
    )
    bootstrap_contract.update(
        {
            "near_duplicate_audit_complete": True,
            "claim_status": (
                "component-cluster CI with completed near-duplicate audit"
            ),
        }
    )

    manifest_resolved = manifest_path.resolve()
    results_resolved = results_dir.resolve()
    result_files = {
        arm: {
            "path": str((results_resolved / f"{arm}.jsonl").resolve()),
            "sha256": sha256_file(results_resolved / f"{arm}.jsonl"),
        }
        for arm in ARMS
    }
    inference_contract_sha256 = canonical_sha256(inference_contract)

    def pair_provenance(control: str, treatment: str) -> Dict[str, Any]:
        return {
            "manifest": str(manifest_resolved),
            "manifest_sha256": sha256_file(manifest_resolved),
            "inference_contract_sha256": inference_contract_sha256,
            "control_result": result_files[control],
            "treatment_result": result_files[treatment],
            "control_adapter": provenance[control],
            "treatment_adapter": provenance[treatment],
        }

    arm_vs_judo: Dict[str, Dict[str, Any]] = {}
    for arm in TREATMENTS_VS_JUDO:
        records = _pair_records(
            selected_ids, manifest, evaluated, clusters, "judo", arm
        )
        arm_vs_judo[arm] = _paired_report(
            records,
            "judo",
            arm,
            draws,
            bootstrap_contract,
            pair_provenance("judo", arm),
        )

    echo_vs_controls: Dict[str, Dict[str, Any]] = {}
    for control in ECHO_CONTROLS:
        records = _pair_records(
            selected_ids, manifest, evaluated, clusters, control, "echo_full"
        )
        echo_vs_controls[control] = _paired_report(
            records,
            control,
            "echo_full",
            draws,
            bootstrap_contract,
            pair_provenance(control, "echo_full"),
        )

    analyzer_path = Path(__file__).resolve()
    helper_path = analyzer_path.with_name("analyze_paired_adapter_eval.py")
    contract_path = analyzer_path.with_name("paired_eval_contract.py")
    interaction = _interaction_report(arm_metrics, draws, bootstrap_contract)
    interaction["provenance"] = {
        "manifest": str(manifest_resolved),
        "manifest_sha256": sha256_file(manifest_resolved),
        "inference_contract_sha256": inference_contract_sha256,
        "factorial_result_files": {
            cell: result_files[arm] for cell, arm in FACTORIAL.items()
        },
        "factorial_adapter_provenance": {
            cell: provenance[arm] for cell, arm in FACTORIAL.items()
        },
    }
    return {
        "analysis_schema_version": ANALYSIS_SCHEMA_VERSION,
        "status": "COMPLETE",
        "scope": "clean5600-full-7-task",
        "rows": len(manifest_rows),
        "manifest": str(manifest_resolved),
        "manifest_sha256": sha256_file(manifest_resolved),
        "results_directory": str(results_resolved),
        "result_files": result_files,
        "implementation": {
            "analyzer": str(analyzer_path),
            "analyzer_sha256": sha256_file(analyzer_path),
            "strict_paired_helper_sha256": sha256_file(helper_path),
            "paired_eval_contract_sha256": sha256_file(contract_path),
        },
        "provenance": {
            "inference_contract_sha256": inference_contract_sha256,
            "inference_contract": inference_contract,
            "arms": provenance,
        },
        "answer_policy": {
            "parser": (
                "exactly one case-sensitive <answer>X</answer> tag; X must be in options"
            ),
            "invalid_scoring": "incorrect under ITT; never coerced",
            "prediction_source": "completion only; stored prediction fields ignored",
        },
        "arm_metrics": arm_metrics,
        "bootstrap_contract": bootstrap_contract,
        "arm_vs_judo": arm_vs_judo,
        "echo_full_vs_controls": echo_vs_controls,
        "writer_reader_interaction": interaction,
        "multiplicity_note": (
            "All comparisons and raw two-sided p-values are reported; no arm was "
            "selected or tuned on this panel. Interpret secondary contrasts with "
            "their paired CIs and the preregistered mechanism hierarchy."
        ),
    }


def write_reports(output_dir: Path, report: Mapping[str, Any]) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    arm_dir = output_dir / "arm_vs_judo"
    control_dir = output_dir / "echo_full_vs_controls"
    for arm, paired in report["arm_vs_judo"].items():
        write_json(arm_dir / f"{arm}_vs_judo.json", paired)
    for control, paired in report["echo_full_vs_controls"].items():
        write_json(control_dir / f"echo_full_vs_{control}.json", paired)
    write_json(
        output_dir / "writer_reader_interaction.json",
        report["writer_reader_interaction"],
    )
    write_json(output_dir / "publication_summary.json", report)
    files = sorted(path for path in output_dir.rglob("*.json"))
    receipt = {
        "schema_version": "echo-caver-publication-analysis-output-receipt-v1",
        "status": "COMPLETE",
        "analysis_schema_version": report["analysis_schema_version"],
        "files": {
            str(path.relative_to(output_dir)): {
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for path in files
        },
    }
    write_json(output_dir / "OUTPUT_RECEIPT.json", receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--bootstrap-replicates", type=int, default=DEFAULT_REPLICATES
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--bootstrap-chunk-size", type=int, default=128)
    args = parser.parse_args()
    report = analyze(
        args.manifest,
        args.results_dir,
        bootstrap_replicates=args.bootstrap_replicates,
        seed=args.seed,
        bootstrap_chunk_size=args.bootstrap_chunk_size,
    )
    receipt = write_reports(args.output_dir, report)
    print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
