#!/usr/bin/env python3
"""Strict paired analysis of a frozen JUDO run and one model-adapter run.

The completion itself is the sole prediction source.  Only one byte-exact
``<answer>X</answer>`` tag is valid; malformed output is an intention-to-treat
error and is never coerced to the normal class.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import random
from typing import Any, Iterable, Mapping, Sequence

from paired_eval_contract import (
    RESULT_METADATA_FIELDS,
    RESULT_SCHEMA_VERSION,
    ad_semantics,
    canonical_sha256,
    manifest_metadata,
    manifest_row_sha256,
    option_keys,
    parse_exact_answer,
    sha256_file,
)


AD_TASK = "Anomaly Detection"
MMAD_TASKS = (
    "Anomaly Detection",
    "Defect Analysis",
    "Defect Classification",
    "Defect Description",
    "Defect Localization",
    "Object Analysis",
    "Object Classification",
)
HEADLINE_METRICS = (
    "itt_accuracy",
    "mmad_overall_macro",
    "non_ad_six_task_macro",
    "ad_balanced_accuracy",
    "ad_tnr",
    "ad_tpr",
)


def loaded_adapter_parameter_count(result: Mapping[str, Any]) -> Any:
    """Read native/generic results while retaining legacy PEFT compatibility."""

    if "loaded_adapter_parameter_count" in result:
        return result.get("loaded_adapter_parameter_count")
    return result.get("loaded_lora_parameter_count")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            try:
                value = json.loads(raw)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {error}") from error
            if not isinstance(value, dict):
                raise ValueError(f"expected object at {path}:{line_number}")
            rows.append(value)
    if not rows:
        raise ValueError(f"empty JSONL: {path}")
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    temporary.replace(path)


def index_unique(rows: Iterable[dict[str, Any]], name: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for position, row in enumerate(rows, 1):
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError(f"{name} row {position} has an empty/non-string sample_id")
        if sample_id in indexed:
            raise ValueError(f"{name} contains duplicate sample_id {sample_id}")
        indexed[sample_id] = row
    return indexed


def safe_rate(numerator: int | float, denominator: int) -> float | None:
    return float(numerator) / denominator if denominator else None


def mean_defined(values: Iterable[float | None]) -> float | None:
    defined = [float(value) for value in values if value is not None]
    return sum(defined) / len(defined) if defined else None


def percentile(values: Sequence[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


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
        if left_root == right_root:
            return
        if left_root > right_root:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root


def component_clusters(rows: Sequence[Mapping[str, Any]]) -> tuple[dict[str, str] | None, str | None]:
    availability = [
        isinstance(row.get("query_component_id"), str)
        and bool(row.get("query_component_id"))
        and isinstance(row.get("reference_component_id"), str)
        and bool(row.get("reference_component_id"))
        for row in rows
    ]
    if not any(availability):
        return None, "query/reference component IDs are unavailable"
    if not all(availability):
        raise ValueError("manifest mixes complete and incomplete query/reference component IDs")
    union = UnionFind()
    for row in rows:
        union.union(
            f"query-reference-component::{row['query_component_id']}",
            f"query-reference-component::{row['reference_component_id']}",
        )
    return (
        {
            str(row["sample_id"]): union.find(
                f"query-reference-component::{row['query_component_id']}"
            )
            for row in rows
        },
        None,
    )


def validate_manifest_row(row: Mapping[str, Any]) -> None:
    sample_id = row.get("sample_id")
    required = (
        "source",
        "question_type",
        "label",
        "correct_answer",
        "options",
        "options_text",
        "question",
        "image",
        "template_image",
    )
    missing = [field for field in required if field not in row]
    if missing:
        raise ValueError(f"manifest {sample_id} missing required fields: {missing}")
    keys = option_keys(row["options"])
    if str(row["correct_answer"]) not in keys:
        raise ValueError(f"manifest {sample_id} correct_answer is absent from options")
    if row["question_type"] == AD_TASK:
        semantics = ad_semantics(row["options"], allow_distractors=True)
        label = str(row["label"])
        if label not in {"normal", "anomaly"}:
            raise ValueError(f"manifest {sample_id} has invalid AD label {label!r}")
        if semantics.get(str(row["correct_answer"])) != label:
            raise ValueError(
                f"manifest {sample_id} AD label/correct_answer/options are inconsistent"
            )


def infer_scope(
    requested: str,
    manifest: Mapping[str, Mapping[str, Any]],
    result_ids: set[str],
) -> tuple[str, set[str]]:
    all_ids = set(manifest)
    ad_ids = {
        sample_id
        for sample_id, row in manifest.items()
        if row.get("question_type") == AD_TASK
    }
    if not ad_ids:
        raise ValueError("manifest contains no Anomaly Detection rows")
    tasks = {str(row.get("question_type")) for row in manifest.values()}
    if requested == "full-7-task":
        if tasks != set(MMAD_TASKS):
            raise ValueError(f"full-7-task requires the canonical seven tasks; got {sorted(tasks)}")
        if result_ids != all_ids:
            raise ValueError(
                f"full result IDs must exactly equal manifest IDs: expected={len(all_ids)} got={len(result_ids)}"
            )
        return requested, all_ids
    if requested == "ad-only":
        if result_ids not in (ad_ids, all_ids):
            raise ValueError(
                "ad-only accepts either exact AD outputs or identical complete-manifest outputs; "
                f"AD={len(ad_ids)} full={len(all_ids)} got={len(result_ids)}"
            )
        return requested, ad_ids
    if requested != "auto":
        raise ValueError(f"unknown scope: {requested}")
    if result_ids == ad_ids:
        return "ad-only", ad_ids
    if result_ids == all_ids:
        if tasks != set(MMAD_TASKS):
            raise ValueError(f"auto full scope is not the canonical seven tasks: {sorted(tasks)}")
        return "full-7-task", all_ids
    raise ValueError(
        "cannot infer scope: paired result IDs are neither exact AD IDs nor exact full manifest IDs"
    )


def require_result_metadata(
    result: Mapping[str, Any],
    manifest_row: Mapping[str, Any],
    manifest_sha256: str,
    name: str,
) -> None:
    sample_id = str(manifest_row["sample_id"])
    if result.get("result_schema_version") != RESULT_SCHEMA_VERSION:
        raise ValueError(
            f"{name} {sample_id} lacks strict result schema {RESULT_SCHEMA_VERSION}; rerun inference"
        )
    if result.get("manifest_sha256") != manifest_sha256:
        raise ValueError(f"{name} {sample_id} manifest_sha256 mismatch")
    expected_row_hash = manifest_row_sha256(manifest_row)
    if result.get("manifest_row_sha256") != expected_row_hash:
        raise ValueError(f"{name} {sample_id} manifest_row_sha256 mismatch")
    expected_metadata = manifest_metadata(manifest_row)
    for field in RESULT_METADATA_FIELDS:
        if field not in result:
            raise ValueError(f"{name} {sample_id} missing metadata field {field}")
        if result[field] != expected_metadata[field]:
            raise ValueError(
                f"{name} {sample_id} metadata mismatch for {field}: "
                f"expected={expected_metadata[field]!r} got={result[field]!r}"
            )
    if not isinstance(result.get("completion"), str):
        raise ValueError(f"{name} {sample_id} completion must be a string")


def validate_run_contracts(
    manifest_rows: Sequence[Mapping[str, Any]],
    manifest_sha256: str,
    baseline: Mapping[str, Mapping[str, Any]],
    candidate: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    result_ids = set(baseline)
    expected_order = [str(row["sample_id"]) for row in manifest_rows if row["sample_id"] in result_ids]
    expected_selected_hash = canonical_sha256(expected_order)

    contracts: dict[str, dict[str, Any]] = {}
    digests: dict[str, str] = {}
    for name, run in (("baseline", baseline), ("candidate", candidate)):
        seen_digests = {row.get("inference_contract_sha256") for row in run.values()}
        if None in seen_digests or len(seen_digests) != 1:
            raise ValueError(f"{name} has missing or mixed inference contract digests")
        digest = str(next(iter(seen_digests)))
        contract_values = [row.get("inference_contract") for row in run.values()]
        if any(not isinstance(value, dict) for value in contract_values):
            raise ValueError(f"{name} has missing/non-object inference_contract")
        contract_hashes = {canonical_sha256(value) for value in contract_values}
        if contract_hashes != {digest}:
            raise ValueError(f"{name} inference contract payload does not match its digest")
        contract = dict(contract_values[0])
        if contract.get("manifest_sha256") != manifest_sha256:
            raise ValueError(f"{name} inference contract manifest hash mismatch")
        if contract.get("selected_ids_sha256") != expected_selected_hash:
            raise ValueError(f"{name} inference contract selected-ID hash mismatch")
        if int(contract.get("selected_rows", -1)) != len(result_ids):
            raise ValueError(f"{name} inference contract row count mismatch")
        expected_model = contract.get("base_model", {}).get("model_reference")
        expected_typo = contract.get("prompt", {}).get("official_leading_n_typo")
        for sample_id, row in run.items():
            if row.get("model") != expected_model:
                raise ValueError(f"{name} {sample_id} flat model provenance disagrees with contract")
            if row.get("official_prompt_typo") != expected_typo:
                raise ValueError(f"{name} {sample_id} flat prompt provenance disagrees with contract")
        contracts[name], digests[name] = contract, digest
    if digests["baseline"] != digests["candidate"] or contracts["baseline"] != contracts["candidate"]:
        raise ValueError("baseline and candidate inference contracts differ")

    baseline_adapters = {
        (
            row.get("adapter"),
            row.get("adapter_identity_sha256"),
            loaded_adapter_parameter_count(row),
        )
        for row in baseline.values()
    }
    if baseline_adapters != {(None, None, 0)}:
        raise ValueError("baseline must be frozen JUDO with no adapter and zero loaded LoRA parameters")

    candidate_identities = {row.get("adapter_identity_sha256") for row in candidate.values()}
    if None in candidate_identities or len(candidate_identities) != 1:
        raise ValueError("candidate must contain exactly one non-null adapter identity")
    candidate_identity = str(next(iter(candidate_identities)))
    candidate_paths = {row.get("adapter") for row in candidate.values()}
    if None in candidate_paths or "" in candidate_paths or len(candidate_paths) != 1:
        raise ValueError("candidate adapter path is missing or mixed")
    for sample_id, row in candidate.items():
        count = loaded_adapter_parameter_count(row)
        if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
            raise ValueError(f"candidate {sample_id} has no positive loaded adapter parameter count")
        provenance = row.get("adapter_provenance")
        if not isinstance(provenance, dict):
            raise ValueError(f"candidate {sample_id} missing adapter provenance")
        if provenance.get("adapter_identity_sha256") != candidate_identity:
            raise ValueError(f"candidate {sample_id} adapter provenance identity mismatch")

    return {
        "inference_contract_sha256": digests["baseline"],
        "inference_contract": contracts["baseline"],
        "candidate_adapter_identity_sha256": candidate_identity,
        "candidate_adapter_path": next(iter(candidate_paths)),
    }


def evaluate_one(result: Mapping[str, Any], manifest_row: Mapping[str, Any]) -> dict[str, Any]:
    labels = option_keys(manifest_row["options"])
    parsed = parse_exact_answer(result["completion"], labels)
    correct_answer = str(manifest_row["correct_answer"])
    record = {
        "prediction": parsed.label,
        "valid": parsed.valid,
        "invalid_reason": None if parsed.valid else parsed.reason,
        "correct": int(parsed.valid and parsed.label == correct_answer),
        "decision": None,
        "predicts_normal": None,
    }
    if manifest_row["question_type"] == AD_TASK and parsed.valid:
        decision = ad_semantics(
            manifest_row["options"],
            allow_distractors=True,
        ).get(str(parsed.label))
        if decision is not None:
            record["decision"] = decision
            record["predicts_normal"] = decision == "normal"
    return record


def build_records(
    selected_ids: set[str],
    manifest: Mapping[str, Mapping[str, Any]],
    baseline: Mapping[str, Mapping[str, Any]],
    candidate: Mapping[str, Mapping[str, Any]],
    clusters: Mapping[str, str] | None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for sample_id in sorted(selected_ids):
        row = manifest[sample_id]
        records.append(
            {
                "sample_id": sample_id,
                "source": str(row["source"]),
                "question_type": str(row["question_type"]),
                "label": str(row["label"]),
                "cluster_id": clusters[sample_id] if clusters else None,
                "baseline": evaluate_one(baseline[sample_id], row),
                "candidate": evaluate_one(candidate[sample_id], row),
            }
        )
    return records


def score_run(records: Sequence[Mapping[str, Any]], arm: str) -> dict[str, Any]:
    rows = list(records)
    valid = sum(bool(row[arm]["valid"]) for row in rows)
    correct = sum(int(row[arm]["correct"]) for row in rows)
    cell_rows: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        cell_rows[(str(row["source"]), str(row["question_type"]))].append(row)

    cell_scores: dict[str, float | None] = {}
    task_scores: dict[str, list[float | None]] = defaultdict(list)
    for (source, task), group in sorted(cell_rows.items()):
        if task == AD_TASK:
            normal = [item for item in group if item["label"] == "normal"]
            anomaly = [item for item in group if item["label"] == "anomaly"]
            tnr = safe_rate(sum(item[arm]["correct"] for item in normal), len(normal))
            tpr = safe_rate(sum(item[arm]["correct"] for item in anomaly), len(anomaly))
            score = mean_defined((tnr, tpr)) if tnr is not None and tpr is not None else None
        else:
            score = safe_rate(sum(item[arm]["correct"] for item in group), len(group))
        cell_scores[f"{source}::{task}"] = score
        task_scores[task].append(score)

    ad = [row for row in rows if row["question_type"] == AD_TASK]
    normal = [row for row in ad if row["label"] == "normal"]
    anomaly = [row for row in ad if row["label"] == "anomaly"]
    tnr = safe_rate(sum(row[arm]["correct"] for row in normal), len(normal))
    tpr = safe_rate(sum(row[arm]["correct"] for row in anomaly), len(anomaly))
    ad_ba = mean_defined((tnr, tpr)) if tnr is not None and tpr is not None else None
    non_ad_cells = [score for key, score in cell_scores.items() if not key.endswith(f"::{AD_TASK}")]
    valid_correct = sum(row[arm]["correct"] for row in rows if row[arm]["valid"])
    invalid_reasons = Counter(
        str(row[arm]["invalid_reason"]) for row in rows if not row[arm]["valid"]
    )
    return {
        "rows": len(rows),
        "correct": correct,
        "invalid": len(rows) - valid,
        "valid_rate": safe_rate(valid, len(rows)),
        "itt_accuracy": safe_rate(correct, len(rows)),
        "valid_only_accuracy": safe_rate(valid_correct, valid),
        "mmad_overall_macro": mean_defined(cell_scores.values()),
        "non_ad_six_task_macro": mean_defined(non_ad_cells),
        "ad_balanced_accuracy": ad_ba,
        "ad_tnr": tnr,
        "ad_tpr": tpr,
        "per_task_macro": {
            task: mean_defined(scores) for task, scores in sorted(task_scores.items())
        },
        "per_source_task": cell_scores,
        "invalid_reasons": dict(sorted(invalid_reasons.items())),
    }


def compact_stratum(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    baseline = score_run(records, "baseline")
    candidate = score_run(records, "candidate")
    keep = (
        "rows",
        "invalid",
        "valid_rate",
        "itt_accuracy",
        "mmad_overall_macro",
        "non_ad_six_task_macro",
        "ad_balanced_accuracy",
        "ad_tnr",
        "ad_tpr",
        "per_task_macro",
    )
    return {
        "baseline": {key: baseline[key] for key in keep},
        "candidate": {key: candidate[key] for key in keep},
        "paired_delta": metric_delta(baseline, candidate),
        "mcnemar_exact": exact_mcnemar(records),
    }


def source_strata(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[str(row["source"])].append(row)
    return {source: compact_stratum(group) for source, group in sorted(grouped.items())}


def exact_mcnemar(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    baseline_only = sum(
        bool(row["baseline"]["correct"]) and not bool(row["candidate"]["correct"])
        for row in records
    )
    candidate_only = sum(
        bool(row["candidate"]["correct"]) and not bool(row["baseline"]["correct"])
        for row in records
    )
    discordant = baseline_only + candidate_only
    if discordant == 0:
        p_value = 1.0
    else:
        smaller = min(baseline_only, candidate_only)
        tail = sum(math.comb(discordant, index) for index in range(smaller + 1))
        p_value = min(1.0, 2.0 * tail / (2**discordant))
    return {
        "baseline_only_correct": baseline_only,
        "candidate_only_correct": candidate_only,
        "discordant": discordant,
        "two_sided_exact_p": p_value,
    }


def mcnemar_strata(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    ad = [row for row in records if row["question_type"] == AD_TASK]
    non_ad = [row for row in records if row["question_type"] != AD_TASK]
    by_source: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    by_task: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        by_source[str(row["source"])].append(row)
        by_task[str(row["question_type"])].append(row)
    return {
        "overall": exact_mcnemar(records),
        "ad": exact_mcnemar(ad) if ad else None,
        "non_ad": exact_mcnemar(non_ad) if non_ad else None,
        "by_source": {key: exact_mcnemar(value) for key, value in sorted(by_source.items())},
        "by_task": {key: exact_mcnemar(value) for key, value in sorted(by_task.items())},
    }


def paired_transitions(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    correctness = Counter(
        {key: 0 for key in ("both_correct", "baseline_only_correct", "candidate_only_correct", "both_wrong")}
    )
    validity = Counter(
        {
            key: 0
            for key in (
                "valid_to_valid",
                "valid_to_invalid",
                "invalid_to_valid",
                "invalid_to_invalid",
            )
        }
    )
    ad_decisions = Counter(
        {
            f"{before}_to_{after}": 0
            for before in ("normal", "anomaly", "invalid")
            for after in ("normal", "anomaly", "invalid")
        }
    )
    invalid_ids: dict[str, list[str]] = {"baseline": [], "candidate": []}
    for row in records:
        base_correct, cand_correct = bool(row["baseline"]["correct"]), bool(row["candidate"]["correct"])
        if base_correct and cand_correct:
            correctness["both_correct"] += 1
        elif base_correct:
            correctness["baseline_only_correct"] += 1
        elif cand_correct:
            correctness["candidate_only_correct"] += 1
        else:
            correctness["both_wrong"] += 1
        base_valid, cand_valid = bool(row["baseline"]["valid"]), bool(row["candidate"]["valid"])
        validity[
            ("valid" if base_valid else "invalid")
            + "_to_"
            + ("valid" if cand_valid else "invalid")
        ] += 1
        for arm in ("baseline", "candidate"):
            if not row[arm]["valid"]:
                invalid_ids[arm].append(str(row["sample_id"]))
        if row["question_type"] == AD_TASK:
            before = row["baseline"]["decision"] or "invalid"
            after = row["candidate"]["decision"] or "invalid"
            ad_decisions[f"{before}_to_{after}"] += 1
    return {
        "correctness": dict(sorted(correctness.items())),
        "validity": dict(sorted(validity.items())),
        "ad_decision": dict(sorted(ad_decisions.items())),
        "invalid_sample_ids": invalid_ids,
    }


def metric_delta(
    baseline: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    for metric in HEADLINE_METRICS:
        left, right = baseline.get(metric), candidate.get(metric)
        result[metric] = None if left is None or right is None else float(right) - float(left)
    return result


def cluster_bootstrap(
    records: Sequence[Mapping[str, Any]],
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    if replicates <= 0:
        return {"available": False, "reason": "bootstrap replicates must be positive"}
    if not records or any(row.get("cluster_id") is None for row in records):
        return {"available": False, "reason": "complete query/reference component IDs are required"}
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[str(row["cluster_id"])].append(row)
    names = sorted(grouped)
    if len(names) < 2:
        return {"available": False, "reason": "at least two independent asset clusters are required"}

    points = {
        "baseline": score_run(records, "baseline"),
        "candidate": score_run(records, "candidate"),
    }
    rng = random.Random(seed)
    draws: dict[str, dict[str, list[float]]] = {
        metric: {"baseline": [], "candidate": [], "delta": []}
        for metric in HEADLINE_METRICS
    }
    for _ in range(replicates):
        sampled: list[Mapping[str, Any]] = []
        for name in rng.choices(names, k=len(names)):
            sampled.extend(grouped[name])
        base = score_run(sampled, "baseline")
        cand = score_run(sampled, "candidate")
        for metric in HEADLINE_METRICS:
            left, right = base[metric], cand[metric]
            if left is None or right is None:
                continue
            draws[metric]["baseline"].append(float(left))
            draws[metric]["candidate"].append(float(right))
            draws[metric]["delta"].append(float(right) - float(left))

    metrics: dict[str, Any] = {}
    for metric in HEADLINE_METRICS:
        delta_values = draws[metric]["delta"]
        left_point, right_point = points["baseline"][metric], points["candidate"][metric]
        metrics[metric] = {
            "baseline_point": left_point,
            "candidate_point": right_point,
            "delta_point": None
            if left_point is None or right_point is None
            else float(right_point) - float(left_point),
            "baseline_ci95": [
                percentile(draws[metric]["baseline"], 0.025),
                percentile(draws[metric]["baseline"], 0.975),
            ],
            "candidate_ci95": [
                percentile(draws[metric]["candidate"], 0.025),
                percentile(draws[metric]["candidate"], 0.975),
            ],
            "delta_ci95": [percentile(delta_values, 0.025), percentile(delta_values, 0.975)],
            "probability_delta_gt_zero": safe_rate(
                sum(value > 0 for value in delta_values), len(delta_values)
            ),
            "defined_replicates": len(delta_values),
        }
    return {
        "available": True,
        "resampling_unit": "query-reference connected component",
        "clusters": len(names),
        "replicates_requested": replicates,
        "seed": seed,
        "metrics": metrics,
    }


def analyze(
    manifest_path: Path,
    baseline_path: Path,
    candidate_path: Path,
    *,
    scope: str = "auto",
    bootstrap_replicates: int = 10000,
    seed: int = 20260820,
) -> dict[str, Any]:
    manifest_rows = read_jsonl(manifest_path)
    manifest = index_unique(manifest_rows, "manifest")
    for row in manifest_rows:
        validate_manifest_row(row)
    baseline = index_unique(read_jsonl(baseline_path), "baseline")
    candidate = index_unique(read_jsonl(candidate_path), "candidate")
    if set(baseline) != set(candidate):
        only_base = sorted(set(baseline) - set(candidate))[:10]
        only_candidate = sorted(set(candidate) - set(baseline))[:10]
        raise ValueError(
            "baseline/candidate IDs differ: "
            f"baseline={len(baseline)} candidate={len(candidate)} "
            f"only_baseline={only_base} only_candidate={only_candidate}"
        )
    unknown = set(baseline) - set(manifest)
    if unknown:
        raise ValueError(f"results contain IDs absent from manifest: {sorted(unknown)[:10]}")
    resolved_scope, selected_ids = infer_scope(scope, manifest, set(baseline))

    manifest_digest = sha256_file(manifest_path)
    for sample_id in sorted(baseline):
        require_result_metadata(baseline[sample_id], manifest[sample_id], manifest_digest, "baseline")
        require_result_metadata(candidate[sample_id], manifest[sample_id], manifest_digest, "candidate")
    run_contract = validate_run_contracts(
        manifest_rows, manifest_digest, baseline, candidate
    )

    selected_manifest_rows = [row for row in manifest_rows if row["sample_id"] in selected_ids]
    clusters, cluster_reason = component_clusters(selected_manifest_rows)
    records = build_records(selected_ids, manifest, baseline, candidate, clusters)
    baseline_scores = score_run(records, "baseline")
    candidate_scores = score_run(records, "candidate")
    bootstrap = cluster_bootstrap(records, bootstrap_replicates, seed)
    if clusters is None:
        bootstrap = {"available": False, "reason": cluster_reason}
    near_duplicate_complete = all(
        row.get("component_near_duplicate_audit_complete") is True
        for row in selected_manifest_rows
    )
    if bootstrap.get("available"):
        bootstrap["near_duplicate_audit_complete"] = near_duplicate_complete
        bootstrap["claim_status"] = (
            "component-cluster CI with completed near-duplicate audit"
            if near_duplicate_complete
            else "exploratory component-cluster CI; near-duplicate audit is incomplete"
        )

    return {
        "analysis_schema_version": "strict-paired-adapter-analysis-v1",
        "scope": resolved_scope,
        "rows": len(records),
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": manifest_digest,
        "baseline": str(baseline_path.resolve()),
        "baseline_sha256": sha256_file(baseline_path),
        "candidate": str(candidate_path.resolve()),
        "candidate_sha256": sha256_file(candidate_path),
        "provenance": run_contract,
        "answer_policy": {
            "parser": "exactly one case-sensitive <answer>X</answer> tag; X must be in options",
            "invalid_scoring": "incorrect under ITT; semantic AD decision is null, never normal",
            "prediction_source": "completion only; stored prediction fields ignored",
        },
        "baseline_metrics": baseline_scores,
        "candidate_metrics": candidate_scores,
        "paired_delta": metric_delta(baseline_scores, candidate_scores),
        "source_strata": source_strata(records),
        "paired_transitions": paired_transitions(records),
        "mcnemar_exact": mcnemar_strata(records),
        "cluster_bootstrap": bootstrap,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scope", choices=("auto", "ad-only", "full-7-task"), default="auto")
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260820)
    args = parser.parse_args()
    report = analyze(
        args.manifest,
        args.baseline,
        args.candidate,
        scope=args.scope,
        bootstrap_replicates=args.bootstrap_replicates,
        seed=args.seed,
    )
    write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
