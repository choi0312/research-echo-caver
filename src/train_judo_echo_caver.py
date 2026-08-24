#!/usr/bin/env python3
"""Train ECHO-CAVER's evidence-tangent residual-reader LoRA.

ECHO leaves the released FREB-CAVER carrier frozen.  CAVER writes visual belief
at decoder layers 18/20/22/24/26; ECHO places small LoRA readers only in the
immediately downstream layers and constrains every LoRA-B update to the column
space of the corresponding frozen CAVER belief writer.  Option-permuted pairs,
frozen-CAVER KL retention, and a wrong-answer repair margin provide the task
objective.  ``--projection-mode none`` is the parameter/data/compute-matched
vanilla selective-LoRA control.

Training is answer-boundary teacher forcing.  Evaluation remains ordinary
end-to-end free generation with one frozen CAVER + PEFT runtime checkpoint.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import nullcontext
import hashlib
import json
import math
from pathlib import Path
import random
import re
import sys
import time
from typing import Any, Iterable, Sequence


ROOT = Path(__file__).resolve().parents[2]
FREB_MODULE_ROOT = ROOT / "external/FREB-CAVER/freb_caver"
for module_root in (FREB_MODULE_ROOT, ROOT / "research/scripts"):
    if str(module_root) not in sys.path:
        sys.path.insert(0, str(module_root))


SCHEMA_VERSION = "echo-caver-training-summary-v1"
DEFAULT_INJECTION_LAYERS = (18, 20, 22, 24, 26)
MODULE_SUFFIXES = ("o_proj", "down_proj")
LENS_DEFAULT_LAYERS = (6, 8, 10, 12, 14)
LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.")
ANSWER_LETTERS = "ABCD"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                raise ValueError(f"blank JSONL line at {path}:{line_number}")
            row = json.loads(raw)
            if not isinstance(row, dict):
                raise ValueError(f"non-object JSONL row at {path}:{line_number}")
            rows.append(row)
    return rows


def pair_index(rows: Sequence[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        pair_id = str(row.get("pair_id", ""))
        view = str(row.get("option_view", ""))
        if not pair_id or view not in {"original", "permuted"}:
            raise ValueError("ECHO rows require pair_id and original/permuted option_view")
        if view in grouped[pair_id]:
            raise ValueError(f"duplicate option view: {pair_id}::{view}")
        grouped[pair_id][view] = row
    pairs = []
    for pair_id, views in sorted(grouped.items()):
        if set(views) != {"original", "permuted"}:
            raise ValueError(f"incomplete option pair: {pair_id}")
        original, permuted = views["original"], views["permuted"]
        for field in ("image", "template_image", "source", "question_type", "question"):
            if original.get(field) != permuted.get(field):
                raise ValueError(f"option pair changes {field}: {pair_id}")
        pairs.append([original, permuted])
    if not pairs:
        raise ValueError("training panel contains no option pairs")
    return pairs


def validate_assets(data_root: Path, rows: Sequence[dict[str, Any]]) -> None:
    missing = []
    for row in rows:
        for field in ("image", "template_image"):
            value = row.get(field)
            if not isinstance(value, str) or not value:
                raise ValueError(f"row {row.get('sample_id')} lacks {field}")
            path = (data_root / value).resolve()
            try:
                path.relative_to(data_root)
            except ValueError as error:
                raise ValueError(f"asset escapes data root: {value}") from error
            if not path.is_file():
                missing.append(path)
    if missing:
        raise FileNotFoundError(f"missing ECHO asset {missing[0]} ({len(missing)} total)")


def select_reader_targets(
    model: Any, layers: Sequence[int], suffixes: Sequence[str]
) -> list[str]:
    layer_set = set(int(value) for value in layers)
    suffix_set = set(suffixes)
    selected = []
    for name, module in model.named_modules():
        if module.__class__.__name__ not in {"Linear", "Conv1D"}:
            continue
        match = LAYER_RE.search(name)
        if match is None or int(match.group(1)) not in layer_set:
            continue
        if name.rsplit(".", 1)[-1] not in suffix_set:
            continue
        if "visual" in name or "native_deep_counterfactual_residual" in name:
            raise ValueError(f"forbidden ECHO target: {name}")
        selected.append(name)
    expected = len(layer_set) * len(suffix_set)
    if len(selected) != expected:
        raise ValueError(
            f"expected {expected} exact reader targets, got {len(selected)}: {selected}"
        )
    return sorted(selected)


def trainable_contract(model: Any, targets: Sequence[str]) -> dict[str, Any]:
    total = trainable = 0
    names = []
    unexpected = []
    for name, parameter in model.named_parameters():
        total += int(parameter.numel())
        if parameter.requires_grad:
            trainable += int(parameter.numel())
            names.append(name)
            if "lora_" not in name:
                unexpected.append(name)
    if trainable <= 0 or unexpected:
        raise ValueError(f"invalid LoRA trainable set: trainable={trainable} unexpected={unexpected[:8]}")
    if any(not any(target in name for name in names) for target in targets):
        raise ValueError("one or more exact reader targets have no trainable LoRA parameter")
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "trainable_fraction": trainable / total,
        "target_module_count": len(targets),
        "trainable_parameter_names": names,
    }


def layer_from_name(name: str) -> int:
    match = LAYER_RE.search(name)
    if match is None:
        raise ValueError(f"cannot derive decoder layer from {name}")
    return int(match.group(1))


def evidence_bases(
    carrier: Any,
    injection_layers: Sequence[int],
    reader_layers: Sequence[int],
    rank: int,
) -> tuple[dict[int, Any], dict[str, Any]]:
    import torch

    if rank <= 0:
        raise ValueError("evidence basis rank must be positive")
    if len(injection_layers) != len(reader_layers):
        raise ValueError("writer/reader layer counts disagree")
    if not hasattr(carrier, "belief_up") or len(carrier.belief_up) != len(injection_layers):
        raise ValueError("released carrier lacks the expected site-specific belief writers")
    bases = {}
    receipts = {}
    for site, (writer_layer, reader_layer) in enumerate(zip(injection_layers, reader_layers)):
        weight = carrier.belief_up[site].weight.detach().float()
        if weight.ndim != 2 or weight.shape[0] <= rank:
            raise ValueError(f"invalid CAVER writer matrix at site {site}: {tuple(weight.shape)}")
        left, singular, _right = torch.linalg.svd(weight, full_matrices=False)
        effective_rank = min(rank, int(left.shape[1]))
        basis = left[:, :effective_rank].contiguous()
        gram_error = float(
            (basis.transpose(0, 1) @ basis - torch.eye(effective_rank, device=basis.device))
            .abs()
            .amax()
            .cpu()
        )
        bases[int(reader_layer)] = basis
        receipts[str(reader_layer)] = {
            "writer_layer": int(writer_layer),
            "reader_layer": int(reader_layer),
            "site": site,
            "writer_shape": list(weight.shape),
            "basis_rank": effective_rank,
            "top_singular_value": float(singular[0].cpu()),
            "retained_singular_energy_fraction": float(
                singular[:effective_rank].square().sum().div(singular.square().sum().clamp_min(1e-20)).cpu()
            ),
            "orthonormality_max_error": gram_error,
            "basis_sha256": hashlib.sha256(
                basis.detach().cpu().numpy().tobytes(order="C")
            ).hexdigest(),
        }
    return bases, receipts


def install_evidence_projection(
    model: Any, bases: dict[int, Any]
) -> tuple[list[Any], dict[str, Any]]:
    import torch

    handles = []
    projected = {}
    for name, parameter in model.named_parameters():
        if "lora_B" not in name or not parameter.requires_grad:
            continue
        layer = layer_from_name(name)
        basis = bases.get(layer)
        if basis is None:
            raise ValueError(f"LoRA-B has no evidence basis: {name}")
        if tuple(parameter.shape)[0] != tuple(basis.shape)[0]:
            raise ValueError(
                f"LoRA-B/evidence output mismatch for {name}: {tuple(parameter.shape)} vs {tuple(basis.shape)}"
            )
        local_basis = basis.to(device=parameter.device, dtype=torch.float32)

        def project_gradient(gradient: Any, *, value: Any = local_basis) -> Any:
            original_dtype = gradient.dtype
            projected_gradient = value @ (value.transpose(0, 1) @ gradient.float())
            return projected_gradient.to(dtype=original_dtype)

        handles.append(parameter.register_hook(project_gradient))
        projected[name] = {
            "layer": layer,
            "shape": list(parameter.shape),
            "basis_rank": int(local_basis.shape[1]),
        }
    expected = sum("lora_B" in name and parameter.requires_grad for name, parameter in model.named_parameters())
    if len(projected) != expected or expected == 0:
        raise ValueError(f"projected LoRA-B count mismatch: {len(projected)} vs {expected}")
    return handles, projected


def project_lora_b_weights(model: Any, bases: dict[int, Any]) -> float:
    import torch

    max_outside = 0.0
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "lora_B" not in name or not parameter.requires_grad:
                continue
            basis = bases[layer_from_name(name)].to(parameter.device, torch.float32)
            value = parameter.data.float()
            projection = basis @ (basis.transpose(0, 1) @ value)
            max_outside = max(max_outside, float((value - projection).abs().amax().cpu()))
            parameter.data.copy_(projection.to(parameter.dtype))
    return max_outside


def option_indices(row: dict[str, Any], device: Any) -> Any:
    import torch

    letters = "ABCD"
    options = row.get("options")
    if not isinstance(options, dict) or list(options) not in [list("AB"), list("ABCD")]:
        raise ValueError("ECHO requires canonical AB or ABCD options")
    return torch.tensor([letters.index(letter) for letter in options], device=device)


def correct_margin(logits: Any, target: int, valid: Any) -> Any:
    import torch

    local = logits.index_select(0, valid)
    target_local = int((valid == int(target)).nonzero(as_tuple=False).item())
    others = torch.cat((local[:target_local], local[target_local + 1 :]))
    return local[target_local] - others.amax()


def option_orbit_losses(
    candidate: Any,
    seed: Any,
    rows: list[dict[str, Any]],
    target: Any,
    *,
    repair_margin: float,
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as functional

    if len(rows) % 2 or candidate.shape != seed.shape or candidate.shape[0] != len(rows):
        raise ValueError("ECHO option-orbit batch is misaligned")
    ce_values = []
    retention_values = []
    repair_values = []
    equivariance_values = []
    seed_correct = seed.argmax(dim=-1).eq(target)
    for index, row in enumerate(rows):
        valid = option_indices(row, candidate.device)
        local_target = int((valid == target[index]).nonzero(as_tuple=False).item())
        candidate_local = candidate[index].index_select(0, valid)
        seed_local = seed[index].index_select(0, valid)
        ce_values.append(
            functional.cross_entropy(
                candidate_local.unsqueeze(0),
                torch.tensor([local_target], device=candidate.device),
            )
        )
        retention_values.append(
            functional.kl_div(
                functional.log_softmax(candidate_local, dim=-1),
                functional.softmax(seed_local.detach(), dim=-1),
                reduction="sum",
            )
        )
        candidate_margin = correct_margin(candidate[index], int(target[index]), valid)
        seed_margin = correct_margin(seed[index], int(target[index]), valid).detach()
        repair_weight = 0.15 if bool(seed_correct[index]) else 2.0
        repair_values.append(
            repair_weight
            * functional.softplus(seed_margin + repair_margin - candidate_margin)
        )
    for start in range(0, len(rows), 2):
        original, permuted = rows[start], rows[start + 1]
        mapping = original.get("original_to_permuted_option")
        if not isinstance(mapping, dict):
            raise ValueError("option pair lacks semantic permutation mapping")
        original_letters = list(original["options"])
        original_indices = torch.tensor(
            ["ABCD".index(letter) for letter in original_letters],
            device=candidate.device,
        )
        permuted_indices = torch.tensor(
            ["ABCD".index(str(mapping[letter])) for letter in original_letters],
            device=candidate.device,
        )
        left = candidate[start].index_select(0, original_indices)
        right = candidate[start + 1].index_select(0, permuted_indices)
        equivariance_values.append(
            0.5
            * (
                functional.kl_div(
                    functional.log_softmax(left, dim=-1),
                    functional.softmax(right.detach(), dim=-1),
                    reduction="sum",
                )
                + functional.kl_div(
                    functional.log_softmax(right, dim=-1),
                    functional.softmax(left.detach(), dim=-1),
                    reduction="sum",
                )
            )
        )
    return {
        "ce": torch.stack(ce_values).mean(),
        "retention_kl": torch.stack(retention_values).mean(),
        "repair_margin": torch.stack(repair_values).mean(),
        "semantic_equivariance": torch.stack(equivariance_values).mean(),
    }


def balanced_pair_schedule(
    pairs: Sequence[list[dict[str, Any]]], *, seed: int
) -> list[int]:
    by_cell: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, pair in enumerate(pairs):
        row = pair[0]
        by_cell[(str(row["source"]), str(row["question_type"]))].append(index)
    rng = random.Random(seed)
    for values in by_cell.values():
        rng.shuffle(values)
    cells = sorted(by_cell)
    rng.shuffle(cells)
    offsets = Counter()
    schedule = []
    rounds = max(len(values) for values in by_cell.values())
    for _ in range(rounds):
        for cell in cells:
            values = by_cell[cell]
            schedule.append(values[offsets[cell] % len(values)])
            offsets[cell] += 1
    return schedule


def semantic_equivariance_error(candidate: Any, pairs: Sequence[list[dict[str, Any]]]) -> float:
    import torch

    errors = []
    for pair_index_value, pair in enumerate(pairs):
        original, _permuted = pair
        mapping = original["original_to_permuted_option"]
        letters = list(original["options"])
        left = candidate[2 * pair_index_value].index_select(
            0, torch.tensor(["ABCD".index(letter) for letter in letters], device=candidate.device)
        )
        right = candidate[2 * pair_index_value + 1].index_select(
            0,
            torch.tensor(
                ["ABCD".index(mapping[letter]) for letter in letters],
                device=candidate.device,
            ),
        )
        errors.append(float((left.log_softmax(-1) - right.log_softmax(-1)).abs().mean().cpu()))
    return sum(errors) / max(1, len(errors))


def teacher_forced_metrics(
    rows: Sequence[dict[str, Any]], logits_by_id: dict[str, list[float]]
) -> dict[str, Any]:
    """Summarize answer-boundary logits, including panels with no AD rows."""

    if not rows:
        raise ValueError("cannot score an empty ECHO validation panel")
    cell_total: Counter[tuple[str, str]] = Counter()
    cell_correct: Counter[tuple[str, str]] = Counter()
    task_total: Counter[str] = Counter()
    task_correct: Counter[str] = Counter()
    ad_total: Counter[str] = Counter()
    ad_correct: Counter[str] = Counter()
    correct = 0
    for row in rows:
        sample_id = str(row["sample_id"])
        values = logits_by_id.get(sample_id)
        if values is None or len(values) != len(ANSWER_LETTERS):
            raise ValueError(f"missing four-way validation logits for {sample_id}")
        target = ANSWER_LETTERS.index(str(row["correct_answer"]))
        predicted = max(range(len(values)), key=lambda index: values[index])
        hit = int(predicted == target)
        source, task = str(row["source"]), str(row["question_type"])
        cell_total[(source, task)] += 1
        cell_correct[(source, task)] += hit
        task_total[task] += 1
        task_correct[task] += hit
        correct += hit
        if task == "Anomaly Detection":
            label = str(row["label"])
            ad_total[label] += 1
            ad_correct[label] += hit
    per_cell = {
        f"{source}|{task}": cell_correct[(source, task)] / cell_total[(source, task)]
        for source, task in sorted(cell_total)
    }
    per_task = {
        task: task_correct[task] / task_total[task] for task in sorted(task_total)
    }
    result: dict[str, Any] = {
        "samples": len(rows),
        "accuracy": correct / len(rows),
        "source_task_macro_accuracy": sum(per_cell.values()) / len(per_cell),
        "per_cell_accuracy": per_cell,
        "per_task_accuracy": per_task,
    }
    if ad_total:
        ad_rates = {
            label: ad_correct[label] / total for label, total in sorted(ad_total.items())
        }
        result["ad_label_accuracy"] = ad_rates
        if {"anomaly", "normal"}.issubset(ad_rates):
            result.update(
                {
                    "ad_anomaly_recall": ad_rates["anomaly"],
                    "ad_normal_specificity": ad_rates["normal"],
                    "ad_balanced_accuracy": (
                        ad_rates["anomaly"] + ad_rates["normal"]
                    )
                    / 2,
                }
            )
    return result


def teacher_trajectory(row: dict[str, Any]) -> str:
    segmentation = str(row.get("teacher_segmentation", "None")).strip() or "None"
    thinking = str(row.get("teacher_thinking", "")).strip()
    answer = str(row["correct_answer"])
    if answer not in ANSWER_LETTERS:
        raise ValueError(f"invalid trajectory answer for {row.get('sample_id')}: {answer}")
    return (
        f"<seg>{segmentation}</seg>\n"
        f"<think>{thinking}</think>\n"
        f"<answer>{answer}</answer>"
    )


def prepare_trajectory_batch(
    processor: Any,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    data_root: Path,
    system_prompt: str,
    prefix_text: str,
    candidates: Any,
    device: Any,
    *,
    direct_text_fn: Any,
    load_image_fn: Any,
    max_tokens: int,
) -> tuple[dict[str, Any], Any, Any]:
    """Build assistant-only trajectory labels and exact answer-branch positions."""

    import torch

    prompts = [direct_text_fn(processor, row, system_prompt, "") for row in rows]
    full_texts = [prompt + teacher_trajectory(row) for prompt, row in zip(prompts, rows)]
    images = []
    for row in rows:
        images.extend(
            (
                load_image_fn(data_root / str(row["image"])),
                load_image_fn(data_root / str(row["template_image"])),
            )
        )
    prompt_batch = processor(
        text=prompts, images=images, padding=True, return_tensors="pt"
    )
    full_batch = processor(
        text=full_texts, images=images, padding=True, return_tensors="pt"
    )
    if int(full_batch["input_ids"].shape[1]) > max_tokens:
        raise ValueError(
            f"ECHO trajectory length {full_batch['input_ids'].shape[1]} exceeds {max_tokens}"
        )
    labels_tensor = full_batch["input_ids"].clone()
    labels_tensor.masked_fill_(full_batch["attention_mask"].eq(0), -100)
    answer_positions = []
    for index, row in enumerate(rows):
        prompt_ids = prompt_batch["input_ids"][index][
            prompt_batch["attention_mask"][index].bool()
        ].tolist()
        full_ids = full_batch["input_ids"][index][
            full_batch["attention_mask"][index].bool()
        ].tolist()
        if full_ids[: len(prompt_ids)] != prompt_ids:
            raise RuntimeError(
                f"trajectory prompt is not a token prefix for {row.get('sample_id')}"
            )
        full_start = int(full_batch["input_ids"].shape[1]) - len(full_ids)
        labels_tensor[index, full_start : full_start + len(prompt_ids)] = -100
        answer = str(row["correct_answer"])
        tail_ids = tokenizer.encode(
            f"<answer>{answer}</answer>", add_special_tokens=False
        )
        boundary_ids = tokenizer.encode(
            prefix_text, add_special_tokens=False
        )
        if (
            not tail_ids
            or len(tail_ids) <= len(boundary_ids)
            or tail_ids[: len(boundary_ids)] != boundary_ids
            or full_ids[-len(tail_ids) :] != tail_ids
        ):
            raise RuntimeError(
                f"trajectory answer boundary does not align for {row.get('sample_id')}"
            )
        target_position = (
            full_start + len(full_ids) - len(tail_ids) + len(boundary_ids)
        )
        expected_branch = int(candidates[ANSWER_LETTERS.index(answer)])
        if int(full_batch["input_ids"][index, target_position]) != expected_branch:
            raise RuntimeError(
                f"trajectory branch token mismatch for {row.get('sample_id')}"
            )
        answer_positions.append(target_position - 1)
    inputs = {
        name: value.to(device) if isinstance(value, torch.Tensor) else value
        for name, value in full_batch.items()
    }
    return (
        inputs,
        labels_tensor.to(device),
        torch.tensor(answer_positions, device=device, dtype=torch.long),
    )


def trajectory_losses(
    candidate_logits: Any,
    seed_logits: Any,
    labels_tensor: Any,
    *,
    topk: int,
) -> dict[str, Any]:
    """Assistant-token CE plus a compact frozen-CAVER trajectory anchor."""

    import torch
    import torch.nn.functional as functional

    shifted_candidate = candidate_logits[:, :-1]
    shifted_seed = seed_logits[:, :-1].detach()
    shifted_labels = labels_tensor[:, 1:]
    valid = shifted_labels.ne(-100)
    if not bool(valid.any()):
        raise RuntimeError("ECHO trajectory batch contains no assistant labels")
    token_ce = functional.cross_entropy(
        shifted_candidate.float().transpose(1, 2),
        shifted_labels,
        reduction="none",
        ignore_index=-100,
    )
    per_sample_ce = (token_ce * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)
    candidate_valid = shifted_candidate[valid].float()
    seed_valid = shifted_seed[valid].float()
    effective_topk = min(int(topk), int(seed_valid.shape[-1]))
    _seed_values, seed_indices = seed_valid.topk(effective_topk, dim=-1)
    seed_support = seed_valid.gather(-1, seed_indices)
    candidate_support = candidate_valid.gather(-1, seed_indices)
    support_kl = functional.kl_div(
        functional.log_softmax(candidate_support, dim=-1),
        functional.softmax(seed_support, dim=-1),
        reduction="none",
    ).sum(dim=-1).mean()
    if not bool(torch.isfinite(support_kl)):
        raise FloatingPointError("non-finite ECHO trajectory retention")
    return {
        "trajectory_ce": per_sample_ce.mean(),
        "trajectory_topk_kl": support_kl,
        "trajectory_supervised_tokens": valid.sum(),
    }


def parse_layers(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("layers must be comma-separated integers") from error
    if not result or len(result) != len(set(result)) or any(layer < 0 for layer in result):
        raise argparse.ArgumentTypeError("layers must be unique non-negative integers")
    return result


def parse_suffixes(value: str) -> tuple[str, ...]:
    result = tuple(part.strip() for part in value.split(",") if part.strip())
    allowed = {"q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "gate_proj", "down_proj"}
    if not result or len(result) != len(set(result)) or not set(result).issubset(allowed):
        raise argparse.ArgumentTypeError(
            "target suffixes must be unique comma-separated decoder projection names"
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--caver-seed", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-pairs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--equivariance-weight", type=float, default=0.25)
    parser.add_argument("--retention-weight", type=float, default=1.0)
    parser.add_argument("--repair-weight", type=float, default=0.75)
    parser.add_argument("--repair-margin", type=float, default=0.25)
    parser.add_argument("--trajectory-weight", type=float, default=0.0)
    parser.add_argument("--trajectory-retention-weight", type=float, default=0.25)
    parser.add_argument("--trajectory-topk", type=int, default=16)
    parser.add_argument("--max-trajectory-tokens", type=int, default=4096)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--basis-rank", type=int, default=32)
    parser.add_argument("--projection-mode", choices=("evidence", "none"), default="evidence")
    parser.add_argument("--reader-layers", type=parse_layers)
    parser.add_argument("--target-suffixes", type=parse_suffixes, default=MODULE_SUFFIXES)
    parser.add_argument(
        "--runtime-gate",
        choices=("none", "image-prefix", "staged-bridge"),
        default="none",
        help="Restrict the additive LoRA branch to a fixed causal token support",
    )
    parser.add_argument(
        "--bridge-image-layers",
        type=parse_layers,
        help="For staged-bridge, reader-layer subset restricted to image tokens",
    )
    parser.add_argument(
        "--external-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Reject every training/validation row not issued by the audited VIADUCT/KSDD2 corpora",
    )
    parser.add_argument("--non-ad-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--save-state-steps", type=int, default=25)
    args = parser.parse_args()
    if min(
        args.steps,
        args.batch_pairs,
        args.lora_rank,
        args.lora_alpha,
        args.save_state_steps,
        args.trajectory_topk,
        args.max_trajectory_tokens,
    ) < 1:
        raise ValueError("invalid positive ECHO training argument")
    if min(
        args.equivariance_weight,
        args.retention_weight,
        args.repair_weight,
        args.trajectory_weight,
        args.trajectory_retention_weight,
    ) < 0:
        raise ValueError("loss weights must be non-negative")

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoProcessor, AutoTokenizer, Qwen2_5_VLForConditionalGeneration
    import transformers
    import peft

    from run_judo_manifest import (
        install_and_load_native_adapter,
        native_adapter_artifact_identity,
    )
    import eval_manifest as eval_entry
    from train_decision_aligned_refdiff import answer_protocol, decision_text, labels
    from train_judo_graft_stage4 import direct_text, load_image, prepare_text_batch
    from train_transport_equivariant_adapter import answer_logits

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data_root = args.data_root.resolve(strict=True)
    train_rows = read_jsonl(args.train_manifest)
    validation_rows = read_jsonl(args.validation_manifest)
    if args.non_ad_only:
        train_rows = [row for row in train_rows if row.get("question_type") != "Anomaly Detection"]
        validation_rows = [row for row in validation_rows if row.get("question_type") != "Anomaly Detection"]
    train_pairs = pair_index(train_rows)
    validation_pairs = pair_index(validation_rows)
    if args.external_only:
        allowed_sources = {"VIADUCT", "KolektorSDD2", "KSDD2"}
        forbidden = []
        for row in train_rows + validation_rows:
            if row.get("source") not in allowed_sources or row.get("mmad_row") is not False:
                forbidden.append(str(row.get("sample_id")))
            if any(
                key in row
                for key in (
                    "mmad_sample_id",
                    "judo_teacher_completion",
                    "teacher_segmentation",
                    "teacher_thinking",
                )
            ):
                forbidden.append(str(row.get("sample_id")))
        if forbidden:
            raise ValueError(
                f"external-only LENS corpus contains forbidden rows: {forbidden[:8]}"
            )
    validate_assets(data_root, train_rows + validation_rows)
    train_assets = {str(row[field]) for row in train_rows for field in ("image", "template_image")}
    validation_assets = {
        str(row[field]) for row in validation_rows for field in ("image", "template_image")
    }
    if train_assets & validation_assets:
        raise ValueError("ECHO train/validation exact-asset leakage")

    device = torch.device("cuda:0")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map={"": "cuda:0"},
        local_files_only=True,
        attn_implementation="sdpa",
    )
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    processor.tokenizer.padding_side = "left"
    prefix_ids, prefix_text, candidates, answer_contract = answer_protocol(tokenizer)

    native_provenance = native_adapter_artifact_identity("caver", args.caver_seed)
    carrier, native_runtime = install_and_load_native_adapter(
        model,
        tokenizer,
        kind="caver",
        directory=args.caver_seed,
        expected_provenance=native_provenance,
        base_model_reference=str(args.model),
    )
    injection_layers = tuple(int(value) for value in native_runtime["injection_layers"])
    if injection_layers != DEFAULT_INJECTION_LAYERS:
        raise ValueError(f"unexpected released CAVER injection layers: {injection_layers}")
    reader_layers = args.reader_layers or (
        LENS_DEFAULT_LAYERS
        if args.runtime_gate in {"image-prefix", "staged-bridge"}
        else tuple(layer + 1 for layer in injection_layers)
    )
    if max(reader_layers) >= 28:
        raise ValueError("reader layers must be valid 28-block Qwen decoder layers")
    if args.projection_mode == "evidence" and len(reader_layers) != len(injection_layers):
        raise ValueError("evidence projection requires one reader per CAVER writer")
    targets = select_reader_targets(model, reader_layers, args.target_suffixes)
    config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=targets,
    )
    model = get_peft_model(model, config)
    contract = trainable_contract(model, targets)
    runtime_gate_receipt: dict[str, Any] = {}
    if args.runtime_gate == "image-prefix":
        from lens_caver_runtime import install_image_prefix_lora

        runtime_gate_receipt = install_image_prefix_lora(model, carrier)
    elif args.runtime_gate == "staged-bridge":
        from lens_bridge_runtime import install_staged_bridge_lora

        if not args.bridge_image_layers:
            raise ValueError("staged-bridge requires --bridge-image-layers")
        image_layers = tuple(args.bridge_image_layers)
        if not set(image_layers).issubset(reader_layers):
            raise ValueError("bridge image layers must be a subset of reader layers")
        bridge_layers = tuple(layer for layer in reader_layers if layer not in image_layers)
        if not bridge_layers:
            raise ValueError("staged-bridge requires at least one prompt-bridge layer")
        runtime_gate_receipt = install_staged_bridge_lora(
            model,
            carrier,
            image_layers=image_layers,
            bridge_layers=bridge_layers,
        )
    basis_map: dict[int, Any] = {}
    basis_receipts: dict[str, Any] = {}
    if args.projection_mode == "evidence":
        basis_map, basis_receipts = evidence_bases(
            carrier,
            injection_layers,
            reader_layers,
            args.basis_rank,
        )
    projection_handles = []
    projected_parameters: dict[str, Any] = {}
    if args.projection_mode == "evidence":
        projection_handles, projected_parameters = install_evidence_projection(model, basis_map)

    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    system_prompt = eval_entry.official_eval.make_system_prompt(16, 16)

    def forward(rows: list[dict[str, Any]], *, peft_enabled: bool) -> Any:
        texts = [decision_text(processor, row, system_prompt, prefix_text) for row in rows]
        inputs = prepare_text_batch(processor, rows, data_root, texts, device)
        context = nullcontext() if peft_enabled else model.disable_adapter()
        with context:
            return answer_logits(model, inputs, candidates)

    def evaluate(pairs: Sequence[list[dict[str, Any]]], *, peft_enabled: bool) -> tuple[dict[str, Any], float, dict[str, list[float]]]:
        flat_rows = [row for pair in pairs for row in pair]
        collected = []
        logits_by_id = {}
        model.eval()
        with torch.no_grad():
            for start in range(0, len(pairs), args.batch_pairs):
                rows = [row for pair in pairs[start : start + args.batch_pairs] for row in pair]
                logits = forward(rows, peft_enabled=peft_enabled)
                collected.append(logits)
                for row, values in zip(rows, logits.detach().cpu().tolist()):
                    logits_by_id[str(row["sample_id"])] = values
        joined = torch.cat(collected, dim=0)
        raw = {row_id: values for row_id, values in logits_by_id.items()}
        return (
            teacher_forced_metrics(flat_rows, raw),
            semantic_equivariance_error(joined, pairs),
            raw,
        )

    parity_pairs = validation_pairs[: min(4, len(validation_pairs))]
    with torch.no_grad():
        parity_rows = [row for pair in parity_pairs for row in pair]
        active_zero = forward(parity_rows, peft_enabled=True)
        disabled = forward(parity_rows, peft_enabled=False)
    zero_parity_max_abs = float((active_zero - disabled).abs().amax().cpu())
    if zero_parity_max_abs != 0.0:
        raise RuntimeError(f"zero-init ECHO is not an exact CAVER identity: {zero_parity_max_abs}")

    seed_metrics, seed_equivariance, seed_logits = evaluate(
        validation_pairs, peft_enabled=False
    )
    atomic_json(
        args.output_dir / "seed_validation.json",
        {
            "metrics": seed_metrics,
            "semantic_equivariance_error": seed_equivariance,
            "logits": seed_logits,
        },
    )

    schedule = balanced_pair_schedule(train_pairs, seed=args.seed)
    history = []
    state_path = args.output_dir / "training_state.pt"
    start_step = 0
    if state_path.is_file():
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        if state.get("schema_version") != "echo-caver-training-state-v1":
            raise ValueError("ECHO resume state schema mismatch")
        model.load_state_dict(state["model"], strict=False)
        optimizer.load_state_dict(state["optimizer"])
        start_step = int(state["next_step"])
        history = list(state["history"])

    tick = time.time()
    torch.cuda.reset_peak_memory_stats(device)
    model.eval()
    first_gradient_receipt = None
    for step in range(start_step, args.steps):
        indices = [
            schedule[(step * args.batch_pairs + offset) % len(schedule)]
            for offset in range(args.batch_pairs)
        ]
        rows = [row for index in indices for row in train_pairs[index]]
        target = labels(rows, device)
        trajectory_components: dict[str, Any] = {}
        if args.trajectory_weight > 0:
            trajectory_inputs, trajectory_labels, answer_positions = prepare_trajectory_batch(
                processor,
                tokenizer,
                rows,
                data_root,
                system_prompt,
                prefix_text,
                candidates,
                device,
                direct_text_fn=direct_text,
                load_image_fn=load_image,
                max_tokens=args.max_trajectory_tokens,
            )
            with torch.no_grad(), model.disable_adapter():
                seed_output = model(
                    **trajectory_inputs,
                    use_cache=False,
                    return_dict=True,
                )
            candidate_output = model(
                **trajectory_inputs,
                use_cache=False,
                return_dict=True,
            )
            batch_indices = torch.arange(len(rows), device=device)
            seed_batch = seed_output.logits[
                batch_indices, answer_positions
            ].index_select(-1, candidates.to(device)).float()
            candidate = candidate_output.logits[
                batch_indices, answer_positions
            ].index_select(-1, candidates.to(device)).float()
            trajectory_components = trajectory_losses(
                candidate_output.logits,
                seed_output.logits,
                trajectory_labels,
                topk=args.trajectory_topk,
            )
        else:
            with torch.no_grad():
                seed_batch = forward(rows, peft_enabled=False)
            candidate = forward(rows, peft_enabled=True)
        components = option_orbit_losses(
            candidate,
            seed_batch,
            rows,
            target,
            repair_margin=args.repair_margin,
        )
        loss = (
            components["ce"]
            + args.equivariance_weight * components["semantic_equivariance"]
            + args.retention_weight * components["retention_kl"]
            + args.repair_weight * components["repair_margin"]
        )
        if trajectory_components:
            loss = (
                loss
                + args.trajectory_weight * trajectory_components["trajectory_ce"]
                + args.trajectory_retention_weight
                * trajectory_components["trajectory_topk_kl"]
            )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(trainable_parameters, 1.0)
        if not bool(torch.isfinite(gradient_norm)):
            raise FloatingPointError("non-finite ECHO gradient")
        if step == 0:
            first_gradient_receipt = {
                name: float(parameter.grad.detach().float().norm().cpu())
                if parameter.grad is not None
                else 0.0
                for name, parameter in model.named_parameters()
                if parameter.requires_grad
            }
            if not any(value > 0 for name, value in first_gradient_receipt.items() if "lora_B" in name):
                raise RuntimeError("task loss did not reach the zero-init LoRA-B writers")
        optimizer.step()
        outside_before_projection = 0.0
        if args.projection_mode == "evidence":
            outside_before_projection = project_lora_b_weights(model, basis_map)
        values = {
            "step": step + 1,
            "total": float(loss.detach().cpu()),
            "gradient_norm": float(gradient_norm.detach().cpu()),
            "outside_subspace_max_abs_before_weight_projection": outside_before_projection,
            "cells": sorted(
                {f"{row['source']}|{row['question_type']}" for row in rows[::2]}
            ),
            **{name: float(value.detach().cpu()) for name, value in components.items()},
            **{
                name: int(value.detach().cpu())
                if name == "trajectory_supervised_tokens"
                else float(value.detach().cpu())
                for name, value in trajectory_components.items()
            },
        }
        history.append(values)
        if step == 0 or (step + 1) % args.save_state_steps == 0 or step + 1 == args.steps:
            temporary = state_path.with_suffix(".tmp")
            torch.save(
                {
                    "schema_version": "echo-caver-training-state-v1",
                    "model": {
                        name: value.detach().cpu()
                        for name, value in model.state_dict().items()
                        if "lora_" in name
                    },
                    "optimizer": optimizer.state_dict(),
                    "next_step": step + 1,
                    "history": history,
                },
                temporary,
            )
            temporary.replace(state_path)
        if step == 0:
            atomic_json(
                args.output_dir / "mechanics_smoke.json",
                {
                    "schema_version": "echo-caver-mechanics-smoke-v1",
                    "status": "passed",
                    "zero_init_exact_identity_max_abs": zero_parity_max_abs,
                    "first_gradient_norms": first_gradient_receipt,
                    "projection_mode": args.projection_mode,
                    "projected_parameters": projected_parameters,
                    "trainable_contract": contract,
                    "max_cuda_memory_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
                },
            )

    training_seconds = time.time() - tick
    candidate_metrics, candidate_equivariance, candidate_logits = evaluate(
        validation_pairs, peft_enabled=True
    )
    adapter_dir = args.output_dir / "adapter"
    model.save_pretrained(adapter_dir, safe_serialization=True)
    processor.save_pretrained(args.output_dir / "processor")
    atomic_json(args.output_dir / "candidate_validation_logits.json", candidate_logits)

    adapter_weights = adapter_dir / "adapter_model.safetensors"
    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "method": {
            "name": "LENS-BRIDGE-CAVER" if args.runtime_gate == "staged-bridge" else (
                "LENS-CAVER" if args.runtime_gate == "image-prefix" else "ECHO-CAVER"
            ),
            "expansion": (
                "Locked Evidence iNjection Support Visual-to-Question Bridge CAVER"
                if args.runtime_gate == "staged-bridge"
                else (
                    "Locked Evidence iNjection Support CAVER"
                    if args.runtime_gate == "image-prefix"
                    else "Evidence-tangent Causal Hidden-state Orthogonal residual-reader LoRA"
                )
            ),
            "single_runtime_model": True,
            "external_router": False,
            "inference_threshold": False,
            "frozen_caver": True,
            "writer_layers": list(injection_layers),
            "reader_layers": list(reader_layers),
            "target_suffixes": list(args.target_suffixes),
            "runtime_gate": args.runtime_gate,
            "runtime_gate_receipt": runtime_gate_receipt,
            "projection_mode": args.projection_mode,
            "basis_source": "left singular subspace of each frozen CAVER belief_up writer",
            "basis_rank": args.basis_rank,
            "weight_reprojection_after_adam": args.projection_mode == "evidence",
            "full_teacher_trajectory_supervision": args.trajectory_weight > 0,
            "trajectory_retention": (
                "frozen-CAVER top-k assistant-token distribution"
                if args.trajectory_weight > 0
                else "disabled"
            ),
        },
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "runtime_versions": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "peft": peft.__version__,
            "gpu": torch.cuda.get_device_name(0),
        },
        "input_provenance": {
            "model": str(args.model.resolve()),
            "caver": native_provenance,
            "train_manifest": str(args.train_manifest.resolve()),
            "train_manifest_sha256": sha256_file(args.train_manifest),
            "validation_manifest": str(args.validation_manifest.resolve()),
            "validation_manifest_sha256": sha256_file(args.validation_manifest),
        },
        "answer_protocol": answer_contract,
        "train_pairs": len(train_pairs),
        "validation_pairs": len(validation_pairs),
        "non_ad_only": args.non_ad_only,
        "external_only": args.external_only,
        "targets": targets,
        "trainable_contract": contract,
        "evidence_bases": basis_receipts,
        "projected_parameters": projected_parameters,
        "zero_init_exact_identity_max_abs": zero_parity_max_abs,
        "seed_validation_metrics": seed_metrics,
        "candidate_validation_metrics": candidate_metrics,
        "seed_semantic_equivariance_error": seed_equivariance,
        "candidate_semantic_equivariance_error": candidate_equivariance,
        "steps": args.steps,
        "training_seconds": training_seconds,
        "max_cuda_memory_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
        "first_gradient_norms": first_gradient_receipt,
        "last_training_metrics": history[-1],
        "adapter_directory": str(adapter_dir.resolve()),
        "adapter_weights_sha256": sha256_file(adapter_weights),
        "scientific_identity_sha256": canonical_sha256(
            {
                "native_adapter_identity": native_provenance["adapter_identity_sha256"],
                "adapter_weights_sha256": sha256_file(adapter_weights),
                "projection_mode": args.projection_mode,
                "reader_layers": list(reader_layers),
                "basis_receipts": basis_receipts,
            }
        ),
    }
    atomic_json(args.output_dir / "echo_training_summary.json", summary)
    if args.runtime_gate in {"image-prefix", "staged-bridge"}:
        atomic_json(adapter_dir / "lens_runtime.json", runtime_gate_receipt)
    for handle in projection_handles:
        handle.remove()
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
