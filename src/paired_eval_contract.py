#!/usr/bin/env python3
"""Pure-Python contracts shared by JUDO inference and paired evaluation.

This module intentionally has no torch/transformers dependency.  It defines
the byte-exact answer parser and the provenance hashes that make a frozen-base
versus PEFT-adapter comparison auditable.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Mapping


CONTRACT_VERSION = "judo-paired-eval-contract-v1"
RESULT_SCHEMA_VERSION = "judo-manifest-result-v2"
EXACT_ANSWER_PATTERN = re.compile(r"<answer>([A-D])</answer>")
# Deliberately match even an unterminated marker (``<answer``).  Once one valid
# exact tag has been removed, any remaining answer marker makes the completion
# ambiguous/malformed.
ANY_ANSWER_MARKUP_PATTERN = re.compile(r"(?i)<\s*/?\s*answer\b")

# Fields copied into every inference result and compared against the immutable
# manifest.  The full row is additionally protected by manifest_row_sha256.
RESULT_METADATA_FIELDS = (
    "source",
    "question_type",
    "label",
    "correct_answer",
    "options",
    "options_text",
    "question",
    "image",
    "template_image",
    "query_component_id",
    "reference_component_id",
    "component_near_duplicate_audit_complete",
    "category",
    "split",
)


@dataclass(frozen=True)
class ExactAnswer:
    """Result of parsing one completion under the byte-exact tag policy."""

    label: str | None
    valid: bool
    reason: str
    exact_tag_count: int


def _reject_nonfinite(value: Any) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("non-finite floats are forbidden in canonical JSON")
    if isinstance(value, Mapping):
        for item in value.values():
            _reject_nonfinite(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _reject_nonfinite(item)


def canonical_json_bytes(value: Any) -> bytes:
    """Return deterministic UTF-8 JSON bytes suitable for a scientific hash."""

    _reject_nonfinite(value)
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def manifest_row_sha256(row: Mapping[str, Any]) -> str:
    """Hash the complete source manifest row, not a lossy projection."""

    return canonical_sha256(dict(row))


def manifest_metadata(row: Mapping[str, Any]) -> dict[str, Any]:
    """Materialize the parity fields written by ``run_judo_manifest.py``."""

    metadata = {field: row.get(field) for field in RESULT_METADATA_FIELDS}
    # Diagnostic manifests sometimes express the intervention target under
    # expected_answer; canonical MMAD manifests use correct_answer.
    metadata["correct_answer"] = row.get("expected_answer", row.get("correct_answer"))
    return metadata


def option_keys(options: Any) -> tuple[str, ...]:
    """Validate and return canonical A-D option labels.

    JSON object member order is not semantic, so an otherwise identical
    ``{"B": ..., "A": ...}`` mapping is canonicalized rather than rejected.
    """

    if not isinstance(options, Mapping) or not options:
        raise ValueError("options must be a non-empty object")
    keys = tuple(sorted(str(key) for key in options))
    if any(key not in {"A", "B", "C", "D"} for key in keys):
        raise ValueError(f"option keys must be uppercase A-D; got {keys}")
    expected = tuple("ABCD"[: len(keys)])
    if keys != expected:
        raise ValueError(f"option keys must be contiguous {expected}; got {keys}")
    return keys


def parse_exact_answer(text: Any, allowed_labels: Iterable[str]) -> ExactAnswer:
    """Parse exactly one case/spacing-sensitive ``<answer>X</answer>`` tag.

    Text outside the tag is allowed because JUDO emits ``<seg>`` and
    ``<think>`` first.  Whitespace inside the tag, lowercase markup/labels,
    duplicate tags, malformed answer markup, or a label absent from the
    sample's options all make the sample invalid.  An invalid sample has no
    semantic decision; callers must score it wrong rather than map it to the
    normal class.
    """

    if not isinstance(text, str):
        return ExactAnswer(None, False, "completion_not_string", 0)
    allowed = tuple(str(label) for label in allowed_labels)
    if not allowed or any(label not in {"A", "B", "C", "D"} for label in allowed):
        raise ValueError(f"invalid allowed labels: {allowed}")
    matches = list(EXACT_ANSWER_PATTERN.finditer(text))
    if not matches:
        return ExactAnswer(None, False, "missing_or_malformed_exact_answer_tag", 0)
    if len(matches) != 1:
        return ExactAnswer(None, False, "multiple_exact_answer_tags", len(matches))

    match = matches[0]
    remainder = text[: match.start()] + text[match.end() :]
    if ANY_ANSWER_MARKUP_PATTERN.search(remainder):
        return ExactAnswer(None, False, "extra_or_malformed_answer_markup", 1)
    label = match.group(1)
    if label not in allowed:
        return ExactAnswer(None, False, "answer_label_not_in_options", 1)
    return ExactAnswer(label, True, "valid", 1)


def ad_semantics(
    options: Any,
    *,
    allow_distractors: bool = False,
) -> dict[str, str]:
    """Return the Yes/No label-to-class mapping for an AD option map.

    Canonical MMAD AD rows are binary.  A small number of source rows retain
    contiguous multiple-choice distractors such as ``Maybe`` and ``Unknown``.
    Callers may explicitly allow those distractors; they remain valid answer
    labels but intentionally have no normal/anomaly semantics.  The default is
    strict so existing training and inference contracts stay fail-closed.
    """

    keys = option_keys(options)
    if not allow_distractors and keys != ("A", "B"):
        raise ValueError(f"AD requires exactly A/B options; got {keys}")
    semantics: dict[str, str] = {}
    for key, raw in options.items():
        value = str(raw).strip().casefold().rstrip(".").strip()
        if value == "yes":
            semantics[str(key)] = "anomaly"
        elif value == "no":
            semantics[str(key)] = "normal"
        elif not allow_distractors:
            raise ValueError(f"AD option {key} is neither Yes nor No: {raw!r}")
    if (
        set(semantics.values()) != {"normal", "anomaly"}
        or len(semantics) != 2
    ):
        raise ValueError(f"AD options must contain one Yes and one No: {options!r}")
    return semantics


def adapter_artifact_identity(adapter_dir: Path) -> dict[str, Any]:
    """Validate and hash a PEFT LoRA adapter without importing PEFT."""

    root = adapter_dir.expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"adapter directory does not exist: {root}")
    config_path = root / "adapter_config.json"
    if not config_path.is_file():
        raise ValueError(f"adapter_config.json missing: {root}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("adapter_config.json must contain an object")
    if str(config.get("peft_type", "")).upper() != "LORA":
        raise ValueError(f"adapter peft_type must be LORA; got {config.get('peft_type')!r}")
    if str(config.get("task_type", "")).upper() != "CAUSAL_LM":
        raise ValueError(f"adapter task_type must be CAUSAL_LM; got {config.get('task_type')!r}")
    targets = config.get("target_modules")
    if not isinstance(targets, list) or not targets:
        raise ValueError("LoRA adapter target_modules must be a non-empty list")

    artifact_paths = sorted(
        path
        for path in root.glob("adapter_model*")
        if path.is_file()
        and (
            path.name.endswith(".safetensors")
            or path.name.endswith(".bin")
            or path.name.endswith(".json")
        )
    )
    if not artifact_paths:
        raise ValueError(f"adapter weights or shard index missing: {root}")
    files = [
        {
            "name": path.name,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in [config_path, *artifact_paths]
    ]
    payload = {
        "schema_version": "peft-lora-artifact-identity-v1",
        "config": config,
        "files": files,
    }
    return {
        "adapter_identity_sha256": canonical_sha256(payload),
        "adapter_config_sha256": sha256_file(config_path),
        "adapter_weight_files": [item for item in files if item["name"] != "adapter_config.json"],
        "adapter_peft_type": str(config["peft_type"]).upper(),
        "adapter_task_type": str(config["task_type"]).upper(),
        "adapter_base_model_name_or_path": config.get("base_model_name_or_path"),
        "adapter_target_modules": sorted(str(value) for value in targets),
    }


def small_model_artifact_identity(model: str) -> dict[str, Any]:
    """Hash small model configuration files while avoiding multi-GB weights."""

    root = Path(model).expanduser()
    if not root.is_dir():
        return {"model_reference": model, "model_config_files": []}
    names = (
        "config.json",
        "generation_config.json",
        "preprocessor_config.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "checkpoint_provenance.json",
    )
    files = []
    for name in names:
        path = root / name
        if path.is_file():
            files.append({"name": name, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return {"model_reference": model, "model_config_files": files}
