#!/usr/bin/env python3
"""Assemble and analyze the sealed three-run A40 ECHO ablation.

This is a fail-closed publication artifact builder.  It accepts the factorial,
evidence/loss-control, and continuation A40 run directories, but it will not
stage or analyze results until every source is intentionally finalized.  An
interrupted wave-1 source must carry a valid ``WAVE1_SEAL.json``; a normally
completed source must carry a valid COMPLETE terminal, checksum ledger, and
role-specific COMPLETE run ledger.

The nine exact result JSONLs and the frozen clean5600 manifest are copied into
a new read-only input tree.  Only after row order, IDs, manifest identity, and
one common inference contract have been established does the script invoke
``analyze_echo_publication_ablation_v1.py`` with the preregistered 50,000
query/reference-component bootstrap replicates.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import sys
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


SCHEMA_VERSION = "echo-publication-ablation-a40-assembly-v1"
EXPECTED_MANIFEST_SHA256 = (
    "cedf23061dc8d95d255fbadd8975de9a7a45a597946b525d83a423f431ee5b33"
)
EXPECTED_ROWS = 5_600
BOOTSTRAP_REPLICATES = 50_000
BOOTSTRAP_SEED = 20260829
BOOTSTRAP_CHUNK_SIZE = 128
HEX64 = re.compile(r"[0-9a-f]{64}")

ARMS_BY_ROLE: Mapping[str, Tuple[str, ...]] = {
    "factorial": ("judo", "freb", "echo_full"),
    "controls": ("unconstrained", "random_subspace", "ce_only"),
    "continuation": ("answer_only", "reader_only", "ce_kl"),
}
COMPLETE_ARMS_BY_ROLE: Mapping[str, Tuple[str, ...]] = {
    "factorial": (
        "judo",
        "freb",
        "echo_full",
        "reader_only",
        "answer_only",
    ),
    "controls": ("unconstrained", "random_subspace", "ce_only", "ce_kl"),
    "continuation": ("answer_only", "reader_only", "ce_kl"),
}
LEDGER_SCHEMA_BY_ROLE = {
    "factorial": "echo-factorial-token-a40x3-run-ledger-v1",
    "controls": "echo-evidence-loss-controls-a40x3-run-ledger-v1",
    "continuation": "echo-continuation-a40x3-run-ledger-v1",
}


class AssemblyContractError(ValueError):
    """Raised before analysis whenever a publication contract is incomplete."""


@dataclass(frozen=True)
class ResultInspection:
    path: Path
    rows: int
    sha256: str
    manifest_sha256: str
    inference_contract_sha256: str


@dataclass(frozen=True)
class RunInspection:
    role: str
    root: Path
    finalization_kind: str
    finalization_evidence: Tuple[Path, ...]
    finalization_sha256: Mapping[str, str]
    hardware_contract: Path
    selected_results: Mapping[str, ResultInspection]
    inference_contract_sha256: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise AssemblyContractError(f"required regular file absent: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AssemblyContractError(f"invalid JSON file {path}: {error}") from error
    if not isinstance(value, dict):
        raise AssemblyContractError(f"JSON object required: {path}")
    return value


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file() or path.is_symlink():
        raise AssemblyContractError(f"required regular JSONL absent: {path}")
    rows: List[Dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise AssemblyContractError(
                        f"JSON object required at {path}:{number}"
                    )
                rows.append(value)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AssemblyContractError(f"invalid JSONL file {path}: {error}") from error
    return rows


def _require_hex64(value: Any, label: str) -> str:
    text = str(value)
    if HEX64.fullmatch(text) is None:
        raise AssemblyContractError(f"{label} is not a lowercase SHA-256: {text!r}")
    return text


def _manifest_ids(manifest_path: Path) -> Tuple[List[str], str]:
    manifest_path = manifest_path.resolve(strict=True)
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise AssemblyContractError("manifest must be a non-symlink regular file")
    digest = sha256_file(manifest_path)
    if digest != EXPECTED_MANIFEST_SHA256:
        raise AssemblyContractError(
            "clean5600 manifest identity mismatch: "
            f"expected={EXPECTED_MANIFEST_SHA256} got={digest}"
        )
    rows = _read_jsonl(manifest_path)
    ids = [str(row.get("sample_id")) for row in rows]
    if len(ids) != EXPECTED_ROWS or len(set(ids)) != EXPECTED_ROWS:
        raise AssemblyContractError(
            f"manifest must contain {EXPECTED_ROWS} unique ordered IDs; "
            f"rows={len(ids)} unique={len(set(ids))}"
        )
    return ids, digest


def inspect_result(
    path: Path,
    expected_ids: Sequence[str],
    expected_manifest_sha256: str,
) -> ResultInspection:
    rows = _read_jsonl(path)
    ids = [str(row.get("sample_id")) for row in rows]
    if len(ids) != len(expected_ids):
        raise AssemblyContractError(
            f"result is incomplete: {path} rows={len(ids)} expected={len(expected_ids)}"
        )
    if len(set(ids)) != len(ids):
        raise AssemblyContractError(f"result contains duplicate sample IDs: {path}")
    if ids != list(expected_ids):
        raise AssemblyContractError(
            f"result order/sample IDs differ from the frozen manifest: {path}"
        )
    manifest_hashes = {str(row.get("manifest_sha256")) for row in rows}
    if manifest_hashes != {expected_manifest_sha256}:
        raise AssemblyContractError(
            f"result manifest SHA mismatch: {path} values={sorted(manifest_hashes)}"
        )
    contracts = {str(row.get("inference_contract_sha256")) for row in rows}
    if len(contracts) != 1:
        raise AssemblyContractError(
            f"result has multiple inference contracts: {path} values={sorted(contracts)}"
        )
    contract = _require_hex64(next(iter(contracts)), f"{path} inference contract")
    return ResultInspection(
        path=path.resolve(strict=True),
        rows=len(rows),
        sha256=sha256_file(path),
        manifest_sha256=expected_manifest_sha256,
        inference_contract_sha256=contract,
    )


def _verify_hardware(root: Path) -> Path:
    path = root / "receipts" / "hardware-contract.json"
    payload = _load_json(path)
    gpus = payload.get("gpus")
    if not isinstance(gpus, list) or len(gpus) != 3:
        raise AssemblyContractError(f"exactly three hardware entries required: {path}")
    names = [str(gpu.get("name")) for gpu in gpus if isinstance(gpu, dict)]
    if names != ["NVIDIA A40"] * 3:
        raise AssemblyContractError(f"exact 3x NVIDIA A40 required: {path} got={names}")
    if payload.get("required_gpu_count") not in (None, 3):
        raise AssemblyContractError(f"hardware required_gpu_count mismatch: {path}")
    if payload.get("required_gpu_name_exact") not in (None, "NVIDIA A40"):
        raise AssemblyContractError(f"hardware required GPU name mismatch: {path}")
    csv_digest = payload.get("hardware_csv_sha256")
    if csv_digest is not None:
        csv_path = root / "receipts" / "hardware.csv"
        if sha256_file(csv_path) != _require_hex64(csv_digest, "hardware CSV SHA"):
            raise AssemblyContractError(f"hardware CSV digest mismatch: {csv_path}")
    return path


def _safe_checksum_member(root: Path, encoded: str) -> Path:
    if not encoded.startswith("./"):
        raise AssemblyContractError(f"unsafe SHA256SUMS member: {encoded!r}")
    relative = PurePosixPath(encoded[2:])
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise AssemblyContractError(f"unsafe SHA256SUMS member: {encoded!r}")
    path = root.joinpath(*relative.parts)
    if path.is_symlink() or not path.is_file():
        raise AssemblyContractError(f"checksum member is not a regular file: {path}")
    try:
        path.resolve(strict=True).relative_to(root.resolve(strict=True))
    except ValueError as error:
        raise AssemblyContractError(f"checksum member escapes run root: {path}") from error
    return path


def _verify_terminal(root: Path) -> Tuple[Path, Path]:
    terminal_path = root / "TERMINAL.json"
    sums_path = root / "SHA256SUMS"
    terminal = _load_json(terminal_path)
    if terminal.get("schema_version") != "runpod-training-terminal-v1":
        raise AssemblyContractError(f"unexpected terminal schema: {terminal_path}")
    if not (
        terminal.get("status") == "COMPLETE"
        and terminal.get("exitCode") == 0
        and terminal.get("checksumsVerifiedBeforeTerminal") is True
    ):
        raise AssemblyContractError(f"source terminal is not successful: {terminal_path}")
    expected_sums_sha = _require_hex64(
        terminal.get("sha256sumsSha256"), "terminal SHA256SUMS digest"
    )
    if sha256_file(sums_path) != expected_sums_sha:
        raise AssemblyContractError(f"SHA256SUMS digest mismatch: {sums_path}")
    seen: set[str] = set()
    for number, line in enumerate(sums_path.read_text(encoding="utf-8").splitlines(), 1):
        match = re.fullmatch(r"([0-9a-f]{64})  (\./.+)", line)
        if match is None:
            raise AssemblyContractError(f"malformed SHA256SUMS line {number}: {line!r}")
        digest, encoded = match.groups()
        if encoded in seen:
            raise AssemblyContractError(f"duplicate SHA256SUMS member: {encoded}")
        seen.add(encoded)
        member = _safe_checksum_member(root, encoded)
        if sha256_file(member) != digest:
            raise AssemblyContractError(f"source checksum mismatch: {member}")
    if "./RUN_LEDGER.json" not in seen:
        raise AssemblyContractError("successful terminal does not seal RUN_LEDGER.json")
    if (root / "FAILED").exists():
        raise AssemblyContractError(f"successful run must not contain FAILED: {root}")
    return terminal_path, sums_path


def _verify_result_receipt(
    arm: str,
    inspection: ResultInspection,
    record: Mapping[str, Any],
    *,
    wave1: bool = False,
) -> None:
    if int(record.get("rows", -1)) != inspection.rows:
        raise AssemblyContractError(f"{arm} row count disagrees with finalization receipt")
    if str(record.get("sha256")) != inspection.sha256:
        raise AssemblyContractError(f"{arm} result SHA disagrees with finalization receipt")
    if wave1:
        # The interrupt seal binds the manifest once at its top level and calls
        # the per-result inference digest ``contract``.
        if set(record) != {"rows", "sha256", "contract"}:
            raise AssemblyContractError(f"unexpected wave-1 result receipt fields for {arm}")
        receipt_contract = str(record.get("contract"))
    else:
        if str(record.get("manifest_sha256")) != inspection.manifest_sha256:
            raise AssemblyContractError(
                f"{arm} manifest SHA disagrees with finalization receipt"
            )
        receipt_contract = str(record.get("inference_contract_sha256"))
    if receipt_contract != inspection.inference_contract_sha256:
        raise AssemblyContractError(
            f"{arm} inference contract disagrees with finalization receipt"
        )


def _verify_training_members(root: Path, payload: Mapping[str, Any]) -> None:
    training = payload.get("training", {})
    if not isinstance(training, dict):
        raise AssemblyContractError("training seal is not an object")
    for arm, record in training.items():
        if not isinstance(record, dict):
            raise AssemblyContractError(f"invalid training receipt for {arm}")
        base = root / "training" / str(arm)
        summary = base / "echo_training_summary.json"
        weights = base / "adapter" / "adapter_model.safetensors"
        if str(record.get("summary_sha256")) != sha256_file(summary):
            raise AssemblyContractError(f"training summary SHA mismatch for {arm}")
        if str(record.get("adapter_weights_sha256")) != sha256_file(weights) and str(
            record.get("weights_sha256")
        ) != sha256_file(weights):
            raise AssemblyContractError(f"training adapter SHA mismatch for {arm}")


def _inspect_wave1(
    role: str,
    root: Path,
    expected_ids: Sequence[str],
    manifest_sha256: str,
) -> RunInspection:
    if role == "continuation":
        raise AssemblyContractError("continuation cannot be accepted as a wave-1 source")
    seal_path = root / "WAVE1_SEAL.json"
    seal = _load_json(seal_path)
    if seal.get("schema_version") != "echo-a40-wave1-interrupt-seal-v1":
        raise AssemblyContractError(f"unexpected wave-1 seal schema: {seal_path}")
    if seal.get("status") != "VERIFIED_BEFORE_SOURCE_STOP":
        raise AssemblyContractError(f"wave-1 seal is not verified: {seal_path}")
    manifest = seal.get("manifest")
    if not isinstance(manifest, dict) or manifest != {
        "rows": EXPECTED_ROWS,
        "sha256": manifest_sha256,
    }:
        raise AssemblyContractError(f"wave-1 manifest contract mismatch: {seal_path}")
    expected_arms = set(ARMS_BY_ROLE[role])
    result_records = seal.get("results")
    if not isinstance(result_records, dict) or set(result_records) != expected_arms:
        raise AssemblyContractError(
            f"wave-1 result set mismatch for {role}: {sorted(result_records or {})}"
        )
    present = {path.stem for path in (root / "results").glob("*.jsonl")}
    if present != expected_arms:
        raise AssemblyContractError(
            f"wave-1 directory has incomplete/unexpected JSONLs: {sorted(present)}"
        )
    selected: Dict[str, ResultInspection] = {}
    for arm in ARMS_BY_ROLE[role]:
        inspection = inspect_result(
            root / "results" / f"{arm}.jsonl", expected_ids, manifest_sha256
        )
        record = result_records[arm]
        if not isinstance(record, dict):
            raise AssemblyContractError(f"invalid wave-1 result receipt for {arm}")
        _verify_result_receipt(arm, inspection, record, wave1=True)
        selected[arm] = inspection
    contracts = {item.inference_contract_sha256 for item in selected.values()}
    sealed_contract = _require_hex64(
        seal.get("paired_inference_contract_sha256"), "wave-1 paired contract"
    )
    if contracts != {sealed_contract}:
        raise AssemblyContractError(f"wave-1 inference contract mismatch: {role}")
    _verify_training_members(root, seal)
    hardware = _verify_hardware(root)
    return RunInspection(
        role=role,
        root=root,
        finalization_kind="WAVE1_SEAL",
        finalization_evidence=(seal_path,),
        finalization_sha256={"WAVE1_SEAL.json": sha256_file(seal_path)},
        hardware_contract=hardware,
        selected_results=selected,
        inference_contract_sha256=sealed_contract,
    )


def _inspect_complete(
    role: str,
    root: Path,
    expected_ids: Sequence[str],
    manifest_sha256: str,
) -> RunInspection:
    terminal_path, sums_path = _verify_terminal(root)
    ledger_path = root / "RUN_LEDGER.json"
    ledger = _load_json(ledger_path)
    if ledger.get("schema_version") != LEDGER_SCHEMA_BY_ROLE[role]:
        raise AssemblyContractError(f"unexpected run-ledger schema: {ledger_path}")
    if ledger.get("status") != "COMPLETE":
        raise AssemblyContractError(f"run ledger is not COMPLETE: {ledger_path}")
    manifest = ledger.get("manifest")
    if not isinstance(manifest, dict) or manifest != {
        "rows": EXPECTED_ROWS,
        "sha256": manifest_sha256,
    }:
        raise AssemblyContractError(f"run-ledger manifest contract mismatch: {ledger_path}")
    records = ledger.get("results")
    complete_arms = set(COMPLETE_ARMS_BY_ROLE[role])
    if not isinstance(records, dict) or set(records) != complete_arms:
        raise AssemblyContractError(
            f"complete result set mismatch for {role}: {sorted(records or {})}"
        )
    present = {path.stem for path in (root / "results").glob("*.jsonl")}
    if present != complete_arms:
        raise AssemblyContractError(
            f"complete directory has incomplete/unexpected JSONLs: {sorted(present)}"
        )
    inspected: Dict[str, ResultInspection] = {}
    for arm in COMPLETE_ARMS_BY_ROLE[role]:
        item = inspect_result(
            root / "results" / f"{arm}.jsonl", expected_ids, manifest_sha256
        )
        record = records[arm]
        if not isinstance(record, dict):
            raise AssemblyContractError(f"invalid run-ledger result receipt for {arm}")
        _verify_result_receipt(arm, item, record)
        inspected[arm] = item
    contracts = {item.inference_contract_sha256 for item in inspected.values()}
    ledger_contract = _require_hex64(
        ledger.get("paired_inference_contract_sha256"), "run-ledger paired contract"
    )
    if contracts != {ledger_contract}:
        raise AssemblyContractError(f"run-ledger inference contract mismatch: {role}")
    _verify_training_members(root, ledger)
    hardware = _verify_hardware(root)
    selected = {arm: inspected[arm] for arm in ARMS_BY_ROLE[role]}
    evidence = (terminal_path, sums_path, ledger_path)
    return RunInspection(
        role=role,
        root=root,
        finalization_kind="COMPLETE_TERMINAL_AND_LEDGER",
        finalization_evidence=evidence,
        finalization_sha256={path.name: sha256_file(path) for path in evidence},
        hardware_contract=hardware,
        selected_results=selected,
        inference_contract_sha256=ledger_contract,
    )


def inspect_run(
    role: str,
    root: Path,
    expected_ids: Sequence[str],
    manifest_sha256: str,
) -> RunInspection:
    root = root.resolve(strict=True)
    if not root.is_dir() or root.is_symlink():
        raise AssemblyContractError(f"run root must be a non-symlink directory: {root}")
    seal_exists = (root / "WAVE1_SEAL.json").is_file()
    terminal_exists = (root / "TERMINAL.json").is_file()
    if seal_exists and terminal_exists:
        raise AssemblyContractError(
            f"ambiguous finalization (both terminal and wave-1 seal): {root}"
        )
    if seal_exists:
        return _inspect_wave1(role, root, expected_ids, manifest_sha256)
    if terminal_exists:
        return _inspect_complete(role, root, expected_ids, manifest_sha256)
    raise AssemblyContractError(
        f"run is partial: neither WAVE1_SEAL.json nor TERMINAL.json exists: {root}"
    )


def _copy_exact(source: Path, destination: Path) -> str:
    if source.is_symlink() or not source.is_file():
        raise AssemblyContractError(f"copy source is not a regular file: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as reader, destination.open("xb") as writer:
        shutil.copyfileobj(reader, writer, length=1024 * 1024)
        writer.flush()
        os.fsync(writer.fileno())
    source_sha = sha256_file(source)
    if sha256_file(destination) != source_sha:
        raise AssemblyContractError(f"post-copy digest mismatch: {destination}")
    destination.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    return source_sha


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def _make_read_only(root: Path) -> None:
    for path in sorted(root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        if path.is_symlink():
            raise AssemblyContractError(f"immutable tree contains symlink: {path}")
        if path.is_file():
            path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        elif path.is_dir():
            path.chmod(
                stat.S_IRUSR
                | stat.S_IXUSR
                | stat.S_IRGRP
                | stat.S_IXGRP
                | stat.S_IROTH
                | stat.S_IXOTH
            )
    root.chmod(
        stat.S_IRUSR
        | stat.S_IXUSR
        | stat.S_IRGRP
        | stat.S_IXGRP
        | stat.S_IROTH
        | stat.S_IXOTH
    )


def assemble_and_analyze(args: argparse.Namespace) -> Dict[str, Any]:
    manifest = args.manifest.resolve(strict=True)
    manifest_ids, manifest_sha = _manifest_ids(manifest)
    sources = {
        "factorial": inspect_run(
            "factorial", args.factorial_run, manifest_ids, manifest_sha
        ),
        "controls": inspect_run(
            "controls", args.controls_run, manifest_ids, manifest_sha
        ),
        "continuation": inspect_run(
            "continuation", args.continuation_run, manifest_ids, manifest_sha
        ),
    }
    all_results = {
        arm: result
        for role in ("factorial", "controls", "continuation")
        for arm, result in sources[role].selected_results.items()
    }
    expected_arms = {
        "judo",
        "freb",
        "echo_full",
        "reader_only",
        "unconstrained",
        "random_subspace",
        "answer_only",
        "ce_only",
        "ce_kl",
    }
    if set(all_results) != expected_arms:
        raise AssemblyContractError(
            f"nine-arm assembly mismatch: got={sorted(all_results)}"
        )
    contracts = {item.inference_contract_sha256 for item in all_results.values()}
    if len(contracts) != 1:
        raise AssemblyContractError(
            f"nine arms do not share one inference contract: {sorted(contracts)}"
        )
    common_contract = next(iter(contracts))

    analyzer = args.analyzer.resolve(strict=True)
    if analyzer.is_symlink() or not analyzer.is_file():
        raise AssemblyContractError(f"analyzer must be a regular file: {analyzer}")
    output = args.output_dir.resolve()
    if output.exists() or output.is_symlink():
        raise AssemblyContractError(f"output directory must be fresh/nonexistent: {output}")
    output.mkdir(parents=True, exist_ok=False)
    frozen = output / "frozen_inputs"
    staged_results = frozen / "results"
    provenance_root = frozen / "source_provenance"
    staged_results.mkdir(parents=True)
    provenance_root.mkdir(parents=True)
    staged_manifest = frozen / "screen5600.jsonl"
    _copy_exact(manifest, staged_manifest)
    staged_receipts: Dict[str, Any] = {}
    for arm in sorted(all_results):
        result = all_results[arm]
        destination = staged_results / f"{arm}.jsonl"
        copied_sha = _copy_exact(result.path, destination)
        staged_receipts[arm] = {
            "source_path": str(result.path),
            "staged_path": str(destination),
            "rows": result.rows,
            "sha256": copied_sha,
            "manifest_sha256": result.manifest_sha256,
            "inference_contract_sha256": result.inference_contract_sha256,
        }
    source_receipts: Dict[str, Any] = {}
    for role, source in sources.items():
        role_dir = provenance_root / role
        role_dir.mkdir()
        copied_evidence: Dict[str, str] = {}
        for evidence in source.finalization_evidence:
            copied_evidence[evidence.name] = _copy_exact(
                evidence, role_dir / evidence.name
            )
        hardware_sha = _copy_exact(
            source.hardware_contract, role_dir / "hardware-contract.json"
        )
        source_receipts[role] = {
            "source_run": str(source.root),
            "finalization_kind": source.finalization_kind,
            "finalization_evidence_sha256": copied_evidence,
            "hardware_contract_sha256": hardware_sha,
            "selected_arms": list(ARMS_BY_ROLE[role]),
            "inference_contract_sha256": source.inference_contract_sha256,
        }
    input_receipt = {
        "schema_version": "echo-publication-ablation-frozen-inputs-v1",
        "status": "VERIFIED_AND_FROZEN",
        "manifest": {
            "path": str(staged_manifest),
            "rows": EXPECTED_ROWS,
            "sha256": manifest_sha,
        },
        "ordered_sample_ids_identical_across_all_nine_arms": True,
        "unique_sample_ids_per_arm": EXPECTED_ROWS,
        "inference_contract_sha256": common_contract,
        "sources": source_receipts,
        "results": staged_receipts,
    }
    input_receipt_path = frozen / "FROZEN_INPUT_RECEIPT.json"
    _write_json(input_receipt_path, input_receipt)
    _make_read_only(frozen)

    analysis_dir = output / "analysis"
    stdout_path = output / "analyzer.stdout.json"
    stderr_path = output / "analyzer.stderr.txt"
    command = [
        sys.executable,
        str(analyzer),
        "--manifest",
        str(staged_manifest),
        "--results-dir",
        str(staged_results),
        "--output-dir",
        str(analysis_dir),
        "--bootstrap-replicates",
        str(BOOTSTRAP_REPLICATES),
        "--seed",
        str(BOOTSTRAP_SEED),
        "--bootstrap-chunk-size",
        str(BOOTSTRAP_CHUNK_SIZE),
    ]
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise AssemblyContractError(
            f"publication analyzer failed rc={completed.returncode}; see {stderr_path}"
        )
    analysis_receipt = _load_json(analysis_dir / "OUTPUT_RECEIPT.json")
    if analysis_receipt.get("status") != "COMPLETE":
        raise AssemblyContractError("publication analyzer output receipt is not COMPLETE")
    summary = _load_json(analysis_dir / "publication_summary.json")
    bootstrap = summary.get("bootstrap_contract")
    if not isinstance(bootstrap, dict) or (
        bootstrap.get("replicates") != BOOTSTRAP_REPLICATES
        or bootstrap.get("seed") != BOOTSTRAP_SEED
        or bootstrap.get("resampling_unit")
        != "query-reference connected component"
    ):
        raise AssemblyContractError("publication bootstrap contract mismatch")
    implementation_files = [
        analyzer,
        analyzer.with_name("analyze_paired_adapter_eval.py"),
        analyzer.with_name("paired_eval_contract.py"),
        Path(__file__).resolve(),
    ]
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "status": "COMPLETE",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "clean5600-nine-arm-A40-publication-ablation",
        "rows_per_arm": EXPECTED_ROWS,
        "arms": sorted(expected_arms),
        "manifest_sha256": manifest_sha,
        "inference_contract_sha256": common_contract,
        "bootstrap": {
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
            "chunk_size": BOOTSTRAP_CHUNK_SIZE,
            "resampling_unit": "query-reference connected component",
        },
        "frozen_input_receipt": {
            "path": str(input_receipt_path),
            "sha256": sha256_file(input_receipt_path),
        },
        "analysis_output_receipt": {
            "path": str(analysis_dir / "OUTPUT_RECEIPT.json"),
            "sha256": sha256_file(analysis_dir / "OUTPUT_RECEIPT.json"),
        },
        "publication_summary": {
            "path": str(analysis_dir / "publication_summary.json"),
            "sha256": sha256_file(analysis_dir / "publication_summary.json"),
        },
        "implementation_sha256": {
            str(path): sha256_file(path) for path in implementation_files
        },
        "analyzer_command": command,
    }
    receipt_path = output / "ASSEMBLY_RECEIPT.json"
    _write_json(receipt_path, receipt)
    _make_read_only(output)
    return receipt


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=project_root / "research/manifests/echo_caver_v1/screen5600.jsonl",
    )
    parser.add_argument("--factorial-run", type=Path, required=True)
    parser.add_argument("--controls-run", type=Path, required=True)
    parser.add_argument("--continuation-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--analyzer",
        type=Path,
        default=script_dir / "analyze_echo_publication_ablation_v1.py",
    )
    args = parser.parse_args()
    receipt = assemble_and_analyze(args)
    print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
