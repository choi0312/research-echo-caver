#!/usr/bin/env bash
set -euo pipefail

if [ "${ECHO_FULL_SECONDARY_WRAPPED:-0}" != "1" ]; then
  exec timeout --signal=TERM --kill-after=120s 14400s env \
    ECHO_FULL_SECONDARY_WRAPPED=1 bash "$0" "$@"
fi

MMAD_REPO="${MMAD_REPO:-/workspace/MMAD}"
RUN_DIR="${RUN_DIR:?RUN_DIR is required}"
PYTHON="${MMAD_REPO}/.venv-h200/bin/python3"
MODEL="/workspace/models/JUDO"
DATA="${MMAD_REPO}/data/mmad-full-assets"
CAVER="${MMAD_REPO}/data/external_training/checkpoints/freb_caver_v0.1.0"
MANIFEST_ROOT="${MMAD_REPO}/research/manifests/m1_full_eval_v1"
FULL="${MANIFEST_ROOT}/full.jsonl"
RECEIPT8="${MANIFEST_ROOT}/receipt8.json"
SHARDS="${MANIFEST_ROOT}/shards8"
RUNNER="${MMAD_REPO}/research/scripts/run_judo_manifest.py"
ANALYZER="${MMAD_REPO}/research/scripts/analyze_sharded_adapter_eval.py"
PAIRED_ANALYZER="${MMAD_REPO}/research/scripts/analyze_paired_adapter_eval.py"
PAIRED_CONTRACT="${MMAD_REPO}/research/scripts/paired_eval_contract.py"
ASSET_PREPARER="${MMAD_REPO}/external/FREB-CAVER/freb_caver/prepare_mmad_manifest_subset.py"
V1_RUN="${MMAD_REPO}/research/runpod/remote_runs/h200x2_echo_caver_screen320_v1-20260824T072025Z-17260"
ADAPTER="${V1_RUN}/training/echo/adapter"

die() { printf 'ECHO-CAVER secondary shards ERROR: %s\n' "$*" >&2; exit 1; }
hash_is() { [ "$(sha256sum "$1" | cut -d' ' -f1)" = "$2" ] || die "SHA drift: $1"; }

[ "${RUNPOD_EXPECTED_GPU_COUNT:-}" = "2" ] || die "two-H200 launcher required"
[ "$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)" = "2" ] || die "physical GPU count mismatch"
[ -x "${PYTHON}" ] || die "bootstrap venv absent"

hash_is "${RUNNER}" f160385a1004749485f2dff060e121ac1bf280770d92914171d7613289b8d6a9
hash_is "${ANALYZER}" 5a9260990743ca695e6913c18560bfdbdf3d14dfa8ff4763858bb07068ac7324
hash_is "${PAIRED_ANALYZER}" 161e5bcc193e55e6b0636e90cb6656a796170023ab3919c73e24f4daacc4056f
hash_is "${PAIRED_CONTRACT}" c18dcc30aedd4deb8d6927bc471c29069def7b21592e5012c0ff246b0e9f0d16
hash_is "${ASSET_PREPARER}" 3b9a7cacb98cebe1c61107a10a3eedf6acd762efedb8b21441c1e993841e5003
hash_is "${FULL}" 61a266302585230610ad58876e76e073ed04f7dccb31c9c61daeaf44a81d4693
hash_is "${RECEIPT8}" 34049b6378763965a61fa964dd41bf48baa86346f38008b9c1c5e7a484954816
hash_is "${CAVER}/native_deep_residual.safetensors" ac3233dbee2de9dd1aaea4acede275ddc2d7a322427795717d9232291e03b0ed
hash_is "${CAVER}/native_deep_residual_identity.json" f2637f2c9074d178ce8fdb2941c09ed1127e2a5de4139dc5f1105503cce71e78
hash_is "${ADAPTER}/adapter_config.json" 91da8bd671662274fdd980560b28e1c493ade09cc14c53bfd7088b827519d8f7
hash_is "${ADAPTER}/adapter_model.safetensors" 21623bfafc3d329128e2753a5600a9197088ef527deeb44687ea0eacd89f4bb9
hash_is "${V1_RUN}/training/echo/echo_training_summary.json" ce848478ff525c455722eefc056b71dd7fd5c16d455a20c4f824960aaccc4fc8

mkdir -p "${RUN_DIR}/logs" "${RUN_DIR}/results" "${RUN_DIR}/analyses" \
  "${RUN_DIR}/receipts"
export TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=20260824
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="${MMAD_REPO}/external/FREB-CAVER/freb_caver:${MMAD_REPO}/research/scripts"
export JUDO_REPO="${MMAD_REPO}/external/JUDO"

"${PYTHON}" - "${RECEIPT8}" "${SHARDS}" \
  "${RUN_DIR}/receipts/secondary_shard_preflight.json" \
  "${RUN_DIR}/receipts/shards4to7_manifest.jsonl" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

receipt_path, shard_root, output, joined_path = map(Path, sys.argv[1:])
receipt = json.loads(receipt_path.read_text())
joined_rows = []
verified = {}
failures = []
for index in range(4, 8):
    name = f"part-{index:02d}-of-08.jsonl"
    path = shard_root / name
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != receipt.get("shard_sha256", {}).get(name):
        failures.append(f"{name} hash mismatch")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    joined_rows.extend(rows)
    verified[name] = {"rows": len(rows), "sha256": digest}
ids = [str(row["sample_id"]) for row in joined_rows]
if len(joined_rows) != 19835 or len(set(ids)) != 19835:
    failures.append("secondary row/disjoint-ID contract")
with joined_path.open("w", encoding="utf-8", newline="\n") as handle:
    for row in joined_rows:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
record = {
    "schema_version": "echo-caver-full-mmad-secondary-preflight-v1",
    "status": "failed" if failures else "passed",
    "rows": len(joined_rows),
    "unique_ids": len(set(ids)),
    "shards": verified,
    "failures": failures,
}
output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
if failures:
    raise SystemExit("; ".join(failures))
PY

"${PYTHON}" "${ASSET_PREPARER}" --manifest "${FULL}" \
  --mmad-root "${DATA}" --workers 24 --archive-workers 5 \
  --ledger "${RUN_DIR}/receipts/full_asset_ledger.json" \
  > "${RUN_DIR}/logs/verify-full-assets.log" 2>&1

common_eval=(
  --model "${MODEL}" --data-root "${DATA}"
  --batch-size 4 --max-new-tokens 1024 --image-size 512 --grid-size 16
  --coverage-threshold 0.02 --attn-implementation sdpa --official-prompt-typo
  --native-adapter-kind caver --native-adapter-directory "${CAVER}"
  --hybrid-adapter-contract echo-caver-v1 --adapter "${ADAPTER}"
)

date -u +%Y-%m-%dT%H:%M:%SZ > "${RUN_DIR}/inference-started-at"
date +%s > "${RUN_DIR}/receipts/inference-started-epoch"
pids=()
names=()
for index in 4 5 6 7; do
  shard="$(printf 'part-%02d-of-08' "${index}")"
  gpu="$((index % 2))"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" "${RUNNER}" \
    --manifest "${SHARDS}/${shard}.jsonl" "${common_eval[@]}" \
    --output "${RUN_DIR}/results/${shard}.jsonl" \
    > "${RUN_DIR}/logs/eval-${shard}.log" 2>&1 &
  pids+=("$!")
  names+=("${shard}")
done
failed=0
for offset in "${!pids[@]}"; do
  if ! wait "${pids[$offset]}"; then
    printf '%s\n' "${names[$offset]} failed" >&2
    failed=1
  fi
done
[ "${failed}" = "0" ] || die "one or more secondary inference shards failed"
date -u +%Y-%m-%dT%H:%M:%SZ > "${RUN_DIR}/inference-finished-at"
date +%s > "${RUN_DIR}/receipts/inference-finished-epoch"

analysis_args=(
  --manifest "${RUN_DIR}/receipts/shards4to7_manifest.jsonl"
  --output "${RUN_DIR}/analyses/echo_shards4to7.json"
)
for index in 4 5 6 7; do
  shard="$(printf 'part-%02d-of-08' "${index}")"
  analysis_args+=(--shard-manifest "${SHARDS}/${shard}.jsonl")
  analysis_args+=(--result "${RUN_DIR}/results/${shard}.jsonl")
done
"${PYTHON}" "${ANALYZER}" "${analysis_args[@]}" \
  > "${RUN_DIR}/logs/analyze-secondary.log" 2>&1

"${PYTHON}" - "${RUN_DIR}" <<'PY'
import json
from pathlib import Path
import sys
import time

root = Path(sys.argv[1])
analysis = json.loads((root / "analyses/echo_shards4to7.json").read_text())
start = int((root / "receipts/inference-started-epoch").read_text())
finish = int((root / "receipts/inference-finished-epoch").read_text())
elapsed = finish - start
ledger = {
    "schema_version": "echo-caver-full-mmad-secondary-shards-run-ledger-v1",
    "status": "COMPLETE",
    "method": "ECHO-CAVER-v1",
    "assigned_shards": [4, 5, 6, 7],
    "rows": 19835,
    "workers": 4,
    "workers_per_gpu": 2,
    "gpus": 2,
    "batch_size_per_worker": 4,
    "adapter_identity_sha256": analysis["adapter"]["adapter_identity_sha256"],
    "adapter_weights_sha256": "21623bfafc3d329128e2753a5600a9197088ef527deeb44687ea0eacd89f4bb9",
    "freb_weights_sha256": "ac3233dbee2de9dd1aaea4acede275ddc2d7a322427795717d9232291e03b0ed",
    "inference_wall_seconds": elapsed,
    "aggregate_samples_per_second": 19835 / elapsed,
    "partial_results_promotable": False,
    "combination_requirement": "shards 0-3 plus this exact disjoint shard set",
    "completed_epoch": int(time.time()),
}
(root / "RUN_LEDGER.json").write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
print(json.dumps(ledger, sort_keys=True))
PY

printf 'ECHO-CAVER full-MMAD secondary shards COMPLETE\n'
