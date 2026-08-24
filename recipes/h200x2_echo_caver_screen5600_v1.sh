#!/usr/bin/env bash
set -euo pipefail

if [ "${ECHO_5600_TIMEOUT_WRAPPED:-0}" != "1" ]; then
  exec timeout --signal=TERM --kill-after=60s 9000s env \
    ECHO_5600_TIMEOUT_WRAPPED=1 bash "$0" "$@"
fi

MMAD_REPO="${MMAD_REPO:-/workspace/MMAD}"
RUN_DIR="${RUN_DIR:?RUN_DIR is required}"
PYTHON="${MMAD_REPO}/.venv-h200/bin/python3"
MODEL="/workspace/models/JUDO"
DATA="${MMAD_REPO}/data/mmad-full-assets"
CAVER="${MMAD_REPO}/data/external_training/checkpoints/freb_caver_v0.1.0"
SCREEN="${MMAD_REPO}/research/manifests/echo_caver_v1/screen5600.jsonl"
SCREEN_SUMMARY="${MMAD_REPO}/research/manifests/echo_caver_v1/screen5600_summary.json"
RUNNER="${MMAD_REPO}/research/scripts/run_judo_manifest.py"
ANALYZER="${MMAD_REPO}/research/scripts/analyze_two_adapter_eval.py"
ASSET_PREPARER="${MMAD_REPO}/external/FREB-CAVER/freb_caver/prepare_mmad_manifest_subset.py"
V1_RUN="${MMAD_REPO}/research/runpod/remote_runs/h200x2_echo_caver_screen320_v1-20260824T072025Z-17260"
ADAPTER="${V1_RUN}/training/echo/adapter"

die() { printf 'ECHO-CAVER screen5600 ERROR: %s\n' "$*" >&2; exit 1; }
hash_is() { [ "$(sha256sum "$1" | cut -d' ' -f1)" = "$2" ] || die "SHA drift: $1"; }

[ "${RUNPOD_EXPECTED_GPU_COUNT:-}" = "2" ] || die "two-H200 launcher required"
[ "$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)" = "2" ] || die "physical GPU count mismatch"
[ -x "${PYTHON}" ] || die "bootstrap venv absent"

hash_is "${RUNNER}" f160385a1004749485f2dff060e121ac1bf280770d92914171d7613289b8d6a9
hash_is "${ANALYZER}" 674aaf526b88282db9daa0379beff5cd0dd8035994c5ae298db8855dc7c18454
hash_is "${ASSET_PREPARER}" 3b9a7cacb98cebe1c61107a10a3eedf6acd762efedb8b21441c1e993841e5003
hash_is "${SCREEN}" cedf23061dc8d95d255fbadd8975de9a7a45a597946b525d83a423f431ee5b33
hash_is "${SCREEN_SUMMARY}" 5567b7e8b922dde5d91c4cd1ab40185b65ab9244ea02c6c4b3f31e90ded30764
hash_is "${CAVER}/native_deep_residual.safetensors" ac3233dbee2de9dd1aaea4acede275ddc2d7a322427795717d9232291e03b0ed
hash_is "${CAVER}/native_deep_residual_identity.json" f2637f2c9074d178ce8fdb2941c09ed1127e2a5de4139dc5f1105503cce71e78
hash_is "${ADAPTER}/adapter_config.json" 91da8bd671662274fdd980560b28e1c493ade09cc14c53bfd7088b827519d8f7
hash_is "${ADAPTER}/adapter_model.safetensors" 21623bfafc3d329128e2753a5600a9197088ef527deeb44687ea0eacd89f4bb9
hash_is "${V1_RUN}/training/echo/echo_training_summary.json" ce848478ff525c455722eefc056b71dd7fd5c16d455a20c4f824960aaccc4fc8

mkdir -p "${RUN_DIR}/logs" "${RUN_DIR}/results" "${RUN_DIR}/analyses" \
  "${RUN_DIR}/receipts"
export TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=20260824
export PYTHONPATH="${MMAD_REPO}/external/FREB-CAVER/freb_caver:${MMAD_REPO}/research/scripts"
export JUDO_REPO="${MMAD_REPO}/external/JUDO"

"${PYTHON}" - "${SCREEN}" "${RUN_DIR}/receipts/screen_preflight.json" <<'PY'
import json
from collections import Counter
from pathlib import Path
import sys

manifest = Path(sys.argv[1])
output = Path(sys.argv[2])
rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
ids = [str(row["sample_id"]) for row in rows]
cells = Counter((str(row["source"]), str(row["question_type"])) for row in rows)
failures = []
if len(rows) != 5600 or len(set(ids)) != 5600:
    failures.append("row/ID contract")
if len(cells) != 28 or set(cells.values()) != {200}:
    failures.append("28 cells x 200 contract")
if any(row.get("selection_used_model_outputs") is not False for row in rows):
    failures.append("output-blind selection contract")
if any(row.get("training_component_excluded") is not True for row in rows):
    failures.append("training-component exclusion contract")
if any(row.get("component_near_duplicate_audit_complete") is not True for row in rows):
    failures.append("near-duplicate component audit contract")
receipt = {
    "schema_version": "echo-caver-screen5600-preflight-v1",
    "status": "failed" if failures else "passed",
    "rows": len(rows),
    "unique_ids": len(set(ids)),
    "source_task_cells": len(cells),
    "rows_per_cell": sorted(set(cells.values())),
    "selection_used_model_outputs": False,
    "training_components_excluded": True,
    "failures": failures,
}
output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
if failures:
    raise SystemExit("; ".join(failures))
PY

# Re-run the pinned subset preparer as a complete asset/CRC preflight.  The
# earlier output-blind prefetch makes this a reuse-only verification in normal operation.
"${PYTHON}" "${ASSET_PREPARER}" --manifest "${SCREEN}" \
  --mmad-root "${DATA}" --workers 16 --archive-workers 5 \
  --ledger "${RUN_DIR}/receipts/screen5600_asset_ledger.json" \
  > "${RUN_DIR}/logs/verify-screen-assets.log" 2>&1

common_eval=(
  --model "${MODEL}" --manifest "${SCREEN}" --data-root "${DATA}"
  --batch-size 4 --max-new-tokens 1024 --image-size 512 --grid-size 16
  --coverage-threshold 0.02 --attn-implementation sdpa --official-prompt-typo
)

CUDA_VISIBLE_DEVICES=0 "${PYTHON}" "${RUNNER}" "${common_eval[@]}" \
  --native-adapter-kind caver --native-adapter-directory "${CAVER}" \
  --output "${RUN_DIR}/results/freb.jsonl" \
  > "${RUN_DIR}/logs/eval-freb.log" 2>&1 &
freb_pid=$!

CUDA_VISIBLE_DEVICES=1 "${PYTHON}" "${RUNNER}" "${common_eval[@]}" \
  --native-adapter-kind caver --native-adapter-directory "${CAVER}" \
  --hybrid-adapter-contract echo-caver-v1 --adapter "${ADAPTER}" \
  --output "${RUN_DIR}/results/echo.jsonl" \
  > "${RUN_DIR}/logs/eval-echo.log" 2>&1 &
echo_pid=$!

wait "${freb_pid}"
wait "${echo_pid}"

"${PYTHON}" "${ANALYZER}" --manifest "${SCREEN}" \
  --control "${RUN_DIR}/results/freb.jsonl" \
  --treatment "${RUN_DIR}/results/echo.jsonl" --scope full-7-task \
  --bootstrap-replicates 50000 --seed 20260824 \
  --output "${RUN_DIR}/analyses/freb_vs_echo.json" \
  > "${RUN_DIR}/logs/analyze-echo.log" 2>&1

"${PYTHON}" - "${RUN_DIR}" <<'PY'
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
report = json.loads((root / "analyses/freb_vs_echo.json").read_text())
baseline = report["control_metrics"]
candidate = report["treatment_metrics"]
delta = candidate["mmad_overall_macro"] - baseline["mmad_overall_macro"]
gate = {
    "schema_version": "echo-caver-screen5600-gate-v1",
    "rows": 5600,
    "source_task_cells": 28,
    "rows_per_cell": 200,
    "baseline": {
        "overall": baseline["mmad_overall_macro"],
        "non_ad": baseline["non_ad_six_task_macro"],
        "ad_balanced_accuracy": baseline["ad_balanced_accuracy"],
    },
    "echo": {
        "overall": candidate["mmad_overall_macro"],
        "non_ad": candidate["non_ad_six_task_macro"],
        "ad_balanced_accuracy": candidate["ad_balanced_accuracy"],
        "delta_overall": delta,
        "delta_ci95": report["cluster_bootstrap"]["metrics"]["mmad_overall_macro"]["delta_ci95"],
        "probability_delta_gt_zero": report["cluster_bootstrap"]["metrics"]["mmad_overall_macro"]["probability_delta_gt_zero"],
        "mcnemar": report["mcnemar_exact"]["overall"],
    },
    "passes_absolute_82": candidate["mmad_overall_macro"] > 0.82,
    "passes_positive_delta": delta > 0,
    "retained_candidate": (
        "ECHO-CAVER-v1" if candidate["mmad_overall_macro"] > 0.82 else None
    ),
    "promotion_rule": "absolute sampled-MMAD Overall strictly greater than 82%",
}
(root / "screen5600_gate.json").write_text(
    json.dumps(gate, indent=2, sort_keys=True) + "\n"
)
print(json.dumps(gate, sort_keys=True))
PY

sha256sum \
  "${RUN_DIR}"/receipts/*.json \
  "${RUN_DIR}"/results/*.jsonl \
  "${RUN_DIR}"/results/*.native_adapter_runtime.json \
  "${RUN_DIR}"/analyses/*.json \
  "${RUN_DIR}/screen5600_gate.json" \
  > "${RUN_DIR}/ECHO_SCREEN5600_RESULTS_SHA256SUMS"
printf 'ECHO-CAVER screen5600 COMPLETE\n'
