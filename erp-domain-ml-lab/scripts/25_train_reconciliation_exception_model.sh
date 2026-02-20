#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_common.sh"

TRAINING_CSV="${1:-}"
if [[ -z "${TRAINING_CSV}" ]]; then
  TRAINING_CSV="$(latest_file "${LAB_OUTPUTS_DIR}/latest_training_csv.txt")"
fi
if [[ -z "${TRAINING_CSV}" || ! -f "${TRAINING_CSV}" ]]; then
  echo "Training CSV not found. Pass a path or run scripts/02_generate_training_data.sh first." >&2
  exit 1
fi

RECON_SEED="${RECON_SEED:-131}"
RECON_EPOCHS="${RECON_EPOCHS:-8}"
RECON_LR="${RECON_LR:-0.08}"
RECON_L2="${RECON_L2:-0.0001}"
RECON_HIDDEN_SIZE="${RECON_HIDDEN_SIZE:-64}"
RECON_FEATURES="${RECON_FEATURES:-2048}"
RECON_BATCH_SIZE="${RECON_BATCH_SIZE:-128}"
RECON_DEVICE="${RECON_DEVICE:-auto}"
RECON_HOLDOUT_RATIO="${RECON_HOLDOUT_RATIO:-0.2}"
RECON_HASH_ALGO="${RECON_HASH_ALGO:-crc32}"
RECON_MAX_HASH_CACHE="${RECON_MAX_HASH_CACHE:-200000}"
RECON_EXCEPTION_THRESHOLD="${RECON_EXCEPTION_THRESHOLD:-0.58}"
RECON_OVERRIDE_MEMORY_JSONL="${RECON_OVERRIDE_MEMORY_JSONL:-${OVERRIDE_REASON_MEMORY_JSONL:-${LAB_DATA_DIR}/labels/override_reason_feedback_memory.jsonl}}"

METRICS_OUT="${LAB_OUTPUTS_DIR}/reconciliation_exception_training_metrics_$(timestamp).json"

CMD=(
  "${PYTHON_BIN}" "${SCRIPT_DIR}/reconciliation_exception_model.py" train
  --training-csv "${TRAINING_CSV}"
  --out "${LAB_MODELS_DIR}"
  --metrics-out "${METRICS_OUT}"
  --seed "${RECON_SEED}"
  --epochs "${RECON_EPOCHS}"
  --learning-rate "${RECON_LR}"
  --l2 "${RECON_L2}"
  --hidden-size "${RECON_HIDDEN_SIZE}"
  --n-features "${RECON_FEATURES}"
  --batch-size "${RECON_BATCH_SIZE}"
  --device "${RECON_DEVICE}"
  --holdout-ratio "${RECON_HOLDOUT_RATIO}"
  --hash-algo "${RECON_HASH_ALGO}"
  --max-hash-cache "${RECON_MAX_HASH_CACHE}"
  --exception-threshold "${RECON_EXCEPTION_THRESHOLD}"
)
if [[ -f "${RECON_OVERRIDE_MEMORY_JSONL}" ]]; then
  CMD+=(--override-memory-jsonl "${RECON_OVERRIDE_MEMORY_JSONL}")
fi

BUNDLE_DIR="$({
  export PYTHONPATH="${AUDITOR_ANALYTICS_DIR}:${PYTHONPATH:-}"
  "${CMD[@]}"
} | tail -n 1)"

if [[ -z "${BUNDLE_DIR}" || ! -d "${BUNDLE_DIR}" ]]; then
  echo "Reconciliation-exception bundle dir not created." >&2
  exit 1
fi

echo "${BUNDLE_DIR}" > "${LAB_OUTPUTS_DIR}/latest_reconciliation_bundle_dir.txt"
echo "${METRICS_OUT}" > "${LAB_OUTPUTS_DIR}/latest_reconciliation_metrics.json"

echo "Reconciliation-exception training complete."
echo "Bundle: ${BUNDLE_DIR}"
echo "Metrics: ${METRICS_OUT}"
