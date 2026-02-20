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

RISK_SEED="${RISK_SEED:-101}"
RISK_EPOCHS="${RISK_EPOCHS:-8}"
RISK_LR="${RISK_LR:-0.08}"
RISK_L2="${RISK_L2:-0.0001}"
RISK_HIDDEN_SIZE="${RISK_HIDDEN_SIZE:-64}"
RISK_FEATURES="${RISK_FEATURES:-2048}"
RISK_BATCH_SIZE="${RISK_BATCH_SIZE:-128}"
RISK_DEVICE="${RISK_DEVICE:-auto}"
RISK_HOLDOUT_RATIO="${RISK_HOLDOUT_RATIO:-0.2}"
RISK_HASH_ALGO="${RISK_HASH_ALGO:-crc32}"
RISK_MAX_HASH_CACHE="${RISK_MAX_HASH_CACHE:-200000}"
RISK_THRESHOLD="${RISK_THRESHOLD:-0.55}"
RISK_OVERRIDE_MEMORY_JSONL="${RISK_OVERRIDE_MEMORY_JSONL:-${OVERRIDE_REASON_MEMORY_JSONL:-${LAB_DATA_DIR}/labels/override_reason_feedback_memory.jsonl}}"

METRICS_OUT="${LAB_OUTPUTS_DIR}/review_risk_training_metrics_$(timestamp).json"

CMD=(
  "${PYTHON_BIN}" "${SCRIPT_DIR}/review_risk_model.py" train
  --training-csv "${TRAINING_CSV}"
  --out "${LAB_MODELS_DIR}"
  --metrics-out "${METRICS_OUT}"
  --seed "${RISK_SEED}"
  --epochs "${RISK_EPOCHS}"
  --learning-rate "${RISK_LR}"
  --l2 "${RISK_L2}"
  --hidden-size "${RISK_HIDDEN_SIZE}"
  --n-features "${RISK_FEATURES}"
  --batch-size "${RISK_BATCH_SIZE}"
  --device "${RISK_DEVICE}"
  --holdout-ratio "${RISK_HOLDOUT_RATIO}"
  --hash-algo "${RISK_HASH_ALGO}"
  --max-hash-cache "${RISK_MAX_HASH_CACHE}"
  --risk-threshold "${RISK_THRESHOLD}"
)
if [[ -f "${RISK_OVERRIDE_MEMORY_JSONL}" ]]; then
  CMD+=(--override-memory-jsonl "${RISK_OVERRIDE_MEMORY_JSONL}")
fi

BUNDLE_DIR="$({
  export PYTHONPATH="${AUDITOR_ANALYTICS_DIR}:${PYTHONPATH:-}"
  "${CMD[@]}"
} | tail -n 1)"

if [[ -z "${BUNDLE_DIR}" || ! -d "${BUNDLE_DIR}" ]]; then
  echo "Review-risk bundle dir not created." >&2
  exit 1
fi

echo "${BUNDLE_DIR}" > "${LAB_OUTPUTS_DIR}/latest_review_risk_bundle_dir.txt"
echo "${METRICS_OUT}" > "${LAB_OUTPUTS_DIR}/latest_review_risk_metrics.json"

echo "Review-risk training complete."
echo "Bundle: ${BUNDLE_DIR}"
echo "Metrics: ${METRICS_OUT}"
