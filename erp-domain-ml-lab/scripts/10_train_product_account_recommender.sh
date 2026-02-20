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

PRODUCT_SYNTH_ROWS="${PRODUCT_SYNTH_ROWS:-5000}"
PRODUCT_SEED="${PRODUCT_SEED:-41}"
PRODUCT_EPOCHS="${PRODUCT_EPOCHS:-8}"
PRODUCT_LR="${PRODUCT_LR:-0.1}"
PRODUCT_L2="${PRODUCT_L2:-0.0001}"
PRODUCT_HIDDEN_SIZE="${PRODUCT_HIDDEN_SIZE:-64}"
PRODUCT_FEATURES="${PRODUCT_FEATURES:-2048}"
PRODUCT_BATCH_SIZE="${PRODUCT_BATCH_SIZE:-128}"
PRODUCT_DEVICE="${PRODUCT_DEVICE:-auto}"
PRODUCT_HOLDOUT_RATIO="${PRODUCT_HOLDOUT_RATIO:-0.2}"
PRODUCT_TOPK="${PRODUCT_TOPK:-3}"
PRODUCT_HASH_ALGO="${PRODUCT_HASH_ALGO:-crc32}"
PRODUCT_MAX_HASH_CACHE="${PRODUCT_MAX_HASH_CACHE:-200000}"
PRODUCT_FEEDBACK_JSONL="${PRODUCT_FEEDBACK_JSONL:-}"

METRICS_OUT="${LAB_OUTPUTS_DIR}/product_account_training_metrics_$(timestamp).json"

CMD=(
  "${PYTHON_BIN}" "${SCRIPT_DIR}/product_account_recommender.py" train
  --training-csv "${TRAINING_CSV}"
  --out "${LAB_MODELS_DIR}"
  --metrics-out "${METRICS_OUT}"
  --synthetic-products "${PRODUCT_SYNTH_ROWS}"
  --seed "${PRODUCT_SEED}"
  --epochs "${PRODUCT_EPOCHS}"
  --learning-rate "${PRODUCT_LR}"
  --l2 "${PRODUCT_L2}"
  --hidden-size "${PRODUCT_HIDDEN_SIZE}"
  --n-features "${PRODUCT_FEATURES}"
  --batch-size "${PRODUCT_BATCH_SIZE}"
  --device "${PRODUCT_DEVICE}"
  --holdout-ratio "${PRODUCT_HOLDOUT_RATIO}"
  --topk "${PRODUCT_TOPK}"
  --hash-algo "${PRODUCT_HASH_ALGO}"
  --max-hash-cache "${PRODUCT_MAX_HASH_CACHE}"
)
if [[ -n "${PRODUCT_FEEDBACK_JSONL}" && -f "${PRODUCT_FEEDBACK_JSONL}" ]]; then
  CMD+=(--feedback-jsonl "${PRODUCT_FEEDBACK_JSONL}")
fi

BUNDLE_DIR="$({
  export PYTHONPATH="${AUDITOR_ANALYTICS_DIR}:${PYTHONPATH:-}"
  "${CMD[@]}"
} | tail -n 1)"

if [[ -z "${BUNDLE_DIR}" || ! -d "${BUNDLE_DIR}" ]]; then
  echo "Product-account bundle dir not created." >&2
  exit 1
fi

echo "${BUNDLE_DIR}" > "${LAB_OUTPUTS_DIR}/latest_product_account_bundle_dir.txt"
echo "${METRICS_OUT}" > "${LAB_OUTPUTS_DIR}/latest_product_account_metrics.json"

echo "Product-account training complete."
echo "Bundle: ${BUNDLE_DIR}"
echo "Metrics: ${METRICS_OUT}"
