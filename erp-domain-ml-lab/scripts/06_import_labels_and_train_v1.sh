#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_common.sh"

REVIEW_CSV="${1:-}"
if [[ -z "${REVIEW_CSV}" ]]; then
  REVIEW_CSV="$(latest_file "${LAB_OUTPUTS_DIR}/latest_review_csv.txt")"
fi
if [[ -z "${REVIEW_CSV}" || ! -f "${REVIEW_CSV}" ]]; then
  echo "Review CSV not found. Pass path or run scripts/05_export_review_csv.sh first." >&2
  exit 1
fi

LABELS_JSONL="${LAB_DATA_DIR}/labels/review_labels_$(timestamp).jsonl"
run_analytics_cli import-review-csv \
  --review-csv "${REVIEW_CSV}" \
  --out-labels-jsonl "${LABELS_JSONL}" >/dev/null

V1_MODEL_DIR="$(
  run_analytics_cli train-transaction-classifier-from-labels \
    --labels-jsonl "${LABELS_JSONL}" \
    --out "${LAB_MODELS_DIR}" \
    --epochs 6
)"

echo "${LABELS_JSONL}" > "${LAB_OUTPUTS_DIR}/latest_labels_jsonl.txt"
echo "${V1_MODEL_DIR}" > "${LAB_OUTPUTS_DIR}/latest_transaction_v1_model_dir.txt"

echo "Fine-tune complete (v1 label loop)."
echo "Labels JSONL: ${LABELS_JSONL}"
echo "v1 model dir: ${V1_MODEL_DIR}"

