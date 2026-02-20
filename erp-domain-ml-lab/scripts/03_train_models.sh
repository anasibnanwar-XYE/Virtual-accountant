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

COA_TRAINING_CSV="${LAB_DATA_DIR}/training/coa_ready_$(timestamp).csv"
"${PYTHON_BIN}" - "${TRAINING_CSV}" "${COA_TRAINING_CSV}" <<'PY'
import csv
import sys

src, dst = sys.argv[1], sys.argv[2]
with open(src, newline="", encoding="utf-8") as f_in:
    reader = csv.DictReader(f_in)
    if reader.fieldnames is None:
        raise SystemExit("CSV has no header")
    kept = 0
    with open(dst, "w", newline="", encoding="utf-8") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=reader.fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in reader:
            if (row.get("journal_lines") or "").strip():
                writer.writerow(row)
                kept += 1
if kept == 0:
    raise SystemExit("No rows with journal_lines for CoA training")
print(kept)
PY

V2_EPOCHS="${V2_EPOCHS:-8}"
V2_LEARNING_RATE="${V2_LEARNING_RATE:-0.1}"
V2_HIDDEN_SIZE="${V2_HIDDEN_SIZE:-64}"
V2_FEATURES="${V2_FEATURES:-2048}"
V2_BATCH_SIZE="${V2_BATCH_SIZE:-128}"
V2_DEVICE="${V2_DEVICE:-auto}"
V2_LOSS_WEIGHTING="${V2_LOSS_WEIGHTING:-class_balanced_materiality}"

COA_EPOCHS="${COA_EPOCHS:-6}"
COA_FEATURES="${COA_FEATURES:-2048}"
COA_BATCH_SIZE="${COA_BATCH_SIZE:-128}"
COA_DEVICE="${COA_DEVICE:-auto}"
COA_LOSS_WEIGHTING="${COA_LOSS_WEIGHTING:-class_balanced_materiality}"

V2_MODEL_DIR="$(
  run_analytics_cli train-transaction-classifier-v2 \
    --dataset-csv "${TRAINING_CSV}" \
    --out "${LAB_MODELS_DIR}" \
    --epochs "${V2_EPOCHS}" \
    --learning-rate "${V2_LEARNING_RATE}" \
    --hidden-size "${V2_HIDDEN_SIZE}" \
    --n-features "${V2_FEATURES}" \
    --batch-size "${V2_BATCH_SIZE}" \
    --device "${V2_DEVICE}" \
    --feature-cache-dir "${LAB_CACHE_DIR}" \
    --write-feature-cache \
    --loss-weighting "${V2_LOSS_WEIGHTING}"
)"

COA_BUNDLE_DIR="$(
  run_analytics_cli train-coa-recommender-v1 \
    --dataset-csv "${COA_TRAINING_CSV}" \
    --out "${LAB_MODELS_DIR}" \
    --epochs "${COA_EPOCHS}" \
    --n-features "${COA_FEATURES}" \
    --batch-size "${COA_BATCH_SIZE}" \
    --device "${COA_DEVICE}" \
    --feature-cache-dir "${LAB_CACHE_DIR}" \
    --write-feature-cache \
    --loss-weighting "${COA_LOSS_WEIGHTING}"
)"

PRODUCT_MODEL_ENABLED="${PRODUCT_MODEL_ENABLED:-true}"
PRODUCT_BUNDLE_DIR=""
if [[ "${PRODUCT_MODEL_ENABLED}" == "true" ]]; then
  bash "${SCRIPT_DIR}/10_train_product_account_recommender.sh" "${TRAINING_CSV}"
  PRODUCT_BUNDLE_DIR="$(latest_file "${LAB_OUTPUTS_DIR}/latest_product_account_bundle_dir.txt")"
fi

echo "${V2_MODEL_DIR}" > "${LAB_OUTPUTS_DIR}/latest_transaction_model_dir.txt"
echo "${COA_BUNDLE_DIR}" > "${LAB_OUTPUTS_DIR}/latest_coa_bundle_dir.txt"

echo "Training complete."
echo "Transaction v2 model: ${V2_MODEL_DIR}"
echo "CoA recommender bundle: ${COA_BUNDLE_DIR}"
if [[ -n "${PRODUCT_BUNDLE_DIR}" ]]; then
  echo "Product-account bundle: ${PRODUCT_BUNDLE_DIR}"
fi
