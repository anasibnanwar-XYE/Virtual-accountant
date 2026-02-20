#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_common.sh"

INPUT_FILE="${1:-}"
if [[ -z "${INPUT_FILE}" || ! -f "${INPUT_FILE}" ]]; then
  echo "Usage: bash scripts/11_run_product_account_suggestions.sh /path/to/products.csv" >&2
  echo "Input can be CSV or JSONL with fields: sku,product_name,category,product_kind,uom,gst_rate,base_price,avg_cost" >&2
  exit 1
fi

BUNDLE_DIR="${2:-}"
if [[ -z "${BUNDLE_DIR}" ]]; then
  BUNDLE_DIR="$(latest_file "${LAB_OUTPUTS_DIR}/latest_product_account_bundle_dir.txt")"
fi
if [[ -z "${BUNDLE_DIR}" || ! -d "${BUNDLE_DIR}" ]]; then
  echo "Product-account bundle not found. Run scripts/10_train_product_account_recommender.sh first." >&2
  exit 1
fi

TOPK="${PRODUCT_TOPK:-3}"
NEIGHBORS="${PRODUCT_NEIGHBORS:-5}"
RUN_DIR="${LAB_OUTPUTS_DIR}/product_account_suggest_$(timestamp)"
SUGGESTIONS_OUT="${RUN_DIR}/product_account_suggestions.jsonl"
MANIFEST_OUT="${RUN_DIR}/product_account_suggestions_manifest.json"
mkdir -p "${RUN_DIR}"

{
  export PYTHONPATH="${AUDITOR_ANALYTICS_DIR}:${PYTHONPATH:-}"
  "${PYTHON_BIN}" "${SCRIPT_DIR}/product_account_recommender.py" suggest \
    --bundle-dir "${BUNDLE_DIR}" \
    --input-file "${INPUT_FILE}" \
    --out "${SUGGESTIONS_OUT}" \
    --manifest-out "${MANIFEST_OUT}" \
    --topk "${TOPK}" \
    --neighbors "${NEIGHBORS}" >/tmp/product_account_suggest_out.txt
}

if [[ ! -f "${SUGGESTIONS_OUT}" ]]; then
  echo "Suggestion output not created." >&2
  exit 1
fi

echo "${RUN_DIR}" > "${LAB_OUTPUTS_DIR}/latest_product_account_suggest_dir.txt"

echo "Product-account suggestion run complete."
echo "Output: ${SUGGESTIONS_OUT}"
echo "Manifest: ${MANIFEST_OUT}"
