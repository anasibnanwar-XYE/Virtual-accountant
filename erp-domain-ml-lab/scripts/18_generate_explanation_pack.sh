#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_common.sh"

RUN_DIR="${1:-}"
if [[ -z "${RUN_DIR}" ]]; then
  RUN_DIR="$(latest_file "${LAB_OUTPUTS_DIR}/latest_advisory_run_dir.txt")"
fi
if [[ -z "${RUN_DIR}" ]]; then
  RUN_DIR="$(latest_file "${LAB_OUTPUTS_DIR}/latest_tx_coa_eval_dir.txt")"
fi
if [[ -z "${RUN_DIR}" || ! -d "${RUN_DIR}" ]]; then
  echo "Run dir not found. Pass arg1 or run scripts/04_run_advisory.sh / scripts/15_eval_tx_coa_scorecard.sh first." >&2
  exit 1
fi

PRODUCT_SUGGEST_DIR="${2:-}"
if [[ -z "${PRODUCT_SUGGEST_DIR}" ]]; then
  PRODUCT_SUGGEST_DIR="$(latest_file "${LAB_OUTPUTS_DIR}/latest_product_account_suggest_dir.txt")"
fi

TX_JSONL="${RUN_DIR}/transaction_v2/advisory_outputs/transaction_classifications_v2.jsonl"
COA_JSONL="${RUN_DIR}/coa_v1/advisory_outputs/coa_mapping_recommendations.jsonl"
ROUTING_JSONL="${RUN_DIR}/workflow_routing/workflow_routing_decisions.jsonl"
if [[ ! -f "${TX_JSONL}" ]]; then
  echo "Missing transaction output: ${TX_JSONL}" >&2
  exit 1
fi
if [[ ! -f "${COA_JSONL}" ]]; then
  echo "Missing CoA output: ${COA_JSONL}" >&2
  exit 1
fi

PACK_DIR="${RUN_DIR}/explanations"
TX_OUT="${PACK_DIR}/transaction_coa_explanations.jsonl"
PRODUCT_OUT="${PACK_DIR}/product_account_explanations.jsonl"
REVIEW_QUEUE_OUT="${PACK_DIR}/transaction_review_priority_queue.jsonl"
SUMMARY_OUT="${PACK_DIR}/explanation_summary.json"
BRIEF_MD="${PACK_DIR}/review_brief.md"
mkdir -p "${PACK_DIR}"

CMD=(
  "${PYTHON_BIN}" "${SCRIPT_DIR}/accountant_explanation_pack.py"
  --tx-jsonl "${TX_JSONL}"
  --coa-jsonl "${COA_JSONL}"
  --out-tx-jsonl "${TX_OUT}"
  --out-review-queue-jsonl "${REVIEW_QUEUE_OUT}"
  --summary-json "${SUMMARY_OUT}"
  --out-md "${BRIEF_MD}"
  --product-min-ml-score "${PRODUCT_MIN_ML_SCORE:-0.80}"
)
if [[ -f "${ROUTING_JSONL}" ]]; then
  CMD+=(--routing-jsonl "${ROUTING_JSONL}")
fi

PRODUCT_JSONL=""
if [[ -n "${PRODUCT_SUGGEST_DIR}" && -d "${PRODUCT_SUGGEST_DIR}" ]]; then
  PRODUCT_JSONL="${PRODUCT_SUGGEST_DIR}/product_account_suggestions.jsonl"
fi
if [[ -n "${PRODUCT_JSONL}" && -f "${PRODUCT_JSONL}" ]]; then
  CMD+=(--product-jsonl "${PRODUCT_JSONL}" --out-product-jsonl "${PRODUCT_OUT}")
fi

"${CMD[@]}" >/tmp/accountant_explanation_pack_out.txt

if [[ ! -f "${SUMMARY_OUT}" ]]; then
  echo "Explanation summary not created: ${SUMMARY_OUT}" >&2
  exit 1
fi

echo "${PACK_DIR}" > "${LAB_OUTPUTS_DIR}/latest_explanation_pack_dir.txt"
echo "${SUMMARY_OUT}" > "${LAB_OUTPUTS_DIR}/latest_explanation_summary_json.txt"

echo "Explanation pack generated."
echo "Run dir: ${RUN_DIR}"
echo "Pack dir: ${PACK_DIR}"
echo "Summary: ${SUMMARY_OUT}"
if [[ -f "${REVIEW_QUEUE_OUT}" ]]; then
  echo "Review priority queue: ${REVIEW_QUEUE_OUT}"
fi
if [[ -f "${ROUTING_JSONL}" ]]; then
  echo "Routing decisions used: ${ROUTING_JSONL}"
fi
if [[ -f "${PRODUCT_OUT}" ]]; then
  echo "Product explanations: ${PRODUCT_OUT}"
fi
echo "Review brief: ${BRIEF_MD}"
