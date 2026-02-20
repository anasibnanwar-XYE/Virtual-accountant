#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_common.sh"

USER_ID="${1:-}"
RUN_DIR_ARG="${2:-}"
COMPANY_CODE_ARG="${3:-${COMPANY_CODE}}"

if [[ -z "${USER_ID}" ]]; then
  echo "Usage: bash scripts/33_apply_user_personalization.sh <user_id> [tx_coa_eval_run_dir] [company_code]" >&2
  exit 1
fi

RUN_DIR="${RUN_DIR_ARG}"
if [[ -z "${RUN_DIR}" ]]; then
  RUN_DIR="$(latest_file "${LAB_OUTPUTS_DIR}/latest_tx_coa_eval_dir.txt")"
fi
if [[ -z "${RUN_DIR}" || ! -d "${RUN_DIR}" ]]; then
  echo "Run directory not found. Pass tx_coa_eval dir or run tx+coa evaluation first." >&2
  exit 1
fi

TX_JSONL="${RUN_DIR}/transaction_v2/advisory_outputs/transaction_classifications_v2.jsonl"
COA_JSONL="${RUN_DIR}/coa_v1/advisory_outputs/coa_mapping_recommendations.jsonl"
if [[ ! -f "${TX_JSONL}" ]]; then
  echo "Missing transaction classifications: ${TX_JSONL}" >&2
  exit 1
fi
if [[ ! -f "${COA_JSONL}" ]]; then
  echo "Missing CoA recommendations: ${COA_JSONL}" >&2
  exit 1
fi

MEMORY_JSONL="${USER_PERSONALIZATION_MEMORY_JSONL:-${LAB_DATA_DIR}/labels/user_personalization_feedback_memory.jsonl}"
if [[ ! -f "${MEMORY_JSONL}" ]]; then
  echo "User personalization memory file not found: ${MEMORY_JSONL}" >&2
  exit 1
fi

SAFE_USER="$(echo "${USER_ID}" | tr '[:upper:]' '[:lower:]' | tr -c 'a-z0-9._-' '_')"
OUT_DIR="${LAB_OUTPUTS_DIR}/user_personalization/${SAFE_USER}/run_$(timestamp)"
mkdir -p "${OUT_DIR}"
OUT_TX_JSONL="${OUT_DIR}/personalized_transaction_classifications_v2.jsonl"
OUT_COA_JSONL="${OUT_DIR}/personalized_coa_mapping_recommendations.jsonl"
REPORT_JSON="${OUT_DIR}/personalization_report.json"
MIN_MEMORY_ROWS="${USER_PERSONALIZATION_MIN_MEMORY_ROWS:-5}"
MIN_FAMILY_MEMORY_ROWS="${USER_PERSONALIZATION_MIN_FAMILY_MEMORY_ROWS:-3}"
GLOBAL_ONLY_ALPHA_SCALE="${USER_PERSONALIZATION_GLOBAL_ONLY_ALPHA_SCALE:-0.40}"
TX_ALPHA="${USER_PERSONALIZATION_TX_ALPHA:-0.25}"
COA_ALPHA="${USER_PERSONALIZATION_COA_ALPHA:-0.20}"
MAX_TX_TOP1_CHANGE_RATE="${USER_PERSONALIZATION_MAX_TX_TOP1_CHANGE_RATE:-0.30}"
MAX_COA_DEBIT_TOP1_CHANGE_RATE="${USER_PERSONALIZATION_MAX_COA_DEBIT_TOP1_CHANGE_RATE:-0.35}"
MAX_COA_CREDIT_TOP1_CHANGE_RATE="${USER_PERSONALIZATION_MAX_COA_CREDIT_TOP1_CHANGE_RATE:-0.35}"
MIN_FAMILY_EVAL_ROWS="${USER_PERSONALIZATION_MIN_FAMILY_EVAL_ROWS:-25}"
MAX_FAMILY_TOP1_CHANGE_RATE="${USER_PERSONALIZATION_MAX_FAMILY_TOP1_CHANGE_RATE:-0.50}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/user_personalization_rerank.py" \
  --tx-jsonl "${TX_JSONL}" \
  --coa-jsonl "${COA_JSONL}" \
  --memory-jsonl "${MEMORY_JSONL}" \
  --user-id "${USER_ID}" \
  --company-code "${COMPANY_CODE_ARG}" \
  --tx-alpha "${TX_ALPHA}" \
  --coa-alpha "${COA_ALPHA}" \
  --min-memory-rows "${MIN_MEMORY_ROWS}" \
  --min-family-memory-rows "${MIN_FAMILY_MEMORY_ROWS}" \
  --global-only-alpha-scale "${GLOBAL_ONLY_ALPHA_SCALE}" \
  --max-tx-top1-change-rate "${MAX_TX_TOP1_CHANGE_RATE}" \
  --max-coa-debit-top1-change-rate "${MAX_COA_DEBIT_TOP1_CHANGE_RATE}" \
  --max-coa-credit-top1-change-rate "${MAX_COA_CREDIT_TOP1_CHANGE_RATE}" \
  --min-family-eval-rows "${MIN_FAMILY_EVAL_ROWS}" \
  --max-family-top1-change-rate "${MAX_FAMILY_TOP1_CHANGE_RATE}" \
  --out-tx-jsonl "${OUT_TX_JSONL}" \
  --out-coa-jsonl "${OUT_COA_JSONL}" \
  --report-out "${REPORT_JSON}" >/tmp/user_personalization_apply_out.txt

echo "${OUT_DIR}" > "${LAB_OUTPUTS_DIR}/latest_user_personalization_dir.txt"
echo "${REPORT_JSON}" > "${LAB_OUTPUTS_DIR}/latest_user_personalization_report_json.txt"
echo "${OUT_TX_JSONL}" > "${LAB_OUTPUTS_DIR}/latest_user_personalized_tx_jsonl.txt"
echo "${OUT_COA_JSONL}" > "${LAB_OUTPUTS_DIR}/latest_user_personalized_coa_jsonl.txt"
echo "${USER_ID}" > "${LAB_OUTPUTS_DIR}/latest_user_personalization_user_id.txt"

echo "User personalization applied."
echo "User: ${USER_ID}"
echo "Personalization weights: tx_alpha=${TX_ALPHA}, coa_alpha=${COA_ALPHA}"
echo "Guardrails: min_memory_rows=${MIN_MEMORY_ROWS}, min_family_memory_rows=${MIN_FAMILY_MEMORY_ROWS}, global_only_alpha_scale=${GLOBAL_ONLY_ALPHA_SCALE}, max_tx_change_rate=${MAX_TX_TOP1_CHANGE_RATE}, max_coa_debit_change_rate=${MAX_COA_DEBIT_TOP1_CHANGE_RATE}, max_coa_credit_change_rate=${MAX_COA_CREDIT_TOP1_CHANGE_RATE}, min_family_eval_rows=${MIN_FAMILY_EVAL_ROWS}, max_family_change_rate=${MAX_FAMILY_TOP1_CHANGE_RATE}"
echo "Run dir: ${OUT_DIR}"
echo "Report: ${REPORT_JSON}"
