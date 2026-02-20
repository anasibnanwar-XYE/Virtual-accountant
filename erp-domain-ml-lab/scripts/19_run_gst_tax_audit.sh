#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_common.sh"

RUN_DIR="${1:-}"
if [[ -z "${RUN_DIR}" ]]; then
  RUN_DIR="$(latest_file "${LAB_OUTPUTS_DIR}/latest_tx_coa_eval_dir.txt")"
fi
if [[ -z "${RUN_DIR}" ]]; then
  RUN_DIR="$(latest_file "${LAB_OUTPUTS_DIR}/latest_advisory_run_dir.txt")"
fi
if [[ -z "${RUN_DIR}" || ! -d "${RUN_DIR}" ]]; then
  echo "Run dir not found. Pass arg1 or run scripts/15_eval_tx_coa_scorecard.sh / scripts/04_run_advisory.sh first." >&2
  exit 1
fi

TX_JSONL="${RUN_DIR}/transaction_v2/advisory_outputs/transaction_classifications_v2.jsonl"
COA_JSONL="${RUN_DIR}/coa_v1/advisory_outputs/coa_mapping_recommendations.jsonl"
if [[ ! -f "${TX_JSONL}" ]]; then
  echo "Missing transaction output: ${TX_JSONL}" >&2
  exit 1
fi
if [[ ! -f "${COA_JSONL}" ]]; then
  echo "Missing CoA output: ${COA_JSONL}" >&2
  exit 1
fi

AUDIT_DIR="${RUN_DIR}/gst_tax_audit"
SUMMARY_JSON="${AUDIT_DIR}/gst_tax_audit_summary.json"
ISSUES_JSONL="${AUDIT_DIR}/gst_tax_audit_issues.jsonl"
mkdir -p "${AUDIT_DIR}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/gst_tax_audit.py" \
  --tx-jsonl "${TX_JSONL}" \
  --coa-jsonl "${COA_JSONL}" \
  --out-summary-json "${SUMMARY_JSON}" \
  --out-issues-jsonl "${ISSUES_JSONL}" \
  --max-major-fail-rate "${GST_AUDIT_MAX_MAJOR_FAIL_RATE:-0.02}" \
  --max-critical-issues "${GST_AUDIT_MAX_CRITICAL_ISSUES:-0}" >/tmp/gst_tax_audit_out.txt

echo "${SUMMARY_JSON}" > "${LAB_OUTPUTS_DIR}/latest_gst_tax_audit_summary.txt"

echo "GST/Tax audit complete."
echo "Run dir: ${RUN_DIR}"
echo "Summary: ${SUMMARY_JSON}"
echo "Issues: ${ISSUES_JSONL}"
