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
  echo "Run dir not found. Pass arg1 or run scripts/15_eval_tx_coa_scorecard.sh first." >&2
  exit 1
fi

TX_JSONL="${RUN_DIR}/transaction_v2/advisory_outputs/transaction_classifications_v2.jsonl"
COA_JSONL="${RUN_DIR}/coa_v1/advisory_outputs/coa_mapping_recommendations.jsonl"
if [[ ! -f "${TX_JSONL}" || ! -f "${COA_JSONL}" ]]; then
  echo "Missing tx/coa outputs under run dir: ${RUN_DIR}" >&2
  exit 1
fi

OUT_DIR="${RUN_DIR}/tax_overlay"
OVERLAY_COA="${OUT_DIR}/coa_mapping_recommendations_tax_overlay.jsonl"
OVERLAY_SUMMARY="${OUT_DIR}/overlay_summary.json"
AUDIT_SUMMARY="${OUT_DIR}/gst_tax_audit_summary_overlay.json"
AUDIT_ISSUES="${OUT_DIR}/gst_tax_audit_issues_overlay.jsonl"
mkdir -p "${OUT_DIR}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/tax_aware_coa_overlay.py" \
  --tx-jsonl "${TX_JSONL}" \
  --coa-jsonl "${COA_JSONL}" \
  --out-jsonl "${OVERLAY_COA}" \
  --summary-json "${OVERLAY_SUMMARY}" \
  --topk-limit "${TAX_OVERLAY_TOPK_LIMIT:-5}" >/tmp/tax_overlay_run_out.txt

"${PYTHON_BIN}" "${SCRIPT_DIR}/gst_tax_audit.py" \
  --tx-jsonl "${TX_JSONL}" \
  --coa-jsonl "${OVERLAY_COA}" \
  --out-summary-json "${AUDIT_SUMMARY}" \
  --out-issues-jsonl "${AUDIT_ISSUES}" \
  --max-major-fail-rate "${GST_AUDIT_MAX_MAJOR_FAIL_RATE:-0.02}" \
  --max-critical-issues "${GST_AUDIT_MAX_CRITICAL_ISSUES:-0}" >/tmp/tax_overlay_audit_out.txt

echo "${AUDIT_SUMMARY}" > "${LAB_OUTPUTS_DIR}/latest_gst_tax_audit_overlay_summary.txt"

echo "Tax overlay + GST audit complete."
echo "Run dir: ${RUN_DIR}"
echo "Overlay summary: ${OVERLAY_SUMMARY}"
echo "Overlay audit summary: ${AUDIT_SUMMARY}"
echo "Overlay audit issues: ${AUDIT_ISSUES}"
