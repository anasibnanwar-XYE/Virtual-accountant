#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_common.sh"

RUN_DIR="${LAB_OUTPUTS_DIR}/virtual_accountant_monitor_$(timestamp)"
mkdir -p "${RUN_DIR}"
REPORT_JSON="${RUN_DIR}/learning_monitor_report.json"

"${PYTHON_BIN}" "${SCRIPT_DIR}/virtual_accountant_learning_monitor.py" \
  --outputs-dir "${LAB_OUTPUTS_DIR}" \
  --out-json "${REPORT_JSON}" >/tmp/virtual_accountant_monitor_out.txt

echo "${RUN_DIR}" > "${LAB_OUTPUTS_DIR}/latest_virtual_accountant_monitor_dir.txt"
echo "${REPORT_JSON}" > "${LAB_OUTPUTS_DIR}/latest_virtual_accountant_monitor_report_json.txt"

OVERALL_STATUS="$(jq -r '.overall_status // "unknown"' "${REPORT_JSON}")"
TX_STATUS="$(jq -r '.models.tx_coa.cycle_status // "unknown"' "${REPORT_JSON}")"
PRODUCT_STATUS="$(jq -r '.models.product_account.cycle_status // "unknown"' "${REPORT_JSON}")"
RISK_STATUS="$(jq -r '.models.review_risk.cycle_status // "unknown"' "${REPORT_JSON}")"
RECON_STATUS="$(jq -r '.models.reconciliation.cycle_status // "unknown"' "${REPORT_JSON}")"
PERSONALIZATION_STATUS="$(jq -r '.models.personalization.status // "unknown"' "${REPORT_JSON}")"
ALERT_COUNT="$(jq -r '.alerts | length' "${REPORT_JSON}")"

echo "Virtual accountant learning monitor complete."
echo "Overall status: ${OVERALL_STATUS}"
echo "TX+CoA: ${TX_STATUS}"
echo "Product-account: ${PRODUCT_STATUS}"
echo "Review-risk: ${RISK_STATUS}"
echo "Reconciliation: ${RECON_STATUS}"
echo "Personalization: ${PERSONALIZATION_STATUS}"
echo "Alerts: ${ALERT_COUNT}"
echo "Run dir: ${RUN_DIR}"
echo "Report: ${REPORT_JSON}"
