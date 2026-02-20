#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_common.sh"

ROWS="${1:-${SYNTHETIC_ROWS:-50000}}"
TRAINING_SEED="${TRAINING_SEED:-1}"
SYNTHETIC_WORKFLOW_PROFILE="${SYNTHETIC_WORKFLOW_PROFILE:-balanced}"
ERP_V2_FLOW_REBALANCE="${ERP_V2_FLOW_REBALANCE:-false}"
ERP_V2_FLOW_TARGETS_JSON="${ERP_V2_FLOW_TARGETS_JSON:-${LAB_DIR}/configs/erp_v2_flow_targets.json}"

OUT_CSV="${LAB_DATA_DIR}/training/orchestrator_training_${ROWS}_$(timestamp).csv"

run_analytics_cli generate-synthetic-training-csv \
  --out-csv "${OUT_CSV}" \
  --rows "${ROWS}" \
  --seed "${TRAINING_SEED}" \
  --period-start "${PERIOD_START}" \
  --period-end "${PERIOD_END}" \
  --workflow-profile "${SYNTHETIC_WORKFLOW_PROFILE}"

if [[ "${ERP_V2_FLOW_REBALANCE}" == "true" && -f "${ERP_V2_FLOW_TARGETS_JSON}" ]]; then
  REBALANCED_CSV="${LAB_DATA_DIR}/training/orchestrator_training_rebalanced_${ROWS}_$(timestamp).csv"
  REBALANCE_REPORT_JSON="${LAB_OUTPUTS_DIR}/training_flow_rebalance_report_$(timestamp).json"
  "${PYTHON_BIN}" "${SCRIPT_DIR}/29_rebalance_training_by_erp_flow.py" \
    --input-csv "${OUT_CSV}" \
    --output-csv "${REBALANCED_CSV}" \
    --profile-json "${ERP_V2_FLOW_TARGETS_JSON}" \
    --seed "${TRAINING_SEED}" \
    --report-out "${REBALANCE_REPORT_JSON}" >/tmp/training_rebalance_out.txt
  OUT_CSV="${REBALANCED_CSV}"
  echo "${REBALANCE_REPORT_JSON}" > "${LAB_OUTPUTS_DIR}/latest_flow_rebalance_report_json.txt"
fi

echo "${OUT_CSV}" > "${LAB_OUTPUTS_DIR}/latest_training_csv.txt"
echo "Training CSV ready: ${OUT_CSV}"
