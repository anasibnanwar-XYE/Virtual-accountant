#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_common.sh"
acquire_lab_lock "compute_training_sweep"

ROWS="${1:-20000}"
SEEDS_CSV="${2:-131,231,331,431}"
PROFILE="${3:-staging_m18}"
FLOW_TARGETS_JSON="${4:-${LAB_DIR}/configs/erp_v2_flow_targets_stage_a.json}"

if [[ ! -f "${FLOW_TARGETS_JSON}" ]]; then
  echo "Flow targets profile not found: ${FLOW_TARGETS_JSON}" >&2
  exit 1
fi

RUN_DIR="${LAB_OUTPUTS_DIR}/compute_sweep_$(timestamp)"
mkdir -p "${RUN_DIR}"
MANIFEST_JSONL="${RUN_DIR}/manifest.jsonl"
SETTINGS_JSON="${RUN_DIR}/settings.json"

"${PYTHON_BIN}" - "${SETTINGS_JSON}" "${ROWS}" "${SEEDS_CSV}" "${PROFILE}" "${FLOW_TARGETS_JSON}" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

Path(sys.argv[1]).write_text(
    json.dumps(
        {
            "schema_version": "v0",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "rows": int(sys.argv[2]),
            "seeds_csv": sys.argv[3],
            "workflow_profile": sys.argv[4],
            "flow_targets_json": sys.argv[5],
            "epochs": {
                "v2": 12,
                "coa": 10,
                "product": 10,
                "risk": 10,
                "reconciliation": 10,
            },
        },
        indent=2,
        sort_keys=True,
    ),
    encoding="utf-8",
)
PY

IFS=',' read -r -a SEEDS <<<"${SEEDS_CSV}"

for seed in "${SEEDS[@]}"; do
  seed="$(echo "${seed}" | xargs)"
  if [[ -z "${seed}" ]]; then
    continue
  fi
  if ! [[ "${seed}" =~ ^[0-9]+$ ]]; then
    echo "Invalid seed: ${seed}" >&2
    continue
  fi

  CYCLE_DIR="${RUN_DIR}/seed_${seed}"
  mkdir -p "${CYCLE_DIR}"
  LOG_FILE="${CYCLE_DIR}/train.log"
  START_UTC="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"

  set +e
  (
    cd "${LAB_DIR}"
    ERP_V2_FLOW_REBALANCE=true \
    ERP_V2_FLOW_TARGETS_JSON="${FLOW_TARGETS_JSON}" \
    SYNTHETIC_WORKFLOW_PROFILE="${PROFILE}" \
    TRAINING_SEED="${seed}" \
    bash scripts/02_generate_training_data.sh "${ROWS}"

    TRAINING_CSV="$(cat outputs/latest_training_csv.txt)"

    V2_EPOCHS=12 \
    COA_EPOCHS=10 \
    PRODUCT_EPOCHS=10 \
    RISK_EPOCHS=10 \
    RECON_EPOCHS=10 \
    TX_COA_INCLUDE_PRODUCT_MODEL=false \
    RUN_REVIEW_RISK_LOOP=true \
    RUN_RECONCILIATION_LOOP=true \
    bash scripts/17_run_full_continual_learning_tick.sh -t "${TRAINING_CSV}"
  ) >"${LOG_FILE}" 2>&1
  RC=$?
  set -e

  END_UTC="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  TX_SUMMARY="$(latest_file "${LAB_OUTPUTS_DIR}/latest_tx_coa_continual_cycle_summary.txt" || true)"
  PROD_SUMMARY="$(latest_file "${LAB_OUTPUTS_DIR}/latest_product_account_continual_cycle_summary.txt" || true)"
  RISK_SUMMARY="$(latest_file "${LAB_OUTPUTS_DIR}/latest_review_risk_continual_cycle_summary.txt" || true)"
  RECON_SUMMARY="$(latest_file "${LAB_OUTPUTS_DIR}/latest_reconciliation_continual_cycle_summary.txt" || true)"
  TRAINING_CSV_LATEST="$(latest_file "${LAB_OUTPUTS_DIR}/latest_training_csv.txt" || true)"
  REBALANCE_REPORT="$(latest_file "${LAB_OUTPUTS_DIR}/latest_flow_rebalance_report_json.txt" || true)"

  "${PYTHON_BIN}" - "${MANIFEST_JSONL}" "${seed}" "${START_UTC}" "${END_UTC}" "${RC}" "${LOG_FILE}" "${TRAINING_CSV_LATEST}" "${REBALANCE_REPORT}" "${TX_SUMMARY}" "${PROD_SUMMARY}" "${RISK_SUMMARY}" "${RECON_SUMMARY}" <<'PY'
import json
import sys
from pathlib import Path

row = {
    "schema_version": "v0",
    "seed": int(sys.argv[2]),
    "started_at_utc": sys.argv[3],
    "ended_at_utc": sys.argv[4],
    "exit_code": int(sys.argv[5]),
    "ok": int(sys.argv[5]) == 0,
    "log_file": sys.argv[6],
    "training_csv": sys.argv[7] or None,
    "flow_rebalance_report_json": sys.argv[8] or None,
    "tx_coa_cycle_summary": sys.argv[9] or None,
    "product_cycle_summary": sys.argv[10] or None,
    "review_risk_cycle_summary": sys.argv[11] or None,
    "reconciliation_cycle_summary": sys.argv[12] or None,
}
with Path(sys.argv[1]).open("a", encoding="utf-8") as f:
    f.write(json.dumps(row, sort_keys=True))
    f.write("\n")
PY

  echo "Compute sweep seed ${seed} complete (exit=${RC}). Log: ${LOG_FILE}"
done

echo "${RUN_DIR}" > "${LAB_OUTPUTS_DIR}/latest_compute_sweep_run_dir.txt"
echo "Compute sweep complete."
echo "Run dir: ${RUN_DIR}"
echo "Manifest: ${MANIFEST_JSONL}"
