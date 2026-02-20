#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_common.sh"

CYCLES="${1:-12}"
SLEEP_SECONDS="${2:-900}"
TRAINING_CSV="${3:-}"
REVIEW_CSV="${4:-}"
BENCHMARK_SNAPSHOT="${5:-}"
PRODUCT_FEEDBACK="${6:-}"
OVERRIDE_FEEDBACK="${7:-}"

if ! [[ "${CYCLES}" =~ ^[0-9]+$ ]] || [[ "${CYCLES}" -le 0 ]]; then
  echo "CYCLES must be a positive integer (arg1)." >&2
  exit 1
fi
if ! [[ "${SLEEP_SECONDS}" =~ ^[0-9]+$ ]] || [[ "${SLEEP_SECONDS}" -lt 0 ]]; then
  echo "SLEEP_SECONDS must be a non-negative integer (arg2)." >&2
  exit 1
fi

RUN_DIR="${LAB_OUTPUTS_DIR}/autopilot_run_$(timestamp)"
mkdir -p "${RUN_DIR}"
MANIFEST_JSONL="${RUN_DIR}/cycles_manifest.jsonl"
SETTINGS_JSON="${RUN_DIR}/settings.json"

"${PYTHON_BIN}" - "${SETTINGS_JSON}" "${CYCLES}" "${SLEEP_SECONDS}" "${TRAINING_CSV}" "${REVIEW_CSV}" "${BENCHMARK_SNAPSHOT}" "${PRODUCT_FEEDBACK}" "${OVERRIDE_FEEDBACK}" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

out = Path(sys.argv[1])
payload = {
    "schema_version": "v0",
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "cycles": int(sys.argv[2]),
    "sleep_seconds": int(sys.argv[3]),
    "training_csv": sys.argv[4] or None,
    "review_csv": sys.argv[5] or None,
    "benchmark_snapshot": sys.argv[6] or None,
    "product_feedback": sys.argv[7] or None,
    "override_feedback": sys.argv[8] or None,
    "stop_on_error": str.lower(str(__import__("os").environ.get("AUTOPILOT_STOP_ON_ERROR", "false"))) == "true",
}
out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
PY

STOP_ON_ERROR="${AUTOPILOT_STOP_ON_ERROR:-false}"

for ((i = 1; i <= CYCLES; i++)); do
  LOG_FILE="${RUN_DIR}/cycle_${i}.log"
  START_UTC="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"

  CMD=(bash "${SCRIPT_DIR}/17_run_full_continual_learning_tick.sh")
  if [[ -n "${TRAINING_CSV}" ]]; then
    CMD+=(-t "${TRAINING_CSV}")
  fi
  if [[ -n "${REVIEW_CSV}" ]]; then
    CMD+=(-r "${REVIEW_CSV}")
  fi
  if [[ -n "${BENCHMARK_SNAPSHOT}" ]]; then
    CMD+=(-s "${BENCHMARK_SNAPSHOT}")
  fi
  if [[ -n "${PRODUCT_FEEDBACK}" ]]; then
    CMD+=(-f "${PRODUCT_FEEDBACK}")
  fi
  if [[ -n "${OVERRIDE_FEEDBACK}" ]]; then
    CMD+=(-o "${OVERRIDE_FEEDBACK}")
  fi

  set +e
  "${CMD[@]}" >"${LOG_FILE}" 2>&1
  RC=$?
  set -e

  END_UTC="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  NEXT_GST_PROFILE=""
  TX_SUMMARY=""
  TX_ROUTING_SUMMARY=""
  PROD_SUMMARY=""
  RISK_SUMMARY=""
  RECON_SUMMARY=""
  GST_AUDIT_SUMMARY=""
  if [[ "${RC}" -eq 0 ]]; then
    TX_SUMMARY="$(latest_file "${LAB_OUTPUTS_DIR}/latest_tx_coa_continual_cycle_summary.txt" || true)"
    TX_ROUTING_SUMMARY="$(latest_file "${LAB_OUTPUTS_DIR}/latest_workflow_routing_summary_json.txt" || true)"
    PROD_SUMMARY="$(latest_file "${LAB_OUTPUTS_DIR}/latest_product_account_continual_cycle_summary.txt" || true)"
    if [[ "${RUN_REVIEW_RISK_LOOP:-true}" == "true" ]]; then
      RISK_SUMMARY="$(latest_file "${LAB_OUTPUTS_DIR}/latest_review_risk_continual_cycle_summary.txt" || true)"
    fi
    if [[ "${RUN_RECONCILIATION_LOOP:-true}" == "true" ]]; then
      RECON_SUMMARY="$(latest_file "${LAB_OUTPUTS_DIR}/latest_reconciliation_continual_cycle_summary.txt" || true)"
    fi
    GST_AUDIT_SUMMARY="$(latest_file "${LAB_OUTPUTS_DIR}/latest_gst_tax_audit_summary.txt" || true)"
    if [[ -f "${LAB_OUTPUTS_DIR}/current_tx_coa_gst_guardrail_profile.txt" ]]; then
      NEXT_GST_PROFILE="$(cat "${LAB_OUTPUTS_DIR}/current_tx_coa_gst_guardrail_profile.txt")"
    fi
  fi

  "${PYTHON_BIN}" - "${MANIFEST_JSONL}" "${i}" "${START_UTC}" "${END_UTC}" "${RC}" "${LOG_FILE}" "${TX_SUMMARY}" "${TX_ROUTING_SUMMARY}" "${PROD_SUMMARY}" "${RISK_SUMMARY}" "${RECON_SUMMARY}" "${GST_AUDIT_SUMMARY}" "${NEXT_GST_PROFILE}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
row = {
    "schema_version": "v0",
    "cycle_index": int(sys.argv[2]),
    "started_at_utc": sys.argv[3],
    "ended_at_utc": sys.argv[4],
    "exit_code": int(sys.argv[5]),
    "ok": int(sys.argv[5]) == 0,
    "log_file": sys.argv[6],
    "tx_coa_cycle_summary": sys.argv[7] or None,
    "tx_coa_routing_summary": sys.argv[8] or None,
    "product_cycle_summary": sys.argv[9] or None,
    "review_risk_cycle_summary": sys.argv[10] or None,
    "reconciliation_cycle_summary": sys.argv[11] or None,
    "gst_audit_summary": sys.argv[12] or None,
    "next_gst_guardrail_profile": sys.argv[13] or None,
}
with path.open("a", encoding="utf-8") as f:
    f.write(json.dumps(row, sort_keys=True))
    f.write("\n")
PY

  echo "Autopilot cycle ${i}/${CYCLES} complete (exit=${RC}). Log: ${LOG_FILE}"
  if [[ "${RC}" -ne 0 && "${STOP_ON_ERROR}" == "true" ]]; then
    echo "Stopping due to AUTOPILOT_STOP_ON_ERROR=true and cycle failure."
    break
  fi

  if [[ "${i}" -lt "${CYCLES}" && "${SLEEP_SECONDS}" -gt 0 ]]; then
    sleep "${SLEEP_SECONDS}"
  fi
done

echo "${RUN_DIR}" > "${LAB_OUTPUTS_DIR}/latest_autopilot_run_dir.txt"
echo "Autopilot run complete."
echo "Run dir: ${RUN_DIR}"
echo "Manifest: ${MANIFEST_JSONL}"
