#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_common.sh"
acquire_lab_lock "full_continual_tick"

TRAINING_CSV=""
REVIEW_CSV=""
BENCHMARK_SNAPSHOT=""
PRODUCT_FEEDBACK_FILES=()
OVERRIDE_FEEDBACK_FILES=()
USER_PERSONALIZATION_FEEDBACK_FILES=()
PERSONALIZATION_USER_ID_OPT=""

while getopts ":t:r:s:f:o:p:u:" opt; do
  case "${opt}" in
    t)
      TRAINING_CSV="${OPTARG}"
      ;;
    r)
      REVIEW_CSV="${OPTARG}"
      ;;
    s)
      BENCHMARK_SNAPSHOT="${OPTARG}"
      ;;
    f)
      PRODUCT_FEEDBACK_FILES+=("${OPTARG}")
      ;;
    o)
      OVERRIDE_FEEDBACK_FILES+=("${OPTARG}")
      ;;
    p)
      USER_PERSONALIZATION_FEEDBACK_FILES+=("${OPTARG}")
      ;;
    u)
      PERSONALIZATION_USER_ID_OPT="${OPTARG}"
      ;;
    *)
      echo "Usage: bash scripts/17_run_full_continual_learning_tick.sh [-t training.csv] [-r reviewed_queue.csv] [-s benchmark_snapshot] [-f product_feedback.csv] [-o override_feedback.csv] [-p user_personalization_feedback.csv] [-u personalization_user_id]" >&2
      exit 1
      ;;
  esac
done

if [[ "${#OVERRIDE_FEEDBACK_FILES[@]}" -gt 0 ]]; then
  bash "${SCRIPT_DIR}/27_ingest_override_reason_feedback.sh" "${OVERRIDE_FEEDBACK_FILES[@]}"
fi
if [[ "${#USER_PERSONALIZATION_FEEDBACK_FILES[@]}" -gt 0 ]]; then
  bash "${SCRIPT_DIR}/32_ingest_user_personalization_feedback.sh" "${USER_PERSONALIZATION_FEEDBACK_FILES[@]}"
fi

TX_COA_CMD=(bash "${SCRIPT_DIR}/16_train_tx_coa_continual.sh")
if [[ -n "${TRAINING_CSV}" ]]; then
  TX_COA_CMD+=(-t "${TRAINING_CSV}")
fi
if [[ -n "${REVIEW_CSV}" ]]; then
  TX_COA_CMD+=(-r "${REVIEW_CSV}")
fi
if [[ -n "${BENCHMARK_SNAPSHOT}" ]]; then
  TX_COA_CMD+=(-s "${BENCHMARK_SNAPSHOT}")
fi
"${TX_COA_CMD[@]}"
bash "${SCRIPT_DIR}/19_run_gst_tax_audit.sh" >/tmp/full_tick_gst_audit_out.txt
GST_AUDIT_SUMMARY="$(latest_file "${LAB_OUTPUTS_DIR}/latest_gst_tax_audit_summary.txt")"
ROUTING_SUMMARY="$(latest_file "${LAB_OUTPUTS_DIR}/latest_workflow_routing_summary_json.txt")"
GST_STAGE_PROFILE=""
if [[ -f "${LAB_OUTPUTS_DIR}/current_tx_coa_gst_guardrail_profile.txt" ]]; then
  GST_STAGE_PROFILE="$(cat "${LAB_OUTPUTS_DIR}/current_tx_coa_gst_guardrail_profile.txt")"
fi
GST_STAGE_STATE=""
if [[ -f "${LAB_OUTPUTS_DIR}/gst_guardrail_stage_state.json" ]]; then
  GST_STAGE_STATE="${LAB_OUTPUTS_DIR}/gst_guardrail_stage_state.json"
fi

PROD_CMD=(bash "${SCRIPT_DIR}/13_train_product_account_continual.sh")
if [[ -n "${TRAINING_CSV}" ]]; then
  PROD_CMD+=(-t "${TRAINING_CSV}")
fi
for src in "${PRODUCT_FEEDBACK_FILES[@]}"; do
  PROD_CMD+=(-f "${src}")
done
"${PROD_CMD[@]}"

RISK_LOOP_STATUS="skipped"
if [[ "${RUN_REVIEW_RISK_LOOP:-true}" == "true" ]]; then
  RISK_CMD=(bash "${SCRIPT_DIR}/24_train_review_risk_continual.sh")
  if [[ -n "${TRAINING_CSV}" ]]; then
    RISK_CMD+=(-t "${TRAINING_CSV}")
  fi
  "${RISK_CMD[@]}"
  RISK_LOOP_STATUS="ran"
fi

RECON_LOOP_STATUS="skipped"
if [[ "${RUN_RECONCILIATION_LOOP:-true}" == "true" ]]; then
  RECON_CMD=(bash "${SCRIPT_DIR}/26_train_reconciliation_continual.sh")
  if [[ -n "${TRAINING_CSV}" ]]; then
    RECON_CMD+=(-t "${TRAINING_CSV}")
  fi
  "${RECON_CMD[@]}"
  RECON_LOOP_STATUS="ran"
fi

PERSONALIZATION_USER_ID_EFFECTIVE="${PERSONALIZATION_USER_ID_OPT:-${PERSONALIZATION_USER_ID:-}}"
PERSONALIZATION_STATUS="skipped"
if [[ -n "${PERSONALIZATION_USER_ID_EFFECTIVE}" ]]; then
  TX_EVAL_DIR="$(latest_file "${LAB_OUTPUTS_DIR}/latest_tx_coa_eval_dir.txt")"
  if [[ -n "${TX_EVAL_DIR}" && -d "${TX_EVAL_DIR}" ]]; then
    if bash "${SCRIPT_DIR}/33_apply_user_personalization.sh" "${PERSONALIZATION_USER_ID_EFFECTIVE}" "${TX_EVAL_DIR}" "${COMPANY_CODE}"; then
      PERSONALIZATION_REPORT="$(latest_file "${LAB_OUTPUTS_DIR}/latest_user_personalization_report_json.txt")"
      if [[ -n "${PERSONALIZATION_REPORT}" && -f "${PERSONALIZATION_REPORT}" ]]; then
        PERSONALIZATION_RUN_STATUS="$("${PYTHON_BIN}" - "${PERSONALIZATION_REPORT}" <<'PY'
import json
import sys
obj = json.load(open(sys.argv[1], encoding="utf-8"))
print(str(obj.get("status") or "unknown"))
PY
)"
        PERSONALIZATION_STATUS="ran:${PERSONALIZATION_RUN_STATUS}"
      else
        PERSONALIZATION_STATUS="ran"
      fi
    else
      PERSONALIZATION_STATUS="failed"
    fi
  fi
fi

echo "Full continual learning tick complete."
echo "TX+CoA loop: ran"
if [[ -n "${GST_AUDIT_SUMMARY}" ]]; then
  echo "GST tax audit summary: ${GST_AUDIT_SUMMARY}"
fi
if [[ -n "${ROUTING_SUMMARY}" ]]; then
  echo "Workflow routing summary: ${ROUTING_SUMMARY}"
fi
if [[ -n "${GST_STAGE_PROFILE}" ]]; then
  echo "GST guardrail profile (next cycle): ${GST_STAGE_PROFILE}"
fi
if [[ -n "${GST_STAGE_STATE}" ]]; then
  echo "GST guardrail stage state: ${GST_STAGE_STATE}"
fi
echo "Product loop: ran"
echo "Review-risk loop: ${RISK_LOOP_STATUS}"
echo "Reconciliation loop: ${RECON_LOOP_STATUS}"
echo "User personalization: ${PERSONALIZATION_STATUS}"
