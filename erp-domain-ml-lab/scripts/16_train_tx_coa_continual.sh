#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_common.sh"
acquire_lab_lock "continual_tx_coa"

TRAINING_CSV=""
BENCHMARK_SNAPSHOT=""
REVIEW_CSV=""

while getopts ":t:s:r:" opt; do
  case "${opt}" in
    t)
      TRAINING_CSV="${OPTARG}"
      ;;
    s)
      BENCHMARK_SNAPSHOT="${OPTARG}"
      ;;
    r)
      REVIEW_CSV="${OPTARG}"
      ;;
    *)
      echo "Usage: bash scripts/16_train_tx_coa_continual.sh [-t training.csv] [-s benchmark_snapshot_dir] [-r review_filled.csv]" >&2
      exit 1
      ;;
  esac
done

if [[ -z "${TRAINING_CSV}" ]]; then
  TRAINING_CSV="$(latest_file "${LAB_OUTPUTS_DIR}/latest_training_csv.txt")"
fi
if [[ -n "${TRAINING_CSV}" ]]; then
  TRAINING_CSV="$(cd "$(dirname "${TRAINING_CSV}")" && pwd)/$(basename "${TRAINING_CSV}")"
fi
if [[ -z "${TRAINING_CSV}" || ! -f "${TRAINING_CSV}" ]]; then
  echo "Training CSV not found. Pass -t or run data generation first." >&2
  exit 1
fi

if [[ -n "${REVIEW_CSV}" ]]; then
  REVIEW_CSV="$(cd "$(dirname "${REVIEW_CSV}")" && pwd)/$(basename "${REVIEW_CSV}")"
  if [[ ! -f "${REVIEW_CSV}" ]]; then
    echo "Review CSV not found: ${REVIEW_CSV}" >&2
    exit 1
  fi
  bash "${SCRIPT_DIR}/06_import_labels_and_train_v1.sh" "${REVIEW_CSV}"
fi

if [[ -z "${BENCHMARK_SNAPSHOT}" ]]; then
  BENCHMARK_SNAPSHOT="${LAB_SNAPSHOTS_DIR}/benchmark_tx_coa_snapshot"
fi
BENCHMARK_SNAPSHOT="$(cd "$(dirname "${BENCHMARK_SNAPSHOT}")" && pwd)/$(basename "${BENCHMARK_SNAPSHOT}")"

if [[ ! -d "${BENCHMARK_SNAPSHOT}" || "${REBUILD_TX_COA_BENCHMARK:-false}" == "true" ]]; then
  rm -rf "${BENCHMARK_SNAPSHOT}"
  run_snapshot_cli synthetic \
    --company-code "${COMPANY_CODE}" \
    --period-start "${PERIOD_START}" \
    --period-end "${PERIOD_END}" \
    --seed "${TX_COA_BENCHMARK_SEED:-4242}" \
    --journal-entries "${TX_COA_BENCHMARK_JOURNALS:-6000}" \
    --out "${BENCHMARK_SNAPSHOT}" >/tmp/tx_coa_benchmark_manifest.txt
fi

if [[ ! -d "${BENCHMARK_SNAPSHOT}" ]]; then
  echo "Benchmark snapshot unavailable: ${BENCHMARK_SNAPSHOT}" >&2
  exit 1
fi

EFFECTIVE_TRAINING_CSV="${TRAINING_CSV}"
OVERRIDE_ENRICH_REPORT=""
OVERRIDE_MEMORY_JSONL="${OVERRIDE_REASON_MEMORY_JSONL:-${LAB_DATA_DIR}/labels/override_reason_feedback_memory.jsonl}"
if [[ "${TX_COA_ENABLE_OVERRIDE_ENRICHMENT:-true}" == "true" ]]; then
  PREP_DIR="${LAB_OUTPUTS_DIR}/continual_learning/tx_coa/prep_$(timestamp)"
  mkdir -p "${PREP_DIR}"
  EFFECTIVE_TRAINING_CSV="${PREP_DIR}/training_enriched.csv"
  OVERRIDE_ENRICH_REPORT="${PREP_DIR}/override_enrich_report.json"

  if [[ -f "${OVERRIDE_MEMORY_JSONL}" ]]; then
    "${PYTHON_BIN}" "${SCRIPT_DIR}/override_reason_enrich_training_csv.py" \
      --training-csv "${TRAINING_CSV}" \
      --override-jsonl "${OVERRIDE_MEMORY_JSONL}" \
      --out-csv "${EFFECTIVE_TRAINING_CSV}" \
      --report-out "${OVERRIDE_ENRICH_REPORT}" >/tmp/tx_coa_override_enrich_out.txt
  else
    cp -f "${TRAINING_CSV}" "${EFFECTIVE_TRAINING_CSV}"
    "${PYTHON_BIN}" - "${OVERRIDE_ENRICH_REPORT}" "${TRAINING_CSV}" "${EFFECTIVE_TRAINING_CSV}" "${OVERRIDE_MEMORY_JSONL}" <<'PY'
import json
import sys
from pathlib import Path

report = {
    "schema_version": "v0",
    "status": "skipped_memory_missing",
    "training_csv": sys.argv[2],
    "override_jsonl": sys.argv[4],
    "out_csv": sys.argv[3],
}
Path(sys.argv[1]).write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
PY
  fi
fi

CHAMP_TX_MODEL_FILE="${LAB_OUTPUTS_DIR}/current_transaction_champion_model_dir.txt"
CHAMP_COA_MODEL_FILE="${LAB_OUTPUTS_DIR}/current_coa_champion_bundle_dir.txt"
CHAMP_SCORECARD_FILE="${LAB_OUTPUTS_DIR}/current_tx_coa_champion_scorecard.json"
GST_STAGE_PROFILE_FILE="${LAB_OUTPUTS_DIR}/current_tx_coa_gst_guardrail_profile.txt"
GST_STAGE_STATE_JSON="${LAB_OUTPUTS_DIR}/gst_guardrail_stage_state.json"

LATEST_TX_MODEL="$(latest_file "${LAB_OUTPUTS_DIR}/latest_transaction_model_dir.txt")"
LATEST_COA_MODEL="$(latest_file "${LAB_OUTPUTS_DIR}/latest_coa_bundle_dir.txt")"

if [[ ! -f "${CHAMP_TX_MODEL_FILE}" && -n "${LATEST_TX_MODEL}" ]]; then
  echo "${LATEST_TX_MODEL}" > "${CHAMP_TX_MODEL_FILE}"
fi
if [[ ! -f "${CHAMP_COA_MODEL_FILE}" && -n "${LATEST_COA_MODEL}" ]]; then
  echo "${LATEST_COA_MODEL}" > "${CHAMP_COA_MODEL_FILE}"
fi

CHAMP_TX_MODEL="$(latest_file "${CHAMP_TX_MODEL_FILE}")"
CHAMP_COA_MODEL="$(latest_file "${CHAMP_COA_MODEL_FILE}")"
CHAMP_SCORECARD="$(latest_file "${CHAMP_SCORECARD_FILE}")"

# If champion scorecard is missing, evaluate current champion once.
if [[ -z "${CHAMP_SCORECARD}" || ! -f "${CHAMP_SCORECARD}" ]]; then
  if [[ -n "${CHAMP_TX_MODEL}" && -d "${CHAMP_TX_MODEL}" && -n "${CHAMP_COA_MODEL}" && -d "${CHAMP_COA_MODEL}" ]]; then
    bash "${SCRIPT_DIR}/15_eval_tx_coa_scorecard.sh" "${BENCHMARK_SNAPSHOT}" "${CHAMP_TX_MODEL}" "${CHAMP_COA_MODEL}"
    CHAMP_SCORECARD="$(latest_file "${LAB_OUTPUTS_DIR}/latest_tx_coa_scorecard.json")"
    if [[ -n "${CHAMP_SCORECARD}" && -f "${CHAMP_SCORECARD}" ]]; then
      echo "${CHAMP_SCORECARD}" > "${CHAMP_SCORECARD_FILE}"
    fi
  fi
fi

# Train challenger tx+coa models.
TRAIN_REUSED_EXISTING="false"
TRAIN_LOG="/tmp/tx_coa_continual_train_$(timestamp).log"
set +e
PRODUCT_MODEL_ENABLED="${TX_COA_INCLUDE_PRODUCT_MODEL:-false}" \
  bash "${SCRIPT_DIR}/03_train_models.sh" "${EFFECTIVE_TRAINING_CSV}" >"${TRAIN_LOG}" 2>&1
TRAIN_EXIT=$?
set -e

if [[ "${TRAIN_EXIT}" -ne 0 ]]; then
  if rg -q "Training manifest already exists with different content" "${TRAIN_LOG}"; then
    RETRY_OFFSET="${TX_COA_RETRY_FEATURE_OFFSET:-13}"
    BASE_V2_FEATURES="${V2_FEATURES:-2048}"
    BASE_COA_FEATURES="${COA_FEATURES:-2048}"
    RETRY_V2_FEATURES="$((BASE_V2_FEATURES + RETRY_OFFSET))"
    RETRY_COA_FEATURES="$((BASE_COA_FEATURES + RETRY_OFFSET))"

    echo "Detected manifest collision; retrying training with feature offsets (V2_FEATURES=${RETRY_V2_FEATURES}, COA_FEATURES=${RETRY_COA_FEATURES})."

    RETRY_LOG="/tmp/tx_coa_continual_train_retry_$(timestamp).log"
    set +e
    PRODUCT_MODEL_ENABLED="${TX_COA_INCLUDE_PRODUCT_MODEL:-false}" \
      V2_FEATURES="${RETRY_V2_FEATURES}" \
      COA_FEATURES="${RETRY_COA_FEATURES}" \
      bash "${SCRIPT_DIR}/03_train_models.sh" "${EFFECTIVE_TRAINING_CSV}" >"${RETRY_LOG}" 2>&1
    RETRY_EXIT=$?
    set -e

    if [[ "${RETRY_EXIT}" -ne 0 ]]; then
      if rg -q "Training manifest already exists with different content" "${RETRY_LOG}"; then
        echo "Manifest collision persisted after retry; reusing current latest tx+coa models as challenger."
        TRAIN_REUSED_EXISTING="true"
      else
        cat "${RETRY_LOG}" >&2
        echo "TX+CoA training retry failed." >&2
        exit "${RETRY_EXIT}"
      fi
    fi
  else
    cat "${TRAIN_LOG}" >&2
    echo "TX+CoA training failed." >&2
    exit "${TRAIN_EXIT}"
  fi
fi

CHALL_TX_MODEL="$(latest_file "${LAB_OUTPUTS_DIR}/latest_transaction_model_dir.txt")"
CHALL_COA_MODEL="$(latest_file "${LAB_OUTPUTS_DIR}/latest_coa_bundle_dir.txt")"

if [[ -z "${CHALL_TX_MODEL}" || ! -d "${CHALL_TX_MODEL}" ]]; then
  echo "Challenger transaction model missing after training." >&2
  exit 1
fi
if [[ -z "${CHALL_COA_MODEL}" || ! -d "${CHALL_COA_MODEL}" ]]; then
  echo "Challenger CoA model missing after training." >&2
  exit 1
fi

bash "${SCRIPT_DIR}/15_eval_tx_coa_scorecard.sh" "${BENCHMARK_SNAPSHOT}" "${CHALL_TX_MODEL}" "${CHALL_COA_MODEL}"
CHALL_SCORECARD="$(latest_file "${LAB_OUTPUTS_DIR}/latest_tx_coa_scorecard.json")"
if [[ -z "${CHALL_SCORECARD}" || ! -f "${CHALL_SCORECARD}" ]]; then
  echo "Challenger scorecard missing." >&2
  exit 1
fi

CHALL_EVAL_DIR="$(cd "$(dirname "${CHALL_SCORECARD}")" && pwd)"
CHALL_ROUTING_POLICY_JSON="${CHALL_EVAL_DIR}/workflow_routing/workflow_routing_policy.json"
CHALL_ROUTING_SUMMARY_JSON="${CHALL_EVAL_DIR}/workflow_routing/workflow_routing_summary.json"
CHALL_ROUTING_DECISIONS_JSONL="${CHALL_EVAL_DIR}/workflow_routing/workflow_routing_decisions.jsonl"
bash "${SCRIPT_DIR}/19_run_gst_tax_audit.sh" "${CHALL_EVAL_DIR}" >/tmp/tx_coa_challenger_gst_audit_out.txt
CHALL_GST_AUDIT_SUMMARY="${CHALL_EVAL_DIR}/gst_tax_audit/gst_tax_audit_summary.json"
if [[ ! -f "${CHALL_GST_AUDIT_SUMMARY}" ]]; then
  echo "Challenger GST audit summary missing: ${CHALL_GST_AUDIT_SUMMARY}" >&2
  exit 1
fi

CHAMP_GST_AUDIT_SUMMARY=""
CHAMP_ROUTING_POLICY_JSON=""
CHAMP_ROUTING_SUMMARY_JSON=""
CHAMP_ROUTING_DECISIONS_JSONL=""
if [[ -n "${CHAMP_SCORECARD}" && -f "${CHAMP_SCORECARD}" ]]; then
  CHAMP_EVAL_DIR="$(cd "$(dirname "${CHAMP_SCORECARD}")" && pwd)"
  CHAMP_ROUTING_POLICY_JSON="${CHAMP_EVAL_DIR}/workflow_routing/workflow_routing_policy.json"
  CHAMP_ROUTING_SUMMARY_JSON="${CHAMP_EVAL_DIR}/workflow_routing/workflow_routing_summary.json"
  CHAMP_ROUTING_DECISIONS_JSONL="${CHAMP_EVAL_DIR}/workflow_routing/workflow_routing_decisions.jsonl"
  CHAMP_GST_AUDIT_SUMMARY="${CHAMP_EVAL_DIR}/gst_tax_audit/gst_tax_audit_summary.json"
  if [[ ! -f "${CHAMP_GST_AUDIT_SUMMARY}" || "${TX_COA_REBUILD_GST_AUDIT:-false}" == "true" ]]; then
    bash "${SCRIPT_DIR}/19_run_gst_tax_audit.sh" "${CHAMP_EVAL_DIR}" >/tmp/tx_coa_champion_gst_audit_out.txt
  fi
  if [[ ! -f "${CHAMP_GST_AUDIT_SUMMARY}" ]]; then
    echo "Champion GST audit summary missing: ${CHAMP_GST_AUDIT_SUMMARY}" >&2
    exit 1
  fi
fi

TX_COA_GST_GUARDRAIL_PROFILE="${TX_COA_GST_GUARDRAIL_PROFILE:-}"
if [[ -z "${TX_COA_GST_GUARDRAIL_PROFILE}" && -f "${GST_STAGE_PROFILE_FILE}" ]]; then
  TX_COA_GST_GUARDRAIL_PROFILE="$(cat "${GST_STAGE_PROFILE_FILE}")"
fi
if [[ -z "${TX_COA_GST_GUARDRAIL_PROFILE}" ]]; then
  TX_COA_GST_GUARDRAIL_PROFILE="${TX_COA_GST_STAGE_INITIAL:-stage1}"
fi
case "${TX_COA_GST_GUARDRAIL_PROFILE}" in
  permissive)
    PROFILE_MAX_GST_MAJOR_FAIL_RATE="1.0"
    PROFILE_MAX_GST_ISSUE_RATE="1.0"
    PROFILE_MAX_GST_CRITICAL_ISSUES="0"
    PROFILE_MAX_GST_MAJOR_FAIL_RATE_DELTA="0.05"
    PROFILE_MAX_GST_ISSUES_TOTAL_DELTA="1000"
    PROFILE_MAX_GST_SALE_MISSING_DELTA="500"
    PROFILE_MAX_GST_PURCHASE_MISSING_DELTA="500"
    ;;
  stage1)
    PROFILE_MAX_GST_MAJOR_FAIL_RATE="0.75"
    PROFILE_MAX_GST_ISSUE_RATE="0.85"
    PROFILE_MAX_GST_CRITICAL_ISSUES="0"
    PROFILE_MAX_GST_MAJOR_FAIL_RATE_DELTA="0.02"
    PROFILE_MAX_GST_ISSUES_TOTAL_DELTA="50"
    PROFILE_MAX_GST_SALE_MISSING_DELTA="25"
    PROFILE_MAX_GST_PURCHASE_MISSING_DELTA="25"
    ;;
  stage2)
    PROFILE_MAX_GST_MAJOR_FAIL_RATE="0.55"
    PROFILE_MAX_GST_ISSUE_RATE="0.70"
    PROFILE_MAX_GST_CRITICAL_ISSUES="0"
    PROFILE_MAX_GST_MAJOR_FAIL_RATE_DELTA="0.0"
    PROFILE_MAX_GST_ISSUES_TOTAL_DELTA="0"
    PROFILE_MAX_GST_SALE_MISSING_DELTA="0"
    PROFILE_MAX_GST_PURCHASE_MISSING_DELTA="0"
    ;;
  stage3)
    PROFILE_MAX_GST_MAJOR_FAIL_RATE="0.35"
    PROFILE_MAX_GST_ISSUE_RATE="0.50"
    PROFILE_MAX_GST_CRITICAL_ISSUES="0"
    PROFILE_MAX_GST_MAJOR_FAIL_RATE_DELTA="-0.01"
    PROFILE_MAX_GST_ISSUES_TOTAL_DELTA="-20"
    PROFILE_MAX_GST_SALE_MISSING_DELTA="-10"
    PROFILE_MAX_GST_PURCHASE_MISSING_DELTA="-10"
    ;;
  *)
    echo "Invalid TX_COA_GST_GUARDRAIL_PROFILE: ${TX_COA_GST_GUARDRAIL_PROFILE} (expected: permissive|stage1|stage2|stage3)" >&2
    exit 1
    ;;
esac

EFFECTIVE_MAX_GST_MAJOR_FAIL_RATE="${TX_COA_MAX_GST_MAJOR_FAIL_RATE:-${PROFILE_MAX_GST_MAJOR_FAIL_RATE}}"
EFFECTIVE_MAX_GST_ISSUE_RATE="${TX_COA_MAX_GST_ISSUE_RATE:-${PROFILE_MAX_GST_ISSUE_RATE}}"
EFFECTIVE_MAX_GST_CRITICAL_ISSUES="${TX_COA_MAX_GST_CRITICAL_ISSUES:-${PROFILE_MAX_GST_CRITICAL_ISSUES}}"
EFFECTIVE_MAX_GST_MAJOR_FAIL_RATE_DELTA="${TX_COA_MAX_GST_MAJOR_FAIL_RATE_DELTA:-${PROFILE_MAX_GST_MAJOR_FAIL_RATE_DELTA}}"
EFFECTIVE_MAX_GST_ISSUES_TOTAL_DELTA="${TX_COA_MAX_GST_ISSUES_TOTAL_DELTA:-${PROFILE_MAX_GST_ISSUES_TOTAL_DELTA}}"
EFFECTIVE_MAX_GST_SALE_MISSING_DELTA="${TX_COA_MAX_GST_SALE_MISSING_DELTA:-${PROFILE_MAX_GST_SALE_MISSING_DELTA}}"
EFFECTIVE_MAX_GST_PURCHASE_MISSING_DELTA="${TX_COA_MAX_GST_PURCHASE_MISSING_DELTA:-${PROFILE_MAX_GST_PURCHASE_MISSING_DELTA}}"

RUN_DIR="${LAB_OUTPUTS_DIR}/continual_learning/tx_coa/cycle_$(timestamp)"
mkdir -p "${RUN_DIR}"
GATE_JSON="${RUN_DIR}/gate_decision.json"
SUMMARY_JSON="${RUN_DIR}/cycle_summary.json"

GATE_CMD=(
  "${PYTHON_BIN}" "${SCRIPT_DIR}/tx_coa_champion_gate.py"
  --challenger-scorecard "${CHALL_SCORECARD}"
  --decision-out "${GATE_JSON}"
  --topk "${WF_TOPK:-3}"
  --max-tx-drop-ppm "${TX_COA_MAX_TX_DROP_PPM:-2000}"
  --max-coa-top1-drop "${TX_COA_MAX_COA_TOP1_DROP:-0.004}"
  --max-coa-top3-drop "${TX_COA_MAX_COA_TOP3_DROP:-0.002}"
  --min-overall-delta "${TX_COA_MIN_OVERALL_DELTA:--0.001}"
  --min-family-examples "${TX_COA_MIN_FAMILY_EXAMPLES:-80}"
  --max-family-tx-drop "${TX_COA_MAX_FAMILY_TX_DROP:-0.02}"
  --max-policy-blocked-top1-rate "${TX_COA_MAX_POLICY_BLOCKED_TOP1_RATE:-0.0}"
  --max-policy-discouraged-top1-rate "${TX_COA_MAX_POLICY_DISCOURAGED_TOP1_RATE:-0.25}"
  --max-policy-blocked-top1-rate-delta "${TX_COA_MAX_POLICY_BLOCKED_TOP1_RATE_DELTA:-0.0}"
  --max-policy-discouraged-top1-rate-delta "${TX_COA_MAX_POLICY_DISCOURAGED_TOP1_RATE_DELTA:-0.05}"
  --challenger-gst-audit "${CHALL_GST_AUDIT_SUMMARY}"
  --max-gst-major-fail-rate "${EFFECTIVE_MAX_GST_MAJOR_FAIL_RATE}"
  --max-gst-examples-with-issues-rate "${EFFECTIVE_MAX_GST_ISSUE_RATE}"
  --max-gst-critical-issues "${EFFECTIVE_MAX_GST_CRITICAL_ISSUES}"
  --max-gst-major-fail-rate-delta "${EFFECTIVE_MAX_GST_MAJOR_FAIL_RATE_DELTA}"
  --max-gst-issues-total-delta "${EFFECTIVE_MAX_GST_ISSUES_TOTAL_DELTA}"
  --max-gst-sale-missing-delta "${EFFECTIVE_MAX_GST_SALE_MISSING_DELTA}"
  --max-gst-purchase-missing-delta "${EFFECTIVE_MAX_GST_PURCHASE_MISSING_DELTA}"
  --threshold-key "${TX_COA_THRESHOLD_KEY:-0.900000}"
  --min-auto-accept-rate-ppm "${TX_COA_MIN_AUTO_ACCEPT_RATE_PPM:-0}"
  --min-auto-accept-accuracy-ppm "${TX_COA_MIN_AUTO_ACCEPT_ACCURACY_PPM:-900000}"
  --max-auto-accept-rate-drop-ppm "${TX_COA_MAX_AUTO_ACCEPT_RATE_DROP_PPM:-80000}"
  --max-auto-accept-accuracy-drop-ppm "${TX_COA_MAX_AUTO_ACCEPT_ACCURACY_DROP_PPM:-30000}"
  --max-review-rate-increase-ppm "${TX_COA_MAX_REVIEW_RATE_INCREASE_PPM:-100000}"
)
if [[ "${TX_COA_ALLOW_PERIOD_LOCK_AUTOACCEPT:-false}" == "true" ]]; then
  GATE_CMD+=(--allow-period-lock-autoaccept)
fi
if [[ -n "${CHAMP_SCORECARD}" && -f "${CHAMP_SCORECARD}" ]]; then
  GATE_CMD+=(--champion-scorecard "${CHAMP_SCORECARD}")
  if [[ -n "${CHAMP_GST_AUDIT_SUMMARY}" && -f "${CHAMP_GST_AUDIT_SUMMARY}" ]]; then
    GATE_CMD+=(--champion-gst-audit "${CHAMP_GST_AUDIT_SUMMARY}")
  fi
fi
"${GATE_CMD[@]}" >/tmp/tx_coa_gate_out.txt

PROMOTE="$(${PYTHON_BIN} - "${GATE_JSON}" <<'PY'
import json
import sys
obj = json.load(open(sys.argv[1], encoding="utf-8"))
print("true" if obj.get("promote") else "false")
PY
)"

STATUS="rejected"
if [[ "${PROMOTE}" == "true" ]]; then
  STATUS="promoted"
  echo "${CHALL_TX_MODEL}" > "${CHAMP_TX_MODEL_FILE}"
  echo "${CHALL_COA_MODEL}" > "${CHAMP_COA_MODEL_FILE}"
  echo "${CHALL_SCORECARD}" > "${CHAMP_SCORECARD_FILE}"
else
  if [[ -n "${CHAMP_TX_MODEL}" && -d "${CHAMP_TX_MODEL}" ]]; then
    echo "${CHAMP_TX_MODEL}" > "${LAB_OUTPUTS_DIR}/latest_transaction_model_dir.txt"
  fi
  if [[ -n "${CHAMP_COA_MODEL}" && -d "${CHAMP_COA_MODEL}" ]]; then
    echo "${CHAMP_COA_MODEL}" > "${LAB_OUTPUTS_DIR}/latest_coa_bundle_dir.txt"
  fi
  echo "${CHALL_TX_MODEL}" > "${LAB_OUTPUTS_DIR}/latest_transaction_rejected_model_dir.txt"
  echo "${CHALL_COA_MODEL}" > "${LAB_OUTPUTS_DIR}/latest_coa_rejected_bundle_dir.txt"
  echo "${CHALL_SCORECARD}" > "${LAB_OUTPUTS_DIR}/latest_tx_coa_rejected_scorecard.json"
fi

"${PYTHON_BIN}" - "${SUMMARY_JSON}" "${STATUS}" "${TRAINING_CSV}" "${EFFECTIVE_TRAINING_CSV}" "${OVERRIDE_ENRICH_REPORT}" "${OVERRIDE_MEMORY_JSONL}" "${BENCHMARK_SNAPSHOT}" "${CHAMP_TX_MODEL}" "${CHAMP_COA_MODEL}" "${CHAMP_SCORECARD}" "${CHAMP_ROUTING_POLICY_JSON}" "${CHAMP_ROUTING_SUMMARY_JSON}" "${CHAMP_ROUTING_DECISIONS_JSONL}" "${CHALL_TX_MODEL}" "${CHALL_COA_MODEL}" "${CHALL_SCORECARD}" "${CHALL_ROUTING_POLICY_JSON}" "${CHALL_ROUTING_SUMMARY_JSON}" "${CHALL_ROUTING_DECISIONS_JSONL}" "${GATE_JSON}" "${TRAIN_REUSED_EXISTING}" "${CHAMP_GST_AUDIT_SUMMARY}" "${CHALL_GST_AUDIT_SUMMARY}" "${TX_COA_GST_GUARDRAIL_PROFILE}" "${EFFECTIVE_MAX_GST_MAJOR_FAIL_RATE}" "${EFFECTIVE_MAX_GST_ISSUE_RATE}" "${EFFECTIVE_MAX_GST_CRITICAL_ISSUES}" "${EFFECTIVE_MAX_GST_MAJOR_FAIL_RATE_DELTA}" "${EFFECTIVE_MAX_GST_ISSUES_TOTAL_DELTA}" "${EFFECTIVE_MAX_GST_SALE_MISSING_DELTA}" "${EFFECTIVE_MAX_GST_PURCHASE_MISSING_DELTA}" <<'PY'
import json
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
status = sys.argv[2]
summary = {
    "schema_version": "v0",
    "status": status,
    "training_csv": sys.argv[3],
    "effective_training_csv": sys.argv[4],
    "override_enrich_report": sys.argv[5] or None,
    "override_memory_jsonl": sys.argv[6] or None,
    "benchmark_snapshot": sys.argv[7],
    "previous_champion_transaction_model": sys.argv[8] or None,
    "previous_champion_coa_model": sys.argv[9] or None,
    "previous_champion_scorecard": sys.argv[10] or None,
    "previous_champion_routing_policy_json": sys.argv[11] or None,
    "previous_champion_routing_summary_json": sys.argv[12] or None,
    "previous_champion_routing_decisions_jsonl": sys.argv[13] or None,
    "challenger_transaction_model": sys.argv[14],
    "challenger_coa_model": sys.argv[15],
    "challenger_scorecard": sys.argv[16],
    "challenger_routing_policy_json": sys.argv[17] or None,
    "challenger_routing_summary_json": sys.argv[18] or None,
    "challenger_routing_decisions_jsonl": sys.argv[19] or None,
    "gate_decision": json.loads(Path(sys.argv[20]).read_text(encoding="utf-8")),
    "training_reused_existing_models": True if str(sys.argv[21]).lower() == "true" else False,
    "previous_champion_gst_audit_summary": sys.argv[22] or None,
    "challenger_gst_audit_summary": sys.argv[23] or None,
    "gst_guardrail_profile": sys.argv[24],
    "gst_guardrails_effective": {
        "max_gst_major_fail_rate": float(sys.argv[25]),
        "max_gst_issue_rate": float(sys.argv[26]),
        "max_gst_critical_issues": int(sys.argv[27]),
        "max_gst_major_fail_rate_delta": float(sys.argv[28]),
        "max_gst_issues_total_delta": int(sys.argv[29]),
        "max_gst_sale_missing_delta": int(sys.argv[30]),
        "max_gst_purchase_missing_delta": int(sys.argv[31]),
    },
}
summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
print(summary_path.as_posix())
PY

if [[ "${TX_COA_GST_STAGE_AUTO:-true}" == "true" ]]; then
  STAGE_UPDATE_JSON="${RUN_DIR}/gst_guardrail_stage_update.json"
  STAGE_CMD=(
    "${PYTHON_BIN}" "${SCRIPT_DIR}/21_update_gst_guardrail_stage.py"
    --cycle-summary "${SUMMARY_JSON}"
    --state-json "${GST_STAGE_STATE_JSON}"
    --out-json "${STAGE_UPDATE_JSON}"
    --initial-stage "${TX_COA_GST_STAGE_INITIAL:-stage1}"
    --promote-stage1-after "${TX_COA_GST_STAGE_PROMOTE_STAGE1_AFTER:-2}"
    --promote-stage2-after "${TX_COA_GST_STAGE_PROMOTE_STAGE2_AFTER:-3}"
    --promote-stage3-after "${TX_COA_GST_STAGE_PROMOTE_STAGE3_AFTER:-5}"
    --demote-after-fails "${TX_COA_GST_STAGE_DEMOTE_AFTER_FAILS:-2}"
    --max-history "${TX_COA_GST_STAGE_MAX_HISTORY:-200}"
  )
  if [[ "${TX_COA_GST_STAGE_REQUIRE_PROMOTED_STATUS:-false}" == "true" ]]; then
    STAGE_CMD+=(--require-promoted-status)
  fi
  if [[ "${TX_COA_GST_STAGE_ALLOW_DEMOTE_TO_PERMISSIVE:-false}" == "true" ]]; then
    STAGE_CMD+=(--allow-demote-to-permissive)
  fi
  "${STAGE_CMD[@]}" >/tmp/tx_coa_stage_update_out.txt

  NEXT_GST_PROFILE="$(${PYTHON_BIN} - "${STAGE_UPDATE_JSON}" <<'PY'
import json
import sys
obj = json.load(open(sys.argv[1], encoding="utf-8"))
print(str(obj.get("next_profile") or "stage1"))
PY
)"
  echo "${NEXT_GST_PROFILE}" > "${GST_STAGE_PROFILE_FILE}"

  "${PYTHON_BIN}" - "${SUMMARY_JSON}" "${STAGE_UPDATE_JSON}" "${GST_STAGE_STATE_JSON}" "${NEXT_GST_PROFILE}" <<'PY'
import json
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
stage_update_path = Path(sys.argv[2])
stage_state_path = Path(sys.argv[3])
next_profile = sys.argv[4]

summary = json.loads(summary_path.read_text(encoding="utf-8"))
summary["gst_stage_update_json"] = stage_update_path.as_posix()
summary["gst_stage_state_json"] = stage_state_path.as_posix()
summary["gst_next_guardrail_profile"] = next_profile
summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
PY
fi

echo "${RUN_DIR}" > "${LAB_OUTPUTS_DIR}/latest_tx_coa_continual_cycle_dir.txt"
echo "${SUMMARY_JSON}" > "${LAB_OUTPUTS_DIR}/latest_tx_coa_continual_cycle_summary.txt"

echo "TX+CoA continual learning cycle complete (${STATUS})."
echo "Cycle dir: ${RUN_DIR}"
echo "Summary: ${SUMMARY_JSON}"
echo "Current TX champion: $(latest_file "${CHAMP_TX_MODEL_FILE}")"
echo "Current CoA champion: $(latest_file "${CHAMP_COA_MODEL_FILE}")"
echo "Current TX+CoA scorecard: $(latest_file "${CHAMP_SCORECARD_FILE}")"
echo "GST guardrail profile: ${TX_COA_GST_GUARDRAIL_PROFILE}"
if [[ -f "${GST_STAGE_PROFILE_FILE}" ]]; then
  echo "GST guardrail profile (next cycle): $(cat "${GST_STAGE_PROFILE_FILE}")"
fi
