#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_common.sh"

SAMPLE_FEEDBACK="${1:-${LAB_DATA_DIR}/training/user_personalization_feedback_sample.csv}"
if [[ ! -f "${SAMPLE_FEEDBACK}" ]]; then
  echo "Sample personalization feedback not found: ${SAMPLE_FEEDBACK}" >&2
  exit 1
fi

if ! command -v jq >/dev/null 2>&1; then
  echo "jq is required for this smoke test." >&2
  exit 1
fi

RUN_DIR="${LAB_OUTPUTS_DIR}/personalization_guardrail_smoke_$(timestamp)"
mkdir -p "${RUN_DIR}"

STRESS_CSV="${RUN_DIR}/stress_feedback.csv"
cat > "${STRESS_CSV}" <<'CSV'
actor_user_id,company_code,workflow_family,record_referenceNumber,suggested_label,chosen_label,suggested_debit_account_code,approved_debit_account_code,suggested_credit_account_code,approved_credit_account_code,reason_code,reason_text,action
CSV
for i in $(seq 1 160); do
  printf 'accountant.guardrail,BBP,sale,SO-GR-%04d,SALE,PURCHASE,AR,CASH,SALES,INVENTORY,STRESS_TEST,Guardrail stress test sample,OVERRIDDEN\n' "${i}" >> "${STRESS_CSV}"
done

bash "${SCRIPT_DIR}/32_ingest_user_personalization_feedback.sh" "${SAMPLE_FEEDBACK}" >/tmp/personalization_guardrail_smoke_ingest_sample.txt
bash "${SCRIPT_DIR}/33_apply_user_personalization.sh" accountant.a >/tmp/personalization_guardrail_smoke_apply_default.txt
DEFAULT_REPORT="$(latest_file "${LAB_OUTPUTS_DIR}/latest_user_personalization_report_json.txt")"
DEFAULT_REPORT_COPY="${RUN_DIR}/default_report.json"
cp -f "${DEFAULT_REPORT}" "${DEFAULT_REPORT_COPY}"
DEFAULT_STATUS="$(jq -r '.status // "unknown"' "${DEFAULT_REPORT_COPY}")"

USER_PERSONALIZATION_MIN_MEMORY_ROWS=999 \
  bash "${SCRIPT_DIR}/33_apply_user_personalization.sh" accountant.a >/tmp/personalization_guardrail_smoke_apply_lowmem.txt
LOWMEM_REPORT="$(latest_file "${LAB_OUTPUTS_DIR}/latest_user_personalization_report_json.txt")"
LOWMEM_REPORT_COPY="${RUN_DIR}/lowmem_report.json"
cp -f "${LOWMEM_REPORT}" "${LOWMEM_REPORT_COPY}"
LOWMEM_STATUS="$(jq -r '.status // "unknown"' "${LOWMEM_REPORT_COPY}")"

bash "${SCRIPT_DIR}/32_ingest_user_personalization_feedback.sh" "${STRESS_CSV}" >/tmp/personalization_guardrail_smoke_ingest_stress.txt
USER_PERSONALIZATION_TX_ALPHA=3.0 \
USER_PERSONALIZATION_COA_ALPHA=2.0 \
USER_PERSONALIZATION_MIN_FAMILY_MEMORY_ROWS=1 \
USER_PERSONALIZATION_MAX_FAMILY_TOP1_CHANGE_RATE=0.05 \
USER_PERSONALIZATION_MAX_TX_TOP1_CHANGE_RATE=0.40 \
USER_PERSONALIZATION_MAX_COA_DEBIT_TOP1_CHANGE_RATE=0.40 \
USER_PERSONALIZATION_MAX_COA_CREDIT_TOP1_CHANGE_RATE=0.40 \
  bash "${SCRIPT_DIR}/33_apply_user_personalization.sh" accountant.guardrail >/tmp/personalization_guardrail_smoke_apply_stress.txt
STRESS_REPORT="$(latest_file "${LAB_OUTPUTS_DIR}/latest_user_personalization_report_json.txt")"
STRESS_REPORT_COPY="${RUN_DIR}/stress_report.json"
cp -f "${STRESS_REPORT}" "${STRESS_REPORT_COPY}"
STRESS_STATUS="$(jq -r '.status // "unknown"' "${STRESS_REPORT_COPY}")"

if [[ "${DEFAULT_STATUS}" != "ok" ]]; then
  echo "Unexpected default status: ${DEFAULT_STATUS}" >&2
  exit 2
fi
if [[ "${LOWMEM_STATUS}" != "low_user_memory_skip" ]]; then
  echo "Unexpected low-memory status: ${LOWMEM_STATUS}" >&2
  exit 3
fi
if [[ "${STRESS_STATUS}" != "guardrail_reverted_to_base" ]]; then
  echo "Unexpected stress status: ${STRESS_STATUS}" >&2
  exit 4
fi

SUMMARY_JSON="${RUN_DIR}/summary.json"
"${PYTHON_BIN}" - "${SUMMARY_JSON}" "${DEFAULT_REPORT_COPY}" "${LOWMEM_REPORT_COPY}" "${STRESS_REPORT_COPY}" <<'PY'
import json
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
paths = [Path(p) for p in sys.argv[2:]]
reports = [json.loads(p.read_text(encoding="utf-8")) for p in paths]

payload = {
    "schema_version": "v0",
    "status": "ok",
    "checks": [
        {"name": "default_apply", "status": reports[0].get("status"), "report": paths[0].as_posix()},
        {"name": "low_memory_skip", "status": reports[1].get("status"), "report": paths[1].as_posix()},
        {"name": "stress_guardrail_revert", "status": reports[2].get("status"), "report": paths[2].as_posix()},
    ],
}
summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
print(summary_path.as_posix())
PY

echo "${RUN_DIR}" > "${LAB_OUTPUTS_DIR}/latest_personalization_guardrail_smoke_dir.txt"
echo "${SUMMARY_JSON}" > "${LAB_OUTPUTS_DIR}/latest_personalization_guardrail_smoke_summary_json.txt"

echo "Personalization guardrail smoke passed."
echo "Run dir: ${RUN_DIR}"
echo "Summary: ${SUMMARY_JSON}"
