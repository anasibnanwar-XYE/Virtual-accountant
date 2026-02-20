#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_common.sh"

if [[ "$#" -lt 1 ]]; then
  echo "Usage: bash scripts/27_ingest_override_reason_feedback.sh /path/to/feedback1.csv [/path/to/feedback2.jsonl ...]" >&2
  exit 1
fi

MEMORY_JSONL="${OVERRIDE_REASON_MEMORY_JSONL:-${LAB_DATA_DIR}/labels/override_reason_feedback_memory.jsonl}"
RUN_DIR="${LAB_OUTPUTS_DIR}/override_reason_ingest_$(timestamp)"
REPORT_JSON="${RUN_DIR}/ingest_report.json"
mkdir -p "${RUN_DIR}"

CMD=(
  "${PYTHON_BIN}" "${SCRIPT_DIR}/override_reason_feedback_ingest.py"
  --out-jsonl "${MEMORY_JSONL}"
  --report-out "${REPORT_JSON}"
)

for src in "$@"; do
  if [[ ! -f "${src}" ]]; then
    echo "Feedback input not found: ${src}" >&2
    exit 1
  fi
  CMD+=(--input-file "${src}")
done

"${CMD[@]}" >/tmp/override_reason_feedback_ingest_out.txt

if [[ ! -f "${MEMORY_JSONL}" ]]; then
  echo "Override reason memory file not created: ${MEMORY_JSONL}" >&2
  exit 1
fi

echo "${MEMORY_JSONL}" > "${LAB_OUTPUTS_DIR}/latest_override_reason_memory_jsonl.txt"
echo "${REPORT_JSON}" > "${LAB_OUTPUTS_DIR}/latest_override_reason_ingest_report.txt"

echo "Override reason feedback ingestion complete."
echo "Feedback memory: ${MEMORY_JSONL}"
echo "Ingest report: ${REPORT_JSON}"
