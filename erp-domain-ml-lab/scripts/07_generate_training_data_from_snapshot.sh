#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_common.sh"

SNAPSHOT_DIR="${1:-}"
if [[ -z "${SNAPSHOT_DIR}" ]]; then
  SNAPSHOT_DIR="$(latest_file "${LAB_OUTPUTS_DIR}/latest_snapshot_dir.txt")"
fi
if [[ -z "${SNAPSHOT_DIR}" || ! -d "${SNAPSHOT_DIR}" ]]; then
  echo "Snapshot directory not found. Pass a path or run scripts/01_capture_snapshot.sh first." >&2
  exit 1
fi

OUT_CSV="${LAB_DATA_DIR}/training/orchestrator_snapshot_training_$(timestamp).csv"
REPORT_JSON="${LAB_OUTPUTS_DIR}/snapshot_training_report_$(timestamp).json"

EXTRA_ARGS=()
if [[ "${INCLUDE_NON_POSTED:-false}" == "true" ]]; then
  EXTRA_ARGS+=("--include-non-posted")
fi
if [[ "${ALLOW_UNKNOWN_LABELS:-false}" == "true" ]]; then
  EXTRA_ARGS+=("--allow-unknown-labels")
fi

"${PYTHON_BIN}" "${SCRIPT_DIR}/build_training_csv_from_snapshot.py" \
  --snapshot-dir "${SNAPSHOT_DIR}" \
  --out-csv "${OUT_CSV}" \
  --label-field "${SNAPSHOT_LABEL_FIELD:-syntheticLabel}" \
  --balance-tolerance "${BALANCE_TOLERANCE:-0.01}" \
  --currency "${CURRENCY_CODE:-INR}" \
  --report-json "${REPORT_JSON}" \
  "${EXTRA_ARGS[@]}"

echo "${OUT_CSV}" > "${LAB_OUTPUTS_DIR}/latest_training_csv.txt"
echo "${REPORT_JSON}" > "${LAB_OUTPUTS_DIR}/latest_snapshot_training_report.txt"

echo "Snapshot training CSV ready: ${OUT_CSV}"
echo "Build report: ${REPORT_JSON}"
