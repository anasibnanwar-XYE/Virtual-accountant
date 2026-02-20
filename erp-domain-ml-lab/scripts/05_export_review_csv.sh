#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_common.sh"

INPUT_PATH="${1:-}"

OUT_CSV="${LAB_DATA_DIR}/labels/review_queue_$(timestamp).csv"
PRIORITY_QUEUE_JSONL="${PRIORITY_REVIEW_QUEUE_JSONL:-}"
PRIORITY_TX_JSONL="${PRIORITY_REVIEW_TX_JSONL:-}"
PACK_DIR=""

if [[ -z "${PRIORITY_QUEUE_JSONL}" ]]; then
  if [[ -n "${INPUT_PATH}" && -f "${INPUT_PATH}" ]]; then
    PRIORITY_QUEUE_JSONL="${INPUT_PATH}"
  elif [[ -n "${INPUT_PATH}" && -d "${INPUT_PATH}" && -f "${INPUT_PATH}/transaction_review_priority_queue.jsonl" ]]; then
    PRIORITY_QUEUE_JSONL="${INPUT_PATH}/transaction_review_priority_queue.jsonl"
    PACK_DIR="${INPUT_PATH}"
  elif [[ -n "${INPUT_PATH}" && -d "${INPUT_PATH}" && -f "${INPUT_PATH}/explanations/transaction_review_priority_queue.jsonl" ]]; then
    PRIORITY_QUEUE_JSONL="${INPUT_PATH}/explanations/transaction_review_priority_queue.jsonl"
    PACK_DIR="${INPUT_PATH}/explanations"
  elif [[ -f "${LAB_OUTPUTS_DIR}/latest_explanation_pack_dir.txt" ]]; then
    CANDIDATE_PACK_DIR="$(latest_file "${LAB_OUTPUTS_DIR}/latest_explanation_pack_dir.txt")"
    if [[ -n "${CANDIDATE_PACK_DIR}" && -f "${CANDIDATE_PACK_DIR}/transaction_review_priority_queue.jsonl" ]]; then
      PRIORITY_QUEUE_JSONL="${CANDIDATE_PACK_DIR}/transaction_review_priority_queue.jsonl"
      PACK_DIR="${CANDIDATE_PACK_DIR}"
    fi
  fi
fi

if [[ -n "${PRIORITY_QUEUE_JSONL}" && -z "${PRIORITY_TX_JSONL}" ]]; then
  if [[ -n "${PACK_DIR}" ]]; then
    RUN_DIR="$(cd "${PACK_DIR}/.." && pwd)"
    CANDIDATE_TX="${RUN_DIR}/transaction_v2/advisory_outputs/transaction_classifications_v2.jsonl"
    if [[ -f "${CANDIDATE_TX}" ]]; then
      PRIORITY_TX_JSONL="${CANDIDATE_TX}"
    fi
  fi
  if [[ -z "${PRIORITY_TX_JSONL}" && -n "${INPUT_PATH}" && -d "${INPUT_PATH}" ]]; then
    CANDIDATE_TX="${INPUT_PATH}/transaction_v2/advisory_outputs/transaction_classifications_v2.jsonl"
    if [[ -f "${CANDIDATE_TX}" ]]; then
      PRIORITY_TX_JSONL="${CANDIDATE_TX}"
    fi
  fi
fi

if [[ -n "${PRIORITY_QUEUE_JSONL}" && -n "${PRIORITY_TX_JSONL}" && -f "${PRIORITY_QUEUE_JSONL}" && -f "${PRIORITY_TX_JSONL}" ]]; then
  REPORT_JSON="${LAB_OUTPUTS_DIR}/review_priority_export_report_$(timestamp).json"
  "${PYTHON_BIN}" "${SCRIPT_DIR}/export_priority_review_csv.py" \
    --tx-jsonl "${PRIORITY_TX_JSONL}" \
    --priority-queue-jsonl "${PRIORITY_QUEUE_JSONL}" \
    --out-csv "${OUT_CSV}" \
    --report-out "${REPORT_JSON}" >/tmp/review_priority_export_out.txt
  echo "${REPORT_JSON}" > "${LAB_OUTPUTS_DIR}/latest_review_priority_export_report_json.txt"

  echo "${OUT_CSV}" > "${LAB_OUTPUTS_DIR}/latest_review_csv.txt"
  echo "Review CSV exported (priority-ranked): ${OUT_CSV}"
  echo "Source queue: ${PRIORITY_QUEUE_JSONL}"
  echo "Fill 'chosen_label' column, then run:"
  echo "bash scripts/06_import_labels_and_train_v1.sh ${OUT_CSV}"
  exit 0
fi

ADVISORY_TX_DIR="${INPUT_PATH:-}"
if [[ -z "${ADVISORY_TX_DIR}" ]]; then
  ADVISORY_TX_DIR="$(latest_file "${LAB_OUTPUTS_DIR}/latest_advisory_transaction_dir.txt")"
fi
if [[ -z "${ADVISORY_TX_DIR}" || ! -d "${ADVISORY_TX_DIR}" ]]; then
  echo "Transaction advisory snapshot dir not found. Run scripts/04_run_advisory.sh or scripts/18_generate_explanation_pack.sh first." >&2
  exit 1
fi

EXPORT_SOURCE_DIR="${ADVISORY_TX_DIR}"
if [[ ! -f "${ADVISORY_TX_DIR}/advisory_outputs/review_queue.jsonl" ]] && [[ -f "${ADVISORY_TX_DIR}/advisory_outputs/review_queue_v2.jsonl" ]]; then
  SHIM_DIR="${LAB_OUTPUTS_DIR}/_review_export_shim_$(timestamp)"
  mkdir -p "${SHIM_DIR}/advisory_outputs"
  cp "${ADVISORY_TX_DIR}/advisory_manifest.json" "${SHIM_DIR}/advisory_manifest.json"
  cp "${ADVISORY_TX_DIR}/advisory_outputs/review_queue_v2.jsonl" "${SHIM_DIR}/advisory_outputs/review_queue.jsonl"
  EXPORT_SOURCE_DIR="${SHIM_DIR}"
fi

run_analytics_cli export-review-csv \
  --advisory-snapshot "${EXPORT_SOURCE_DIR}" \
  --out-csv "${OUT_CSV}" >/dev/null

echo "${OUT_CSV}" > "${LAB_OUTPUTS_DIR}/latest_review_csv.txt"
echo "Review CSV exported (default queue): ${OUT_CSV}"
echo "Fill 'chosen_label' column, then run:"
echo "bash scripts/06_import_labels_and_train_v1.sh ${OUT_CSV}"
