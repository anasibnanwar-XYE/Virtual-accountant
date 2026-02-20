#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_common.sh"

TRAINING_CSV=""
REVIEW_CSV=""
PRODUCT_FEEDBACK_FILES=()

while getopts ":t:r:f:" opt; do
  case "${opt}" in
    t)
      TRAINING_CSV="${OPTARG}"
      ;;
    r)
      REVIEW_CSV="${OPTARG}"
      ;;
    f)
      PRODUCT_FEEDBACK_FILES+=("${OPTARG}")
      ;;
    *)
      echo "Usage: bash scripts/14_run_continual_learning_tick.sh [-t training.csv] [-r reviewed_queue.csv] [-f product_feedback.csv]" >&2
      exit 1
      ;;
  esac
done

if [[ -n "${REVIEW_CSV}" ]]; then
  if [[ ! -f "${REVIEW_CSV}" ]]; then
    echo "Reviewed CSV not found: ${REVIEW_CSV}" >&2
    exit 1
  fi
  bash "${SCRIPT_DIR}/06_import_labels_and_train_v1.sh" "${REVIEW_CSV}"
fi

CONT_CMD=(bash "${SCRIPT_DIR}/13_train_product_account_continual.sh")
if [[ -n "${TRAINING_CSV}" ]]; then
  CONT_CMD+=(-t "${TRAINING_CSV}")
fi
for src in "${PRODUCT_FEEDBACK_FILES[@]}"; do
  CONT_CMD+=(-f "${src}")
done
"${CONT_CMD[@]}"

echo "Continual learning tick complete."
echo "Transaction label loop: $([[ -n "${REVIEW_CSV}" ]] && echo "ran" || echo "skipped")"
echo "Product continual loop: ran"
