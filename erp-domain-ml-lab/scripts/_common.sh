#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAB_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ -f "${LAB_DIR}/.env" ]]; then
  # Preserve explicit runtime env overrides so .env does not silently
  # force heavier defaults during ad-hoc/cron training runs.
  PRESERVE_ENV_KEYS=(
    V2_EPOCHS V2_LEARNING_RATE V2_HIDDEN_SIZE V2_FEATURES V2_BATCH_SIZE V2_DEVICE V2_LOSS_WEIGHTING
    COA_EPOCHS COA_FEATURES COA_BATCH_SIZE COA_DEVICE COA_LOSS_WEIGHTING
    PRODUCT_EPOCHS PRODUCT_SYNTH_ROWS PRODUCT_SEED PRODUCT_LR PRODUCT_L2 PRODUCT_HIDDEN_SIZE PRODUCT_FEATURES PRODUCT_BATCH_SIZE PRODUCT_DEVICE PRODUCT_HOLDOUT_RATIO PRODUCT_TOPK
    RUN_REVIEW_RISK_LOOP RUN_RECONCILIATION_LOOP TX_COA_INCLUDE_PRODUCT_MODEL
    USER_PERSONALIZATION_TX_ALPHA USER_PERSONALIZATION_COA_ALPHA USER_PERSONALIZATION_MIN_MEMORY_ROWS USER_PERSONALIZATION_MIN_FAMILY_MEMORY_ROWS USER_PERSONALIZATION_GLOBAL_ONLY_ALPHA_SCALE USER_PERSONALIZATION_MAX_TX_TOP1_CHANGE_RATE USER_PERSONALIZATION_MAX_COA_DEBIT_TOP1_CHANGE_RATE USER_PERSONALIZATION_MAX_COA_CREDIT_TOP1_CHANGE_RATE USER_PERSONALIZATION_MIN_FAMILY_EVAL_ROWS USER_PERSONALIZATION_MAX_FAMILY_TOP1_CHANGE_RATE
  )
  declare -A PRESERVE_ENV_VALUE=()
  declare -A PRESERVE_ENV_SET=()
  for key in "${PRESERVE_ENV_KEYS[@]}"; do
    if [[ -n "${!key+x}" ]]; then
      PRESERVE_ENV_SET["${key}"]="1"
      PRESERVE_ENV_VALUE["${key}"]="${!key}"
    fi
  done

  set -a
  # shellcheck disable=SC1091
  source "${LAB_DIR}/.env"
  set +a

  for key in "${PRESERVE_ENV_KEYS[@]}"; do
    if [[ -n "${PRESERVE_ENV_SET[$key]:-}" ]]; then
      export "${key}=${PRESERVE_ENV_VALUE[$key]}"
    fi
  done
fi

: "${AUDITOR_ANALYTICS_DIR:=/home/realnigga/RustroverProjects/AUDITOR/analytics}"
: "${BASE_URL:=http://localhost:8080}"
: "${COMPANY_CODE:=BBP}"
: "${PERIOD_START:=2025-01-01}"
: "${PERIOD_END:=2025-12-31}"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "${LAB_DIR}/.venv/bin/python3" ]]; then
    PYTHON_BIN="${LAB_DIR}/.venv/bin/python3"
  else
    PYTHON_BIN="python3"
  fi
fi

: "${LAB_DATA_DIR:=${LAB_DIR}/data}"
: "${LAB_SNAPSHOTS_DIR:=${LAB_DIR}/snapshots}"
: "${LAB_MODELS_DIR:=${LAB_DIR}/models}"
: "${LAB_OUTPUTS_DIR:=${LAB_DIR}/outputs}"
: "${LAB_CACHE_DIR:=${LAB_DIR}/cache}"
: "${LAB_LOCKS_DIR:=${LAB_DIR}/locks}"

mkdir -p \
  "${LAB_DATA_DIR}/training" \
  "${LAB_DATA_DIR}/labels" \
  "${LAB_SNAPSHOTS_DIR}" \
  "${LAB_MODELS_DIR}" \
  "${LAB_OUTPUTS_DIR}" \
  "${LAB_CACHE_DIR}" \
  "${LAB_LOCKS_DIR}"

timestamp() {
  date +"%Y%m%d_%H%M%S"
}

require_var() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    echo "Missing required environment variable: ${name}" >&2
    exit 1
  fi
}

run_analytics_cli() {
  (
    cd "${AUDITOR_ANALYTICS_DIR}"
    "${PYTHON_BIN}" -m ledgerstudio_analytics.cli "$@"
  )
}

run_snapshot_cli() {
  (
    cd "${AUDITOR_ANALYTICS_DIR}"
    "${PYTHON_BIN}" -m ledgerstudio_analytics.connectors.orchestrator_erp.cli "$@"
  )
}

latest_file() {
  local path="$1"
  if [[ -f "${path}" ]]; then
    cat "${path}"
  fi
}

acquire_lab_lock() {
  local lock_name="$1"
  local wait_sec="${2:-${LAB_LOCK_WAIT_SEC:-600}}"
  local lock_file="${LAB_LOCKS_DIR}/${lock_name}.lock"

  if ! command -v flock >/dev/null 2>&1; then
    echo "WARN: 'flock' not found; proceeding without lock (${lock_name})." >&2
    return 0
  fi

  # Keep FD in a deterministic var name per lock to avoid collisions.
  local fd_var="LAB_LOCK_FD_${lock_name//[^a-zA-Z0-9_]/_}"
  if [[ -n "${!fd_var:-}" ]]; then
    return 0
  fi

  # shellcheck disable=SC1083
  exec {lock_fd}> "${lock_file}"
  if ! flock -w "${wait_sec}" "${lock_fd}"; then
    echo "Could not acquire lock '${lock_name}' within ${wait_sec}s (${lock_file})." >&2
    exit 2
  fi
  printf -v "${fd_var}" "%s" "${lock_fd}"
  export "${fd_var}"
}
