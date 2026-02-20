#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_common.sh"

MODE="${1:-synthetic}" # synthetic | live
SNAPSHOT_NAME="${2:-erp_snapshot_$(timestamp)}"
SNAPSHOT_DIR="${LAB_SNAPSHOTS_DIR}/${SNAPSHOT_NAME}"

case "${MODE}" in
  synthetic)
    SNAPSHOT_SEED="${SNAPSHOT_SEED:-7}"
    SNAPSHOT_JOURNAL_ENTRIES="${SNAPSHOT_JOURNAL_ENTRIES:-2500}"
    run_snapshot_cli synthetic \
      --company-code "${COMPANY_CODE}" \
      --period-start "${PERIOD_START}" \
      --period-end "${PERIOD_END}" \
      --seed "${SNAPSHOT_SEED}" \
      --journal-entries "${SNAPSHOT_JOURNAL_ENTRIES}" \
      --out "${SNAPSHOT_DIR}" >/tmp/erp_ml_snapshot_manifest.txt
    ;;
  live)
    require_var BASE_URL
    require_var COMPANY_CODE
    require_var EMAIL
    require_var PASSWORD_ENV_VAR
    if [[ -z "${!PASSWORD_ENV_VAR:-}" ]]; then
      echo "Env var ${PASSWORD_ENV_VAR} is not set." >&2
      exit 1
    fi
    run_snapshot_cli snapshot \
      --base-url "${BASE_URL}" \
      --company-code "${COMPANY_CODE}" \
      --email "${EMAIL}" \
      --password-env "${PASSWORD_ENV_VAR}" \
      --period-start "${PERIOD_START}" \
      --period-end "${PERIOD_END}" \
      --out "${SNAPSHOT_DIR}" >/tmp/erp_ml_snapshot_manifest.txt
    ;;
  *)
    echo "Invalid mode: ${MODE}. Use 'synthetic' or 'live'." >&2
    exit 1
    ;;
esac

echo "${SNAPSHOT_DIR}" > "${LAB_OUTPUTS_DIR}/latest_snapshot_dir.txt"
echo "Snapshot ready: ${SNAPSHOT_DIR}"
echo "Manifest: $(cat /tmp/erp_ml_snapshot_manifest.txt)"

