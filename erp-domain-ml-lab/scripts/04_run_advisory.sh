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

V2_MODEL_DIR="$(latest_file "${LAB_OUTPUTS_DIR}/latest_transaction_model_dir.txt")"
COA_BUNDLE_DIR="$(latest_file "${LAB_OUTPUTS_DIR}/latest_coa_bundle_dir.txt")"
if [[ -z "${V2_MODEL_DIR}" || ! -d "${V2_MODEL_DIR}" ]]; then
  echo "Missing transaction model dir. Run scripts/03_train_models.sh first." >&2
  exit 1
fi
if [[ -z "${COA_BUNDLE_DIR}" || ! -d "${COA_BUNDLE_DIR}" ]]; then
  echo "Missing CoA bundle dir. Run scripts/03_train_models.sh first." >&2
  exit 1
fi

RUN_DIR="${LAB_OUTPUTS_DIR}/advisory_run_$(timestamp)"
TX_OUT="${RUN_DIR}/transaction_v2"
COA_OUT="${RUN_DIR}/coa_v1"
RISK_OUT="${RUN_DIR}/journal_risk"

TX_MANIFEST="$(
  run_analytics_cli run-transaction-classifier-v2 \
    --erp-snapshot "${SNAPSHOT_DIR}" \
    --model-dir "${V2_MODEL_DIR}" \
    --out "${TX_OUT}" \
    --device auto \
    --feature-cache-dir "${LAB_CACHE_DIR}" \
    --write-feature-cache
)"

COA_MANIFEST="$(
  run_analytics_cli run-coa-recommender-v1 \
    --erp-snapshot "${SNAPSHOT_DIR}" \
    --model-dir "${COA_BUNDLE_DIR}" \
    --out "${COA_OUT}" \
    --device auto \
    --feature-cache-dir "${LAB_CACHE_DIR}" \
    --write-feature-cache
)"

RISK_MANIFEST="$(
  run_analytics_cli run-journal-risk \
    --erp-snapshot "${SNAPSHOT_DIR}" \
    --out "${RISK_OUT}"
)"

echo "${RUN_DIR}" > "${LAB_OUTPUTS_DIR}/latest_advisory_run_dir.txt"
echo "${TX_OUT}" > "${LAB_OUTPUTS_DIR}/latest_advisory_transaction_dir.txt"

echo "Advisory run complete: ${RUN_DIR}"
echo "Transaction manifest: ${TX_MANIFEST}"
echo "CoA manifest: ${COA_MANIFEST}"
echo "Risk manifest: ${RISK_MANIFEST}"

