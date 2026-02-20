#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_common.sh"

SNAPSHOT_DIR="${1:-}"
if [[ -z "${SNAPSHOT_DIR}" ]]; then
  SNAPSHOT_DIR="$(latest_file "${LAB_OUTPUTS_DIR}/latest_snapshot_dir.txt")"
fi
if [[ -n "${SNAPSHOT_DIR}" ]]; then
  SNAPSHOT_DIR="$(cd "$(dirname "${SNAPSHOT_DIR}")" && pwd)/$(basename "${SNAPSHOT_DIR}")"
fi
if [[ -z "${SNAPSHOT_DIR}" || ! -d "${SNAPSHOT_DIR}" ]]; then
  echo "Snapshot dir not found. Pass arg1 or set latest snapshot." >&2
  exit 1
fi

TX_MODEL_DIR="${2:-}"
if [[ -z "${TX_MODEL_DIR}" ]]; then
  TX_MODEL_DIR="$(latest_file "${LAB_OUTPUTS_DIR}/latest_transaction_model_dir.txt")"
fi
if [[ -n "${TX_MODEL_DIR}" ]]; then
  TX_MODEL_DIR="$(cd "$(dirname "${TX_MODEL_DIR}")" && pwd)/$(basename "${TX_MODEL_DIR}")"
fi
if [[ -z "${TX_MODEL_DIR}" || ! -d "${TX_MODEL_DIR}" ]]; then
  echo "Transaction model dir not found. Pass arg2 or train first." >&2
  exit 1
fi

COA_MODEL_DIR="${3:-}"
if [[ -z "${COA_MODEL_DIR}" ]]; then
  COA_MODEL_DIR="$(latest_file "${LAB_OUTPUTS_DIR}/latest_coa_bundle_dir.txt")"
fi
if [[ -n "${COA_MODEL_DIR}" ]]; then
  COA_MODEL_DIR="$(cd "$(dirname "${COA_MODEL_DIR}")" && pwd)/$(basename "${COA_MODEL_DIR}")"
fi
if [[ -z "${COA_MODEL_DIR}" || ! -d "${COA_MODEL_DIR}" ]]; then
  echo "CoA model dir not found. Pass arg3 or train first." >&2
  exit 1
fi

TOPK="${WF_TOPK:-3}"
TX_THRESHOLD="${WF_TX_THRESHOLD:-0.90}"
RUN_DIR="${LAB_OUTPUTS_DIR}/tx_coa_eval_$(timestamp)"
TX_OUT="${RUN_DIR}/transaction_v2"
COA_OUT="${RUN_DIR}/coa_v1"
SCORECARD_JSON="${RUN_DIR}/workflow_scorecard.json"
mkdir -p "${RUN_DIR}"

run_analytics_cli run-transaction-classifier-v2 \
  --erp-snapshot "${SNAPSHOT_DIR}" \
  --model-dir "${TX_MODEL_DIR}" \
  --out "${TX_OUT}" \
  --topk "${TOPK}" \
  --confidence-margin-threshold "${TX_THRESHOLD}" \
  --always-review-labels PERIOD_LOCK \
  --eval-with-snapshot-labels \
  --eval-thresholds 0.95,0.90,0.80 \
  --device auto \
  --feature-cache-dir "${LAB_CACHE_DIR}" \
  --write-feature-cache >/tmp/tx_coa_eval_tx_manifest.txt

run_analytics_cli run-coa-recommender-v1 \
  --erp-snapshot "${SNAPSHOT_DIR}" \
  --model-dir "${COA_MODEL_DIR}" \
  --out "${COA_OUT}" \
  --topk "${TOPK}" \
  --device auto \
  --feature-cache-dir "${LAB_CACHE_DIR}" \
  --write-feature-cache >/tmp/tx_coa_eval_coa_manifest.txt

"${PYTHON_BIN}" "${SCRIPT_DIR}/workflow_scorecard.py" \
  --snapshot-dir "${SNAPSHOT_DIR}" \
  --tx-jsonl "${TX_OUT}/advisory_outputs/transaction_classifications_v2.jsonl" \
  --coa-jsonl "${COA_OUT}/advisory_outputs/coa_mapping_recommendations.jsonl" \
  --tx-eval-json "${TX_OUT}/advisory_outputs/eval_snapshot_v2.json" \
  --topk "${TOPK}" \
  --out-json "${SCORECARD_JSON}" >/tmp/workflow_scorecard_out.txt

if [[ ! -f "${SCORECARD_JSON}" ]]; then
  echo "Scorecard JSON was not created." >&2
  exit 1
fi

bash "${SCRIPT_DIR}/28_generate_workflow_routing.sh" "${RUN_DIR}" "${SCORECARD_JSON}" >/tmp/tx_coa_eval_routing_out.txt

echo "${RUN_DIR}" > "${LAB_OUTPUTS_DIR}/latest_tx_coa_eval_dir.txt"
echo "${SCORECARD_JSON}" > "${LAB_OUTPUTS_DIR}/latest_tx_coa_scorecard.json"

echo "TX+CoA evaluation complete."
echo "Run dir: ${RUN_DIR}"
echo "Scorecard: ${SCORECARD_JSON}"
