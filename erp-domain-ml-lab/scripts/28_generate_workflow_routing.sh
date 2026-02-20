#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_common.sh"

RUN_DIR="${1:-}"
if [[ -z "${RUN_DIR}" ]]; then
  RUN_DIR="$(latest_file "${LAB_OUTPUTS_DIR}/latest_tx_coa_eval_dir.txt")"
fi
if [[ -z "${RUN_DIR}" ]]; then
  RUN_DIR="$(latest_file "${LAB_OUTPUTS_DIR}/latest_advisory_run_dir.txt")"
fi
if [[ -z "${RUN_DIR}" || ! -d "${RUN_DIR}" ]]; then
  echo "Run dir not found. Pass arg1 or run tx+coa evaluation/advisory first." >&2
  exit 1
fi

SCORECARD_JSON="${2:-}"
if [[ -z "${SCORECARD_JSON}" ]]; then
  SCORECARD_JSON="${RUN_DIR}/workflow_scorecard.json"
fi
if [[ ! -f "${SCORECARD_JSON}" ]]; then
  SCORECARD_JSON="$(latest_file "${LAB_OUTPUTS_DIR}/latest_tx_coa_scorecard.json")"
fi
if [[ -z "${SCORECARD_JSON}" || ! -f "${SCORECARD_JSON}" ]]; then
  echo "Workflow scorecard not found. Pass arg2 or run scripts/15_eval_tx_coa_scorecard.sh first." >&2
  exit 1
fi

TX_JSONL="${RUN_DIR}/transaction_v2/advisory_outputs/transaction_classifications_v2.jsonl"
COA_JSONL="${RUN_DIR}/coa_v1/advisory_outputs/coa_mapping_recommendations.jsonl"
if [[ ! -f "${TX_JSONL}" ]]; then
  echo "Missing transaction output: ${TX_JSONL}" >&2
  exit 1
fi
if [[ ! -f "${COA_JSONL}" ]]; then
  echo "Missing CoA output: ${COA_JSONL}" >&2
  exit 1
fi

ROUTING_DIR="${RUN_DIR}/workflow_routing"
POLICY_JSON="${ROUTING_DIR}/workflow_routing_policy.json"
POLICY_SUMMARY_JSON="${ROUTING_DIR}/workflow_routing_policy_summary.json"
ROUTE_JSONL="${ROUTING_DIR}/workflow_routing_decisions.jsonl"
ROUTE_SUMMARY_JSON="${ROUTING_DIR}/workflow_routing_summary.json"
mkdir -p "${ROUTING_DIR}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/workflow_routing_policy.py" \
  --scorecard "${SCORECARD_JSON}" \
  --out-policy-json "${POLICY_JSON}" \
  --out-summary-json "${POLICY_SUMMARY_JSON}" \
  --min-family-examples "${ROUTING_MIN_FAMILY_EXAMPLES:-80}" \
  --target-auto-accept-rate-ppm "${ROUTING_TARGET_AUTO_ACCEPT_RATE_PPM:-850000}" \
  --min-auto-accept-accuracy-ppm "${ROUTING_MIN_AUTO_ACCEPT_ACCURACY_PPM:-980000}" \
  --preferred-threshold "${ROUTING_PREFERRED_THRESHOLD:-0.90}" \
  --conservative-threshold "${ROUTING_CONSERVATIVE_THRESHOLD:-0.95}" >/tmp/workflow_routing_policy_out.txt

"${PYTHON_BIN}" "${SCRIPT_DIR}/workflow_routing_apply.py" \
  --tx-jsonl "${TX_JSONL}" \
  --coa-jsonl "${COA_JSONL}" \
  --routing-policy-json "${POLICY_JSON}" \
  --out-jsonl "${ROUTE_JSONL}" \
  --summary-json "${ROUTE_SUMMARY_JSON}" \
  --default-threshold "${ROUTING_DEFAULT_THRESHOLD:-0.90}" >/tmp/workflow_routing_apply_out.txt

echo "${ROUTING_DIR}" > "${LAB_OUTPUTS_DIR}/latest_workflow_routing_dir.txt"
echo "${POLICY_JSON}" > "${LAB_OUTPUTS_DIR}/latest_workflow_routing_policy_json.txt"
echo "${ROUTE_JSONL}" > "${LAB_OUTPUTS_DIR}/latest_workflow_routing_decisions_jsonl.txt"
echo "${ROUTE_SUMMARY_JSON}" > "${LAB_OUTPUTS_DIR}/latest_workflow_routing_summary_json.txt"

echo "Workflow routing generation complete."
echo "Run dir: ${RUN_DIR}"
echo "Routing dir: ${ROUTING_DIR}"
echo "Policy: ${POLICY_JSON}"
echo "Routing decisions: ${ROUTE_JSONL}"
echo "Routing summary: ${ROUTE_SUMMARY_JSON}"
