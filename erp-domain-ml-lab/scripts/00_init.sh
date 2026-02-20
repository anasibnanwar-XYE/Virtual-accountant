#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_common.sh"

mkdir -p \
  "${LAB_DATA_DIR}/training" \
  "${LAB_DATA_DIR}/labels" \
  "${LAB_SNAPSHOTS_DIR}" \
  "${LAB_MODELS_DIR}" \
  "${LAB_OUTPUTS_DIR}" \
  "${LAB_CACHE_DIR}"

if [[ ! -f "${LAB_DIR}/.env" ]]; then
  cp "${LAB_DIR}/.env.example" "${LAB_DIR}/.env"
fi

if [[ ! -d "${LAB_DIR}/.venv" ]]; then
  python3 -m venv "${LAB_DIR}/.venv"
fi
"${LAB_DIR}/.venv/bin/python3" -m pip install --upgrade pip >/dev/null
"${LAB_DIR}/.venv/bin/python3" -m pip install -e "${AUDITOR_ANALYTICS_DIR}" >/dev/null

echo "Initialized ML lab at: ${LAB_DIR}"
echo "Edit config file: ${LAB_DIR}/.env"
echo "Python env ready: ${LAB_DIR}/.venv"
