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

bash "${SCRIPT_DIR}/07_generate_training_data_from_snapshot.sh" "${SNAPSHOT_DIR}"
TRAINING_CSV="$(latest_file "${LAB_OUTPUTS_DIR}/latest_training_csv.txt")"
bash "${SCRIPT_DIR}/03_train_models.sh" "${TRAINING_CSV}"
