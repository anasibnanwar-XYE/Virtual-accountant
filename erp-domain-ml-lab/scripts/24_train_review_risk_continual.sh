#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_common.sh"
acquire_lab_lock "continual_review_risk"

TRAINING_CSV=""

while getopts ":t:" opt; do
  case "${opt}" in
    t)
      TRAINING_CSV="${OPTARG}"
      ;;
    *)
      echo "Usage: bash scripts/24_train_review_risk_continual.sh [-t training.csv]" >&2
      exit 1
      ;;
  esac
done

if [[ -z "${TRAINING_CSV}" ]]; then
  TRAINING_CSV="$(latest_file "${LAB_OUTPUTS_DIR}/latest_training_csv.txt")"
fi
if [[ -z "${TRAINING_CSV}" || ! -f "${TRAINING_CSV}" ]]; then
  echo "Training CSV not found. Provide -t or run scripts/02_generate_training_data.sh first." >&2
  exit 1
fi

CHAMPION_BUNDLE_FILE="${LAB_OUTPUTS_DIR}/current_review_risk_champion_bundle_dir.txt"
CHAMPION_METRICS_FILE="${LAB_OUTPUTS_DIR}/current_review_risk_champion_metrics.json"
FINGERPRINT_STATE_FILE="${LAB_OUTPUTS_DIR}/current_review_risk_training_fingerprint.json"

LATEST_BUNDLE="$(latest_file "${LAB_OUTPUTS_DIR}/latest_review_risk_bundle_dir.txt")"
LATEST_METRICS="$(latest_file "${LAB_OUTPUTS_DIR}/latest_review_risk_metrics.json")"

if [[ ! -f "${CHAMPION_BUNDLE_FILE}" && -n "${LATEST_BUNDLE}" ]]; then
  echo "${LATEST_BUNDLE}" > "${CHAMPION_BUNDLE_FILE}"
fi
if [[ ! -f "${CHAMPION_METRICS_FILE}" && -n "${LATEST_METRICS}" ]]; then
  echo "${LATEST_METRICS}" > "${CHAMPION_METRICS_FILE}"
fi

CHAMPION_BUNDLE="$(latest_file "${CHAMPION_BUNDLE_FILE}")"
CHAMPION_METRICS="$(latest_file "${CHAMPION_METRICS_FILE}")"

RISK_SEED_EFFECTIVE="${RISK_SEED:-101}"
RISK_EPOCHS_EFFECTIVE="${RISK_EPOCHS:-8}"
RISK_LR_EFFECTIVE="${RISK_LR:-0.08}"
RISK_L2_EFFECTIVE="${RISK_L2:-0.0001}"
RISK_HIDDEN_SIZE_EFFECTIVE="${RISK_HIDDEN_SIZE:-64}"
RISK_FEATURES_EFFECTIVE="${RISK_FEATURES:-2048}"
RISK_BATCH_SIZE_EFFECTIVE="${RISK_BATCH_SIZE:-128}"
RISK_DEVICE_EFFECTIVE="${RISK_DEVICE:-auto}"
RISK_HOLDOUT_RATIO_EFFECTIVE="${RISK_HOLDOUT_RATIO:-0.2}"
RISK_HASH_ALGO_EFFECTIVE="${RISK_HASH_ALGO:-crc32}"
RISK_MAX_HASH_CACHE_EFFECTIVE="${RISK_MAX_HASH_CACHE:-200000}"
RISK_THRESHOLD_EFFECTIVE="${RISK_THRESHOLD:-0.55}"
RISK_OVERRIDE_MEMORY_JSONL_EFFECTIVE="${RISK_OVERRIDE_MEMORY_JSONL:-${OVERRIDE_REASON_MEMORY_JSONL:-${LAB_DATA_DIR}/labels/override_reason_feedback_memory.jsonl}}"

CURRENT_TRAINING_FINGERPRINT="$(${PYTHON_BIN} - "${TRAINING_CSV}" "${RISK_SEED_EFFECTIVE}" "${RISK_EPOCHS_EFFECTIVE}" "${RISK_LR_EFFECTIVE}" "${RISK_L2_EFFECTIVE}" "${RISK_HIDDEN_SIZE_EFFECTIVE}" "${RISK_FEATURES_EFFECTIVE}" "${RISK_BATCH_SIZE_EFFECTIVE}" "${RISK_DEVICE_EFFECTIVE}" "${RISK_HOLDOUT_RATIO_EFFECTIVE}" "${RISK_HASH_ALGO_EFFECTIVE}" "${RISK_MAX_HASH_CACHE_EFFECTIVE}" "${RISK_THRESHOLD_EFFECTIVE}" "${RISK_OVERRIDE_MEMORY_JSONL_EFFECTIVE}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path


def sha256_file(path: str) -> str | None:
    p = Path(path)
    if not p.exists() or not p.is_file():
        return None
    h = hashlib.sha256()
    with p.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def canonical(path: str) -> str:
    p = Path(path)
    try:
        return p.resolve().as_posix()
    except OSError:
        return p.as_posix()


payload = {
    "schema_version": "v0",
    "training_csv_path": canonical(sys.argv[1]),
    "training_csv_sha256": sha256_file(sys.argv[1]),
    "override_memory_path": canonical(sys.argv[14]),
    "override_memory_sha256": sha256_file(sys.argv[14]),
    "params": {
        "seed": sys.argv[2],
        "epochs": sys.argv[3],
        "learning_rate": sys.argv[4],
        "l2": sys.argv[5],
        "hidden_size": sys.argv[6],
        "n_features": sys.argv[7],
        "batch_size": sys.argv[8],
        "device": sys.argv[9],
        "holdout_ratio": sys.argv[10],
        "hash_algo": sys.argv[11],
        "max_hash_cache": sys.argv[12],
        "risk_threshold": sys.argv[13],
    },
}
raw = json.dumps(payload, sort_keys=True).encode("utf-8")
print(hashlib.sha256(raw).hexdigest())
PY
)"

PREVIOUS_TRAINING_FINGERPRINT=""
if [[ -f "${FINGERPRINT_STATE_FILE}" ]]; then
  PREVIOUS_TRAINING_FINGERPRINT="$(${PYTHON_BIN} - "${FINGERPRINT_STATE_FILE}" <<'PY'
import json
import sys
from pathlib import Path

p = Path(sys.argv[1])
if not p.exists():
    print("")
else:
    obj = json.loads(p.read_text(encoding="utf-8"))
    print(str(obj.get("training_fingerprint") or ""))
PY
)"
fi

RUN_DIR="${LAB_OUTPUTS_DIR}/continual_learning/review_risk/cycle_$(timestamp)"
mkdir -p "${RUN_DIR}"
GATE_JSON="${RUN_DIR}/gate_decision.json"
SUMMARY_JSON="${RUN_DIR}/cycle_summary.json"

if [[ "${RISK_CONTINUAL_SKIP_IF_NO_CHANGE:-true}" == "true" ]] && \
   [[ -n "${PREVIOUS_TRAINING_FINGERPRINT}" ]] && \
   [[ "${CURRENT_TRAINING_FINGERPRINT}" == "${PREVIOUS_TRAINING_FINGERPRINT}" ]] && \
   [[ -n "${CHAMPION_BUNDLE}" && -d "${CHAMPION_BUNDLE}" ]] && \
   [[ -n "${CHAMPION_METRICS}" && -f "${CHAMPION_METRICS}" ]]; then
  "${PYTHON_BIN}" - "${GATE_JSON}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = {
    "schema_version": "v0",
    "promote": False,
    "skipped": True,
    "reason": "no_data_or_param_change",
    "violations": [],
}
path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
PY

  "${PYTHON_BIN}" - "${SUMMARY_JSON}" "${TRAINING_CSV}" "${RISK_OVERRIDE_MEMORY_JSONL_EFFECTIVE}" "${CHAMPION_BUNDLE}" "${CHAMPION_METRICS}" "${GATE_JSON}" "${CURRENT_TRAINING_FINGERPRINT}" "${FINGERPRINT_STATE_FILE}" <<'PY'
import json
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
summary = {
    "schema_version": "v0",
    "status": "skipped_no_data_change",
    "training_csv": sys.argv[2],
    "override_memory_jsonl": sys.argv[3] or None,
    "previous_champion_bundle": sys.argv[4] or None,
    "previous_champion_metrics": sys.argv[5] or None,
    "challenger_bundle": sys.argv[4] or None,
    "challenger_metrics": sys.argv[5] or None,
    "gate_decision": json.loads(Path(sys.argv[6]).read_text(encoding="utf-8")),
    "training_fingerprint": sys.argv[7],
    "fingerprint_state_file": sys.argv[8],
    "training_skipped": True,
}
summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
print(summary_path.as_posix())
PY

  "${PYTHON_BIN}" - "${FINGERPRINT_STATE_FILE}" "${CURRENT_TRAINING_FINGERPRINT}" "${TRAINING_CSV}" "${RISK_OVERRIDE_MEMORY_JSONL_EFFECTIVE}" "${CHAMPION_BUNDLE}" "${CHAMPION_METRICS}" "${SUMMARY_JSON}" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

state_path = Path(sys.argv[1])
state = {
    "schema_version": "v0",
    "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    "training_fingerprint": sys.argv[2],
    "training_csv": sys.argv[3],
    "override_memory_jsonl": sys.argv[4] or None,
    "bundle": sys.argv[5] or None,
    "metrics": sys.argv[6] or None,
    "training_skipped": True,
    "cycle_summary": sys.argv[7],
}
state_path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
PY

  echo "${CHAMPION_BUNDLE}" > "${LAB_OUTPUTS_DIR}/latest_review_risk_bundle_dir.txt"
  echo "${CHAMPION_METRICS}" > "${LAB_OUTPUTS_DIR}/latest_review_risk_metrics.json"
  echo "${RUN_DIR}" > "${LAB_OUTPUTS_DIR}/latest_review_risk_continual_cycle_dir.txt"
  echo "${SUMMARY_JSON}" > "${LAB_OUTPUTS_DIR}/latest_review_risk_continual_cycle_summary.txt"

  echo "Review-risk continual learning cycle complete (skipped_no_data_change)."
  echo "Cycle dir: ${RUN_DIR}"
  echo "Summary: ${SUMMARY_JSON}"
  echo "Current champion bundle: ${CHAMPION_BUNDLE}"
  echo "Current champion metrics: ${CHAMPION_METRICS}"
  exit 0
fi

RISK_SEED="${RISK_SEED_EFFECTIVE}" \
RISK_EPOCHS="${RISK_EPOCHS_EFFECTIVE}" \
RISK_LR="${RISK_LR_EFFECTIVE}" \
RISK_L2="${RISK_L2_EFFECTIVE}" \
RISK_HIDDEN_SIZE="${RISK_HIDDEN_SIZE_EFFECTIVE}" \
RISK_FEATURES="${RISK_FEATURES_EFFECTIVE}" \
RISK_BATCH_SIZE="${RISK_BATCH_SIZE_EFFECTIVE}" \
RISK_DEVICE="${RISK_DEVICE_EFFECTIVE}" \
RISK_HOLDOUT_RATIO="${RISK_HOLDOUT_RATIO_EFFECTIVE}" \
RISK_HASH_ALGO="${RISK_HASH_ALGO_EFFECTIVE}" \
RISK_MAX_HASH_CACHE="${RISK_MAX_HASH_CACHE_EFFECTIVE}" \
RISK_THRESHOLD="${RISK_THRESHOLD_EFFECTIVE}" \
RISK_OVERRIDE_MEMORY_JSONL="${RISK_OVERRIDE_MEMORY_JSONL_EFFECTIVE}" \
bash "${SCRIPT_DIR}/23_train_review_risk_model.sh" "${TRAINING_CSV}"

CHALLENGER_BUNDLE="$(latest_file "${LAB_OUTPUTS_DIR}/latest_review_risk_bundle_dir.txt")"
CHALLENGER_METRICS="$(latest_file "${LAB_OUTPUTS_DIR}/latest_review_risk_metrics.json")"

if [[ -z "${CHALLENGER_BUNDLE}" || ! -d "${CHALLENGER_BUNDLE}" ]]; then
  echo "Challenger bundle missing after train." >&2
  exit 1
fi
if [[ -z "${CHALLENGER_METRICS}" || ! -f "${CHALLENGER_METRICS}" ]]; then
  echo "Challenger metrics missing after train." >&2
  exit 1
fi

GATE_CMD=(
  "${PYTHON_BIN}" "${SCRIPT_DIR}/review_risk_champion_gate.py"
  --challenger-metrics "${CHALLENGER_METRICS}"
  --decision-out "${GATE_JSON}"
  --max-fnr-increase "${RISK_MAX_FNR_INCREASE:-0.02}"
  --max-recall-drop "${RISK_MAX_RECALL_DROP:-0.02}"
  --min-f1-delta "${RISK_MIN_F1_DELTA:--0.01}"
  --min-precision "${RISK_MIN_PRECISION:-0.55}"
  --min-accuracy "${RISK_MIN_ACCURACY:-0.65}"
  --max-predicted-positive-rate-increase "${RISK_MAX_PRED_POS_RATE_INCREASE:-0.08}"
  --max-predicted-positive-rate-drop "${RISK_MAX_PRED_POS_RATE_DROP:-0.08}"
)
if [[ -n "${CHAMPION_METRICS}" && -f "${CHAMPION_METRICS}" ]]; then
  GATE_CMD+=(--champion-metrics "${CHAMPION_METRICS}")
fi
"${GATE_CMD[@]}" >/tmp/review_risk_champion_gate_out.txt

PROMOTE="$(${PYTHON_BIN} - "${GATE_JSON}" <<'PY'
import json
import sys
obj = json.load(open(sys.argv[1], encoding="utf-8"))
print("true" if obj.get("promote") else "false")
PY
)"

STATUS="rejected"
if [[ "${PROMOTE}" == "true" ]]; then
  STATUS="promoted"
  echo "${CHALLENGER_BUNDLE}" > "${CHAMPION_BUNDLE_FILE}"
  echo "${CHALLENGER_METRICS}" > "${CHAMPION_METRICS_FILE}"
else
  if [[ -n "${CHAMPION_BUNDLE}" && -d "${CHAMPION_BUNDLE}" ]]; then
    echo "${CHAMPION_BUNDLE}" > "${LAB_OUTPUTS_DIR}/latest_review_risk_bundle_dir.txt"
  fi
  if [[ -n "${CHAMPION_METRICS}" && -f "${CHAMPION_METRICS}" ]]; then
    echo "${CHAMPION_METRICS}" > "${LAB_OUTPUTS_DIR}/latest_review_risk_metrics.json"
  fi
  echo "${CHALLENGER_BUNDLE}" > "${LAB_OUTPUTS_DIR}/latest_review_risk_rejected_bundle_dir.txt"
  echo "${CHALLENGER_METRICS}" > "${LAB_OUTPUTS_DIR}/latest_review_risk_rejected_metrics.json"
fi

"${PYTHON_BIN}" - "${SUMMARY_JSON}" "${STATUS}" "${TRAINING_CSV}" "${RISK_OVERRIDE_MEMORY_JSONL_EFFECTIVE}" "${CHAMPION_BUNDLE}" "${CHAMPION_METRICS}" "${CHALLENGER_BUNDLE}" "${CHALLENGER_METRICS}" "${GATE_JSON}" "${CURRENT_TRAINING_FINGERPRINT}" "${FINGERPRINT_STATE_FILE}" <<'PY'
import json
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
status = sys.argv[2]
summary = {
    "schema_version": "v0",
    "status": status,
    "training_csv": sys.argv[3],
    "override_memory_jsonl": sys.argv[4] or None,
    "previous_champion_bundle": sys.argv[5] or None,
    "previous_champion_metrics": sys.argv[6] or None,
    "challenger_bundle": sys.argv[7],
    "challenger_metrics": sys.argv[8],
    "gate_decision": json.loads(Path(sys.argv[9]).read_text(encoding="utf-8")),
    "training_fingerprint": sys.argv[10],
    "fingerprint_state_file": sys.argv[11],
    "training_skipped": False,
}
summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
print(summary_path.as_posix())
PY

"${PYTHON_BIN}" - "${FINGERPRINT_STATE_FILE}" "${CURRENT_TRAINING_FINGERPRINT}" "${TRAINING_CSV}" "${RISK_OVERRIDE_MEMORY_JSONL_EFFECTIVE}" "${CHALLENGER_BUNDLE}" "${CHALLENGER_METRICS}" "${SUMMARY_JSON}" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

state_path = Path(sys.argv[1])
state = {
    "schema_version": "v0",
    "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    "training_fingerprint": sys.argv[2],
    "training_csv": sys.argv[3],
    "override_memory_jsonl": sys.argv[4] or None,
    "bundle": sys.argv[5],
    "metrics": sys.argv[6],
    "training_skipped": False,
    "cycle_summary": sys.argv[7],
}
state_path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
PY

echo "${RUN_DIR}" > "${LAB_OUTPUTS_DIR}/latest_review_risk_continual_cycle_dir.txt"
echo "${SUMMARY_JSON}" > "${LAB_OUTPUTS_DIR}/latest_review_risk_continual_cycle_summary.txt"

echo "Review-risk continual learning cycle complete (${STATUS})."
echo "Cycle dir: ${RUN_DIR}"
echo "Summary: ${SUMMARY_JSON}"
echo "Current champion bundle: $(latest_file "${CHAMPION_BUNDLE_FILE}")"
echo "Current champion metrics: $(latest_file "${CHAMPION_METRICS_FILE}")"
