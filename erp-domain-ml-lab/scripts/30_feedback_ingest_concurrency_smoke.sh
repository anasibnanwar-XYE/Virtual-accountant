#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_common.sh"

WORKERS="${1:-8}"
if ! [[ "${WORKERS}" =~ ^[0-9]+$ ]] || [[ "${WORKERS}" -lt 1 ]]; then
  echo "Usage: bash scripts/30_feedback_ingest_concurrency_smoke.sh [workers>=1]" >&2
  exit 1
fi

RUN_DIR="${LAB_OUTPUTS_DIR}/feedback_concurrency_smoke_$(timestamp)"
mkdir -p "${RUN_DIR}"

INPUT_CSV="${RUN_DIR}/sample_feedback.csv"
OUT_JSONL="${RUN_DIR}/memory.jsonl"
SUMMARY_JSON="${RUN_DIR}/summary.json"

cat > "${INPUT_CSV}" <<'CSV'
sku,product_name,product_kind,category,uom,gst_rate,base_price,avg_cost,revenue_account_code,cogs_account_code,inventory_account_code,tax_account_code,discount_account_code
FG-LOCK-001,Decorative Premium,FINISHED_GOOD,DECORATIVE_INTERIOR,L,18,1450,990,SALES,COGS,INVENTORY,GST_OUTPUT,DISCOUNT_ALLOWED
RM-LOCK-002,Industrial Solvent A,RAW_MATERIAL,SOLVENTS,KG,18,420,310,SALES,FREIGHT_IN,INVENTORY,GST_INPUT,DISCOUNT_RECEIVED
CSV

pids=()
for i in $(seq 1 "${WORKERS}"); do
  report="${RUN_DIR}/worker_${i}.json"
  (
    "${PYTHON_BIN}" "${SCRIPT_DIR}/product_feedback_ingest.py" \
      --input-file "${INPUT_CSV}" \
      --out-jsonl "${OUT_JSONL}" \
      --report-out "${report}" >/dev/null
  ) &
  pids+=("$!")
done

for pid in "${pids[@]}"; do
  wait "${pid}"
done

"${PYTHON_BIN}" - "${OUT_JSONL}" "${SUMMARY_JSON}" "${WORKERS}" <<'PY'
import json
import sys
from pathlib import Path

out_path = Path(sys.argv[1])
summary_path = Path(sys.argv[2])
workers = int(sys.argv[3])

rows = []
if out_path.exists():
    with out_path.open("r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            rows.append(json.loads(raw))

keys = set()
for r in rows:
    keys.add(
        (
            str(r.get("sku") or ""),
            str(r.get("revenue_account_code") or ""),
            str(r.get("cogs_account_code") or ""),
            str(r.get("inventory_account_code") or ""),
            str(r.get("tax_account_code") or ""),
            str(r.get("discount_account_code") or ""),
        )
    )

summary = {
    "schema_version": "v0",
    "workers": workers,
    "rows_written": len(rows),
    "unique_keys": len(keys),
    "expected_unique_keys": 2,
    "status": "ok" if len(keys) == 2 else "failed",
}
summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
print(json.dumps(summary, sort_keys=True))
if summary["status"] != "ok":
    raise SystemExit(2)
PY

echo "Feedback concurrency smoke complete."
echo "Run dir: ${RUN_DIR}"
echo "Summary: ${SUMMARY_JSON}"
