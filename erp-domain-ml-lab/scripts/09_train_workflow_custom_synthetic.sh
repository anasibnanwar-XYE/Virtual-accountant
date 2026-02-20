#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_common.sh"

ROWS="${1:-30000}"
TRAIN_SEED="${2:-17}"
HOLDOUT_SEED="${3:-77}"
HOLDOUT_JOURNALS="${4:-4000}"

WF_V2_EPOCHS="${WF_V2_EPOCHS:-6}"
WF_COA_EPOCHS="${WF_COA_EPOCHS:-6}"
WF_TOPK="${WF_TOPK:-3}"
WF_TX_THRESHOLD="${WF_TX_THRESHOLD:-0.90}"
WF_PROFILE="${WF_PROFILE:-staging_m18}"
ERP_V2_FLOW_REBALANCE="${ERP_V2_FLOW_REBALANCE:-true}"
ERP_V2_FLOW_TARGETS_JSON="${ERP_V2_FLOW_TARGETS_JSON:-${LAB_DIR}/configs/erp_v2_flow_targets.json}"

TRAINING_CSV="${LAB_DATA_DIR}/training/workflow_synth_train_${ROWS}_${TRAIN_SEED}_$(timestamp).csv"
COA_TRAINING_CSV="${LAB_DATA_DIR}/training/workflow_synth_coa_${ROWS}_${TRAIN_SEED}_$(timestamp).csv"
HOLDOUT_DIR="${LAB_SNAPSHOTS_DIR}/workflow_holdout_${HOLDOUT_SEED}_$(timestamp)"
RUN_DIR="${LAB_OUTPUTS_DIR}/workflow_custom_run_$(timestamp)"
TX_OUT="${RUN_DIR}/transaction_v2"
COA_OUT="${RUN_DIR}/coa_v1"
RISK_OUT="${RUN_DIR}/journal_risk"
SUMMARY_JSON="${RUN_DIR}/workflow_summary.json"

run_analytics_cli generate-synthetic-training-csv \
  --out-csv "${TRAINING_CSV}" \
  --rows "${ROWS}" \
  --seed "${TRAIN_SEED}" \
  --period-start "${PERIOD_START}" \
  --period-end "${PERIOD_END}" \
  --workflow-profile "${WF_PROFILE}"

FLOW_REBALANCE_REPORT=""
if [[ "${ERP_V2_FLOW_REBALANCE}" == "true" && -f "${ERP_V2_FLOW_TARGETS_JSON}" ]]; then
  REBALANCED_TRAINING_CSV="${LAB_DATA_DIR}/training/workflow_synth_train_rebalanced_${ROWS}_${TRAIN_SEED}_$(timestamp).csv"
  FLOW_REBALANCE_REPORT="${RUN_DIR}/flow_rebalance_report.json"
  mkdir -p "${RUN_DIR}"
  "${PYTHON_BIN}" "${SCRIPT_DIR}/29_rebalance_training_by_erp_flow.py" \
    --input-csv "${TRAINING_CSV}" \
    --output-csv "${REBALANCED_TRAINING_CSV}" \
    --profile-json "${ERP_V2_FLOW_TARGETS_JSON}" \
    --seed "${TRAIN_SEED}" \
    --report-out "${FLOW_REBALANCE_REPORT}" >/tmp/workflow_rebalance_out.txt
  TRAINING_CSV="${REBALANCED_TRAINING_CSV}"
  echo "${FLOW_REBALANCE_REPORT}" > "${LAB_OUTPUTS_DIR}/latest_flow_rebalance_report_json.txt"
fi

"${PYTHON_BIN}" - "${TRAINING_CSV}" "${COA_TRAINING_CSV}" <<'PY'
import csv
import sys

src, dst = sys.argv[1], sys.argv[2]
with open(src, newline="", encoding="utf-8") as f_in:
    reader = csv.DictReader(f_in)
    if reader.fieldnames is None:
        raise SystemExit("CSV has no header")
    kept = 0
    with open(dst, "w", newline="", encoding="utf-8") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=reader.fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in reader:
            if (row.get("journal_lines") or "").strip():
                writer.writerow(row)
                kept += 1
if kept == 0:
    raise SystemExit("No rows with journal_lines for CoA training")
print(kept)
PY

V2_MODEL_DIR="$(
  run_analytics_cli train-transaction-classifier-v2 \
    --dataset-csv "${TRAINING_CSV}" \
    --out "${LAB_MODELS_DIR}" \
    --epochs "${WF_V2_EPOCHS}" \
    --learning-rate "${V2_LEARNING_RATE:-0.1}" \
    --hidden-size "${V2_HIDDEN_SIZE:-64}" \
    --n-features "${V2_FEATURES:-2048}" \
    --batch-size "${V2_BATCH_SIZE:-128}" \
    --device "${V2_DEVICE:-auto}" \
    --feature-cache-dir "${LAB_CACHE_DIR}" \
    --write-feature-cache \
    --loss-weighting "${V2_LOSS_WEIGHTING:-class_balanced_materiality}"
)"

COA_MODEL_DIR="$(
  run_analytics_cli train-coa-recommender-v1 \
    --dataset-csv "${COA_TRAINING_CSV}" \
    --out "${LAB_MODELS_DIR}" \
    --epochs "${WF_COA_EPOCHS}" \
    --n-features "${COA_FEATURES:-2048}" \
    --batch-size "${COA_BATCH_SIZE:-128}" \
    --device "${COA_DEVICE:-auto}" \
    --feature-cache-dir "${LAB_CACHE_DIR}" \
    --write-feature-cache \
    --loss-weighting "${COA_LOSS_WEIGHTING:-class_balanced_materiality}"
)"

PRODUCT_MODEL_ENABLED="${PRODUCT_MODEL_ENABLED:-true}"
PRODUCT_BUNDLE_DIR=""
PRODUCT_METRICS_JSON=""
if [[ "${PRODUCT_MODEL_ENABLED}" == "true" ]]; then
  bash "${SCRIPT_DIR}/10_train_product_account_recommender.sh" "${TRAINING_CSV}"
  PRODUCT_BUNDLE_DIR="$(latest_file "${LAB_OUTPUTS_DIR}/latest_product_account_bundle_dir.txt")"
  PRODUCT_METRICS_JSON="$(latest_file "${LAB_OUTPUTS_DIR}/latest_product_account_metrics.json")"
fi

run_snapshot_cli synthetic \
  --company-code "${COMPANY_CODE}" \
  --period-start "${PERIOD_START}" \
  --period-end "${PERIOD_END}" \
  --seed "${HOLDOUT_SEED}" \
  --journal-entries "${HOLDOUT_JOURNALS}" \
  --out "${HOLDOUT_DIR}" >/tmp/workflow_custom_holdout_manifest.txt

run_analytics_cli run-transaction-classifier-v2 \
  --erp-snapshot "${HOLDOUT_DIR}" \
  --model-dir "${V2_MODEL_DIR}" \
  --out "${TX_OUT}" \
  --topk "${WF_TOPK}" \
  --confidence-margin-threshold "${WF_TX_THRESHOLD}" \
  --always-review-labels PERIOD_LOCK \
  --eval-with-snapshot-labels \
  --eval-thresholds 0.95,0.90,0.80 \
  --device auto \
  --feature-cache-dir "${LAB_CACHE_DIR}" \
  --write-feature-cache >/tmp/workflow_custom_tx_manifest.txt

run_analytics_cli run-coa-recommender-v1 \
  --erp-snapshot "${HOLDOUT_DIR}" \
  --model-dir "${COA_MODEL_DIR}" \
  --out "${COA_OUT}" \
  --topk "${WF_TOPK}" \
  --device auto \
  --feature-cache-dir "${LAB_CACHE_DIR}" \
  --write-feature-cache >/tmp/workflow_custom_coa_manifest.txt

run_analytics_cli run-journal-risk \
  --erp-snapshot "${HOLDOUT_DIR}" \
  --out "${RISK_OUT}" >/tmp/workflow_custom_risk_manifest.txt

"${PYTHON_BIN}" - "${HOLDOUT_DIR}" "${TX_OUT}" "${COA_OUT}" "${SUMMARY_JSON}" "${PRODUCT_BUNDLE_DIR}" "${PRODUCT_METRICS_JSON}" <<'PY'
import json
import sys
from decimal import Decimal
from pathlib import Path

holdout_dir = Path(sys.argv[1])
tx_out = Path(sys.argv[2])
coa_out = Path(sys.argv[3])
summary_path = Path(sys.argv[4])
product_bundle = Path(sys.argv[5]) if len(sys.argv) > 5 and sys.argv[5] else None
product_metrics_path = Path(sys.argv[6]) if len(sys.argv) > 6 and sys.argv[6] else None

eval_obj = json.loads((tx_out / "advisory_outputs" / "eval_snapshot_v2.json").read_text(encoding="utf-8"))
tx_manifest = json.loads((tx_out / "advisory_manifest.json").read_text(encoding="utf-8"))

journal_truth = {}
for line in (holdout_dir / "normalized" / "journal_entries.jsonl").open(encoding="utf-8"):
    entry = json.loads(line)
    key = (
        str(entry.get("publicId")),
        str(entry.get("id")),
        str(entry.get("referenceNumber")),
    )
    debit = {}
    credit = {}
    for ln in entry.get("lines") or []:
        if not isinstance(ln, dict):
            continue
        account = str(ln.get("accountCode") or "").strip()
        if not account:
            continue
        d = ln.get("debit")
        c = ln.get("credit")
        if d is not None:
            amt = abs(Decimal(str(d)))
            debit[account] = debit.get(account, Decimal("0")) + amt
        if c is not None:
            amt = abs(Decimal(str(c)))
            credit[account] = credit.get(account, Decimal("0")) + amt
    if debit and credit:
        debit_lbl = sorted(debit.items(), key=lambda t: (-t[1], t[0]))[0][0]
        credit_lbl = sorted(credit.items(), key=lambda t: (-t[1], t[0]))[0][0]
        journal_truth[key] = (debit_lbl, credit_lbl)

n = 0
d1 = d3 = c1 = c3 = 0
for line in (coa_out / "advisory_outputs" / "coa_mapping_recommendations.jsonl").open(encoding="utf-8"):
    rec = json.loads(line)
    key = (
        str(rec.get("record", {}).get("publicId")),
        str(rec.get("record", {}).get("id")),
        str(rec.get("record", {}).get("referenceNumber")),
    )
    truth = journal_truth.get(key)
    if truth is None:
        continue
    td, tc = truth
    d_preds = [p.get("label") for p in rec.get("debit_recommendations") or []]
    c_preds = [p.get("label") for p in rec.get("credit_recommendations") or []]
    if d_preds:
        if d_preds[0] == td:
            d1 += 1
        if td in d_preds[:3]:
            d3 += 1
    if c_preds:
        if c_preds[0] == tc:
            c1 += 1
        if tc in c_preds[:3]:
            c3 += 1
    n += 1

summary = {
    "schema_version": "v0",
    "holdout_classified": int(tx_manifest.get("counts", {}).get("classified", 0)),
    "transaction_eval": {
        "overall_accuracy_ppm": eval_obj.get("overall", {}).get("accuracy_ppm"),
        "overall_examples": eval_obj.get("overall", {}).get("examples"),
        "threshold_stats": eval_obj.get("threshold_stats", {}),
    },
    "coa_eval": {
        "examples": n,
        "debit_top1": (d1 / n) if n else 0.0,
        "debit_top3": (d3 / n) if n else 0.0,
        "credit_top1": (c1 / n) if n else 0.0,
        "credit_top3": (c3 / n) if n else 0.0,
    },
}
if product_bundle is not None and product_bundle.exists():
    summary["product_account_bundle"] = product_bundle.as_posix()
if product_metrics_path is not None and product_metrics_path.exists():
    try:
        summary["product_account_metrics"] = json.loads(product_metrics_path.read_text(encoding="utf-8"))
    except Exception:
        summary["product_account_metrics"] = {"error": "unable_to_parse_metrics"}
summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
print(summary_path.as_posix())
PY

echo "${TRAINING_CSV}" > "${LAB_OUTPUTS_DIR}/latest_training_csv.txt"
echo "${HOLDOUT_DIR}" > "${LAB_OUTPUTS_DIR}/latest_snapshot_dir.txt"
echo "${V2_MODEL_DIR}" > "${LAB_OUTPUTS_DIR}/latest_transaction_model_dir.txt"
echo "${COA_MODEL_DIR}" > "${LAB_OUTPUTS_DIR}/latest_coa_bundle_dir.txt"
echo "${RUN_DIR}" > "${LAB_OUTPUTS_DIR}/latest_advisory_run_dir.txt"
echo "${TX_OUT}" > "${LAB_OUTPUTS_DIR}/latest_advisory_transaction_dir.txt"

echo "Workflow-custom run complete."
echo "Training CSV: ${TRAINING_CSV}"
if [[ -n "${FLOW_REBALANCE_REPORT}" ]]; then
  echo "Flow rebalance report: ${FLOW_REBALANCE_REPORT}"
fi
echo "Holdout snapshot: ${HOLDOUT_DIR}"
echo "Transaction model: ${V2_MODEL_DIR}"
echo "CoA model: ${COA_MODEL_DIR}"
if [[ -n "${PRODUCT_BUNDLE_DIR}" ]]; then
  echo "Product-account bundle: ${PRODUCT_BUNDLE_DIR}"
fi
echo "Run dir: ${RUN_DIR}"
echo "Summary JSON: ${SUMMARY_JSON}"
