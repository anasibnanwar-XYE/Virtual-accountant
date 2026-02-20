#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SALE_LABELS = {"SALE", "SALE_RETURN"}
PURCHASE_LABELS = {"PURCHASE"}
TAX_SETTLEMENT_LABELS = {"TAX_SETTLEMENT"}

GST_OUTPUT = "GST_OUTPUT"
GST_INPUT = "GST_INPUT"
TAX_PAYABLE = "TAX_PAYABLE"
LIQUID_ACCOUNTS = {"BANK", "CASH"}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            obj = json.loads(raw)
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True))
            f.write("\n")


def _as_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    return str(value).strip()


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _pred_label(tx_row: dict[str, Any]) -> str:
    direct = _as_str(tx_row.get("suggested_label")).upper()
    if direct:
        return direct
    preds = list(tx_row.get("predictions") or [])
    if preds and isinstance(preds[0], dict):
        return _as_str(preds[0].get("label")).upper()
    return ""


def _is_taxable(numeric: dict[str, Any]) -> bool:
    return (
        _as_float(numeric.get("gst_treatment_taxable"), 0.0) > 0.5
        or _as_float(numeric.get("has_tax_line"), 0.0) > 0.5
        or _as_float(numeric.get("tax_rate"), 0.0) > 0.0
    )


def _is_tax_settlement(label: str, numeric: dict[str, Any]) -> bool:
    return (
        label in TAX_SETTLEMENT_LABELS
        or _as_float(numeric.get("doc_type_tax_payment"), 0.0) > 0.5
        or _as_float(numeric.get("workflow_tax_settlement"), 0.0) > 0.5
    )


def _labels(recs: list[dict[str, Any]]) -> list[str]:
    out: list[str] = []
    for rec in recs:
        if not isinstance(rec, dict):
            continue
        label = _as_str(rec.get("label")).upper()
        if label:
            out.append(label)
    return out


def _score_of(rec: dict[str, Any], fallback: float) -> float:
    return _as_float(rec.get("score"), fallback)


def _ensure_label(
    recs: list[dict[str, Any]],
    *,
    label: str,
    policy_reason: str,
    insert_after_top1: bool,
    topk_limit: int,
) -> bool:
    labels = _labels(recs)
    if label in labels:
        return False

    top_score = _score_of(recs[0], 0.5) if recs else 0.5
    tail_score = _score_of(recs[-1], 0.02) if recs else 0.02
    blend = max(0.02, min(top_score * 0.94, max(tail_score, 0.02) + 0.02))
    new_rec = {
        "label": label,
        "score": f"{blend:.6f}",
        "raw_score": f"{max(blend - 0.03, 0.0):.6f}",
        "margin": "0.000000",
        "policy_band": "preferred",
        "policy_delta": "0.030000",
        "policy_reason": policy_reason,
    }
    insert_pos = 1 if insert_after_top1 and recs else 0
    recs.insert(insert_pos, new_rec)
    if topk_limit > 0 and len(recs) > topk_limit:
        del recs[topk_limit:]
    return True


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Apply tax-aware overlay to CoA recommendations")
    p.add_argument("--tx-jsonl", required=True)
    p.add_argument("--coa-jsonl", required=True)
    p.add_argument("--out-jsonl", required=True)
    p.add_argument("--summary-json", required=True)
    p.add_argument("--topk-limit", type=int, default=5)
    return p


def main() -> int:
    args = _parser().parse_args()
    tx_path = Path(args.tx_jsonl)
    coa_path = Path(args.coa_jsonl)
    out_path = Path(args.out_jsonl)
    summary_path = Path(args.summary_json)

    if not tx_path.exists() or not tx_path.is_file():
        raise SystemExit(f"Missing tx jsonl: {tx_path}")
    if not coa_path.exists() or not coa_path.is_file():
        raise SystemExit(f"Missing coa jsonl: {coa_path}")

    tx_rows = _read_jsonl(tx_path)
    coa_rows = _read_jsonl(coa_path)
    tx_by_id = {_as_str(r.get("example_id")): r for r in tx_rows if _as_str(r.get("example_id"))}

    updated: list[dict[str, Any]] = []
    counts = Counter()
    per_rule = Counter()

    for coa in coa_rows:
        rec = dict(coa)
        ex_id = _as_str(rec.get("example_id"))
        tx = tx_by_id.get(ex_id)
        if tx is None:
            updated.append(rec)
            counts["missing_tx"] += 1
            continue

        label = _pred_label(tx)
        numeric = dict(tx.get("numeric") or {})
        taxable = _is_taxable(numeric)
        settlement = _is_tax_settlement(label, numeric)

        debits = [dict(x) for x in list(rec.get("debit_recommendations") or []) if isinstance(x, dict)]
        credits = [dict(x) for x in list(rec.get("credit_recommendations") or []) if isinstance(x, dict)]

        changed = False
        if label in SALE_LABELS and taxable:
            if _ensure_label(
                credits,
                label=GST_OUTPUT,
                policy_reason="tax_overlay_sale_output",
                insert_after_top1=True,
                topk_limit=int(args.topk_limit),
            ):
                changed = True
                per_rule["sale_output_tax_added"] += 1

        if label in PURCHASE_LABELS and taxable:
            if _ensure_label(
                debits,
                label=GST_INPUT,
                policy_reason="tax_overlay_purchase_input",
                insert_after_top1=True,
                topk_limit=int(args.topk_limit),
            ):
                changed = True
                per_rule["purchase_input_tax_added"] += 1

        if settlement:
            if _ensure_label(
                debits,
                label=TAX_PAYABLE,
                policy_reason="tax_overlay_settlement_liability",
                insert_after_top1=True,
                topk_limit=int(args.topk_limit),
            ):
                changed = True
                per_rule["settlement_liability_added"] += 1

            debit_labels = set(_labels(debits))
            credit_labels = set(_labels(credits))
            if not (debit_labels & LIQUID_ACCOUNTS or credit_labels & LIQUID_ACCOUNTS):
                if _ensure_label(
                    credits,
                    label="BANK",
                    policy_reason="tax_overlay_settlement_liquid",
                    insert_after_top1=True,
                    topk_limit=int(args.topk_limit),
                ):
                    changed = True
                    per_rule["settlement_liquid_added"] += 1

        if changed:
            counts["rows_modified"] += 1
            policy_summary = dict(rec.get("policy_summary") or {})
            rec["policy_summary"] = {
                **policy_summary,
                "tax_overlay_applied": True,
            }
            rec["debit_recommendations"] = debits
            rec["credit_recommendations"] = credits
        else:
            counts["rows_unmodified"] += 1

        updated.append(rec)

    _write_jsonl(out_path, updated)
    summary = {
        "schema_version": "v0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "tx_jsonl": tx_path.as_posix(),
            "coa_jsonl": coa_path.as_posix(),
        },
        "outputs": {
            "overlay_coa_jsonl": out_path.as_posix(),
        },
        "counts": {
            "rows_total": len(updated),
            "rows_modified": int(counts.get("rows_modified", 0)),
            "rows_unmodified": int(counts.get("rows_unmodified", 0)),
            "rows_missing_tx": int(counts.get("missing_tx", 0)),
        },
        "rule_counts": dict(sorted(per_rule.items(), key=lambda t: t[0])),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(summary_path.as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
