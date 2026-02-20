#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

LABEL_TO_FAMILY = {
    "SALE": "sale",
    "SALE_RETURN": "sale",
    "PURCHASE": "purchase",
    "PAYMENT": "payment",
    "COGS": "cogs",
    "WRITE_OFF": "write_off",
    "INVENTORY_COUNT": "inventory_count",
    "PERIOD_LOCK": "period_lock",
    "PAYROLL": "payroll",
    "PAYROLL_JOURNAL": "payroll",
    "TAX_SETTLEMENT": "tax_settlement",
}

DOC_TO_FAMILY = {
    "INVOICE": "sale",
    "DEALER_RECEIPT": "payment",
    "RAW_MATERIAL_PURCHASE": "purchase",
    "SUPPLIER_PAYMENT": "payment",
    "SETTLEMENT_SPLIT": "settlement_split",
    "TAX_PAYMENT": "tax_settlement",
    "PAYROLL_JOURNAL": "payroll",
    "PERIOD_LOCK": "period_lock",
    "INVENTORY_ADJUSTMENT": "inventory_count",
}


@dataclass(frozen=True)
class TruthRecord:
    label: str
    family: str
    debit_label: str
    credit_label: str
    doc_type: str


def _to_decimal(value: Any) -> Decimal:
    try:
        if value is None:
            return Decimal("0")
        return abs(Decimal(str(value)))
    except Exception:
        return Decimal("0")


def _record_key(obj: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(obj.get("publicId") or ""),
        str(obj.get("id") or ""),
        str(obj.get("referenceNumber") or ""),
    )


def _family_for(label: str, doc_type: str) -> str:
    label_up = str(label or "").strip().upper()
    if label_up in LABEL_TO_FAMILY:
        return LABEL_TO_FAMILY[label_up]
    doc_up = str(doc_type or "").strip().upper()
    if doc_up in DOC_TO_FAMILY:
        return DOC_TO_FAMILY[doc_up]
    return "other"


def _top_label(lines: list[dict[str, Any]], side: str) -> str:
    totals: dict[str, Decimal] = {}
    for line in lines:
        if not isinstance(line, dict):
            continue
        account = str(line.get("accountCode") or "").strip().upper()
        if not account:
            continue
        amt = _to_decimal(line.get(side))
        if amt <= 0:
            continue
        totals[account] = totals.get(account, Decimal("0")) + amt
    if not totals:
        return ""
    return sorted(totals.items(), key=lambda t: (-t[1], t[0]))[0][0]


def _read_truth(snapshot_dir: Path) -> dict[tuple[str, str, str], TruthRecord]:
    truth_path = snapshot_dir / "normalized" / "journal_entries.jsonl"
    if not truth_path.exists():
        raise SystemExit(f"Missing snapshot journal entries: {truth_path}")

    out: dict[tuple[str, str, str], TruthRecord] = {}
    with truth_path.open("r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            obj = json.loads(raw)
            if not isinstance(obj, dict):
                continue
            key = _record_key(obj)
            label = str(obj.get("syntheticLabel") or obj.get("docType") or "UNKNOWN").strip().upper()
            doc_type = str(obj.get("docType") or "").strip().upper()
            family = _family_for(label, doc_type)
            lines = list(obj.get("lines") or [])
            debit_label = _top_label(lines, "debit")
            credit_label = _top_label(lines, "credit")
            out[key] = TruthRecord(
                label=label,
                family=family,
                debit_label=debit_label,
                credit_label=credit_label,
                doc_type=doc_type,
            )
    return out


def _safe_ratio(n: int, d: int) -> float:
    if d <= 0:
        return 0.0
    return float(n) / float(d)


def _read_tx(
    tx_jsonl: Path,
    truth: dict[tuple[str, str, str], TruthRecord],
) -> dict[str, Any]:
    total = 0
    correct = 0
    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    by_label_counts: Counter[str] = Counter()
    by_label_correct: Counter[str] = Counter()
    by_family_counts: Counter[str] = Counter()
    by_family_correct: Counter[str] = Counter()

    with tx_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            rec = json.loads(raw)
            if not isinstance(rec, dict):
                continue
            rec_obj = dict(rec.get("record") or {})
            key = _record_key(rec_obj)
            t = truth.get(key)
            if t is None:
                continue

            pred = str(rec.get("suggested_label") or "").strip().upper()
            if not pred:
                preds = list(rec.get("predictions") or [])
                if preds:
                    pred = str(preds[0].get("label") or "").strip().upper()

            total += 1
            by_label_counts[t.label] += 1
            by_family_counts[t.family] += 1
            confusion[t.label][pred] += 1
            if pred == t.label:
                correct += 1
                by_label_correct[t.label] += 1
                by_family_correct[t.family] += 1

    by_label = {
        label: {
            "examples": int(by_label_counts[label]),
            "accuracy": _safe_ratio(int(by_label_correct[label]), int(by_label_counts[label])),
        }
        for label in sorted(by_label_counts.keys())
    }
    by_family = {
        family: {
            "examples": int(by_family_counts[family]),
            "accuracy": _safe_ratio(int(by_family_correct[family]), int(by_family_counts[family])),
        }
        for family in sorted(by_family_counts.keys())
    }

    return {
        "examples": total,
        "accuracy": _safe_ratio(correct, total),
        "accuracy_ppm": int(round(_safe_ratio(correct, total) * 1_000_000.0)),
        "by_label": by_label,
        "by_family": by_family,
        "confusion": {
            label: dict(sorted(counter.items(), key=lambda t: t[0]))
            for label, counter in sorted(confusion.items(), key=lambda t: t[0])
        },
    }


def _read_coa(
    coa_jsonl: Path,
    truth: dict[tuple[str, str, str], TruthRecord],
    *,
    topk: int,
) -> dict[str, Any]:
    total = 0
    d1 = d3 = c1 = c3 = 0

    band_counts: Counter[str] = Counter()
    top1_band_counts: Counter[str] = Counter()
    by_label_counts: Counter[str] = Counter()
    by_label_d1: Counter[str] = Counter()
    by_label_d3: Counter[str] = Counter()
    by_label_c1: Counter[str] = Counter()
    by_label_c3: Counter[str] = Counter()

    by_family_counts: Counter[str] = Counter()
    by_family_d1: Counter[str] = Counter()
    by_family_d3: Counter[str] = Counter()
    by_family_c1: Counter[str] = Counter()
    by_family_c3: Counter[str] = Counter()
    by_family_debit_top1_band: dict[str, Counter[str]] = defaultdict(Counter)
    by_family_credit_top1_band: dict[str, Counter[str]] = defaultdict(Counter)

    with coa_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            rec = json.loads(raw)
            if not isinstance(rec, dict):
                continue
            rec_obj = dict(rec.get("record") or {})
            key = _record_key(rec_obj)
            t = truth.get(key)
            if t is None:
                continue
            if not t.debit_label or not t.credit_label:
                continue

            debits = [str(p.get("label") or "").strip().upper() for p in (rec.get("debit_recommendations") or [])]
            credits = [str(p.get("label") or "").strip().upper() for p in (rec.get("credit_recommendations") or [])]
            if not debits or not credits:
                continue

            total += 1
            by_label_counts[t.label] += 1
            by_family_counts[t.family] += 1

            if debits[0] == t.debit_label:
                d1 += 1
                by_label_d1[t.label] += 1
                by_family_d1[t.family] += 1
            if t.debit_label in debits[:topk]:
                d3 += 1
                by_label_d3[t.label] += 1
                by_family_d3[t.family] += 1

            if credits[0] == t.credit_label:
                c1 += 1
                by_label_c1[t.label] += 1
                by_family_c1[t.family] += 1
            if t.credit_label in credits[:topk]:
                c3 += 1
                by_label_c3[t.label] += 1
                by_family_c3[t.family] += 1

            debit_top1_band = str((rec.get("debit_recommendations") or [{}])[0].get("policy_band") or "").strip().lower()
            credit_top1_band = str((rec.get("credit_recommendations") or [{}])[0].get("policy_band") or "").strip().lower()
            if debit_top1_band:
                top1_band_counts[f"debit:{debit_top1_band}"] += 1
                by_family_debit_top1_band[t.family][debit_top1_band] += 1
            if credit_top1_band:
                top1_band_counts[f"credit:{credit_top1_band}"] += 1
                by_family_credit_top1_band[t.family][credit_top1_band] += 1

            for p in (rec.get("debit_recommendations") or []):
                band = str(p.get("policy_band") or "").strip().lower()
                if band:
                    band_counts[f"debit:{band}"] += 1
            for p in (rec.get("credit_recommendations") or []):
                band = str(p.get("policy_band") or "").strip().lower()
                if band:
                    band_counts[f"credit:{band}"] += 1

    by_label: dict[str, dict[str, float | int]] = {}
    for label in sorted(by_label_counts.keys()):
        n = int(by_label_counts[label])
        by_label[label] = {
            "examples": n,
            "debit_top1": _safe_ratio(int(by_label_d1[label]), n),
            f"debit_top{topk}": _safe_ratio(int(by_label_d3[label]), n),
            "credit_top1": _safe_ratio(int(by_label_c1[label]), n),
            f"credit_top{topk}": _safe_ratio(int(by_label_c3[label]), n),
        }

    by_family: dict[str, dict[str, float | int]] = {}
    for family in sorted(by_family_counts.keys()):
        n = int(by_family_counts[family])
        by_family[family] = {
            "examples": n,
            "debit_top1": _safe_ratio(int(by_family_d1[family]), n),
            f"debit_top{topk}": _safe_ratio(int(by_family_d3[family]), n),
            "credit_top1": _safe_ratio(int(by_family_c1[family]), n),
            f"credit_top{topk}": _safe_ratio(int(by_family_c3[family]), n),
            "policy_debit_top1_blocked_rate": _safe_ratio(int(by_family_debit_top1_band[family].get("blocked", 0)), n),
            "policy_debit_top1_discouraged_rate": _safe_ratio(int(by_family_debit_top1_band[family].get("discouraged", 0)), n),
            "policy_credit_top1_blocked_rate": _safe_ratio(int(by_family_credit_top1_band[family].get("blocked", 0)), n),
            "policy_credit_top1_discouraged_rate": _safe_ratio(int(by_family_credit_top1_band[family].get("discouraged", 0)), n),
        }

    debit_top1_counts = {
        "preferred": int(top1_band_counts.get("debit:preferred", 0)),
        "allowed": int(top1_band_counts.get("debit:allowed", 0)),
        "discouraged": int(top1_band_counts.get("debit:discouraged", 0)),
        "blocked": int(top1_band_counts.get("debit:blocked", 0)),
    }
    credit_top1_counts = {
        "preferred": int(top1_band_counts.get("credit:preferred", 0)),
        "allowed": int(top1_band_counts.get("credit:allowed", 0)),
        "discouraged": int(top1_band_counts.get("credit:discouraged", 0)),
        "blocked": int(top1_band_counts.get("credit:blocked", 0)),
    }

    return {
        "examples": total,
        "debit_top1": _safe_ratio(d1, total),
        f"debit_top{topk}": _safe_ratio(d3, total),
        "credit_top1": _safe_ratio(c1, total),
        f"credit_top{topk}": _safe_ratio(c3, total),
        "by_label": by_label,
        "by_family": by_family,
        "policy_band_counts": dict(sorted(band_counts.items(), key=lambda t: t[0])),
        "policy_top1": {
            "examples": total,
            "debit": {
                "counts": debit_top1_counts,
                "blocked_rate": _safe_ratio(int(debit_top1_counts["blocked"]), total),
                "discouraged_rate": _safe_ratio(int(debit_top1_counts["discouraged"]), total),
            },
            "credit": {
                "counts": credit_top1_counts,
                "blocked_rate": _safe_ratio(int(credit_top1_counts["blocked"]), total),
                "discouraged_rate": _safe_ratio(int(credit_top1_counts["discouraged"]), total),
            },
        },
    }


def _merge_workflow_view(tx: dict[str, Any], coa: dict[str, Any], *, topk: int) -> dict[str, Any]:
    families = set(tx.get("by_family", {}).keys()) | set(coa.get("by_family", {}).keys())
    out: dict[str, Any] = {}
    for family in sorted(families):
        tx_f = dict(tx.get("by_family", {}).get(family) or {})
        coa_f = dict(coa.get("by_family", {}).get(family) or {})
        out[family] = {
            "examples_tx": int(tx_f.get("examples", 0)),
            "tx_accuracy": float(tx_f.get("accuracy", 0.0)),
            "examples_coa": int(coa_f.get("examples", 0)),
            "coa_debit_top1": float(coa_f.get("debit_top1", 0.0)),
            f"coa_debit_top{topk}": float(coa_f.get(f"debit_top{topk}", 0.0)),
            "coa_credit_top1": float(coa_f.get("credit_top1", 0.0)),
            f"coa_credit_top{topk}": float(coa_f.get(f"credit_top{topk}", 0.0)),
            "coa_policy_debit_top1_blocked_rate": float(coa_f.get("policy_debit_top1_blocked_rate", 0.0)),
            "coa_policy_debit_top1_discouraged_rate": float(coa_f.get("policy_debit_top1_discouraged_rate", 0.0)),
            "coa_policy_credit_top1_blocked_rate": float(coa_f.get("policy_credit_top1_blocked_rate", 0.0)),
            "coa_policy_credit_top1_discouraged_rate": float(coa_f.get("policy_credit_top1_discouraged_rate", 0.0)),
        }
    return out


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build workflow-sliced scorecard from tx+coa advisory outputs")
    p.add_argument("--snapshot-dir", required=True)
    p.add_argument("--tx-jsonl", required=True)
    p.add_argument("--coa-jsonl", required=True)
    p.add_argument("--tx-eval-json", default=None)
    p.add_argument("--topk", type=int, default=3)
    p.add_argument("--out-json", required=True)
    return p


def main() -> int:
    args = _parser().parse_args()

    snapshot_dir = Path(args.snapshot_dir)
    tx_jsonl = Path(args.tx_jsonl)
    coa_jsonl = Path(args.coa_jsonl)
    out_json = Path(args.out_json)

    if not tx_jsonl.exists() or not tx_jsonl.is_file():
        raise SystemExit(f"Missing tx jsonl: {tx_jsonl}")
    if not coa_jsonl.exists() or not coa_jsonl.is_file():
        raise SystemExit(f"Missing coa jsonl: {coa_jsonl}")

    truth = _read_truth(snapshot_dir)
    tx = _read_tx(tx_jsonl, truth)
    coa = _read_coa(coa_jsonl, truth, topk=args.topk)

    tx_eval: dict[str, Any] | None = None
    if args.tx_eval_json:
        tx_eval_path = Path(args.tx_eval_json)
        if tx_eval_path.exists() and tx_eval_path.is_file():
            tx_eval = json.loads(tx_eval_path.read_text(encoding="utf-8"))

    summary = {
        "schema_version": "v0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "snapshot_dir": snapshot_dir.as_posix(),
        "inputs": {
            "tx_jsonl": tx_jsonl.as_posix(),
            "coa_jsonl": coa_jsonl.as_posix(),
            "tx_eval_json": Path(args.tx_eval_json).as_posix() if args.tx_eval_json else None,
        },
        "truth_records": len(truth),
        "topk": int(args.topk),
        "tx": tx,
        "coa": coa,
        "workflow_families": _merge_workflow_view(tx, coa, topk=args.topk),
    }
    if tx_eval is not None:
        summary["tx_eval_snapshot"] = {
            "overall": tx_eval.get("overall"),
            "threshold_stats": tx_eval.get("threshold_stats"),
            "calibration": tx_eval.get("calibration"),
        }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(out_json.as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
