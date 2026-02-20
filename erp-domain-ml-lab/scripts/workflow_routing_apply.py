#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
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


def _as_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    return str(value).strip()


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    return bool(value)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or not path.is_file():
        raise SystemExit(f"Input not found: {path}")
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


def _top_label_and_score(tx: dict[str, Any]) -> tuple[str, float]:
    suggested = _as_str(tx.get("suggested_label")).upper()
    predictions = [p for p in list(tx.get("predictions") or []) if isinstance(p, dict)]
    if suggested:
        for p in predictions:
            if _as_str(p.get("label")).upper() == suggested:
                return suggested, _as_float(p.get("score"), 0.0)
    if predictions:
        first = predictions[0]
        return _as_str(first.get("label")).upper(), _as_float(first.get("score"), 0.0)
    return "", 0.0


def _calibrated_score(tx: dict[str, Any], fallback: float) -> float:
    conf = dict(tx.get("confidence") or {})
    score = _as_float(conf.get("calibrated_score"), fallback)
    if score <= 0:
        score = _as_float(conf.get("score"), fallback)
    return score


def _policy_top1_bands(coa: dict[str, Any] | None) -> tuple[str, str]:
    if not isinstance(coa, dict):
        return ("unknown", "unknown")
    debits = [x for x in list(coa.get("debit_recommendations") or []) if isinstance(x, dict)]
    credits = [x for x in list(coa.get("credit_recommendations") or []) if isinstance(x, dict)]
    debit_band = _as_str((debits[0] if debits else {}).get("policy_band"), "unknown").lower()
    credit_band = _as_str((credits[0] if credits else {}).get("policy_band"), "unknown").lower()
    return (debit_band, credit_band)


def _route_action(
    *,
    label: str,
    calibrated_score: float,
    threshold: float,
    always_review_labels: set[str],
    debit_band: str,
    credit_band: str,
) -> tuple[str, str, str]:
    label_up = label.upper()
    if label_up in always_review_labels:
        return ("manual_review_required", "P1", "always_review_label")
    if debit_band == "blocked" or credit_band == "blocked":
        return ("reject_or_override_required", "P1", "policy_blocked")
    if debit_band == "discouraged" or credit_band == "discouraged":
        return ("manual_review_required", "P2", "policy_discouraged")
    if calibrated_score < threshold:
        return ("manual_review_required", "P2", "below_family_threshold")
    return ("quick_confirm", "P3", "threshold_and_policy_ok")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Apply workflow routing policy to tx+coa outputs.")
    p.add_argument("--tx-jsonl", required=True)
    p.add_argument("--coa-jsonl", required=True)
    p.add_argument("--routing-policy-json", required=True)
    p.add_argument("--out-jsonl", required=True)
    p.add_argument("--summary-json", required=True)
    p.add_argument("--default-threshold", type=float, default=0.90)
    return p


def main() -> int:
    args = _build_parser().parse_args()

    tx_rows = _read_jsonl(Path(args.tx_jsonl))
    coa_rows = _read_jsonl(Path(args.coa_jsonl))

    policy_path = Path(args.routing_policy_json)
    if not policy_path.exists() or not policy_path.is_file():
        raise SystemExit(f"Routing policy not found: {policy_path}")
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    family_policy = dict(policy.get("families") or {})
    always_review_labels = {
        str(x).upper()
        for x in list(dict(policy.get("global") or {}).get("always_review_labels") or [])
        if str(x).strip()
    }

    coa_by_example = {
        _as_str(r.get("example_id")): r
        for r in coa_rows
        if _as_str(r.get("example_id"))
    }

    actions: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    by_family_action: dict[str, Counter[str]] = defaultdict(Counter)
    out_rows: list[dict[str, Any]] = []

    for tx in tx_rows:
        example_id = _as_str(tx.get("example_id"))
        label, score_raw = _top_label_and_score(tx)
        score = _calibrated_score(tx, score_raw)
        family = LABEL_TO_FAMILY.get(label.upper(), "other")
        fam_cfg = dict(family_policy.get(family) or {})
        threshold = _as_float(fam_cfg.get("threshold"), float(args.default_threshold))

        coa = coa_by_example.get(example_id)
        debit_band, credit_band = _policy_top1_bands(coa)
        auto_accept_model = _as_bool(dict(tx.get("confidence") or {}).get("auto_accept"))
        action, priority, reason = _route_action(
            label=label,
            calibrated_score=score,
            threshold=threshold,
            always_review_labels=always_review_labels,
            debit_band=debit_band,
            credit_band=credit_band,
        )

        actions[action] += 1
        reasons[reason] += 1
        by_family_action[family][action] += 1

        out_rows.append(
            {
                "schema_version": "v0",
                "example_id": example_id,
                "reference": _as_str(dict(tx.get("record") or {}).get("referenceNumber")),
                "label": label,
                "workflow_family": family,
                "routing": {
                    "action": action,
                    "priority": priority,
                    "reason_code": reason,
                    "family_threshold": float(f"{threshold:.6f}"),
                    "calibrated_score": float(f"{score:.6f}"),
                    "model_auto_accept": bool(auto_accept_model),
                },
                "policy": {
                    "debit_top1_band": debit_band,
                    "credit_top1_band": credit_band,
                },
            }
        )

    out_jsonl = Path(args.out_jsonl)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("w", encoding="utf-8") as f:
        for row in out_rows:
            f.write(json.dumps(row, sort_keys=True))
            f.write("\n")

    summary = {
        "schema_version": "v0",
        "routing_policy_json": policy_path.as_posix(),
        "tx_jsonl": Path(args.tx_jsonl).as_posix(),
        "coa_jsonl": Path(args.coa_jsonl).as_posix(),
        "out_jsonl": out_jsonl.as_posix(),
        "records": len(out_rows),
        "action_counts": dict(sorted(actions.items(), key=lambda t: t[0])),
        "reason_counts": dict(sorted(reasons.items(), key=lambda t: t[0])),
        "family_action_counts": {
            fam: dict(sorted(cnt.items(), key=lambda t: t[0]))
            for fam, cnt in sorted(by_family_action.items(), key=lambda t: t[0])
        },
    }

    summary_path = Path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(out_jsonl.as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
