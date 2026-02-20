#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

CSV_FIELDS = [
    "example_id",
    "actor_user_id",
    "company_code",
    "suggested_label",
    "chosen_label",
    "suggested_debit_account_code",
    "approved_debit_account_code",
    "suggested_credit_account_code",
    "approved_credit_account_code",
    "text",
    "record_kind",
    "record_publicId",
    "record_id",
    "record_referenceNumber",
    "record_entryDate",
    "review_rank",
    "priority",
    "action",
    "reason_code",
    "workflow_family",
    "urgency_score",
    "tx_calibrated_score",
    "tx_threshold",
    "tx_margin",
    "policy_severity",
]


def _as_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            obj = json.loads(raw)
            if isinstance(obj, dict):
                out.append(obj)
    return out


def _top_label(tx: dict[str, Any]) -> str:
    suggested = _as_str(tx.get("suggested_label")).upper()
    if suggested:
        return suggested
    preds = [p for p in list(tx.get("predictions") or []) if isinstance(p, dict)]
    if preds:
        return _as_str(preds[0].get("label")).upper()
    return ""


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Export prioritized review queue CSV from tx output + ranked queue JSONL.")
    p.add_argument("--tx-jsonl", required=True)
    p.add_argument("--priority-queue-jsonl", required=True)
    p.add_argument("--out-csv", required=True)
    p.add_argument("--report-out", default=None)
    return p


def main() -> int:
    args = _build_parser().parse_args()

    tx_path = Path(args.tx_jsonl)
    queue_path = Path(args.priority_queue_jsonl)
    out_csv = Path(args.out_csv)
    report_out = Path(args.report_out) if args.report_out else None

    if not tx_path.exists() or not tx_path.is_file():
        raise SystemExit(f"Transaction JSONL not found: {tx_path}")
    if not queue_path.exists() or not queue_path.is_file():
        raise SystemExit(f"Priority queue JSONL not found: {queue_path}")

    tx_rows = _read_jsonl(tx_path)
    queue_rows = _read_jsonl(queue_path)
    tx_by_id = {_as_str(r.get("example_id")): r for r in tx_rows if _as_str(r.get("example_id"))}

    skipped_missing_tx = 0
    written = 0

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        for q in sorted(
            queue_rows,
            key=lambda row: (_as_int(row.get("review_rank"), 10**9), _as_str(row.get("example_id"))),
        ):
            example_id = _as_str(q.get("example_id"))
            tx = tx_by_id.get(example_id)
            if tx is None:
                skipped_missing_tx += 1
                continue

            record = dict(tx.get("record") or {})
            row = {
                "example_id": example_id,
                "actor_user_id": "",
                "company_code": "",
                "suggested_label": _top_label(tx),
                "chosen_label": "",
                "suggested_debit_account_code": "",
                "approved_debit_account_code": "",
                "suggested_credit_account_code": "",
                "approved_credit_account_code": "",
                "text": _as_str(tx.get("text")),
                "record_kind": _as_str(record.get("kind")),
                "record_publicId": _as_str(record.get("publicId")),
                "record_id": _as_str(record.get("id")),
                "record_referenceNumber": _as_str(record.get("referenceNumber")),
                "record_entryDate": _as_str(record.get("entryDate")),
                "review_rank": _as_str(q.get("review_rank")),
                "priority": _as_str(q.get("priority")),
                "action": _as_str(q.get("action")),
                "reason_code": _as_str(q.get("reason_code")),
                "workflow_family": _as_str(q.get("workflow_family")),
                "urgency_score": f"{_as_float(q.get('urgency_score'), 0.0):.6f}",
                "tx_calibrated_score": f"{_as_float(q.get('tx_calibrated_score'), 0.0):.6f}",
                "tx_threshold": f"{_as_float(q.get('tx_threshold'), 0.0):.6f}",
                "tx_margin": f"{_as_float(q.get('tx_margin'), 0.0):.6f}",
                "policy_severity": _as_str(q.get("policy_severity")),
            }
            writer.writerow(row)
            written += 1

    report = {
        "schema_version": "v0",
        "tx_jsonl": tx_path.as_posix(),
        "priority_queue_jsonl": queue_path.as_posix(),
        "out_csv": out_csv.as_posix(),
        "tx_records": len(tx_rows),
        "queue_records": len(queue_rows),
        "rows_written": written,
        "rows_skipped_missing_tx": skipped_missing_tx,
    }

    if report_out is not None:
        report_out.parent.mkdir(parents=True, exist_ok=True)
        report_out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
