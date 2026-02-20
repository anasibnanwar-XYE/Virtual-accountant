#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path
from typing import Any


CSV_FIELDS = [
    "type",
    "reference",
    "date",
    "sku",
    "qty",
    "price",
    "cost",
    "tax_rate",
    "party",
    "notes",
    "doc_type",
    "doc_status",
    "memo",
    "payment_method",
    "gst_treatment",
    "gst_inclusive",
    "currency",
    "journal_lines",
]

ZERO = Decimal("0")
TAX_ZERO = Decimal("0.0000")

LABEL_RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"^CN[-_]"), "SALE_RETURN"),
    (re.compile(r"^(SO|INV|SALE)[-_]"), "SALE"),
    (re.compile(r"^(PO|PUR|BILL)[-_]"), "PURCHASE"),
    (re.compile(r"^(RCPT|RECEIPT|PMT|PAY)[-_]"), "PAYMENT"),
    (re.compile(r"^COGS[-_]"), "COGS"),
    (re.compile(r"^WO[-_]"), "WRITE_OFF"),
    (re.compile(r"^COUNT[-_]"), "INVENTORY_COUNT"),
    (re.compile(r"^LOCK[-_]"), "PERIOD_LOCK"),
    (re.compile(r"^OPEN[-_]?STOCK"), "OPENING_BALANCE"),
]

DOC_TYPE_BY_LABEL = {
    "SALE": "INVOICE",
    "PURCHASE": "RAW_MATERIAL_PURCHASE",
    "PAYMENT": "SETTLEMENT",
    "SALE_RETURN": "SALE_RETURN",
    "COGS": "INVENTORY_COSTING",
    "WRITE_OFF": "INVENTORY_ADJUSTMENT",
    "INVENTORY_COUNT": "INVENTORY_COUNT",
    "PERIOD_LOCK": "PERIOD_CONTROL",
    "OPENING_BALANCE": "OPENING_STOCK",
}

NOTES_BY_LABEL = {
    "SALE": "snapshot sale journal",
    "PURCHASE": "snapshot purchase journal",
    "PAYMENT": "snapshot payment journal",
    "SALE_RETURN": "snapshot sales return journal",
    "COGS": "snapshot cogs journal",
    "WRITE_OFF": "snapshot write-off journal",
    "INVENTORY_COUNT": "snapshot inventory count journal",
    "PERIOD_LOCK": "snapshot period control journal",
    "OPENING_BALANCE": "snapshot opening balance journal",
}


@dataclass(frozen=True, slots=True)
class LinePosting:
    account_code: str
    direction: str
    amount: Decimal
    description: str


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            raw = line.rstrip("\n")
            if not raw:
                raise ValueError(f"Empty JSONL line in {path} at {line_no}")
            obj = json.loads(raw)
            if not isinstance(obj, dict):
                raise ValueError(f"Invalid JSONL object in {path} at {line_no}")
            records.append(obj)
    return records


def _to_decimal(value: Any) -> Decimal:
    if value is None:
        return ZERO
    if isinstance(value, Decimal):
        return value
    if isinstance(value, (int, float)):
        return Decimal(str(value))
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return ZERO
        try:
            return Decimal(s)
        except InvalidOperation as e:
            raise ValueError(f"Invalid decimal value: {value}") from e
    raise ValueError(f"Unsupported amount type: {type(value)!r}")


def _decimal_string(value: Decimal, *, places: str = "0.01") -> str:
    q = value.quantize(Decimal(places), rounding=ROUND_HALF_UP)
    s = format(q, "f")
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s if s else "0"


def _load_party_code_by_id(snapshot_dir: Path, file_name: str, prefix: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for rec in _read_jsonl(snapshot_dir / "normalized" / file_name):
        entity_id = rec.get("id")
        if entity_id is None:
            continue
        code = str(rec.get("code") or "").strip()
        if code:
            out[str(entity_id)] = code
        else:
            out[str(entity_id)] = f"{prefix}-{entity_id}"
    return out


def _account_code_map(snapshot_dir: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for rec in _read_jsonl(snapshot_dir / "normalized" / "accounts.jsonl"):
        account_id = rec.get("id")
        if account_id is None:
            continue
        code = str(rec.get("code") or "").strip()
        if code:
            out[str(account_id)] = code
    return out


def _normalize_postings(entry: dict[str, Any], *, account_code_by_id: dict[str, str]) -> list[LinePosting]:
    lines = entry.get("lines")
    if not isinstance(lines, list):
        return []
    postings: list[LinePosting] = []
    for line in lines:
        if not isinstance(line, dict):
            continue
        account_code = str(line.get("accountCode") or "").strip()
        if not account_code:
            account_id = line.get("accountId")
            if account_id is not None:
                account_code = account_code_by_id.get(str(account_id), "")
        account_code = account_code.strip()
        if not account_code:
            continue
        description = str(line.get("description") or "").strip()
        debit = line.get("debit")
        credit = line.get("credit")
        if debit is not None:
            amount = _to_decimal(debit).copy_abs()
            if amount > ZERO:
                postings.append(
                    LinePosting(
                        account_code=account_code,
                        direction="D",
                        amount=amount,
                        description=description,
                    )
                )
        if credit is not None:
            amount = _to_decimal(credit).copy_abs()
            if amount > ZERO:
                postings.append(
                    LinePosting(
                        account_code=account_code,
                        direction="C",
                        amount=amount,
                        description=description,
                    )
                )
    return postings


def _serialize_journal_lines(postings: list[LinePosting]) -> str:
    segments = []
    for p in postings:
        desc = p.description if p.description else "line"
        segments.append(f"{p.account_code} | {p.direction} | {_decimal_string(p.amount)} | {desc}")
    return " || ".join(segments)


def _infer_label(reference: str, memo: str, postings: list[LinePosting]) -> str:
    reference_norm = reference.strip().upper()
    for pattern, label in LABEL_RULES:
        if pattern.search(reference_norm):
            return label

    debit_accounts = {p.account_code.upper() for p in postings if p.direction == "D"}
    credit_accounts = {p.account_code.upper() for p in postings if p.direction == "C"}
    all_accounts = debit_accounts | credit_accounts
    memo_norm = memo.upper()

    if "SALES_RETURN" in all_accounts or "CREDIT NOTE" in memo_norm:
        return "SALE_RETURN"
    if "COGS" in debit_accounts and "INVENTORY" in credit_accounts:
        return "COGS"
    if "WRITE_OFF" in all_accounts:
        return "WRITE_OFF"
    if "SALES" in credit_accounts and ("AR" in debit_accounts or "CASH" in debit_accounts or "BANK" in debit_accounts):
        return "SALE"
    if "AP" in credit_accounts and ("INVENTORY" in debit_accounts or "GST_INPUT" in debit_accounts):
        return "PURCHASE"
    if ("CASH" in debit_accounts or "BANK" in debit_accounts) and "AR" in credit_accounts:
        return "PAYMENT"
    if ("CASH" in credit_accounts or "BANK" in credit_accounts) and "AP" in debit_accounts:
        return "PAYMENT"
    if "LOCK" in memo_norm:
        return "PERIOD_LOCK"
    return ""


def _tax_rate_from_postings(postings: list[LinePosting], total_amount: Decimal) -> Decimal:
    gst_amount = ZERO
    for p in postings:
        if "GST_" in p.account_code.upper():
            gst_amount += p.amount
    if gst_amount <= ZERO or total_amount <= ZERO:
        return TAX_ZERO
    base_amount = total_amount - gst_amount
    if base_amount <= ZERO:
        return TAX_ZERO
    rate = gst_amount / base_amount
    if rate < TAX_ZERO:
        return TAX_ZERO
    return rate.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)


def _party_value(
    entry: dict[str, Any],
    *,
    dealer_code_by_id: dict[str, str],
    supplier_code_by_id: dict[str, str],
) -> str:
    dealer_id = entry.get("dealerId")
    if dealer_id is not None:
        return dealer_code_by_id.get(str(dealer_id), f"GST-DEALER-{dealer_id}")
    supplier_id = entry.get("supplierId")
    if supplier_id is not None:
        return supplier_code_by_id.get(str(supplier_id), f"SUP-RAW-{supplier_id}")
    return ""


def _sorted_entries(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        entries,
        key=lambda e: (
            str(e.get("entryDate") or ""),
            str(e.get("referenceNumber") or ""),
            str(e.get("publicId") or ""),
            str(e.get("id") or ""),
        ),
    )


def build_training_rows(
    *,
    snapshot_dir: Path,
    include_non_posted: bool,
    allow_unknown_labels: bool,
    label_field: str,
    balance_tolerance: Decimal,
    currency: str,
) -> tuple[list[dict[str, str]], dict[str, int]]:
    entries = _read_jsonl(snapshot_dir / "normalized" / "journal_entries.jsonl")
    account_code_by_id = _account_code_map(snapshot_dir)
    dealer_code_by_id = _load_party_code_by_id(snapshot_dir, "dealers.jsonl", "GST-DEALER")
    supplier_code_by_id = _load_party_code_by_id(snapshot_dir, "suppliers.jsonl", "SUP-RAW")

    stats: dict[str, int] = {
        "entries_seen": len(entries),
        "rows_written": 0,
        "skipped_non_posted": 0,
        "skipped_no_postings": 0,
        "skipped_unbalanced": 0,
        "skipped_unknown_label": 0,
    }
    rows: list[dict[str, str]] = []

    for entry in _sorted_entries(entries):
        status = str(entry.get("status") or "").strip().upper()
        if not include_non_posted and status != "POSTED":
            stats["skipped_non_posted"] += 1
            continue

        postings = _normalize_postings(entry, account_code_by_id=account_code_by_id)
        if not postings:
            stats["skipped_no_postings"] += 1
            continue

        debit_total = sum((p.amount for p in postings if p.direction == "D"), start=ZERO)
        credit_total = sum((p.amount for p in postings if p.direction == "C"), start=ZERO)
        if (debit_total - credit_total).copy_abs() > balance_tolerance:
            stats["skipped_unbalanced"] += 1
            continue

        memo = str(entry.get("memo") or "").strip()
        reference = str(entry.get("referenceNumber") or "").strip()
        if not reference:
            reference = str(entry.get("publicId") or entry.get("id") or "").strip()
        if not reference:
            reference = "UNSPECIFIED-REF"

        label = ""
        if label_field:
            label = str(entry.get(label_field) or "").strip().upper()
        if not label:
            label = _infer_label(reference, memo, postings)
        if not label and not allow_unknown_labels:
            stats["skipped_unknown_label"] += 1
            continue
        if not label:
            label = "UNKNOWN"

        total_amount = max(debit_total, credit_total)
        tax_rate = _tax_rate_from_postings(postings, total_amount)
        has_gst = any("GST_" in p.account_code.upper() for p in postings)
        party = _party_value(
            entry,
            dealer_code_by_id=dealer_code_by_id,
            supplier_code_by_id=supplier_code_by_id,
        )
        journal_lines = _serialize_journal_lines(postings)
        notes = NOTES_BY_LABEL.get(label, "snapshot journal import")
        doc_type = str(entry.get("docType") or entry.get("doc_type") or DOC_TYPE_BY_LABEL.get(label, "JOURNAL_ENTRY"))
        payment_method = str(entry.get("paymentMethod") or entry.get("payment_method") or "")
        gst_treatment = str(entry.get("gstTreatment") or entry.get("gst_treatment") or ("TAXABLE" if has_gst else "NON_GST"))
        gst_inclusive_raw = entry.get("gstInclusive")
        if gst_inclusive_raw is None:
            gst_inclusive_raw = entry.get("gst_inclusive")
        gst_inclusive = str(gst_inclusive_raw if gst_inclusive_raw is not None else "false")

        row = {
            "type": label,
            "reference": reference,
            "date": str(entry.get("entryDate") or ""),
            "sku": "",
            "qty": "1",
            "price": _decimal_string(total_amount),
            "cost": "0",
            "tax_rate": _decimal_string(tax_rate, places="0.0001"),
            "party": party,
            "notes": notes,
            "doc_type": doc_type,
            "doc_status": status,
            "memo": memo,
            "payment_method": payment_method,
            "gst_treatment": gst_treatment,
            "gst_inclusive": gst_inclusive,
            "currency": currency,
            "journal_lines": journal_lines,
        }
        rows.append(row)
        stats["rows_written"] += 1

    return rows, stats


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build a training CSV from ORCHESTRATOR ERP snapshot journal entries."
    )
    p.add_argument("--snapshot-dir", required=True, help="Path to ERP snapshot directory")
    p.add_argument("--out-csv", required=True, help="Output CSV file path")
    p.add_argument(
        "--include-non-posted",
        action="store_true",
        help="Include non-POSTED journal entries (default: only POSTED)",
    )
    p.add_argument(
        "--allow-unknown-labels",
        action="store_true",
        help="Include rows where label cannot be inferred (default: skip)",
    )
    p.add_argument(
        "--label-field",
        default="syntheticLabel",
        help="Label field in journal entries to prefer before inference (default: syntheticLabel)",
    )
    p.add_argument(
        "--balance-tolerance",
        default="0.01",
        help="Allowed debit-credit absolute difference before dropping row (default: 0.01)",
    )
    p.add_argument("--currency", default="INR", help="Currency value to store in CSV (default: INR)")
    p.add_argument("--report-json", default=None, help="Optional path to write a build report JSON")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    snapshot_dir = Path(args.snapshot_dir).resolve()
    out_csv = Path(args.out_csv).resolve()

    if out_csv.exists():
        raise SystemExit(f"Output CSV already exists: {out_csv}")
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    balance_tolerance = _to_decimal(args.balance_tolerance)
    rows, stats = build_training_rows(
        snapshot_dir=snapshot_dir,
        include_non_posted=bool(args.include_non_posted),
        allow_unknown_labels=bool(args.allow_unknown_labels),
        label_field=str(args.label_field or ""),
        balance_tolerance=balance_tolerance,
        currency=str(args.currency),
    )
    if not rows:
        raise SystemExit("No training rows produced from snapshot")

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    report = {
        "schema_version": "v0",
        "dataset_format": "orchestrator_erp_training_csv_v2",
        "snapshot_dir": snapshot_dir.as_posix(),
        "out_csv": out_csv.as_posix(),
        "settings": {
            "include_non_posted": bool(args.include_non_posted),
            "allow_unknown_labels": bool(args.allow_unknown_labels),
            "label_field": str(args.label_field or ""),
            "balance_tolerance": str(balance_tolerance),
            "currency": str(args.currency),
        },
        "stats": stats,
    }

    if args.report_json:
        report_path = Path(args.report_json).resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
