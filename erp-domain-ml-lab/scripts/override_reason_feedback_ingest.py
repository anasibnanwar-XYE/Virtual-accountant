#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any

ALIASES = {
    "reference": ["reference", "reference_number", "ref", "doc_reference"],
    "type": ["type", "transaction_type", "label", "predicted_type", "suggested_label"],
    "doc_type": ["doc_type", "document_type"],
    "doc_status": ["doc_status", "status"],
    "party": ["party", "party_code", "customer_code", "supplier_code", "dealer_code"],
    "payment_method": ["payment_method", "pay_method"],
    "gst_treatment": ["gst_treatment", "tax_treatment"],
    "currency": ["currency", "currency_code"],
    "suggested_label": ["suggested_label", "predicted_label", "model_label", "old_label"],
    "approved_label": ["approved_label", "chosen_label", "corrected_label", "new_label"],
    "suggested_debit_account_code": [
        "suggested_debit_account_code",
        "predicted_debit_account_code",
        "old_debit_account_code",
    ],
    "suggested_credit_account_code": [
        "suggested_credit_account_code",
        "predicted_credit_account_code",
        "old_credit_account_code",
    ],
    "approved_debit_account_code": [
        "approved_debit_account_code",
        "chosen_debit_account_code",
        "corrected_debit_account_code",
        "new_debit_account_code",
    ],
    "approved_credit_account_code": [
        "approved_credit_account_code",
        "chosen_credit_account_code",
        "corrected_credit_account_code",
        "new_credit_account_code",
    ],
    "reason_code": ["reason_code", "override_reason_code", "issue_code"],
    "reason_text": ["reason_text", "override_reason_text", "reason", "comment", "notes"],
    "source": ["source", "feedback_source"],
    "action": ["action", "decision", "feedback_action"],
}


def _load_records(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix in {".json", ".jsonl"}:
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

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return []
        return [dict(row) for row in reader]


def _pick(obj: dict[str, Any], names: list[str], default: Any = "") -> Any:
    for name in names:
        if name in obj and obj[name] not in (None, ""):
            return obj[name]
    return default


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _clean_upper(value: Any) -> str:
    return _clean_text(value).upper()


def _normalize_reason_code(value: str) -> str:
    code = value.strip().upper()
    if not code:
        return ""
    code = code.replace("-", "_").replace(" ", "_")
    while "__" in code:
        code = code.replace("__", "_")
    return code


def _normalize_action(value: str) -> str:
    action = value.strip().upper()
    if action in {"ACCEPTED", "APPROVED", "AUTO_ACCEPTED", "CONFIRMED"}:
        return "ACCEPTED"
    if action in {"OVERRIDDEN", "REJECTED", "EDITED", "CORRECTED"}:
        return "OVERRIDDEN"
    if action in {"REVIEW", "MANUAL_REVIEW"}:
        return "REVIEW"
    return action or ""


def _correction_scope(
    *,
    suggested_label: str,
    approved_label: str,
    suggested_debit: str,
    approved_debit: str,
    suggested_credit: str,
    approved_credit: str,
) -> str:
    changed_label = bool(approved_label and approved_label != suggested_label)
    changed_debit = bool(approved_debit and approved_debit != suggested_debit)
    changed_credit = bool(approved_credit and approved_credit != suggested_credit)

    if changed_label and (changed_debit or changed_credit):
        return "LABEL_AND_COA"
    if changed_label:
        return "LABEL_ONLY"
    if changed_debit and changed_credit:
        return "COA_BOTH"
    if changed_debit:
        return "COA_DEBIT"
    if changed_credit:
        return "COA_CREDIT"
    return "NO_CHANGE"


def _normalize_record(obj: dict[str, Any], source_name: str) -> dict[str, Any] | None:
    reference = _clean_upper(_pick(obj, ALIASES["reference"]))
    typ = _clean_upper(_pick(obj, ALIASES["type"]))
    suggested_label = _clean_upper(_pick(obj, ALIASES["suggested_label"], typ))
    approved_label = _clean_upper(_pick(obj, ALIASES["approved_label"]))
    suggested_debit = _clean_upper(_pick(obj, ALIASES["suggested_debit_account_code"]))
    suggested_credit = _clean_upper(_pick(obj, ALIASES["suggested_credit_account_code"]))
    approved_debit = _clean_upper(_pick(obj, ALIASES["approved_debit_account_code"]))
    approved_credit = _clean_upper(_pick(obj, ALIASES["approved_credit_account_code"]))
    reason_code = _normalize_reason_code(_clean_text(_pick(obj, ALIASES["reason_code"])))
    reason_text = _clean_text(_pick(obj, ALIASES["reason_text"]))
    action = _normalize_action(_clean_text(_pick(obj, ALIASES["action"])))

    if not reference and not reason_text and not reason_code:
        return None

    scope = _correction_scope(
        suggested_label=suggested_label,
        approved_label=approved_label,
        suggested_debit=suggested_debit,
        approved_debit=approved_debit,
        suggested_credit=suggested_credit,
        approved_credit=approved_credit,
    )

    return {
        "reference": reference,
        "type": typ,
        "doc_type": _clean_upper(_pick(obj, ALIASES["doc_type"])),
        "doc_status": _clean_upper(_pick(obj, ALIASES["doc_status"])),
        "party": _clean_upper(_pick(obj, ALIASES["party"])),
        "payment_method": _clean_upper(_pick(obj, ALIASES["payment_method"])),
        "gst_treatment": _clean_upper(_pick(obj, ALIASES["gst_treatment"])),
        "currency": _clean_upper(_pick(obj, ALIASES["currency"], "INR")) or "INR",
        "suggested_label": suggested_label,
        "approved_label": approved_label,
        "suggested_debit_account_code": suggested_debit,
        "suggested_credit_account_code": suggested_credit,
        "approved_debit_account_code": approved_debit,
        "approved_credit_account_code": approved_credit,
        "correction_scope": scope,
        "reason_code": reason_code,
        "reason_text": reason_text,
        "action": action,
        "source": _clean_text(_pick(obj, ALIASES["source"], source_name)) or source_name,
    }


def _dedupe_key(obj: dict[str, Any]) -> str:
    key_obj = {
        "reference": obj.get("reference"),
        "type": obj.get("type"),
        "suggested_label": obj.get("suggested_label"),
        "approved_label": obj.get("approved_label"),
        "suggested_debit_account_code": obj.get("suggested_debit_account_code"),
        "suggested_credit_account_code": obj.get("suggested_credit_account_code"),
        "approved_debit_account_code": obj.get("approved_debit_account_code"),
        "approved_credit_account_code": obj.get("approved_credit_account_code"),
        "reason_code": obj.get("reason_code"),
        "reason_text": obj.get("reason_text"),
    }
    payload = json.dumps(key_obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_existing_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    keys: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except Exception:
                continue
            if isinstance(obj, dict):
                keys.add(_dedupe_key(obj))
    return keys


def _append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _lock_path(out_path: Path) -> Path:
    return out_path.with_name(f"{out_path.name}.lock")


@contextmanager
def _exclusive_lock(lock_path: Path):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Ingest override-reason feedback into canonical JSONL memory.")
    p.add_argument("--input-file", action="append", required=True, help="CSV/JSONL source file (repeatable)")
    p.add_argument("--out-jsonl", required=True, help="Canonical override-reason memory JSONL")
    p.add_argument("--report-out", default=None)
    return p


def main() -> int:
    args = _build_parser().parse_args()

    out_path = Path(args.out_jsonl)
    lock_path = _lock_path(out_path)

    candidates_by_file: dict[str, list[tuple[str, dict[str, Any]]]] = {}
    duplicate_in_batch_by_file: dict[str, int] = {}
    candidate_seen: set[str] = set()
    scanned_by_file: dict[str, int] = {}
    skipped_by_file: dict[str, int] = {}

    accepted: list[dict[str, Any]] = []
    scanned = 0
    skipped = 0
    duplicate = 0

    for input_file in args.input_file:
        src = Path(input_file)
        if not src.exists() or not src.is_file():
            raise SystemExit(f"Input not found: {src}")

        records = _load_records(src)
        src_key = src.as_posix()
        candidates_by_file.setdefault(src_key, [])
        duplicate_in_batch_by_file.setdefault(src_key, 0)
        scanned_by_file[src_key] = len(records)
        skipped_by_file[src_key] = 0

        for rec in records:
            scanned += 1
            if not isinstance(rec, dict):
                skipped += 1
                skipped_by_file[src_key] += 1
                continue

            norm = _normalize_record(rec, src.name)
            if norm is None:
                skipped += 1
                skipped_by_file[src_key] += 1
                continue

            key = _dedupe_key(norm)
            if key in candidate_seen:
                duplicate += 1
                duplicate_in_batch_by_file[src_key] += 1
                continue

            candidate_seen.add(key)
            candidates_by_file[src_key].append((key, norm))

    duplicate_existing_by_file: dict[str, int] = {k: 0 for k in candidates_by_file}
    accepted_by_file: dict[str, int] = {k: 0 for k in candidates_by_file}

    with _exclusive_lock(lock_path):
        existing_keys = _load_existing_keys(out_path)
        for input_file in args.input_file:
            src = Path(input_file).as_posix()
            for key, norm in candidates_by_file.get(src, []):
                if key in existing_keys:
                    duplicate += 1
                    duplicate_existing_by_file[src] += 1
                    continue
                existing_keys.add(key)
                accepted.append(norm)
                accepted_by_file[src] += 1
        if accepted:
            _append_jsonl(out_path, accepted)

    per_file: list[dict[str, Any]] = []
    for input_file in args.input_file:
        src = Path(input_file).as_posix()
        per_file.append(
            {
                "input_file": src,
                "records_scanned": int(scanned_by_file.get(src, 0)),
                "accepted": int(accepted_by_file.get(src, 0)),
                "skipped": int(skipped_by_file.get(src, 0)),
                "duplicate_in_batch": int(duplicate_in_batch_by_file.get(src, 0)),
                "duplicate_existing": int(duplicate_existing_by_file.get(src, 0)),
            }
        )

    report = {
        "schema_version": "v0",
        "out_jsonl": out_path.as_posix(),
        "lock_file": lock_path.as_posix(),
        "records_scanned": scanned,
        "accepted": len(accepted),
        "skipped": skipped,
        "duplicate": duplicate,
        "per_file": per_file,
    }

    if args.report_out:
        report_path = Path(args.report_out)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
