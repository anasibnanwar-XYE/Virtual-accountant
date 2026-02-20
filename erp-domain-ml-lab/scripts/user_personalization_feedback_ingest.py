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
    "user_id": ["user_id", "actor_user_id", "user", "email", "actor", "assignee"],
    "company_code": ["company_code", "company", "tenant", "company_id", "companyId"],
    "workflow_family": ["workflow_family", "family", "workflow", "workflow_context_family"],
    "reference": ["reference", "record_referenceNumber", "reference_number", "doc_reference"],
    "doc_type": ["doc_type", "document_type", "record_doc_type"],
    "suggested_label": ["suggested_label", "predicted_label", "old_label", "label"],
    "approved_label": ["approved_label", "chosen_label", "corrected_label", "new_label"],
    "suggested_debit_account_code": [
        "suggested_debit_account_code",
        "predicted_debit_account_code",
        "old_debit_account_code",
        "debit_suggested",
    ],
    "approved_debit_account_code": [
        "approved_debit_account_code",
        "chosen_debit_account_code",
        "corrected_debit_account_code",
        "new_debit_account_code",
        "debit_approved",
    ],
    "suggested_credit_account_code": [
        "suggested_credit_account_code",
        "predicted_credit_account_code",
        "old_credit_account_code",
        "credit_suggested",
    ],
    "approved_credit_account_code": [
        "approved_credit_account_code",
        "chosen_credit_account_code",
        "corrected_credit_account_code",
        "new_credit_account_code",
        "credit_approved",
    ],
    "reason_code": ["reason_code", "override_reason_code", "issue_code"],
    "reason_text": ["reason_text", "reason", "comment", "notes"],
    "action": ["action", "decision", "feedback_action"],
    "source": ["source", "feedback_source"],
}

FAMILY_CANON = {
    "SALE": "sale",
    "O2C": "sale",
    "PURCHASE": "purchase",
    "P2P": "purchase",
    "PAYMENT": "payment",
    "SETTLEMENT_SPLIT": "settlement_split",
    "TAX_SETTLEMENT": "tax_settlement",
    "PAYROLL": "payroll",
    "SALE_RETURN": "sale_return",
    "COGS": "cogs",
    "WRITE_OFF": "write_off",
    "INVENTORY_COUNT": "inventory_count",
    "PERIOD_LOCK": "period_lock",
}


def _pick(obj: dict[str, Any], keys: list[str], default: Any = "") -> Any:
    for k in keys:
        if k in obj and obj[k] not in (None, ""):
            return obj[k]
    return default


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _up(value: Any) -> str:
    return _clean(value).upper()


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


def _family(value: str) -> str:
    raw = _up(value).replace("-", "_").replace(" ", "_")
    if raw in FAMILY_CANON:
        return FAMILY_CANON[raw]
    return raw.lower() if raw else ""


def _action(value: str) -> str:
    a = _up(value)
    if a in {"APPROVED", "ACCEPTED", "CONFIRMED"}:
        return "ACCEPTED"
    if a in {"OVERRIDDEN", "REJECTED", "EDITED", "CORRECTED"}:
        return "OVERRIDDEN"
    if a in {"REVIEW", "MANUAL_REVIEW"}:
        return "REVIEW"
    return a


def _normalize(obj: dict[str, Any], source_name: str) -> dict[str, Any] | None:
    user_id = _clean(_pick(obj, ALIASES["user_id"]))
    if not user_id:
        return None

    approved_label = _up(_pick(obj, ALIASES["approved_label"]))
    approved_debit = _up(_pick(obj, ALIASES["approved_debit_account_code"]))
    approved_credit = _up(_pick(obj, ALIASES["approved_credit_account_code"]))
    if not approved_label and not approved_debit and not approved_credit:
        return None

    family_raw = _clean(_pick(obj, ALIASES["workflow_family"]))
    return {
        "user_id": user_id.lower(),
        "company_code": _up(_pick(obj, ALIASES["company_code"])),
        "workflow_family": _family(family_raw),
        "reference": _up(_pick(obj, ALIASES["reference"])),
        "doc_type": _up(_pick(obj, ALIASES["doc_type"])),
        "suggested_label": _up(_pick(obj, ALIASES["suggested_label"])),
        "approved_label": approved_label,
        "suggested_debit_account_code": _up(_pick(obj, ALIASES["suggested_debit_account_code"])),
        "approved_debit_account_code": approved_debit,
        "suggested_credit_account_code": _up(_pick(obj, ALIASES["suggested_credit_account_code"])),
        "approved_credit_account_code": approved_credit,
        "reason_code": _up(_pick(obj, ALIASES["reason_code"])),
        "reason_text": _clean(_pick(obj, ALIASES["reason_text"])),
        "action": _action(_clean(_pick(obj, ALIASES["action"]))),
        "source": _clean(_pick(obj, ALIASES["source"], source_name)) or source_name,
    }


def _dedupe_key(row: dict[str, Any]) -> str:
    payload = {
        "user_id": row.get("user_id"),
        "company_code": row.get("company_code"),
        "workflow_family": row.get("workflow_family"),
        "reference": row.get("reference"),
        "doc_type": row.get("doc_type"),
        "approved_label": row.get("approved_label"),
        "approved_debit_account_code": row.get("approved_debit_account_code"),
        "approved_credit_account_code": row.get("approved_credit_account_code"),
        "reason_code": row.get("reason_code"),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


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
    p = argparse.ArgumentParser(description="Ingest user-personalization feedback memory JSONL.")
    p.add_argument("--input-file", action="append", required=True)
    p.add_argument("--out-jsonl", required=True)
    p.add_argument("--report-out", default=None)
    return p


def main() -> int:
    args = _build_parser().parse_args()

    out_path = Path(args.out_jsonl)
    lock_path = _lock_path(out_path)

    candidates_by_file: dict[str, list[tuple[str, dict[str, Any]]]] = {}
    duplicate_in_batch_by_file: dict[str, int] = {}
    skipped_by_file: dict[str, int] = {}
    scanned_by_file: dict[str, int] = {}
    candidate_seen: set[str] = set()

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
        duplicate_in_batch_by_file[src_key] = 0
        skipped_by_file[src_key] = 0
        scanned_by_file[src_key] = len(records)

        for rec in records:
            scanned += 1
            if not isinstance(rec, dict):
                skipped += 1
                skipped_by_file[src_key] += 1
                continue
            norm = _normalize(rec, src.name)
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

    accepted: list[dict[str, Any]] = []
    accepted_by_file: dict[str, int] = {Path(i).as_posix(): 0 for i in args.input_file}
    duplicate_existing_by_file: dict[str, int] = {Path(i).as_posix(): 0 for i in args.input_file}

    with _exclusive_lock(lock_path):
        existing = _load_existing_keys(out_path)
        for input_file in args.input_file:
            src_key = Path(input_file).as_posix()
            for key, row in candidates_by_file.get(src_key, []):
                if key in existing:
                    duplicate += 1
                    duplicate_existing_by_file[src_key] += 1
                    continue
                existing.add(key)
                accepted.append(row)
                accepted_by_file[src_key] += 1
        if accepted:
            _append_jsonl(out_path, accepted)

    files = []
    for input_file in args.input_file:
        src_key = Path(input_file).as_posix()
        files.append(
            {
                "input_file": src_key,
                "records_scanned": int(scanned_by_file.get(src_key, 0)),
                "accepted": int(accepted_by_file.get(src_key, 0)),
                "skipped": int(skipped_by_file.get(src_key, 0)),
                "duplicate_in_batch": int(duplicate_in_batch_by_file.get(src_key, 0)),
                "duplicate_existing": int(duplicate_existing_by_file.get(src_key, 0)),
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
        "files": files,
    }

    if args.report_out:
        rpt = Path(args.report_out)
        rpt.parent.mkdir(parents=True, exist_ok=True)
        rpt.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
