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

REQUIRED_TARGETS = [
    "revenue_account_code",
    "cogs_account_code",
    "inventory_account_code",
    "tax_account_code",
    "discount_account_code",
]

DEFAULT_BY_KIND = {
    "FINISHED_GOOD": {
        "revenue_account_code": "SALES",
        "cogs_account_code": "COGS",
        "inventory_account_code": "INVENTORY",
        "tax_account_code": "GST_OUTPUT",
        "discount_account_code": "DISCOUNT_ALLOWED",
    },
    "RAW_MATERIAL": {
        "revenue_account_code": "SALES",
        "cogs_account_code": "FREIGHT_IN",
        "inventory_account_code": "INVENTORY",
        "tax_account_code": "GST_INPUT",
        "discount_account_code": "DISCOUNT_RECEIVED",
    },
}

ALIASES = {
    "sku": ["sku", "sku_code", "product_sku", "item_code", "item_sku"],
    "product_name": ["product_name", "name", "item_name"],
    "product_kind": ["product_kind", "kind", "item_kind", "product_type"],
    "category": ["category", "product_category", "item_category"],
    "uom": ["uom", "unit", "unit_of_measure"],
    "gst_rate": ["gst_rate", "tax_rate", "vat_rate"],
    "base_price": ["base_price", "price", "list_price", "selling_price"],
    "avg_cost": ["avg_cost", "cost", "standard_cost", "purchase_cost"],
    "revenue_account_code": [
        "revenue_account_code",
        "revenue_account",
        "sales_account_code",
        "sales_account",
        "chosen_revenue_account_code",
        "approved_revenue_account_code",
    ],
    "cogs_account_code": [
        "cogs_account_code",
        "cogs_account",
        "cost_account_code",
        "chosen_cogs_account_code",
        "approved_cogs_account_code",
    ],
    "inventory_account_code": [
        "inventory_account_code",
        "inventory_account",
        "stock_account_code",
        "chosen_inventory_account_code",
        "approved_inventory_account_code",
    ],
    "tax_account_code": [
        "tax_account_code",
        "gst_account_code",
        "tax_account",
        "chosen_tax_account_code",
        "approved_tax_account_code",
    ],
    "discount_account_code": [
        "discount_account_code",
        "discount_account",
        "chosen_discount_account_code",
        "approved_discount_account_code",
    ],
}


def _load_records(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix in {".jsonl", ".json"}:
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


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        text = str(value or "").strip()
        if not text:
            return default
        return float(text)
    except Exception:
        return default


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _clean_account(value: Any) -> str:
    return _clean_text(value).upper()


def _normalize_kind(value: str) -> str:
    kind = value.strip().upper()
    if kind in {"FG", "FINISHED", "FINISHED_GOODS", "FINISHED_GOOD"}:
        return "FINISHED_GOOD"
    if kind in {"RM", "RAW", "RAW_MATERIALS", "RAW_MATERIAL"}:
        return "RAW_MATERIAL"
    return "FINISHED_GOOD"


def _flatten_json_like_record(obj: dict[str, Any]) -> dict[str, Any]:
    flat = dict(obj)
    candidate = obj.get("candidate")
    if isinstance(candidate, dict):
        for key, value in candidate.items():
            flat.setdefault(key, value)

    approved = obj.get("approved_accounts")
    if isinstance(approved, dict):
        for key, value in approved.items():
            flat.setdefault(f"{key}_account_code", value)

    recommendations = obj.get("recommendations")
    if isinstance(recommendations, dict):
        for target in ["revenue", "cogs", "inventory", "tax", "discount"]:
            target_obj = recommendations.get(target)
            if isinstance(target_obj, dict):
                best = target_obj.get("combined_best")
                if isinstance(best, dict):
                    flat.setdefault(f"{target}_account_code", best.get("label"))
    return flat


def _normalize_record(obj: dict[str, Any], source: str) -> dict[str, Any] | None:
    raw = _flatten_json_like_record(obj)

    sku = _clean_text(_pick(raw, ALIASES["sku"]))
    if not sku:
        return None

    kind = _normalize_kind(_clean_text(_pick(raw, ALIASES["product_kind"], "FINISHED_GOOD")))
    defaults = DEFAULT_BY_KIND[kind]

    normalized = {
        "sku": sku.upper(),
        "product_name": _clean_text(_pick(raw, ALIASES["product_name"], sku)),
        "product_kind": kind,
        "category": _clean_text(_pick(raw, ALIASES["category"], "GENERIC")).upper(),
        "uom": _clean_text(_pick(raw, ALIASES["uom"], "PCS")).upper(),
        "gst_rate": _as_float(_pick(raw, ALIASES["gst_rate"], 0.0)),
        "base_price": _as_float(_pick(raw, ALIASES["base_price"], 0.0)),
        "avg_cost": _as_float(_pick(raw, ALIASES["avg_cost"], 0.0)),
        "source": source,
    }

    for target in REQUIRED_TARGETS:
        account = _clean_account(_pick(raw, ALIASES[target], defaults[target]))
        normalized[target] = account or defaults[target]

    return normalized


def _dedupe_key(obj: dict[str, Any]) -> str:
    key_obj = {
        "sku": obj.get("sku"),
        "product_kind": obj.get("product_kind"),
        "category": obj.get("category"),
        "revenue_account_code": obj.get("revenue_account_code"),
        "cogs_account_code": obj.get("cogs_account_code"),
        "inventory_account_code": obj.get("inventory_account_code"),
        "tax_account_code": obj.get("tax_account_code"),
        "discount_account_code": obj.get("discount_account_code"),
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
    p = argparse.ArgumentParser(description="Ingest product-account feedback into canonical JSONL memory.")
    p.add_argument("--input-file", action="append", required=True, help="CSV/JSONL source file (repeatable)")
    p.add_argument("--out-jsonl", required=True, help="Canonical feedback memory JSONL")
    p.add_argument("--report-out", default=None)
    return p


def main() -> int:
    args = _build_parser().parse_args()

    out_path = Path(args.out_jsonl)
    lock_path = _lock_path(out_path)

    candidates_by_file: dict[str, list[tuple[str, dict[str, Any]]]] = {}
    duplicate_in_batch_by_file: dict[str, int] = {}
    candidate_seen: set[str] = set()
    scanned = 0
    skipped = 0
    duplicate = 0
    skipped_by_file: dict[str, int] = {}
    scanned_by_file: dict[str, int] = {}

    for input_file in args.input_file:
        src = Path(input_file)
        if not src.exists() or not src.is_file():
            raise SystemExit(f"Input not found: {src}")

        records = _load_records(src)
        src_key = src.as_posix()
        candidates_by_file.setdefault(src_key, [])
        duplicate_in_batch_by_file.setdefault(src_key, 0)
        skipped_by_file.setdefault(src_key, 0)
        scanned_by_file[src_key] = len(records)

        for rec in records:
            scanned += 1
            if not isinstance(rec, dict):
                skipped += 1
                skipped_by_file[src_key] += 1
                continue
            norm = _normalize_record(rec, source=src.name)
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
    duplicate_existing_by_file: dict[str, int] = {k: 0 for k in candidates_by_file}
    accepted_by_file: dict[str, int] = {k: 0 for k in candidates_by_file}

    with _exclusive_lock(lock_path):
        existing_keys = _load_existing_keys(out_path)
        for src_key in args.input_file:
            src = Path(src_key).as_posix()
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
    for src_key in args.input_file:
        src = Path(src_key).as_posix()
        per_file.append(
            {
                "input_file": src,
                "rows_scanned": int(scanned_by_file.get(src, 0)),
                "rows_accepted": int(accepted_by_file.get(src, 0)),
                "rows_skipped": int(skipped_by_file.get(src, 0)),
                "rows_duplicate_in_batch": int(duplicate_in_batch_by_file.get(src, 0)),
                "rows_duplicate_existing": int(duplicate_existing_by_file.get(src, 0)),
            }
        )

    report = {
        "schema_version": "v0",
        "out_jsonl": out_path.as_posix(),
        "lock_file": lock_path.as_posix(),
        "rows_scanned": scanned,
        "rows_accepted": len(accepted),
        "rows_skipped": skipped,
        "rows_duplicate": duplicate,
        "files": per_file,
    }

    if args.report_out:
        rpt = Path(args.report_out)
        rpt.parent.mkdir(parents=True, exist_ok=True)
        rpt.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
