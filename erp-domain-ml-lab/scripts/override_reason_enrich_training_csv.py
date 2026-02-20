#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _clean_upper(value: Any) -> str:
    return _clean_text(value).upper()


def _compress_ws(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _append_hint(existing: str, hint: str) -> str:
    base = _compress_ws(existing)
    if not hint:
        return base
    if hint in base:
        return base
    if not base:
        return hint
    return f"{base} | {hint}"


def _safe_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or not path.is_file():
        return []
    out: list[dict[str, Any]] = []
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
                out.append(obj)
    return out


def _dominant(values: list[str]) -> str:
    if not values:
        return ""
    ctr = Counter(v for v in values if v)
    if not ctr:
        return ""
    return ctr.most_common(1)[0][0]


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Enrich ERP training CSV with override reason memory.")
    p.add_argument("--training-csv", required=True)
    p.add_argument("--override-jsonl", required=True)
    p.add_argument("--out-csv", required=True)
    p.add_argument("--report-out", default=None)
    p.add_argument("--enable-type-fallback", action="store_true")
    p.add_argument("--max-reason-text-chars", type=int, default=120)
    return p


def main() -> int:
    args = _build_parser().parse_args()

    training_csv = Path(args.training_csv)
    override_jsonl = Path(args.override_jsonl)
    out_csv = Path(args.out_csv)
    report_out = Path(args.report_out) if args.report_out else None

    if not training_csv.exists() or not training_csv.is_file():
        raise SystemExit(f"Training CSV not found: {training_csv}")
    if not override_jsonl.exists() or not override_jsonl.is_file():
        raise SystemExit(f"Override memory JSONL not found: {override_jsonl}")

    override_rows = _safe_jsonl(override_jsonl)

    by_exact: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for obj in override_rows:
        reference = _clean_upper(obj.get("reference"))
        typ = _clean_upper(obj.get("type"))
        if typ:
            by_type[typ].append(obj)
        if reference and typ:
            by_exact[(reference, typ)].append(obj)

    total_rows = 0
    exact_matched_rows = 0
    type_fallback_rows = 0
    relabeled_rows = 0
    notes_enriched_rows = 0
    memo_enriched_rows = 0

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with training_csv.open("r", encoding="utf-8", newline="") as f_in:
        reader = csv.DictReader(f_in)
        if reader.fieldnames is None:
            raise SystemExit(f"CSV has no header: {training_csv}")

        fieldnames = list(reader.fieldnames)
        if "notes" not in fieldnames:
            fieldnames.append("notes")
        if "memo" not in fieldnames:
            fieldnames.append("memo")

        with out_csv.open("w", encoding="utf-8", newline="") as f_out:
            writer = csv.DictWriter(f_out, fieldnames=fieldnames, lineterminator="\n")
            writer.writeheader()

            for row in reader:
                total_rows += 1
                row_ref = _clean_upper(row.get("reference"))
                row_type = _clean_upper(row.get("type"))

                matches = list(by_exact.get((row_ref, row_type), []))
                used_type_fallback = False
                if not matches and args.enable_type_fallback and row_type:
                    matches = list(by_type.get(row_type, []))
                    used_type_fallback = bool(matches)

                if matches:
                    if used_type_fallback:
                        type_fallback_rows += 1
                    else:
                        exact_matched_rows += 1

                    approved_label = _dominant([_clean_upper(m.get("approved_label")) for m in matches])
                    reason_code = _dominant([_clean_upper(m.get("reason_code")) for m in matches])
                    scope = _dominant([_clean_upper(m.get("correction_scope")) for m in matches])
                    reason_text = _dominant(
                        [
                            _compress_ws(_clean_text(m.get("reason_text")))[: int(args.max_reason_text_chars)]
                            for m in matches
                            if _clean_text(m.get("reason_text"))
                        ]
                    )

                    if approved_label and approved_label != row_type:
                        row["type"] = approved_label
                        relabeled_rows += 1

                    hint_parts: list[str] = []
                    if reason_code:
                        hint_parts.append(f"override_reason_code={reason_code}")
                    if scope:
                        hint_parts.append(f"override_scope={scope}")
                    if approved_label:
                        hint_parts.append(f"override_approved_label={approved_label}")
                    if reason_text:
                        hint_parts.append(f"override_reason_text={reason_text}")
                    hint = "; ".join(hint_parts)

                    if hint:
                        old_notes = _clean_text(row.get("notes"))
                        new_notes = _append_hint(old_notes, hint)
                        if new_notes != old_notes:
                            notes_enriched_rows += 1
                        row["notes"] = new_notes

                        old_memo = _clean_text(row.get("memo"))
                        new_memo = _append_hint(old_memo, hint)
                        if new_memo != old_memo:
                            memo_enriched_rows += 1
                        row["memo"] = new_memo

                writer.writerow(row)

    report = {
        "schema_version": "v0",
        "training_csv": training_csv.as_posix(),
        "override_jsonl": override_jsonl.as_posix(),
        "out_csv": out_csv.as_posix(),
        "override_rows_loaded": len(override_rows),
        "rows_total": total_rows,
        "rows_exact_match": exact_matched_rows,
        "rows_type_fallback_match": type_fallback_rows,
        "rows_relabeled": relabeled_rows,
        "rows_notes_enriched": notes_enriched_rows,
        "rows_memo_enriched": memo_enriched_rows,
    }

    if report_out is not None:
        report_out.parent.mkdir(parents=True, exist_ok=True)
        report_out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
