#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _up(value: Any) -> str:
    return str(value or "").strip().upper()


def _family(row: dict[str, str]) -> str:
    typ = _up(row.get("type"))
    doc = _up(row.get("doc_type"))
    ref = _up(row.get("reference"))
    status = _up(row.get("doc_status"))
    text = " ".join(
        [
            typ,
            doc,
            ref,
            status,
            _up(row.get("party")),
            _up(row.get("memo")),
            _up(row.get("notes")),
            _up(row.get("journal_lines")),
            _up(row.get("payment_method")),
            _up(row.get("gst_treatment")),
        ]
    )

    if typ == "PERIOD_LOCK" or "PERIOD_LOCK" in text or "LOCK" in text:
        return "period_close"
    if "PAYROLL" in text or "SALARY" in text or "WAGES" in text:
        return "payroll"
    if "OVERRIDE" in text or "APPROVAL" in text or "EXCEPTION" in text:
        return "approvals_override"
    if "RETURN" in text or "REVERS" in text or typ.endswith("_RETURN"):
        return "returns_reversal"
    if typ in {"TAX_SETTLEMENT", "GST_SETTLEMENT"}:
        return "tax_settlement"
    if "GST" in text and ("SETTLEMENT" in text or "TAX_PAYMENT" in text or "LIABILITY" in text):
        return "tax_settlement"
    if "SETTLEMENT_SPLIT" in text or ("SETTLEMENT" in text and "SPLIT" in text):
        return "banking_settlement"
    if typ == "PAYMENT":
        if "TAX" in text or "GST" in text:
            return "tax_settlement"
        if "SUPPLIER" in text or "VENDOR" in text or "PURCHASE" in text:
            return "p2p"
        if "DEALER" in text or "CUSTOMER" in text or "SALE" in text or "INVOICE" in text:
            return "o2c"
        return "banking_settlement"
    if typ in {"PURCHASE", "GRN"} or "PURCHASE" in text or "RAW_MATERIAL_PURCHASE" in text:
        return "p2p"
    if typ in {"SALE"} or "SALE" in text or "INVOICE" in text or "DEALER_RECEIPT" in text:
        return "o2c"
    if typ in {"WRITE_OFF", "INVENTORY_COUNT", "COGS"}:
        return "inventory_production"
    if "INVENTORY" in text or "PRODUCTION" in text or "DISPATCH" in text:
        return "inventory_production"
    return "general"


def _load_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise SystemExit(f"Missing CSV header: {path}")
        rows = [dict(r) for r in reader]
    return list(reader.fieldnames), rows


def _write_rows(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _load_profile(path: Path) -> dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    families = obj.get("families")
    if not isinstance(families, dict) or not families:
        raise SystemExit(f"Invalid flow profile; missing families: {path}")
    return obj


def _target_counts(
    *,
    total: int,
    available_families: set[str],
    weights: dict[str, float],
    default_weight: float,
) -> dict[str, int]:
    effective_weights: dict[str, float] = {}
    for fam in sorted(available_families):
        raw = weights.get(fam, default_weight)
        w = float(raw) if raw is not None else float(default_weight)
        effective_weights[fam] = max(0.0, w)
    weight_sum = sum(effective_weights.values())
    if weight_sum <= 0:
        equal = 1.0 / max(1, len(effective_weights))
        effective_weights = {k: equal for k in effective_weights}
        weight_sum = 1.0

    raw_targets: dict[str, float] = {k: total * (v / weight_sum) for k, v in effective_weights.items()}
    base: dict[str, int] = {k: int(math.floor(v)) for k, v in raw_targets.items()}
    remainder = total - sum(base.values())
    if remainder > 0:
        order = sorted(raw_targets.keys(), key=lambda k: (raw_targets[k] - base[k]), reverse=True)
        for fam in order[:remainder]:
            base[fam] += 1
    return base


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Rebalance synthetic ERP training CSV by ERP v2 workflow families.")
    p.add_argument("--input-csv", required=True)
    p.add_argument("--output-csv", required=True)
    p.add_argument("--profile-json", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--inject-missing-families", action="store_true", default=True)
    p.add_argument("--no-inject-missing-families", dest="inject_missing_families", action="store_false")
    p.add_argument("--report-out", default=None)
    return p


def _make_row(
    *,
    fieldnames: list[str],
    base_row: dict[str, str],
    overrides: dict[str, Any],
) -> dict[str, str]:
    out = dict(base_row)
    for k, v in overrides.items():
        if k in fieldnames:
            out[k] = str(v)
    return out


def _synthesize_rows_for_family(
    *,
    family: str,
    count: int,
    fieldnames: list[str],
    rows: list[dict[str, str]],
    rng: random.Random,
) -> list[dict[str, str]]:
    if count <= 0:
        return []
    if not rows:
        return []

    out: list[dict[str, str]] = []
    for i in range(count):
        base = dict(rng.choice(rows))
        amount = round(rng.uniform(1500.0, 25000.0), 2)
        if family == "o2c":
            tax_rate = 18.0
            tax_amount = round(amount * tax_rate / 100.0, 2)
            gross = round(amount + tax_amount, 2)
            row = _make_row(
                fieldnames=fieldnames,
                base_row=base,
                overrides={
                    "type": "SALE",
                    "reference": f"INV-SYN-O2C-{i + 1:06d}",
                    "doc_type": "INVOICE",
                    "doc_status": "POSTED",
                    "party": "DEALER_SYNTH",
                    "notes": "synthetic o2c dispatch invoice flow",
                    "memo": "dealer invoice with gst output",
                    "payment_method": "BANK_TRANSFER",
                    "gst_treatment": "TAXABLE",
                    "gst_inclusive": "false",
                    "qty": "10",
                    "price": f"{amount:.2f}",
                    "cost": f"{round(amount * 0.72, 2):.2f}",
                    "tax_rate": f"{tax_rate:.2f}",
                    "journal_lines": (
                        f"ACCOUNTS_RECEIVABLE | D | {gross:.2f} | dealer receivable || "
                        f"SALES | C | {amount:.2f} | invoice revenue || "
                        f"GST_OUTPUT | C | {tax_amount:.2f} | output gst"
                    ),
                },
            )
            out.append(row)
            continue

        if family == "approvals_override":
            row = _make_row(
                fieldnames=fieldnames,
                base_row=base,
                overrides={
                    "type": "PAYMENT",
                    "reference": f"OVR-SYN-{i + 1:06d}",
                    "doc_type": "JOURNAL_ENTRY",
                    "doc_status": "POSTED",
                    "party": "SYSTEM",
                    "notes": "manual override approved by accounting manager",
                    "memo": "override approval required for manual adjustment",
                    "payment_method": "BANK_TRANSFER",
                    "gst_treatment": "NON_GST",
                    "gst_inclusive": "false",
                    "qty": "",
                    "price": "",
                    "cost": "",
                    "tax_rate": "0",
                    "journal_lines": (
                        f"EXPENSE_MISC | D | {amount:.2f} | manual override adjustment || "
                        f"CASH | C | {amount:.2f} | approved override settlement"
                    ),
                },
            )
            out.append(row)
            continue

        # Fallback for any future family without explicit template.
        out.append(base)

    return out


def main() -> int:
    args = _build_parser().parse_args()
    rng = random.Random(int(args.seed))

    in_path = Path(args.input_csv)
    out_path = Path(args.output_csv)
    profile_path = Path(args.profile_json)

    fieldnames, rows = _load_rows(in_path)
    if not rows:
        _write_rows(out_path, fieldnames, rows)
        report = {
            "schema_version": "v0",
            "status": "empty_input",
            "input_csv": in_path.as_posix(),
            "output_csv": out_path.as_posix(),
            "profile_json": profile_path.as_posix(),
            "seed": int(args.seed),
        }
        if args.report_out:
            rp = Path(args.report_out)
            rp.parent.mkdir(parents=True, exist_ok=True)
            rp.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps(report, sort_keys=True))
        return 0

    profile = _load_profile(profile_path)
    weights = {str(k): float(v) for k, v in dict(profile.get("families") or {}).items()}
    default_weight = float(profile.get("default_weight", 0.02))

    buckets: dict[str, list[dict[str, str]]] = defaultdict(list)
    before_family = Counter()
    for row in rows:
        fam = _family(row)
        before_family[fam] += 1
        buckets[fam].append(row)

    available_families = set(buckets.keys())
    if args.inject_missing_families:
        available_families |= set(weights.keys())

    target = _target_counts(
        total=len(rows),
        available_families=available_families,
        weights=weights,
        default_weight=default_weight,
    )

    rebalanced: list[dict[str, str]] = []
    injected_by_family: dict[str, int] = {}
    for fam, need in target.items():
        source = buckets.get(fam, [])
        if need <= 0:
            continue
        if not source:
            if args.inject_missing_families:
                injected = _synthesize_rows_for_family(
                    family=fam,
                    count=need,
                    fieldnames=fieldnames,
                    rows=rows,
                    rng=rng,
                )
                if injected:
                    rebalanced.extend(injected)
                    injected_by_family[fam] = len(injected)
            continue
        if need <= len(source):
            pick = rng.sample(source, k=need)
        else:
            pick = list(source)
            pick.extend(rng.choice(source) for _ in range(need - len(source)))
        rebalanced.extend(pick)

    if len(rebalanced) < len(rows):
        deficit = len(rows) - len(rebalanced)
        rebalanced.extend(rng.choice(rows) for _ in range(deficit))
    elif len(rebalanced) > len(rows):
        rng.shuffle(rebalanced)
        rebalanced = rebalanced[: len(rows)]

    rng.shuffle(rebalanced)
    _write_rows(out_path, fieldnames, rebalanced)

    after_family = Counter(_family(r) for r in rebalanced)
    missing_profile_families = sorted([f for f in weights.keys() if before_family.get(f, 0) == 0])

    report = {
        "schema_version": "v0",
        "status": "ok",
        "input_csv": in_path.as_posix(),
        "output_csv": out_path.as_posix(),
        "profile_json": profile_path.as_posix(),
        "profile_name": profile.get("profile_name"),
        "seed": int(args.seed),
        "rows": {
            "before": len(rows),
            "after": len(rebalanced),
        },
        "family_counts_before": dict(sorted(before_family.items())),
        "family_counts_after": dict(sorted(after_family.items())),
        "target_counts": dict(sorted(target.items())),
        "missing_profile_families_in_source": missing_profile_families,
        "synthetic_rows_injected_by_family": dict(sorted(injected_by_family.items())),
        "inject_missing_families": bool(args.inject_missing_families),
    }

    if args.report_out:
        rp = Path(args.report_out)
        rp.parent.mkdir(parents=True, exist_ok=True)
        rp.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
