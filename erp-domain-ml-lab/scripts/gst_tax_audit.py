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

OUTPUT_TAX_ACCOUNTS = {"GST_OUTPUT", "TAX_PAYABLE"}
INPUT_TAX_ACCOUNTS = {"GST_INPUT"}
TAX_ACCOUNT_UNION = OUTPUT_TAX_ACCOUNTS | INPUT_TAX_ACCOUNTS
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


def _labels(rows: list[dict[str, Any]]) -> list[str]:
    out: list[str] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        label = _as_str(row.get("label")).upper()
        if label:
            out.append(label)
    return out


def _first_label(rows: list[dict[str, Any]]) -> str:
    labels = _labels(rows)
    if not labels:
        return ""
    return labels[0]


def _has_any(labels: list[str], expected: set[str]) -> bool:
    return any(label in expected for label in labels)


def _predicted_label(tx_row: dict[str, Any]) -> str:
    direct = _as_str(tx_row.get("suggested_label")).upper()
    if direct:
        return direct
    preds = list(tx_row.get("predictions") or [])
    if preds and isinstance(preds[0], dict):
        return _as_str(preds[0].get("label")).upper()
    return ""


def _issue(
    *,
    code: str,
    severity: str,
    reason: str,
    tx_row: dict[str, Any],
    coa_row: dict[str, Any],
) -> dict[str, Any]:
    record = dict(tx_row.get("record") or {})
    debit_top = list(coa_row.get("debit_recommendations") or [])
    credit_top = list(coa_row.get("credit_recommendations") or [])
    return {
        "schema_version": "v0",
        "issue_code": code,
        "severity": severity,
        "reason": reason,
        "example_id": _as_str(tx_row.get("example_id")),
        "reference_number": _as_str(record.get("referenceNumber")),
        "record_id": record.get("id"),
        "entry_date": _as_str(record.get("entryDate")),
        "predicted_label": _predicted_label(tx_row),
        "debit_top1": _first_label(debit_top),
        "credit_top1": _first_label(credit_top),
        "debit_topk": _labels(debit_top),
        "credit_topk": _labels(credit_top),
        "policy_bands_top1": {
            "debit": _as_str((debit_top[0] if debit_top else {}).get("policy_band"), "unknown").lower(),
            "credit": _as_str((credit_top[0] if credit_top else {}).get("policy_band"), "unknown").lower(),
        },
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True))
            f.write("\n")


def _safe_rate(n: int, d: int) -> float:
    if d <= 0:
        return 0.0
    return float(n) / float(d)


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="GST/tax accounting audit for tx+coa advisory outputs")
    p.add_argument("--tx-jsonl", required=True)
    p.add_argument("--coa-jsonl", required=True)
    p.add_argument("--out-summary-json", required=True)
    p.add_argument("--out-issues-jsonl", required=True)
    p.add_argument("--max-major-fail-rate", type=float, default=0.02)
    p.add_argument("--max-critical-issues", type=int, default=0)
    return p


def main() -> int:
    args = _parser().parse_args()

    tx_path = Path(args.tx_jsonl)
    coa_path = Path(args.coa_jsonl)
    summary_path = Path(args.out_summary_json)
    issues_path = Path(args.out_issues_jsonl)

    if not tx_path.exists() or not tx_path.is_file():
        raise SystemExit(f"Missing transaction output: {tx_path}")
    if not coa_path.exists() or not coa_path.is_file():
        raise SystemExit(f"Missing coa output: {coa_path}")

    tx_rows = _read_jsonl(tx_path)
    coa_rows = _read_jsonl(coa_path)
    coa_by_example = {_as_str(r.get("example_id")): r for r in coa_rows if _as_str(r.get("example_id"))}

    issues: list[dict[str, Any]] = []
    check_count: Counter[str] = Counter()
    check_fail: Counter[str] = Counter()
    severity_count: Counter[str] = Counter()
    label_examples: Counter[str] = Counter()
    label_issues: Counter[str] = Counter()

    audited = 0
    matched = 0

    for tx_row in tx_rows:
        example_id = _as_str(tx_row.get("example_id"))
        if not example_id:
            continue
        audited += 1
        coa_row = coa_by_example.get(example_id)
        if coa_row is None:
            continue
        matched += 1

        numeric = dict(tx_row.get("numeric") or {})
        label = _predicted_label(tx_row)
        if label:
            label_examples[label] += 1

        debit_rows = list(coa_row.get("debit_recommendations") or [])
        credit_rows = list(coa_row.get("credit_recommendations") or [])
        debit_labels = _labels(debit_rows)
        credit_labels = _labels(credit_rows)
        debit_top1 = _first_label(debit_rows)
        credit_top1 = _first_label(credit_rows)

        taxable = (
            _as_float(numeric.get("gst_treatment_taxable"), 0.0) > 0.5
            or _as_float(numeric.get("has_tax_line"), 0.0) > 0.5
            or _as_float(numeric.get("tax_rate"), 0.0) > 0.0
        )
        explicit_non_gst = (
            _as_float(numeric.get("gst_treatment_non_gst"), 0.0) > 0.5
            or _as_float(numeric.get("gst_treatment_exempt"), 0.0) > 0.5
        )
        no_tax_line = _as_float(numeric.get("has_tax_line"), 0.0) <= 0.5 and _as_float(numeric.get("tax_rate"), 0.0) <= 0.0
        non_gst = explicit_non_gst and no_tax_line
        tax_payment_context = (
            _as_float(numeric.get("doc_type_tax_payment"), 0.0) > 0.5
            or _as_float(numeric.get("workflow_tax_settlement"), 0.0) > 0.5
            or label in TAX_SETTLEMENT_LABELS
        )
        rcm_hint = "rcm" in _as_str(tx_row.get("text")).lower()

        # 1) Taxable sale should surface GST output liability on credit side top-k.
        if label in SALE_LABELS and taxable:
            code = "SALE_TAX_OUTPUT_MISSING_CREDIT_TOPK"
            check_count[code] += 1
            if not _has_any(credit_labels, OUTPUT_TAX_ACCOUNTS):
                check_fail[code] += 1
                row = _issue(
                    code=code,
                    severity="major",
                    reason="Taxable sale-like transaction has no GST output/tax payable in credit top-k.",
                    tx_row=tx_row,
                    coa_row=coa_row,
                )
                issues.append(row)
                severity_count[row["severity"]] += 1
                if label:
                    label_issues[label] += 1

        # 2) Taxable purchase should surface GST input on debit side top-k.
        if label in PURCHASE_LABELS and taxable:
            code = "PURCHASE_TAX_INPUT_MISSING_DEBIT_TOPK"
            check_count[code] += 1
            if not _has_any(debit_labels, INPUT_TAX_ACCOUNTS):
                check_fail[code] += 1
                row = _issue(
                    code=code,
                    severity="major",
                    reason="Taxable purchase-like transaction has no GST input credit in debit top-k.",
                    tx_row=tx_row,
                    coa_row=coa_row,
                )
                issues.append(row)
                severity_count[row["severity"]] += 1
                if label:
                    label_issues[label] += 1

        # 3) Non-GST records should not prioritize GST accounts on top-1.
        if non_gst:
            code = "NON_GST_TOP1_POINTS_TO_TAX_ACCOUNT"
            check_count[code] += 1
            if debit_top1 in TAX_ACCOUNT_UNION or credit_top1 in TAX_ACCOUNT_UNION:
                check_fail[code] += 1
                row = _issue(
                    code=code,
                    severity="minor",
                    reason="Non-GST/exempt context still has GST/tax account at top-1.",
                    tx_row=tx_row,
                    coa_row=coa_row,
                )
                issues.append(row)
                severity_count[row["severity"]] += 1
                if label:
                    label_issues[label] += 1

        # 4) Tax payment/settlement should include liability and bank/cash hints.
        if tax_payment_context:
            code_liab = "TAX_SETTLEMENT_MISSING_LIABILITY_SIDE"
            code_liq = "TAX_SETTLEMENT_MISSING_BANK_CASH_SIDE"
            check_count[code_liab] += 1
            check_count[code_liq] += 1
            if not (_has_any(debit_labels, OUTPUT_TAX_ACCOUNTS) or _has_any(credit_labels, OUTPUT_TAX_ACCOUNTS)):
                check_fail[code_liab] += 1
                row = _issue(
                    code=code_liab,
                    severity="major",
                    reason="Tax settlement context does not include tax liability account in top-k.",
                    tx_row=tx_row,
                    coa_row=coa_row,
                )
                issues.append(row)
                severity_count[row["severity"]] += 1
                if label:
                    label_issues[label] += 1
            if not (_has_any(debit_labels, LIQUID_ACCOUNTS) or _has_any(credit_labels, LIQUID_ACCOUNTS)):
                check_fail[code_liq] += 1
                row = _issue(
                    code=code_liq,
                    severity="major",
                    reason="Tax settlement context does not include bank/cash account in top-k.",
                    tx_row=tx_row,
                    coa_row=coa_row,
                )
                issues.append(row)
                severity_count[row["severity"]] += 1
                if label:
                    label_issues[label] += 1

        # 5) Reverse-charge hint should still include tax payable/output side.
        if rcm_hint and taxable:
            code = "RCM_HINT_WITHOUT_TAX_PAYABLE_SIGNAL"
            check_count[code] += 1
            if not (_has_any(debit_labels, OUTPUT_TAX_ACCOUNTS) or _has_any(credit_labels, OUTPUT_TAX_ACCOUNTS)):
                check_fail[code] += 1
                row = _issue(
                    code=code,
                    severity="minor",
                    reason="Narrative suggests RCM, but no GST output/tax payable account appears in top-k.",
                    tx_row=tx_row,
                    coa_row=coa_row,
                )
                issues.append(row)
                severity_count[row["severity"]] += 1
                if label:
                    label_issues[label] += 1

        # 6) Policy invalid top-1 should never appear.
        debit_band = _as_str((debit_rows[0] if debit_rows else {}).get("policy_band"), "unknown").lower()
        credit_band = _as_str((credit_rows[0] if credit_rows else {}).get("policy_band"), "unknown").lower()
        if debit_band == "blocked" or credit_band == "blocked":
            code = "POLICY_BLOCKED_TOP1"
            check_count[code] += 1
            check_fail[code] += 1
            row = _issue(
                code=code,
                severity="critical",
                reason="Top-1 recommendation is in blocked policy band.",
                tx_row=tx_row,
                coa_row=coa_row,
            )
            issues.append(row)
            severity_count[row["severity"]] += 1
            if label:
                label_issues[label] += 1

    check_rows: list[dict[str, Any]] = []
    for code in sorted(check_count.keys()):
        applied = int(check_count[code])
        failed = int(check_fail[code])
        check_rows.append(
            {
                "check_code": code,
                "applied_examples": applied,
                "failed_examples": failed,
                "fail_rate": _safe_rate(failed, applied),
            }
        )

    major_issues = int(severity_count.get("major", 0))
    critical_issues = int(severity_count.get("critical", 0))
    major_fail_base = sum(int(c["applied_examples"]) for c in check_rows if c["check_code"] != "POLICY_BLOCKED_TOP1")
    major_fail_rate = _safe_rate(major_issues, major_fail_base)

    passed = (critical_issues <= int(args.max_critical_issues)) and (major_fail_rate <= float(args.max_major_fail_rate))

    by_label = {
        label: {
            "examples": int(label_examples[label]),
            "issues": int(label_issues[label]),
            "issue_rate": _safe_rate(int(label_issues[label]), int(label_examples[label])),
        }
        for label in sorted(label_examples.keys())
    }

    examples_with_issues = len({_as_str(i.get("example_id")) for i in issues if _as_str(i.get("example_id"))})
    summary = {
        "schema_version": "v0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "tx_jsonl": tx_path.as_posix(),
            "coa_jsonl": coa_path.as_posix(),
        },
        "thresholds": {
            "max_major_fail_rate": float(args.max_major_fail_rate),
            "max_critical_issues": int(args.max_critical_issues),
        },
        "counts": {
            "audited_examples": int(audited),
            "matched_tx_coa_examples": int(matched),
            "issues_total": int(len(issues)),
            "examples_with_issues": int(examples_with_issues),
            "critical_issues": critical_issues,
            "major_issues": major_issues,
            "minor_issues": int(severity_count.get("minor", 0)),
        },
        "rates": {
            "examples_with_issues_rate": _safe_rate(examples_with_issues, matched),
            "major_fail_rate": major_fail_rate,
        },
        "gate": {
            "passed": bool(passed),
            "reasons": []
            if passed
            else [
                (
                    f"critical_issues={critical_issues} exceeds limit {int(args.max_critical_issues)}"
                    if critical_issues > int(args.max_critical_issues)
                    else f"major_fail_rate={major_fail_rate:.6f} exceeds limit {float(args.max_major_fail_rate):.6f}"
                )
            ],
        },
        "check_results": check_rows,
        "issue_counts_by_code": dict(sorted(Counter(i["issue_code"] for i in issues).items(), key=lambda t: t[0])),
        "issue_counts_by_severity": dict(sorted(severity_count.items(), key=lambda t: t[0])),
        "by_predicted_label": by_label,
        "outputs": {
            "issues_jsonl": issues_path.as_posix(),
            "summary_json": summary_path.as_posix(),
        },
        "notes": [
            "This is a policy-oriented audit over advisory outputs, not legal/tax filing advice.",
            "Checks are tuned for synthetic ERP workflows and should be calibrated with accountant feedback.",
        ],
    }

    _write_jsonl(issues_path, issues)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(summary_path.as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
