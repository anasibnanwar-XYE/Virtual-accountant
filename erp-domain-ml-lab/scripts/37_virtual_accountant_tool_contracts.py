#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


TOOL_IDS = (
    "categorize_transaction",
    "detect_anomalies",
    "forecast_cashflow",
    "generate_monthly_summary",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        raise SystemExit(f"Invalid JSON: {path} ({e})")


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(obj, indent=2, sort_keys=True)
    path.write_text(payload + "\n", encoding="utf-8")


def _sha256_obj(obj: Any) -> str:
    raw = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _require_dict(payload: Any, name: str, errors: list[str]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        errors.append(f"{name} must be object")
        return {}
    return dict(payload)


def _require_str(payload: dict[str, Any], key: str, errors: list[str]) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        errors.append(f"{key} must be non-empty string")
        return ""
    return value


def _require_number(payload: dict[str, Any], key: str, errors: list[str]) -> float:
    value = payload.get(key)
    if not _is_number(value):
        errors.append(f"{key} must be number")
        return 0.0
    return float(value)


def _require_list(payload: dict[str, Any], key: str, errors: list[str]) -> list[Any]:
    value = payload.get(key)
    if not isinstance(value, list):
        errors.append(f"{key} must be array")
        return []
    return list(value)


def _validate_probability(value: float, field: str, errors: list[str]) -> None:
    if value < 0.0 or value > 1.0:
        errors.append(f"{field} must be in [0,1]")


def validate_categorize_transaction(payload: Any) -> list[str]:
    errors: list[str] = []
    obj = _require_dict(payload, "payload", errors)
    _require_str(obj, "task", errors)

    prediction = _require_dict(obj.get("prediction"), "prediction", errors)
    _require_str(prediction, "category", errors)
    _require_str(prediction, "code", errors)
    confidence = _require_number(prediction, "confidence", errors)
    _validate_probability(confidence, "prediction.confidence", errors)

    top_k = _require_list(obj, "top_k", errors)
    if not top_k:
        errors.append("top_k must contain at least one candidate")
    for idx, item in enumerate(top_k):
        row = _require_dict(item, f"top_k[{idx}]", errors)
        _require_str(row, "category", errors)
        _require_str(row, "code", errors)
        score = _require_number(row, "confidence", errors)
        _validate_probability(score, f"top_k[{idx}].confidence", errors)

    reason_codes = _require_list(obj, "reason_codes", errors)
    for idx, reason in enumerate(reason_codes):
        if not isinstance(reason, str) or not reason.strip():
            errors.append(f"reason_codes[{idx}] must be non-empty string")

    warnings = _require_list(obj, "warnings", errors)
    for idx, warning in enumerate(warnings):
        if not isinstance(warning, str):
            errors.append(f"warnings[{idx}] must be string")

    return errors


def validate_detect_anomalies(payload: Any) -> list[str]:
    errors: list[str] = []
    obj = _require_dict(payload, "payload", errors)
    _require_str(obj, "task", errors)
    _require_str(obj, "window", errors)

    anomalies = _require_list(obj, "anomalies", errors)
    for idx, item in enumerate(anomalies):
        row = _require_dict(item, f"anomalies[{idx}]", errors)
        _require_str(row, "reference", errors)
        _require_str(row, "severity", errors)
        score = _require_number(row, "score", errors)
        _validate_probability(score, f"anomalies[{idx}].score", errors)
        reasons = _require_list(row, "reason_codes", errors)
        if not reasons:
            errors.append(f"anomalies[{idx}].reason_codes must not be empty")

    total = obj.get("total_anomalies")
    if not isinstance(total, int) or total < 0:
        errors.append("total_anomalies must be non-negative integer")

    warnings = _require_list(obj, "warnings", errors)
    for idx, warning in enumerate(warnings):
        if not isinstance(warning, str):
            errors.append(f"warnings[{idx}] must be string")

    return errors


def validate_forecast_cashflow(payload: Any) -> list[str]:
    errors: list[str] = []
    obj = _require_dict(payload, "payload", errors)
    _require_str(obj, "task", errors)
    _require_str(obj, "currency", errors)

    horizon = obj.get("horizon_days")
    if not isinstance(horizon, int) or horizon <= 0:
        errors.append("horizon_days must be positive integer")

    points = _require_list(obj, "forecast_points", errors)
    if not points:
        errors.append("forecast_points must not be empty")
    for idx, item in enumerate(points):
        row = _require_dict(item, f"forecast_points[{idx}]", errors)
        _require_str(row, "date", errors)
        _require_number(row, "cash_in", errors)
        _require_number(row, "cash_out", errors)
        _require_number(row, "net_cash", errors)
        conf = _require_number(row, "confidence", errors)
        _validate_probability(conf, f"forecast_points[{idx}].confidence", errors)

    warnings = _require_list(obj, "warnings", errors)
    for idx, warning in enumerate(warnings):
        if not isinstance(warning, str):
            errors.append(f"warnings[{idx}] must be string")

    return errors


def validate_generate_monthly_summary(payload: Any) -> list[str]:
    errors: list[str] = []
    obj = _require_dict(payload, "payload", errors)
    _require_str(obj, "task", errors)
    _require_str(obj, "period", errors)
    _require_str(obj, "currency", errors)

    kpis = _require_dict(obj.get("kpis"), "kpis", errors)
    for key in ("revenue", "expense", "gross_profit", "net_profit", "gst_payable"):
        _require_number(kpis, key, errors)

    highlights = _require_list(obj, "highlights", errors)
    if not highlights:
        errors.append("highlights must not be empty")
    for idx, line in enumerate(highlights):
        if not isinstance(line, str) or not line.strip():
            errors.append(f"highlights[{idx}] must be non-empty string")

    evidence = _require_list(obj, "evidence_refs", errors)
    for idx, ref in enumerate(evidence):
        if not isinstance(ref, str) or not ref.strip():
            errors.append(f"evidence_refs[{idx}] must be non-empty string")

    warnings = _require_list(obj, "warnings", errors)
    for idx, warning in enumerate(warnings):
        if not isinstance(warning, str):
            errors.append(f"warnings[{idx}] must be string")

    return errors


VALIDATORS: dict[str, Callable[[Any], list[str]]] = {
    "categorize_transaction": validate_categorize_transaction,
    "detect_anomalies": validate_detect_anomalies,
    "forecast_cashflow": validate_forecast_cashflow,
    "generate_monthly_summary": validate_generate_monthly_summary,
}


SAMPLE_PAYLOADS: dict[str, dict[str, Any]] = {
    "categorize_transaction": {
        "task": "expense_category",
        "prediction": {"category": "Office Supplies", "code": "EXP_OFFICE_SUP", "confidence": 0.82},
        "top_k": [
            {"category": "Office Supplies", "code": "EXP_OFFICE_SUP", "confidence": 0.82},
            {"category": "Repairs & Maintenance", "code": "EXP_REPAIR", "confidence": 0.11},
            {"category": "Software Subscriptions", "code": "EXP_SOFTWARE", "confidence": 0.07},
        ],
        "reason_codes": ["VENDOR_MATCH_OFFICE", "ITEM_KEYWORD_STATIONERY"],
        "warnings": [],
    },
    "detect_anomalies": {
        "task": "transaction_anomaly_scan",
        "window": "2026-02",
        "total_anomalies": 2,
        "anomalies": [
            {
                "reference": "JE-2026-02-000145",
                "severity": "HIGH",
                "score": 0.93,
                "reason_codes": ["DUPLICATE_INVOICE_PATTERN", "AMOUNT_OUTLIER"],
            },
            {
                "reference": "JE-2026-02-000233",
                "severity": "MEDIUM",
                "score": 0.71,
                "reason_codes": ["NEW_VENDOR_HIGH_VALUE"],
            },
        ],
        "warnings": [],
    },
    "forecast_cashflow": {
        "task": "cashflow_forecast",
        "currency": "INR",
        "horizon_days": 30,
        "forecast_points": [
            {"date": "2026-03-01", "cash_in": 275000.0, "cash_out": 228000.0, "net_cash": 47000.0, "confidence": 0.84},
            {"date": "2026-03-02", "cash_in": 194000.0, "cash_out": 176500.0, "net_cash": 17500.0, "confidence": 0.82},
        ],
        "warnings": ["FORECAST_BASED_ON_SYNTHETIC_HISTORY"],
    },
    "generate_monthly_summary": {
        "task": "monthly_close_summary",
        "period": "2026-02",
        "currency": "INR",
        "kpis": {
            "revenue": 2850000.0,
            "expense": 2165000.0,
            "gross_profit": 685000.0,
            "net_profit": 492000.0,
            "gst_payable": 138500.0,
        },
        "highlights": [
            "Revenue up 8.2% month-over-month led by finished goods sales.",
            "GST payable increased due to higher taxable sales mix.",
        ],
        "evidence_refs": ["tb:2026-02", "pnl:2026-02", "gst:outward:2026-02"],
        "warnings": [],
    },
}


def _validate(tool: str, payload: Any) -> dict[str, Any]:
    validator = VALIDATORS.get(tool)
    if validator is None:
        raise SystemExit(f"Unsupported tool '{tool}'. Expected one of: {', '.join(TOOL_IDS)}")
    errors = validator(payload)
    return {
        "schema_version": "v1",
        "tool": tool,
        "validated_at_utc": _utc_now(),
        "ok": len(errors) == 0,
        "error_count": len(errors),
        "errors": errors,
    }


def _cmd_sample(args: argparse.Namespace) -> int:
    payload = SAMPLE_PAYLOADS[args.tool]
    if args.out:
        _write_json(Path(args.out), payload)
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    payload = _read_json(Path(args.payload_json))
    report = _validate(args.tool, payload)
    if args.report_out:
        _write_json(Path(args.report_out), report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 2


def _cmd_audit(args: argparse.Namespace) -> int:
    request_obj = _read_json(Path(args.request_json))
    response_obj = _read_json(Path(args.response_json))
    report = _validate(args.tool, response_obj)

    status = "valid" if report["ok"] else "invalid"
    if args.status:
        status = str(args.status).strip().lower()

    event = {
        "schema_version": "v1",
        "audit_type": "virtual_accountant_tool_call",
        "logged_at_utc": _utc_now(),
        "tool": args.tool,
        "actor": str(args.actor or "system"),
        "status": status,
        "request_sha256": _sha256_obj(request_obj),
        "response_sha256": _sha256_obj(response_obj),
        "validation": report,
    }
    out = Path(args.audit_log_jsonl)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, sort_keys=True) + "\n")

    print(json.dumps({"status": status, "audit_log_jsonl": out.as_posix(), "ok": report["ok"]}, indent=2, sort_keys=True))
    return 0 if report["ok"] else 2


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Virtual accountant tool-contract validation + audit logging.")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("sample", help="Print/write sample payload for a tool")
    s.add_argument("--tool", required=True, choices=TOOL_IDS)
    s.add_argument("--out", default=None)

    v = sub.add_parser("validate", help="Validate a tool response payload")
    v.add_argument("--tool", required=True, choices=TOOL_IDS)
    v.add_argument("--payload-json", required=True)
    v.add_argument("--report-out", default=None)

    a = sub.add_parser("audit", help="Validate and append audit trail for a tool call")
    a.add_argument("--tool", required=True, choices=TOOL_IDS)
    a.add_argument("--request-json", required=True)
    a.add_argument("--response-json", required=True)
    a.add_argument("--audit-log-jsonl", required=True)
    a.add_argument("--actor", default="system")
    a.add_argument("--status", default=None)
    return p


def main() -> int:
    args = _build_parser().parse_args()
    if args.cmd == "sample":
        return _cmd_sample(args)
    if args.cmd == "validate":
        return _cmd_validate(args)
    if args.cmd == "audit":
        return _cmd_audit(args)
    raise SystemExit(f"Unsupported command: {args.cmd}")


if __name__ == "__main__":
    raise SystemExit(main())
