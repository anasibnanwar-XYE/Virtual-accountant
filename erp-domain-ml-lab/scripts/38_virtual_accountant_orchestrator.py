#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import tempfile
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from statistics import mean
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
LAB_ROOT = SCRIPT_DIR.parent
OUTPUTS_DIR = LAB_ROOT / "outputs"
CONTRACTS_SCRIPT = SCRIPT_DIR / "37_virtual_accountant_tool_contracts.py"
REVIEW_RISK_SCRIPT = SCRIPT_DIR / "review_risk_model.py"
RECON_SCRIPT = SCRIPT_DIR / "reconciliation_exception_model.py"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _detect_python_bin() -> str:
    candidates = [
        LAB_ROOT / ".venv" / "bin" / "python3",
        Path("/home/realnigga/erp-domain-ml-lab/.venv/bin/python3"),
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate.as_posix()
    return "python3"


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        raise SystemExit(f"Invalid JSON file {path}: {e}")


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        raw = str(value).strip()
        if not raw:
            return default
        return float(raw)
    except Exception:
        return default


def _parse_iso_date(value: Any) -> date | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        return date.fromisoformat(raw[:10])
    except Exception:
        return None


def _parse_journal_lines(raw: Any) -> list[tuple[str, str, float]]:
    text = str(raw or "").strip()
    if not text:
        return []
    out: list[tuple[str, str, float]] = []
    for part in text.split("||"):
        piece = part.strip()
        if not piece:
            continue
        cols = [c.strip() for c in piece.split("|")]
        if len(cols) < 3:
            continue
        account = cols[0].upper()
        drcr = cols[1].upper()
        amt = _to_float(cols[2], 0.0)
        out.append((account, drcr, amt))
    return out


def _load_rows_from_input(payload: dict[str, Any]) -> list[dict[str, Any]]:
    tx_rows = payload.get("transactions")
    if isinstance(tx_rows, list):
        return [dict(r) for r in tx_rows if isinstance(r, dict)]

    input_file = str(payload.get("input_file") or "").strip()
    if not input_file:
        raise SystemExit("Input must contain either 'transactions' array or 'input_file' path")
    path = Path(input_file)
    if not path.exists() or not path.is_file():
        raise SystemExit(f"Input file not found: {path}")
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            return [dict(r) for r in reader]
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            obj = json.loads(raw)
            if isinstance(obj, dict):
                rows.append(dict(obj))
    return rows


def _tool_categorize_transaction(payload: dict[str, Any]) -> dict[str, Any]:
    row = payload.get("transaction")
    if not isinstance(row, dict):
        rows = _load_rows_from_input(payload)
        if not rows:
            raise SystemExit("No transaction rows available for categorization")
        row = rows[0]

    typ = str(row.get("type") or "").strip().upper()
    doc_type = str(row.get("doc_type") or "").strip().upper()
    journal_entries = _parse_journal_lines(row.get("journal_lines"))
    accounts = {acc for acc, _dc, _amt in journal_entries}

    label_map = {
        "SALE": ("Revenue", "SALE"),
        "SALE_RETURN": ("Revenue Return", "SALE_RETURN"),
        "PURCHASE": ("Procurement", "PURCHASE"),
        "PAYMENT": ("Payments", "PAYMENT"),
        "SETTLEMENT_SPLIT": ("Settlement", "SETTLEMENT_SPLIT"),
        "TAX_SETTLEMENT": ("Tax Settlement", "TAX_SETTLEMENT"),
        "PAYROLL": ("Payroll", "PAYROLL"),
        "COGS": ("Cost Of Goods", "COGS"),
        "WRITE_OFF": ("Write Off", "WRITE_OFF"),
        "INVENTORY_COUNT": ("Inventory Adjustment", "INVENTORY_COUNT"),
        "PERIOD_LOCK": ("Period Lock", "PERIOD_LOCK"),
        "OPENING_BALANCE": ("Opening Balance", "OPENING_BALANCE"),
        "JOURNAL_ADJUSTMENT": ("Journal Adjustment", "JOURNAL_ADJUSTMENT"),
    }
    category, code = label_map.get(typ, ("General Accounting", "JOURNAL_ADJUSTMENT"))

    confidence = 0.55
    reason_codes: list[str] = []
    warnings: list[str] = []
    if typ in label_map:
        confidence = 0.91
        reason_codes.append("TYPE_SIGNAL_STRONG")
    if doc_type:
        reason_codes.append("DOC_TYPE_SIGNAL_PRESENT")
        confidence += 0.02
    if any("GST" in a or "TAX" in a for a in accounts):
        reason_codes.append("TAX_ACCOUNT_PRESENT")
        confidence += 0.01
    if not journal_entries and typ != "PERIOD_LOCK":
        warnings.append("MISSING_JOURNAL_LINES")
        confidence -= 0.08
    confidence = max(0.05, min(0.99, confidence))

    top_k = [
        {"category": category, "code": code, "confidence": round(confidence, 6)},
    ]
    fallbacks = [x for x in ("PAYMENT", "PURCHASE", "SALE") if x != code]
    for idx, fb in enumerate(fallbacks[:2], start=1):
        fb_cat, _fb_code = label_map.get(fb, ("General Accounting", fb))
        top_k.append({"category": fb_cat, "code": fb, "confidence": round(max(0.01, confidence - (0.2 * idx)), 6)})

    return {
        "task": "transaction_category",
        "prediction": {"category": category, "code": code, "confidence": round(confidence, 6)},
        "top_k": top_k,
        "reason_codes": sorted(set(reason_codes)) or ["WEAK_SIGNAL"],
        "warnings": warnings,
    }


def _read_pointer(pointer_name: str) -> str:
    path = OUTPUTS_DIR / pointer_name
    if not path.exists():
        raise SystemExit(f"Missing pointer file: {path}")
    value = path.read_text(encoding="utf-8").strip()
    if not value:
        raise SystemExit(f"Empty pointer file: {path}")
    return value


def _run_python(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit(f"Command failed ({proc.returncode}): {' '.join(cmd)}\n{proc.stderr.strip()}")


def _run_scoring_models(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    warnings: list[str] = []
    if not rows:
        return [], [], ["NO_ROWS_FOR_SCORING"]

    py = _detect_python_bin()
    risk_bundle = _read_pointer("current_review_risk_champion_bundle_dir.txt")
    recon_bundle = _read_pointer("current_reconciliation_champion_bundle_dir.txt")

    with tempfile.TemporaryDirectory(prefix="va_orchestrator_") as tdir:
        troot = Path(tdir)
        input_jsonl = troot / "input_rows.jsonl"
        risk_out = troot / "risk.jsonl"
        recon_out = troot / "recon.jsonl"

        with input_jsonl.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, sort_keys=True) + "\n")

        _run_python(
            [
                py,
                REVIEW_RISK_SCRIPT.as_posix(),
                "score",
                "--bundle-dir",
                risk_bundle,
                "--input-file",
                input_jsonl.as_posix(),
                "--out",
                risk_out.as_posix(),
            ]
        )
        _run_python(
            [
                py,
                RECON_SCRIPT.as_posix(),
                "score",
                "--bundle-dir",
                recon_bundle,
                "--input-file",
                input_jsonl.as_posix(),
                "--out",
                recon_out.as_posix(),
            ]
        )

        def _read_jsonl(path: Path) -> list[dict[str, Any]]:
            out: list[dict[str, Any]] = []
            with path.open("r", encoding="utf-8") as f:
                for raw in f:
                    raw = raw.strip()
                    if not raw:
                        continue
                    obj = json.loads(raw)
                    if isinstance(obj, dict):
                        out.append(dict(obj))
            return out

        return _read_jsonl(risk_out), _read_jsonl(recon_out), warnings


def _tool_detect_anomalies(payload: dict[str, Any]) -> dict[str, Any]:
    rows = _load_rows_from_input(payload)
    window = str(payload.get("window") or datetime.now(timezone.utc).strftime("%Y-%m"))
    min_score = _to_float(payload.get("min_score"), 0.55)

    risk_rows, recon_rows, warnings = _run_scoring_models(rows)
    by_ref: dict[str, dict[str, Any]] = {}
    for rr in risk_rows:
        ref = str(rr.get("reference") or "").strip().upper()
        if not ref:
            continue
        by_ref.setdefault(ref, {})["risk"] = rr
    for rc in recon_rows:
        ref = str(rc.get("reference") or "").strip().upper()
        if not ref:
            continue
        by_ref.setdefault(ref, {})["recon"] = rc

    anomalies: list[dict[str, Any]] = []
    for ref, obj in sorted(by_ref.items()):
        risk_pred = dict(dict(obj.get("risk") or {}).get("prediction") or {})
        recon_pred = dict(dict(obj.get("recon") or {}).get("prediction") or {})
        risk_score = _to_float(risk_pred.get("review_required_probability"), 0.0)
        recon_score = _to_float(recon_pred.get("exception_probability"), 0.0)
        score = max(risk_score, recon_score)
        if score < min_score:
            continue

        reason_codes: list[str] = []
        if risk_score >= min_score:
            reason_codes.append("REVIEW_RISK_HIGH")
        if recon_score >= min_score:
            reason_codes.append("RECON_EXCEPTION_HIGH")
        if str(recon_pred.get("reconciliation_priority") or "").upper() == "P1":
            reason_codes.append("RECON_PRIORITY_P1")
        if str(risk_pred.get("risk_band") or "").upper() == "HIGH":
            reason_codes.append("RISK_BAND_HIGH")

        severity = "LOW"
        if score >= 0.85:
            severity = "HIGH"
        elif score >= 0.65:
            severity = "MEDIUM"

        anomalies.append(
            {
                "reference": ref,
                "severity": severity,
                "score": round(score, 6),
                "reason_codes": sorted(set(reason_codes)) or ["ANOMALY_SIGNAL"],
            }
        )

    return {
        "task": "transaction_anomaly_scan",
        "window": window,
        "total_anomalies": len(anomalies),
        "anomalies": anomalies,
        "warnings": warnings,
    }


def _load_training_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or not path.is_file():
        raise SystemExit(f"Training CSV not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as f:
        return [dict(r) for r in csv.DictReader(f)]


def _tool_forecast_cashflow(payload: dict[str, Any]) -> dict[str, Any]:
    training_csv = str(payload.get("training_csv") or "").strip()
    if not training_csv:
        raise SystemExit("forecast_cashflow requires 'training_csv'")
    rows = _load_training_rows(Path(training_csv))

    horizon_days = int(_to_float(payload.get("horizon_days"), 30))
    horizon_days = max(1, min(180, horizon_days))
    currency = str(payload.get("currency") or "INR").strip().upper() or "INR"

    daily_net: dict[date, float] = defaultdict(float)
    for row in rows:
        dt = _parse_iso_date(row.get("date"))
        if dt is None:
            continue
        for account, dc, amt in _parse_journal_lines(row.get("journal_lines")):
            if "CASH" not in account and "BANK" not in account:
                continue
            if dc == "D":
                daily_net[dt] += amt
            elif dc == "C":
                daily_net[dt] -= amt

    warnings: list[str] = []
    points: list[dict[str, Any]] = []
    if not daily_net:
        warnings.append("NO_CASH_HISTORY_FOUND")
        base_day = date.today()
        for i in range(horizon_days):
            d = base_day + timedelta(days=i + 1)
            points.append({"date": d.isoformat(), "cash_in": 0.0, "cash_out": 0.0, "net_cash": 0.0, "confidence": 0.2})
    else:
        ordered_dates = sorted(daily_net.keys())
        series = [daily_net[d] for d in ordered_dates]
        window = min(7, len(series))
        baseline = float(mean(series[-window:])) if series else 0.0
        drift = 0.0
        if len(series) >= 2:
            drift = (series[-1] - series[-2]) * 0.15
        conf = min(0.9, 0.45 + (len(series) / 200.0))
        if len(series) < 14:
            warnings.append("LOW_HISTORY_FOR_FORECAST")
        start = ordered_dates[-1]
        for i in range(horizon_days):
            d = start + timedelta(days=i + 1)
            net = baseline + (drift * i)
            points.append(
                {
                    "date": d.isoformat(),
                    "cash_in": round(max(net, 0.0), 2),
                    "cash_out": round(max(-net, 0.0), 2),
                    "net_cash": round(net, 2),
                    "confidence": round(max(0.2, min(0.99, conf)), 6),
                }
            )

    return {
        "task": "cashflow_forecast",
        "currency": currency,
        "horizon_days": horizon_days,
        "forecast_points": points,
        "warnings": warnings,
    }


def _tool_generate_monthly_summary(payload: dict[str, Any]) -> dict[str, Any]:
    training_csv = str(payload.get("training_csv") or "").strip()
    if not training_csv:
        raise SystemExit("generate_monthly_summary requires 'training_csv'")
    period = str(payload.get("period") or "").strip()
    if not period:
        period = datetime.now(timezone.utc).strftime("%Y-%m")
    currency = str(payload.get("currency") or "INR").strip().upper() or "INR"
    rows = _load_training_rows(Path(training_csv))

    revenue = 0.0
    expense = 0.0
    cogs = 0.0
    gst_payable = 0.0
    count = 0
    for row in rows:
        dt = _parse_iso_date(row.get("date"))
        if dt is None:
            continue
        if not dt.isoformat().startswith(period):
            continue
        count += 1
        for account, dc, amt in _parse_journal_lines(row.get("journal_lines")):
            if "SALES" in account and dc == "C":
                revenue += amt
            if ("COGS" in account or "EXPENSE" in account or "WRITE_OFF" in account or "FREIGHT" in account) and dc == "D":
                expense += amt
            if "COGS" in account and dc == "D":
                cogs += amt
            if ("GST_OUTPUT" in account or "TAX_PAYABLE" in account or "TDS_PAYABLE" in account):
                if dc == "C":
                    gst_payable += amt
                elif dc == "D":
                    gst_payable -= amt

    gross_profit = revenue - cogs
    net_profit = revenue - expense
    warnings: list[str] = []
    if count == 0:
        warnings.append("NO_ROWS_FOR_PERIOD")
    elif count < 30:
        warnings.append("LOW_PERIOD_VOLUME")

    highlights = [
        f"Period {period}: processed {count} transactions for summary.",
        f"Net profit estimate is {net_profit:.2f} {currency} with GST payable {gst_payable:.2f} {currency}.",
    ]

    return {
        "task": "monthly_close_summary",
        "period": period,
        "currency": currency,
        "kpis": {
            "revenue": round(revenue, 2),
            "expense": round(expense, 2),
            "gross_profit": round(gross_profit, 2),
            "net_profit": round(net_profit, 2),
            "gst_payable": round(gst_payable, 2),
        },
        "highlights": highlights,
        "evidence_refs": [f"training_csv:{Path(training_csv).name}", f"period:{period}"],
        "warnings": warnings,
    }


TOOL_IMPL = {
    "categorize_transaction": _tool_categorize_transaction,
    "detect_anomalies": _tool_detect_anomalies,
    "forecast_cashflow": _tool_forecast_cashflow,
    "generate_monthly_summary": _tool_generate_monthly_summary,
}


def _run_contract_validate(tool: str, payload: dict[str, Any]) -> dict[str, Any]:
    py = _detect_python_bin()
    with tempfile.TemporaryDirectory(prefix="va_contract_validate_") as tdir:
        troot = Path(tdir)
        payload_path = troot / "payload.json"
        payload_path.write_text(json.dumps(payload), encoding="utf-8")
        cmd = [
            py,
            CONTRACTS_SCRIPT.as_posix(),
            "validate",
            "--tool",
            tool,
            "--payload-json",
            payload_path.as_posix(),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode not in {0, 2}:
            raise SystemExit(f"Contract validate command failed: {proc.stderr.strip()}")
        try:
            return json.loads(proc.stdout.strip() or "{}")
        except Exception as e:
            raise SystemExit(f"Failed to parse contract validation output: {e}")


def _run_contract_audit(
    *,
    tool: str,
    request_obj: dict[str, Any],
    response_obj: dict[str, Any],
    audit_log_jsonl: Path,
    actor: str,
) -> dict[str, Any]:
    py = _detect_python_bin()
    with tempfile.TemporaryDirectory(prefix="va_contract_audit_") as tdir:
        troot = Path(tdir)
        req_path = troot / "request.json"
        res_path = troot / "response.json"
        req_path.write_text(json.dumps(request_obj), encoding="utf-8")
        res_path.write_text(json.dumps(response_obj), encoding="utf-8")
        cmd = [
            py,
            CONTRACTS_SCRIPT.as_posix(),
            "audit",
            "--tool",
            tool,
            "--request-json",
            req_path.as_posix(),
            "--response-json",
            res_path.as_posix(),
            "--audit-log-jsonl",
            audit_log_jsonl.as_posix(),
            "--actor",
            actor,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode not in {0, 2}:
            raise SystemExit(f"Contract audit command failed: {proc.stderr.strip()}")
        try:
            return json.loads(proc.stdout.strip() or "{}")
        except Exception as e:
            raise SystemExit(f"Failed to parse contract audit output: {e}")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Runtime orchestrator for virtual-accountant tool calls.")
    p.add_argument("--tool", required=True, choices=sorted(TOOL_IMPL.keys()))
    p.add_argument("--input-json", required=True, help="Tool request JSON payload")
    p.add_argument("--out-json", required=True, help="Output response JSON path")
    p.add_argument(
        "--audit-log-jsonl",
        default=(OUTPUTS_DIR / "virtual_accountant_audit" / "tool_calls.jsonl").as_posix(),
        help="Audit log JSONL path",
    )
    p.add_argument("--actor", default="system")
    return p


def main() -> int:
    args = _build_parser().parse_args()
    tool = str(args.tool)
    request_obj = _load_json(Path(args.input_json))
    if not isinstance(request_obj, dict):
        raise SystemExit("input-json must contain a JSON object")

    tool_payload = TOOL_IMPL[tool](dict(request_obj))
    validation = _run_contract_validate(tool, tool_payload)
    audit = _run_contract_audit(
        tool=tool,
        request_obj=request_obj,
        response_obj=tool_payload,
        audit_log_jsonl=Path(args.audit_log_jsonl),
        actor=str(args.actor),
    )

    out_obj = {
        "schema_version": "v1",
        "generated_at_utc": _utc_now(),
        "tool": tool,
        "request": request_obj,
        "result": tool_payload,
        "validation": validation,
        "audit": audit,
    }
    _write_json(Path(args.out_json), out_obj)
    print(Path(args.out_json).as_posix())
    return 0 if bool(validation.get("ok")) else 2


if __name__ == "__main__":
    raise SystemExit(main())
