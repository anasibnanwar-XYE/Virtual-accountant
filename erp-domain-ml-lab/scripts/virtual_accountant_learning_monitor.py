#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _read_pointer(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    raw = path.read_text(encoding="utf-8").strip()
    return raw or None


def _load_json(path_str: str | None) -> dict[str, Any] | None:
    if not path_str:
        return None
    p = Path(path_str)
    if not p.exists() or not p.is_file():
        return None
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    return []


def _pick_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _tx_coa_summary(cycle: dict[str, Any] | None, scorecard: dict[str, Any] | None) -> dict[str, Any]:
    gate = dict((cycle or {}).get("gate_decision") or {})
    score_tx = dict((scorecard or {}).get("tx") or {})
    score_coa = dict((scorecard or {}).get("coa") or {})
    challenger_scores = dict(gate.get("scores", {}).get("challenger") or {})
    return {
        "cycle_status": (cycle or {}).get("status"),
        "promote": gate.get("promote"),
        "violations": _as_list(gate.get("violations")),
        "training_reused_existing_models": (cycle or {}).get("training_reused_existing_models"),
        "gst_guardrail_profile": (cycle or {}).get("gst_guardrail_profile"),
        "gst_next_guardrail_profile": (cycle or {}).get("gst_next_guardrail_profile"),
        "tx_accuracy": _pick_float(score_tx.get("accuracy")),
        "tx_accuracy_ppm": score_tx.get("accuracy_ppm"),
        "coa_debit_top1": _pick_float(score_coa.get("debit_top1")),
        "coa_credit_top1": _pick_float(score_coa.get("credit_top1")),
        "coa_debit_top3": _pick_float(score_coa.get("debit_top3")),
        "coa_credit_top3": _pick_float(score_coa.get("credit_top3")),
        "gst_major_fail_rate": _pick_float(challenger_scores.get("gst_major_fail_rate")),
        "gst_issue_rate": _pick_float(challenger_scores.get("gst_examples_with_issues_rate")),
    }


def _product_summary(cycle: dict[str, Any] | None, metrics: dict[str, Any] | None) -> dict[str, Any]:
    gate = dict((cycle or {}).get("gate_decision") or {})
    targets = dict(((metrics or {}).get("metrics") or {}).get("targets") or {})
    return {
        "cycle_status": (cycle or {}).get("status"),
        "promote": gate.get("promote"),
        "violations": _as_list(gate.get("violations")),
        "training_skipped": (cycle or {}).get("training_skipped"),
        "holdout_top1": {
            "revenue": _pick_float(targets.get("revenue", {}).get("holdout", {}).get("top1")),
            "cogs": _pick_float(targets.get("cogs", {}).get("holdout", {}).get("top1")),
            "inventory": _pick_float(targets.get("inventory", {}).get("holdout", {}).get("top1")),
            "tax": _pick_float(targets.get("tax", {}).get("holdout", {}).get("top1")),
            "discount": _pick_float(targets.get("discount", {}).get("holdout", {}).get("top1")),
        },
    }


def _binary_model_summary(cycle: dict[str, Any] | None, metrics: dict[str, Any] | None) -> dict[str, Any]:
    gate = dict((cycle or {}).get("gate_decision") or {})
    holdout = dict(((metrics or {}).get("metrics") or {}).get("holdout") or {})
    return {
        "cycle_status": (cycle or {}).get("status"),
        "promote": gate.get("promote"),
        "violations": _as_list(gate.get("violations")),
        "training_skipped": (cycle or {}).get("training_skipped"),
        "holdout_accuracy": _pick_float(holdout.get("accuracy")),
        "holdout_precision": _pick_float(holdout.get("precision")),
        "holdout_recall": _pick_float(holdout.get("recall")),
        "holdout_f1": _pick_float(holdout.get("f1")),
        "holdout_predicted_positive_rate": _pick_float(holdout.get("predicted_positive_rate")),
    }


def _personalization_summary(report: dict[str, Any] | None) -> dict[str, Any]:
    return {
        "status": (report or {}).get("status"),
        "user_id": (report or {}).get("user_id"),
        "company_code": (report or {}).get("company_code"),
        "memory_rows_user": (report or {}).get("memory_rows_user"),
        "tx_top1_change_rate": _pick_float((report or {}).get("tx_top1_change_rate")),
        "coa_debit_top1_change_rate": _pick_float((report or {}).get("coa_debit_top1_change_rate")),
        "coa_credit_top1_change_rate": _pick_float((report or {}).get("coa_credit_top1_change_rate")),
        "violations": _as_list((report or {}).get("violations")),
    }


def _alerts(report: dict[str, Any]) -> list[str]:
    alerts: list[str] = []
    tx = dict(report.get("models", {}).get("tx_coa") or {})
    product = dict(report.get("models", {}).get("product_account") or {})
    risk = dict(report.get("models", {}).get("review_risk") or {})
    recon = dict(report.get("models", {}).get("reconciliation") or {})
    pers = dict(report.get("models", {}).get("personalization") or {})

    if tx.get("cycle_status") in {"rejected", "failed"}:
        alerts.append(f"tx_coa:{tx.get('cycle_status')}")
    if product.get("cycle_status") in {"rejected", "failed"}:
        alerts.append(f"product_account:{product.get('cycle_status')}")
    if risk.get("cycle_status") in {"rejected", "failed"}:
        alerts.append(f"review_risk:{risk.get('cycle_status')}")
    if recon.get("cycle_status") in {"rejected", "failed"}:
        alerts.append(f"reconciliation:{recon.get('cycle_status')}")
    if pers.get("status") in {"guardrail_reverted_to_base", "low_user_memory_skip", "failed"}:
        alerts.append(f"personalization:{pers.get('status')}")
    return alerts


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build controlled learning-monitor report for virtual accountant models.")
    p.add_argument("--outputs-dir", required=True)
    p.add_argument("--out-json", required=True)
    return p


def main() -> int:
    args = _build_parser().parse_args()
    outputs_dir = Path(args.outputs_dir)
    out_json = Path(args.out_json)

    pointers = {
        "tx_cycle_summary": _read_pointer(outputs_dir / "latest_tx_coa_continual_cycle_summary.txt"),
        "tx_scorecard": _read_pointer(outputs_dir / "latest_tx_coa_scorecard.json"),
        "product_cycle_summary": _read_pointer(outputs_dir / "latest_product_account_continual_cycle_summary.txt"),
        "product_metrics": _read_pointer(outputs_dir / "latest_product_account_metrics.json"),
        "review_risk_cycle_summary": _read_pointer(outputs_dir / "latest_review_risk_continual_cycle_summary.txt"),
        "review_risk_metrics": _read_pointer(outputs_dir / "latest_review_risk_metrics.json"),
        "reconciliation_cycle_summary": _read_pointer(outputs_dir / "latest_reconciliation_continual_cycle_summary.txt"),
        "reconciliation_metrics": _read_pointer(outputs_dir / "latest_reconciliation_metrics.json"),
        "personalization_report": _read_pointer(outputs_dir / "latest_user_personalization_report_json.txt"),
        "personalization_guardrail_smoke": _read_pointer(outputs_dir / "latest_personalization_guardrail_smoke_summary_json.txt"),
    }

    tx_cycle = _load_json(pointers["tx_cycle_summary"])
    tx_score = _load_json(pointers["tx_scorecard"])
    product_cycle = _load_json(pointers["product_cycle_summary"])
    product_metrics = _load_json(pointers["product_metrics"])
    risk_cycle = _load_json(pointers["review_risk_cycle_summary"])
    risk_metrics = _load_json(pointers["review_risk_metrics"])
    recon_cycle = _load_json(pointers["reconciliation_cycle_summary"])
    recon_metrics = _load_json(pointers["reconciliation_metrics"])
    pers_report = _load_json(pointers["personalization_report"])
    pers_smoke = _load_json(pointers["personalization_guardrail_smoke"])

    report: dict[str, Any] = {
        "schema_version": "v0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "paths": pointers,
        "models": {
            "tx_coa": _tx_coa_summary(tx_cycle, tx_score),
            "product_account": _product_summary(product_cycle, product_metrics),
            "review_risk": _binary_model_summary(risk_cycle, risk_metrics),
            "reconciliation": _binary_model_summary(recon_cycle, recon_metrics),
            "personalization": _personalization_summary(pers_report),
        },
        "guardrail_smoke": {
            "status": (pers_smoke or {}).get("status"),
            "checks": _as_list((pers_smoke or {}).get("checks")),
        },
    }
    report["alerts"] = _alerts(report)
    report["overall_status"] = "attention_required" if report["alerts"] else "healthy"

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(out_json.as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
