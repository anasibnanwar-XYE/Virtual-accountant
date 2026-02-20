#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _up(value: Any) -> str:
    return _clean(value).upper()


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _safe_jsonl(path: Path) -> list[dict[str, Any]]:
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


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def _family_from_tx(row: dict[str, Any]) -> str:
    num = dict(row.get("numeric") or {})
    if _as_float(num.get("workflow_sale"), 0.0) > 0:
        return "sale"
    if _as_float(num.get("workflow_purchase"), 0.0) > 0:
        return "purchase"
    if _as_float(num.get("workflow_payment"), 0.0) > 0:
        return "payment"
    if _as_float(num.get("workflow_settlement_split"), 0.0) > 0:
        return "settlement_split"
    if _as_float(num.get("workflow_tax_settlement"), 0.0) > 0:
        return "tax_settlement"
    if _as_float(num.get("workflow_payroll"), 0.0) > 0:
        return "payroll"
    if _as_float(num.get("workflow_sale_return"), 0.0) > 0:
        return "sale_return"
    if _as_float(num.get("workflow_cogs"), 0.0) > 0:
        return "cogs"
    if _as_float(num.get("workflow_write_off"), 0.0) > 0:
        return "write_off"
    if _as_float(num.get("workflow_inventory_count"), 0.0) > 0:
        return "inventory_count"
    if _as_float(num.get("workflow_period_lock"), 0.0) > 0:
        return "period_lock"
    lbl = _up(row.get("suggested_label"))
    if lbl in {"SALE", "PURCHASE", "PAYMENT", "SALE_RETURN", "COGS", "WRITE_OFF", "PERIOD_LOCK"}:
        return lbl.lower()
    return "unknown"


def _norm_family(value: Any) -> str:
    raw = _up(value).replace("-", "_").replace(" ", "_")
    mapping = {
        "O2C": "sale",
        "P2P": "purchase",
        "SALE": "sale",
        "PURCHASE": "purchase",
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
    if raw in mapping:
        return mapping[raw]
    return raw.lower() if raw else "unknown"


def _load_personalization(
    *,
    path: Path,
    user_id: str,
    company_code: str,
) -> tuple[
    list[dict[str, Any]],
    dict[str, Counter[str]],
    dict[str, Counter[str]],
    dict[str, Counter[str]],
    Counter[str],
    Counter[str],
    Counter[str],
    Counter[str],
]:
    rows = _safe_jsonl(path) if path.exists() else []
    user = user_id.strip().lower()
    company = _up(company_code)

    filtered: list[dict[str, Any]] = []
    tx_by_family: dict[str, Counter[str]] = defaultdict(Counter)
    debit_by_family: dict[str, Counter[str]] = defaultdict(Counter)
    credit_by_family: dict[str, Counter[str]] = defaultdict(Counter)
    tx_global: Counter[str] = Counter()
    debit_global: Counter[str] = Counter()
    credit_global: Counter[str] = Counter()
    memory_rows_by_family: Counter[str] = Counter()

    for row in rows:
        uid = _clean(row.get("user_id")).lower()
        if uid != user:
            continue
        row_company = _up(row.get("company_code"))
        if company and row_company and row_company != company:
            continue
        fam = _norm_family(row.get("workflow_family"))
        memory_rows_by_family[fam] += 1
        approved_label = _up(row.get("approved_label"))
        approved_debit = _up(row.get("approved_debit_account_code"))
        approved_credit = _up(row.get("approved_credit_account_code"))

        if approved_label:
            tx_by_family[fam][approved_label] += 1
            tx_global[approved_label] += 1
        if approved_debit:
            debit_by_family[fam][approved_debit] += 1
            debit_global[approved_debit] += 1
        if approved_credit:
            credit_by_family[fam][approved_credit] += 1
            credit_global[approved_credit] += 1
        filtered.append(row)

    return (
        filtered,
        tx_by_family,
        debit_by_family,
        credit_by_family,
        tx_global,
        debit_global,
        credit_global,
        memory_rows_by_family,
    )


def _prob(counter: Counter[str], label: str) -> float:
    total = sum(counter.values())
    if total <= 0:
        return 0.0
    return float(counter.get(label, 0)) / float(total)


def _rerank_predictions(
    preds: list[dict[str, Any]],
    *,
    family_counter: Counter[str],
    global_counter: Counter[str],
    alpha: float,
    alpha_scale: float = 1.0,
) -> tuple[list[dict[str, Any]], bool]:
    adjusted: list[dict[str, Any]] = []
    for pred in preds:
        label = _up(pred.get("label"))
        base = _as_float(pred.get("score"), 0.0)
        fam_p = _prob(family_counter, label)
        glob_p = _prob(global_counter, label)
        bonus = (alpha * alpha_scale) * (0.7 * fam_p + 0.3 * glob_p)
        score_adj = base + bonus
        out = dict(pred)
        out["personalization_bonus"] = f"{bonus:.6f}"
        out["score"] = f"{score_adj:.6f}"
        adjusted.append(out)

    before_top = _up(preds[0].get("label")) if preds else ""
    adjusted.sort(key=lambda x: (_as_float(x.get("score"), 0.0), _up(x.get("label"))), reverse=True)
    after_top = _up(adjusted[0].get("label")) if adjusted else ""
    return adjusted, bool(before_top and after_top and before_top != after_top)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Apply per-user personalization reranking to tx+coa advisory outputs.")
    p.add_argument("--tx-jsonl", required=True)
    p.add_argument("--coa-jsonl", required=True)
    p.add_argument("--memory-jsonl", required=True)
    p.add_argument("--user-id", required=True)
    p.add_argument("--company-code", default="")
    p.add_argument("--tx-alpha", type=float, default=0.25)
    p.add_argument("--coa-alpha", type=float, default=0.20)
    p.add_argument("--min-memory-rows", type=int, default=5)
    p.add_argument("--min-family-memory-rows", type=int, default=3)
    p.add_argument("--global-only-alpha-scale", type=float, default=0.40)
    p.add_argument("--max-tx-top1-change-rate", type=float, default=0.30)
    p.add_argument("--max-coa-debit-top1-change-rate", type=float, default=0.35)
    p.add_argument("--max-coa-credit-top1-change-rate", type=float, default=0.35)
    p.add_argument("--min-family-eval-rows", type=int, default=25)
    p.add_argument("--max-family-top1-change-rate", type=float, default=0.50)
    p.add_argument("--out-tx-jsonl", required=True)
    p.add_argument("--out-coa-jsonl", required=True)
    p.add_argument("--report-out", required=True)
    return p


def main() -> int:
    args = _build_parser().parse_args()

    tx_path = Path(args.tx_jsonl)
    coa_path = Path(args.coa_jsonl)
    mem_path = Path(args.memory_jsonl)
    out_tx = Path(args.out_tx_jsonl)
    out_coa = Path(args.out_coa_jsonl)
    report_path = Path(args.report_out)

    if not tx_path.exists() or not tx_path.is_file():
        raise SystemExit(f"Transaction JSONL not found: {tx_path}")
    if not coa_path.exists() or not coa_path.is_file():
        raise SystemExit(f"CoA JSONL not found: {coa_path}")
    if not mem_path.exists() or not mem_path.is_file():
        raise SystemExit(f"Personalization memory JSONL not found: {mem_path}")

    (
        mem_rows,
        tx_by_family,
        debit_by_family,
        credit_by_family,
        tx_global,
        debit_global,
        credit_global,
        memory_rows_by_family,
    ) = _load_personalization(path=mem_path, user_id=args.user_id, company_code=args.company_code)

    tx_rows = _safe_jsonl(tx_path)
    coa_rows = _safe_jsonl(coa_path)

    if not mem_rows:
        _write_jsonl(out_tx, tx_rows)
        _write_jsonl(out_coa, coa_rows)
        report = {
            "schema_version": "v1",
            "status": "no_user_memory",
            "user_id": _clean(args.user_id).lower(),
            "company_code": _up(args.company_code),
            "memory_jsonl": mem_path.as_posix(),
            "memory_rows_user": 0,
            "tx_rows": len(tx_rows),
            "coa_rows": len(coa_rows),
            "guardrails": {
                "min_memory_rows": int(args.min_memory_rows),
                "min_family_memory_rows": int(args.min_family_memory_rows),
                "global_only_alpha_scale": float(args.global_only_alpha_scale),
                "max_tx_top1_change_rate": float(args.max_tx_top1_change_rate),
                "max_coa_debit_top1_change_rate": float(args.max_coa_debit_top1_change_rate),
                "max_coa_credit_top1_change_rate": float(args.max_coa_credit_top1_change_rate),
                "min_family_eval_rows": int(args.min_family_eval_rows),
                "max_family_top1_change_rate": float(args.max_family_top1_change_rate),
            },
            "out_tx_jsonl": out_tx.as_posix(),
            "out_coa_jsonl": out_coa.as_posix(),
        }
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps(report, sort_keys=True))
        return 0

    if len(mem_rows) < int(args.min_memory_rows):
        _write_jsonl(out_tx, tx_rows)
        _write_jsonl(out_coa, coa_rows)
        report = {
            "schema_version": "v1",
            "status": "low_user_memory_skip",
            "user_id": _clean(args.user_id).lower(),
            "company_code": _up(args.company_code),
            "memory_jsonl": mem_path.as_posix(),
            "memory_rows_user": len(mem_rows),
            "tx_rows": len(tx_rows),
            "coa_rows": len(coa_rows),
            "guardrails": {
                "min_memory_rows": int(args.min_memory_rows),
                "min_family_memory_rows": int(args.min_family_memory_rows),
                "global_only_alpha_scale": float(args.global_only_alpha_scale),
                "max_tx_top1_change_rate": float(args.max_tx_top1_change_rate),
                "max_coa_debit_top1_change_rate": float(args.max_coa_debit_top1_change_rate),
                "max_coa_credit_top1_change_rate": float(args.max_coa_credit_top1_change_rate),
                "min_family_eval_rows": int(args.min_family_eval_rows),
                "max_family_top1_change_rate": float(args.max_family_top1_change_rate),
            },
            "violations": [f"memory_rows_user:{len(mem_rows)} < {int(args.min_memory_rows)}"],
            "out_tx_jsonl": out_tx.as_posix(),
            "out_coa_jsonl": out_coa.as_posix(),
        }
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps(report, sort_keys=True))
        return 0

    tx_top1_changed = 0
    tx_family_rows: Counter[str] = Counter()
    tx_family_changed: Counter[str] = Counter()
    tx_out: list[dict[str, Any]] = []
    for row in tx_rows:
        fam = _family_from_tx(row)
        tx_family_rows[fam] += 1
        fam_support_rows = int(memory_rows_by_family.get(fam, 0))
        use_family_prior = fam_support_rows >= int(args.min_family_memory_rows)
        alpha_scale = 1.0 if use_family_prior else float(args.global_only_alpha_scale)
        preds = [p for p in list(row.get("predictions") or []) if isinstance(p, dict)]
        adjusted, changed = _rerank_predictions(
            preds,
            family_counter=(tx_by_family.get(fam, Counter()) if use_family_prior else Counter()),
            global_counter=tx_global,
            alpha=float(args.tx_alpha),
            alpha_scale=alpha_scale,
        )
        out_row = dict(row)
        out_row["predictions"] = adjusted
        if adjusted:
            out_row["suggested_label"] = _up(adjusted[0].get("label"))
        out_row["personalization"] = {
            "user_id": _clean(args.user_id).lower(),
            "company_code": _up(args.company_code),
            "workflow_family": fam,
            "applied": True,
            "family_support_rows": fam_support_rows,
            "family_evidence_mode": ("family_plus_global" if use_family_prior else "global_only_low_evidence"),
            "alpha_scale": alpha_scale,
            "tx_alpha": float(args.tx_alpha),
        }
        if changed:
            tx_top1_changed += 1
            tx_family_changed[fam] += 1
        tx_out.append(out_row)

    coa_top1_changed_debit = 0
    coa_top1_changed_credit = 0
    coa_family_rows: Counter[str] = Counter()
    coa_family_changed_debit: Counter[str] = Counter()
    coa_family_changed_credit: Counter[str] = Counter()
    coa_out: list[dict[str, Any]] = []
    for row in coa_rows:
        fam = _norm_family(((row.get("workflow_context") or {}).get("family")))
        coa_family_rows[fam] += 1
        fam_support_rows = int(memory_rows_by_family.get(fam, 0))
        use_family_prior = fam_support_rows >= int(args.min_family_memory_rows)
        alpha_scale = 1.0 if use_family_prior else float(args.global_only_alpha_scale)

        debit_preds = [p for p in list(row.get("debit_recommendations") or []) if isinstance(p, dict)]
        debit_adj, debit_changed = _rerank_predictions(
            debit_preds,
            family_counter=(debit_by_family.get(fam, Counter()) if use_family_prior else Counter()),
            global_counter=debit_global,
            alpha=float(args.coa_alpha),
            alpha_scale=alpha_scale,
        )
        credit_preds = [p for p in list(row.get("credit_recommendations") or []) if isinstance(p, dict)]
        credit_adj, credit_changed = _rerank_predictions(
            credit_preds,
            family_counter=(credit_by_family.get(fam, Counter()) if use_family_prior else Counter()),
            global_counter=credit_global,
            alpha=float(args.coa_alpha),
            alpha_scale=alpha_scale,
        )

        out_row = dict(row)
        out_row["debit_recommendations"] = debit_adj
        out_row["credit_recommendations"] = credit_adj
        out_row["personalization"] = {
            "user_id": _clean(args.user_id).lower(),
            "company_code": _up(args.company_code),
            "workflow_family": fam,
            "applied": True,
            "family_support_rows": fam_support_rows,
            "family_evidence_mode": ("family_plus_global" if use_family_prior else "global_only_low_evidence"),
            "alpha_scale": alpha_scale,
            "coa_alpha": float(args.coa_alpha),
        }
        if debit_changed:
            coa_top1_changed_debit += 1
            coa_family_changed_debit[fam] += 1
        if credit_changed:
            coa_top1_changed_credit += 1
            coa_family_changed_credit[fam] += 1
        coa_out.append(out_row)

    tx_rows_n = len(tx_rows)
    coa_rows_n = len(coa_rows)
    tx_top1_change_rate = (float(tx_top1_changed) / float(tx_rows_n)) if tx_rows_n > 0 else 0.0
    coa_debit_top1_change_rate = (float(coa_top1_changed_debit) / float(coa_rows_n)) if coa_rows_n > 0 else 0.0
    coa_credit_top1_change_rate = (float(coa_top1_changed_credit) / float(coa_rows_n)) if coa_rows_n > 0 else 0.0

    violations: list[str] = []
    if tx_top1_change_rate > float(args.max_tx_top1_change_rate):
        violations.append(
            f"tx_top1_change_rate:{tx_top1_change_rate:.6f} > {float(args.max_tx_top1_change_rate):.6f}"
        )
    if coa_debit_top1_change_rate > float(args.max_coa_debit_top1_change_rate):
        violations.append(
            f"coa_debit_top1_change_rate:{coa_debit_top1_change_rate:.6f} > {float(args.max_coa_debit_top1_change_rate):.6f}"
        )
    if coa_credit_top1_change_rate > float(args.max_coa_credit_top1_change_rate):
        violations.append(
            f"coa_credit_top1_change_rate:{coa_credit_top1_change_rate:.6f} > {float(args.max_coa_credit_top1_change_rate):.6f}"
        )

    tx_family_change_summary: dict[str, dict[str, float | int]] = {}
    for fam, rows in tx_family_rows.items():
        changed = int(tx_family_changed.get(fam, 0))
        rate = (float(changed) / float(rows)) if rows > 0 else 0.0
        tx_family_change_summary[fam] = {
            "rows": int(rows),
            "changed_top1": changed,
            "change_rate": rate,
        }
        if int(rows) >= int(args.min_family_eval_rows) and rate > float(args.max_family_top1_change_rate):
            violations.append(
                f"tx_family_top1_change_rate:{fam}:{rate:.6f} > {float(args.max_family_top1_change_rate):.6f}"
            )

    coa_debit_family_change_summary: dict[str, dict[str, float | int]] = {}
    coa_credit_family_change_summary: dict[str, dict[str, float | int]] = {}
    for fam, rows in coa_family_rows.items():
        changed_debit = int(coa_family_changed_debit.get(fam, 0))
        changed_credit = int(coa_family_changed_credit.get(fam, 0))
        rate_debit = (float(changed_debit) / float(rows)) if rows > 0 else 0.0
        rate_credit = (float(changed_credit) / float(rows)) if rows > 0 else 0.0
        coa_debit_family_change_summary[fam] = {
            "rows": int(rows),
            "changed_top1": changed_debit,
            "change_rate": rate_debit,
        }
        coa_credit_family_change_summary[fam] = {
            "rows": int(rows),
            "changed_top1": changed_credit,
            "change_rate": rate_credit,
        }
        if int(rows) >= int(args.min_family_eval_rows) and rate_debit > float(args.max_family_top1_change_rate):
            violations.append(
                f"coa_debit_family_top1_change_rate:{fam}:{rate_debit:.6f} > {float(args.max_family_top1_change_rate):.6f}"
            )
        if int(rows) >= int(args.min_family_eval_rows) and rate_credit > float(args.max_family_top1_change_rate):
            violations.append(
                f"coa_credit_family_top1_change_rate:{fam}:{rate_credit:.6f} > {float(args.max_family_top1_change_rate):.6f}"
            )

    # Guardrail: if personalization would shift too many top-1 decisions, keep base outputs.
    if violations:
        _write_jsonl(out_tx, tx_rows)
        _write_jsonl(out_coa, coa_rows)
        report = {
            "schema_version": "v1",
            "status": "guardrail_reverted_to_base",
            "user_id": _clean(args.user_id).lower(),
            "company_code": _up(args.company_code),
            "memory_jsonl": mem_path.as_posix(),
            "memory_rows_user": len(mem_rows),
            "tx_rows": tx_rows_n,
            "coa_rows": coa_rows_n,
            "tx_top1_changed": tx_top1_changed,
            "coa_top1_changed_debit": coa_top1_changed_debit,
            "coa_top1_changed_credit": coa_top1_changed_credit,
            "tx_top1_change_rate": tx_top1_change_rate,
            "coa_debit_top1_change_rate": coa_debit_top1_change_rate,
            "coa_credit_top1_change_rate": coa_credit_top1_change_rate,
            "tx_family_change_summary": tx_family_change_summary,
            "coa_debit_family_change_summary": coa_debit_family_change_summary,
            "coa_credit_family_change_summary": coa_credit_family_change_summary,
            "guardrails": {
                "min_memory_rows": int(args.min_memory_rows),
                "min_family_memory_rows": int(args.min_family_memory_rows),
                "global_only_alpha_scale": float(args.global_only_alpha_scale),
                "max_tx_top1_change_rate": float(args.max_tx_top1_change_rate),
                "max_coa_debit_top1_change_rate": float(args.max_coa_debit_top1_change_rate),
                "max_coa_credit_top1_change_rate": float(args.max_coa_credit_top1_change_rate),
                "min_family_eval_rows": int(args.min_family_eval_rows),
                "max_family_top1_change_rate": float(args.max_family_top1_change_rate),
            },
            "violations": violations,
            "out_tx_jsonl": out_tx.as_posix(),
            "out_coa_jsonl": out_coa.as_posix(),
        }
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps(report, sort_keys=True))
        return 0

    _write_jsonl(out_tx, tx_out)
    _write_jsonl(out_coa, coa_out)

    report = {
        "schema_version": "v1",
        "status": "ok",
        "user_id": _clean(args.user_id).lower(),
        "company_code": _up(args.company_code),
        "memory_jsonl": mem_path.as_posix(),
        "memory_rows_user": len(mem_rows),
        "tx_rows": tx_rows_n,
        "coa_rows": coa_rows_n,
        "tx_top1_changed": tx_top1_changed,
        "coa_top1_changed_debit": coa_top1_changed_debit,
        "coa_top1_changed_credit": coa_top1_changed_credit,
        "tx_top1_change_rate": tx_top1_change_rate,
        "coa_debit_top1_change_rate": coa_debit_top1_change_rate,
        "coa_credit_top1_change_rate": coa_credit_top1_change_rate,
        "tx_family_change_summary": tx_family_change_summary,
        "coa_debit_family_change_summary": coa_debit_family_change_summary,
        "coa_credit_family_change_summary": coa_credit_family_change_summary,
        "guardrails": {
            "min_memory_rows": int(args.min_memory_rows),
            "min_family_memory_rows": int(args.min_family_memory_rows),
            "global_only_alpha_scale": float(args.global_only_alpha_scale),
            "max_tx_top1_change_rate": float(args.max_tx_top1_change_rate),
            "max_coa_debit_top1_change_rate": float(args.max_coa_debit_top1_change_rate),
            "max_coa_credit_top1_change_rate": float(args.max_coa_credit_top1_change_rate),
            "min_family_eval_rows": int(args.min_family_eval_rows),
            "max_family_top1_change_rate": float(args.max_family_top1_change_rate),
        },
        "out_tx_jsonl": out_tx.as_posix(),
        "out_coa_jsonl": out_coa.as_posix(),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
