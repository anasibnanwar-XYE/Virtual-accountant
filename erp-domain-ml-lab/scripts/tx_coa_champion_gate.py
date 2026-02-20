#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    if not path.exists() or not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _gst_major_fail_rate(summary: dict[str, Any] | None) -> float | None:
    if summary is None:
        return None
    return float(summary.get("rates", {}).get("major_fail_rate", 0.0))


def _gst_examples_with_issues_rate(summary: dict[str, Any] | None) -> float | None:
    if summary is None:
        return None
    return float(summary.get("rates", {}).get("examples_with_issues_rate", 0.0))


def _gst_critical_issues(summary: dict[str, Any] | None) -> int | None:
    if summary is None:
        return None
    return int(summary.get("counts", {}).get("critical_issues", 0))


def _gst_issues_total(summary: dict[str, Any] | None) -> int | None:
    if summary is None:
        return None
    return int(summary.get("counts", {}).get("issues_total", 0))


def _gst_issue_code_count(summary: dict[str, Any] | None, issue_code: str) -> int | None:
    if summary is None:
        return None
    return int(summary.get("issue_counts_by_code", {}).get(issue_code, 0))


def _tx_accuracy(scorecard: dict[str, Any]) -> float:
    return float(scorecard.get("tx", {}).get("accuracy", 0.0))


def _tx_accuracy_ppm(scorecard: dict[str, Any]) -> int:
    return int(scorecard.get("tx", {}).get("accuracy_ppm", 0))


def _coa_top1_mean(scorecard: dict[str, Any]) -> float:
    coa = scorecard.get("coa", {})
    return (float(coa.get("debit_top1", 0.0)) + float(coa.get("credit_top1", 0.0))) / 2.0


def _coa_top3_mean(scorecard: dict[str, Any], topk: int) -> float:
    coa = scorecard.get("coa", {})
    return (
        float(coa.get(f"debit_top{topk}", 0.0))
        + float(coa.get(f"credit_top{topk}", 0.0))
    ) / 2.0


def _overall_score(scorecard: dict[str, Any], topk: int) -> float:
    # Weighted toward transaction correctness and CoA top-1 precision.
    return (
        0.50 * _tx_accuracy(scorecard)
        + 0.35 * _coa_top1_mean(scorecard)
        + 0.15 * _coa_top3_mean(scorecard, topk)
    )


def _policy_rate(scorecard: dict[str, Any], side: str, rate_kind: str) -> float:
    return float(
        scorecard.get("coa", {})
        .get("policy_top1", {})
        .get(side, {})
        .get(rate_kind, 0.0)
    )


def _select_threshold_stats(scorecard: dict[str, Any], preferred_threshold: str = "0.900000") -> dict[str, Any] | None:
    thresholds = (
        scorecard.get("tx_eval_snapshot", {})
        .get("threshold_stats", {})
    )
    if not isinstance(thresholds, dict) or not thresholds:
        return None

    selected: dict[str, Any] | None = None
    if preferred_threshold in thresholds:
        selected = thresholds.get(preferred_threshold)
    else:
        # Fallback to closest threshold key if exact key not present.
        keys = []
        for k in thresholds.keys():
            try:
                keys.append((abs(float(k) - 0.9), k))
            except Exception:
                continue
        if keys:
            selected = thresholds.get(sorted(keys, key=lambda t: t[0])[0][1])
    return selected if isinstance(selected, dict) else None


def _threshold_stat_ppm(scorecard: dict[str, Any], metric_key: str, preferred_threshold: str = "0.900000") -> int | None:
    selected = _select_threshold_stats(scorecard, preferred_threshold=preferred_threshold)
    if selected is None:
        return None
    try:
        return int(selected.get(metric_key, 0))
    except Exception:
        return None


def _period_lock_auto_accept_ppm(scorecard: dict[str, Any], preferred_threshold: str = "0.900000") -> int | None:
    selected = _select_threshold_stats(scorecard, preferred_threshold=preferred_threshold)
    if selected is None:
        return None
    period_lock = (
        selected.get("by_true_label", {})
        .get("PERIOD_LOCK", {})
    )
    if not isinstance(period_lock, dict):
        return None
    try:
        return int(period_lock.get("auto_accept_rate_ppm", 0))
    except Exception:
        return None


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Champion/challenger gate for transaction + CoA scorecards")
    p.add_argument("--challenger-scorecard", required=True)
    p.add_argument("--champion-scorecard", default=None)
    p.add_argument("--challenger-gst-audit", default=None)
    p.add_argument("--champion-gst-audit", default=None)
    p.add_argument("--decision-out", default=None)

    p.add_argument("--topk", type=int, default=3)
    p.add_argument("--max-tx-drop-ppm", type=int, default=2000)
    p.add_argument("--max-coa-top1-drop", type=float, default=0.004)
    p.add_argument("--max-coa-top3-drop", type=float, default=0.002)
    p.add_argument("--min-overall-delta", type=float, default=-0.001)

    p.add_argument("--min-family-examples", type=int, default=80)
    p.add_argument("--max-family-tx-drop", type=float, default=0.02)
    p.add_argument("--max-policy-blocked-top1-rate", type=float, default=0.0)
    p.add_argument("--max-policy-discouraged-top1-rate", type=float, default=0.25)
    p.add_argument("--max-policy-blocked-top1-rate-delta", type=float, default=0.0)
    p.add_argument("--max-policy-discouraged-top1-rate-delta", type=float, default=0.05)
    p.add_argument("--allow-period-lock-autoaccept", action="store_true")
    p.add_argument("--max-gst-major-fail-rate", type=float, default=1.0)
    p.add_argument("--max-gst-examples-with-issues-rate", type=float, default=1.0)
    p.add_argument("--max-gst-critical-issues", type=int, default=0)
    p.add_argument("--max-gst-major-fail-rate-delta", type=float, default=0.02)
    p.add_argument("--max-gst-issues-total-delta", type=int, default=100)
    p.add_argument("--max-gst-sale-missing-delta", type=int, default=100)
    p.add_argument("--max-gst-purchase-missing-delta", type=int, default=100)
    p.add_argument("--threshold-key", default="0.900000")
    p.add_argument("--min-auto-accept-rate-ppm", type=int, default=0)
    p.add_argument("--min-auto-accept-accuracy-ppm", type=int, default=900000)
    p.add_argument("--max-auto-accept-rate-drop-ppm", type=int, default=80000)
    p.add_argument("--max-auto-accept-accuracy-drop-ppm", type=int, default=30000)
    p.add_argument("--max-review-rate-increase-ppm", type=int, default=100000)
    return p


def main() -> int:
    args = _parser().parse_args()

    challenger_path = Path(args.challenger_scorecard)
    champion_path = Path(args.champion_scorecard) if args.champion_scorecard else None
    challenger_gst_path = Path(args.challenger_gst_audit) if args.challenger_gst_audit else None
    champion_gst_path = Path(args.champion_gst_audit) if args.champion_gst_audit else None

    challenger = _load(challenger_path)
    if challenger is None:
        raise SystemExit(f"Missing challenger scorecard: {challenger_path}")

    champion = _load(champion_path)
    challenger_gst = _load(challenger_gst_path)
    champion_gst = _load(champion_gst_path)

    ch_tx = _tx_accuracy(challenger)
    ch_tx_ppm = _tx_accuracy_ppm(challenger)
    ch_coa_top1 = _coa_top1_mean(challenger)
    ch_coa_top3 = _coa_top3_mean(challenger, args.topk)
    ch_overall = _overall_score(challenger, args.topk)
    ch_pol_debit_blocked = _policy_rate(challenger, "debit", "blocked_rate")
    ch_pol_credit_blocked = _policy_rate(challenger, "credit", "blocked_rate")
    ch_pol_debit_disc = _policy_rate(challenger, "debit", "discouraged_rate")
    ch_pol_credit_disc = _policy_rate(challenger, "credit", "discouraged_rate")
    ch_pol_blocked_max = max(ch_pol_debit_blocked, ch_pol_credit_blocked)
    ch_pol_disc_max = max(ch_pol_debit_disc, ch_pol_credit_disc)
    ch_period_lock_autoaccept_ppm = _period_lock_auto_accept_ppm(challenger, preferred_threshold=args.threshold_key)
    ch_auto_accept_rate_ppm = _threshold_stat_ppm(challenger, "auto_accept_rate_ppm", preferred_threshold=args.threshold_key)
    ch_auto_accept_accuracy_ppm = _threshold_stat_ppm(challenger, "auto_accept_accuracy_ppm", preferred_threshold=args.threshold_key)
    ch_review_rate_ppm = _threshold_stat_ppm(challenger, "review_rate_ppm", preferred_threshold=args.threshold_key)
    ch_gst_major_fail_rate = _gst_major_fail_rate(challenger_gst)
    ch_gst_examples_with_issues_rate = _gst_examples_with_issues_rate(challenger_gst)
    ch_gst_critical_issues = _gst_critical_issues(challenger_gst)
    ch_gst_issues_total = _gst_issues_total(challenger_gst)
    ch_gst_sale_missing = _gst_issue_code_count(challenger_gst, "SALE_TAX_OUTPUT_MISSING_CREDIT_TOPK")
    ch_gst_purchase_missing = _gst_issue_code_count(challenger_gst, "PURCHASE_TAX_INPUT_MISSING_DEBIT_TOPK")

    cmp_tx = _tx_accuracy(champion) if champion else None
    cmp_tx_ppm = _tx_accuracy_ppm(champion) if champion else None
    cmp_coa_top1 = _coa_top1_mean(champion) if champion else None
    cmp_coa_top3 = _coa_top3_mean(champion, args.topk) if champion else None
    cmp_overall = _overall_score(champion, args.topk) if champion else None
    cmp_pol_debit_blocked = _policy_rate(champion, "debit", "blocked_rate") if champion else None
    cmp_pol_credit_blocked = _policy_rate(champion, "credit", "blocked_rate") if champion else None
    cmp_pol_debit_disc = _policy_rate(champion, "debit", "discouraged_rate") if champion else None
    cmp_pol_credit_disc = _policy_rate(champion, "credit", "discouraged_rate") if champion else None
    cmp_pol_blocked_max = max(cmp_pol_debit_blocked, cmp_pol_credit_blocked) if champion else None
    cmp_pol_disc_max = max(cmp_pol_debit_disc, cmp_pol_credit_disc) if champion else None
    cmp_period_lock_autoaccept_ppm = _period_lock_auto_accept_ppm(champion, preferred_threshold=args.threshold_key) if champion else None
    cmp_auto_accept_rate_ppm = _threshold_stat_ppm(champion, "auto_accept_rate_ppm", preferred_threshold=args.threshold_key) if champion else None
    cmp_auto_accept_accuracy_ppm = _threshold_stat_ppm(champion, "auto_accept_accuracy_ppm", preferred_threshold=args.threshold_key) if champion else None
    cmp_review_rate_ppm = _threshold_stat_ppm(champion, "review_rate_ppm", preferred_threshold=args.threshold_key) if champion else None
    cmp_gst_major_fail_rate = _gst_major_fail_rate(champion_gst)
    cmp_gst_examples_with_issues_rate = _gst_examples_with_issues_rate(champion_gst)
    cmp_gst_critical_issues = _gst_critical_issues(champion_gst)
    cmp_gst_issues_total = _gst_issues_total(champion_gst)
    cmp_gst_sale_missing = _gst_issue_code_count(champion_gst, "SALE_TAX_OUTPUT_MISSING_CREDIT_TOPK")
    cmp_gst_purchase_missing = _gst_issue_code_count(champion_gst, "PURCHASE_TAX_INPUT_MISSING_DEBIT_TOPK")

    violations: list[str] = []

    if not args.allow_period_lock_autoaccept:
        if ch_period_lock_autoaccept_ppm is not None and int(ch_period_lock_autoaccept_ppm) > 0:
            violations.append(
                f"period_lock_autoaccept_ppm:{int(ch_period_lock_autoaccept_ppm)} > 0"
            )

    if ch_pol_blocked_max > float(args.max_policy_blocked_top1_rate):
        violations.append(
            f"policy_blocked_top1_rate:{ch_pol_blocked_max:.6f} > {float(args.max_policy_blocked_top1_rate):.6f}"
        )
    if ch_pol_disc_max > float(args.max_policy_discouraged_top1_rate):
        violations.append(
            f"policy_discouraged_top1_rate:{ch_pol_disc_max:.6f} > {float(args.max_policy_discouraged_top1_rate):.6f}"
        )
    if ch_auto_accept_rate_ppm is not None and int(ch_auto_accept_rate_ppm) < int(args.min_auto_accept_rate_ppm):
        violations.append(
            f"auto_accept_rate_ppm:{int(ch_auto_accept_rate_ppm)} < {int(args.min_auto_accept_rate_ppm)}"
        )
    if ch_auto_accept_accuracy_ppm is not None and int(ch_auto_accept_accuracy_ppm) < int(args.min_auto_accept_accuracy_ppm):
        violations.append(
            f"auto_accept_accuracy_ppm:{int(ch_auto_accept_accuracy_ppm)} < {int(args.min_auto_accept_accuracy_ppm)}"
        )
    if ch_gst_critical_issues is not None and int(ch_gst_critical_issues) > int(args.max_gst_critical_issues):
        violations.append(
            f"gst_critical_issues:{int(ch_gst_critical_issues)} > {int(args.max_gst_critical_issues)}"
        )
    if ch_gst_major_fail_rate is not None and float(ch_gst_major_fail_rate) > float(args.max_gst_major_fail_rate):
        violations.append(
            f"gst_major_fail_rate:{float(ch_gst_major_fail_rate):.6f} > {float(args.max_gst_major_fail_rate):.6f}"
        )
    if (
        ch_gst_examples_with_issues_rate is not None
        and float(ch_gst_examples_with_issues_rate) > float(args.max_gst_examples_with_issues_rate)
    ):
        violations.append(
            f"gst_examples_with_issues_rate:{float(ch_gst_examples_with_issues_rate):.6f} > {float(args.max_gst_examples_with_issues_rate):.6f}"
        )

    if champion is not None:
        tx_drop_ppm = int(cmp_tx_ppm) - int(ch_tx_ppm)
        if tx_drop_ppm > int(args.max_tx_drop_ppm):
            violations.append(
                f"tx_drop_ppm:{tx_drop_ppm} > {int(args.max_tx_drop_ppm)}"
            )

        coa_top1_drop = float(cmp_coa_top1) - float(ch_coa_top1)
        if coa_top1_drop > float(args.max_coa_top1_drop):
            violations.append(
                f"coa_top1_drop:{coa_top1_drop:.6f} > {float(args.max_coa_top1_drop):.6f}"
            )

        coa_top3_drop = float(cmp_coa_top3) - float(ch_coa_top3)
        if coa_top3_drop > float(args.max_coa_top3_drop):
            violations.append(
                f"coa_top3_drop:{coa_top3_drop:.6f} > {float(args.max_coa_top3_drop):.6f}"
            )

        overall_delta = float(ch_overall) - float(cmp_overall)
        if overall_delta < float(args.min_overall_delta):
            violations.append(
                f"overall_drop:{overall_delta:.6f} < {float(args.min_overall_delta):.6f}"
            )

        blocked_delta = float(ch_pol_blocked_max) - float(cmp_pol_blocked_max)
        if blocked_delta > float(args.max_policy_blocked_top1_rate_delta):
            violations.append(
                f"policy_blocked_top1_rate_delta:{blocked_delta:.6f} > {float(args.max_policy_blocked_top1_rate_delta):.6f}"
            )

        discouraged_delta = float(ch_pol_disc_max) - float(cmp_pol_disc_max)
        if discouraged_delta > float(args.max_policy_discouraged_top1_rate_delta):
            violations.append(
                f"policy_discouraged_top1_rate_delta:{discouraged_delta:.6f} > {float(args.max_policy_discouraged_top1_rate_delta):.6f}"
            )
        if cmp_gst_major_fail_rate is not None and ch_gst_major_fail_rate is not None:
            gst_major_fail_rate_delta = float(ch_gst_major_fail_rate) - float(cmp_gst_major_fail_rate)
            if gst_major_fail_rate_delta > float(args.max_gst_major_fail_rate_delta):
                violations.append(
                    f"gst_major_fail_rate_delta:{gst_major_fail_rate_delta:.6f} > {float(args.max_gst_major_fail_rate_delta):.6f}"
                )
        if cmp_gst_issues_total is not None and ch_gst_issues_total is not None:
            gst_issues_total_delta = int(ch_gst_issues_total) - int(cmp_gst_issues_total)
            if gst_issues_total_delta > int(args.max_gst_issues_total_delta):
                violations.append(
                    f"gst_issues_total_delta:{gst_issues_total_delta} > {int(args.max_gst_issues_total_delta)}"
                )
        if cmp_gst_sale_missing is not None and ch_gst_sale_missing is not None:
            gst_sale_missing_delta = int(ch_gst_sale_missing) - int(cmp_gst_sale_missing)
            if gst_sale_missing_delta > int(args.max_gst_sale_missing_delta):
                violations.append(
                    f"gst_sale_missing_delta:{gst_sale_missing_delta} > {int(args.max_gst_sale_missing_delta)}"
                )
        if cmp_gst_purchase_missing is not None and ch_gst_purchase_missing is not None:
            gst_purchase_missing_delta = int(ch_gst_purchase_missing) - int(cmp_gst_purchase_missing)
            if gst_purchase_missing_delta > int(args.max_gst_purchase_missing_delta):
                violations.append(
                    f"gst_purchase_missing_delta:{gst_purchase_missing_delta} > {int(args.max_gst_purchase_missing_delta)}"
                )
        if cmp_auto_accept_rate_ppm is not None and ch_auto_accept_rate_ppm is not None:
            auto_accept_rate_drop_ppm = int(cmp_auto_accept_rate_ppm) - int(ch_auto_accept_rate_ppm)
            if auto_accept_rate_drop_ppm > int(args.max_auto_accept_rate_drop_ppm):
                violations.append(
                    f"auto_accept_rate_drop_ppm:{auto_accept_rate_drop_ppm} > {int(args.max_auto_accept_rate_drop_ppm)}"
                )
        if cmp_auto_accept_accuracy_ppm is not None and ch_auto_accept_accuracy_ppm is not None:
            auto_accept_accuracy_drop_ppm = int(cmp_auto_accept_accuracy_ppm) - int(ch_auto_accept_accuracy_ppm)
            if auto_accept_accuracy_drop_ppm > int(args.max_auto_accept_accuracy_drop_ppm):
                violations.append(
                    f"auto_accept_accuracy_drop_ppm:{auto_accept_accuracy_drop_ppm} > {int(args.max_auto_accept_accuracy_drop_ppm)}"
                )
        if cmp_review_rate_ppm is not None and ch_review_rate_ppm is not None:
            review_rate_increase_ppm = int(ch_review_rate_ppm) - int(cmp_review_rate_ppm)
            if review_rate_increase_ppm > int(args.max_review_rate_increase_ppm):
                violations.append(
                    f"review_rate_increase_ppm:{review_rate_increase_ppm} > {int(args.max_review_rate_increase_ppm)}"
                )

        challenger_families = dict(challenger.get("workflow_families") or {})
        champion_families = dict(champion.get("workflow_families") or {})
        for family in sorted(set(challenger_families.keys()) & set(champion_families.keys())):
            ch_f = dict(challenger_families.get(family) or {})
            cmp_f = dict(champion_families.get(family) or {})
            n_ch = int(ch_f.get("examples_tx", 0))
            n_cmp = int(cmp_f.get("examples_tx", 0))
            if min(n_ch, n_cmp) < int(args.min_family_examples):
                continue
            drop = float(cmp_f.get("tx_accuracy", 0.0)) - float(ch_f.get("tx_accuracy", 0.0))
            if drop > float(args.max_family_tx_drop):
                violations.append(
                    f"family_tx_drop:{family}:{drop:.6f} > {float(args.max_family_tx_drop):.6f}"
                )

    promote = champion is None or not violations

    report = {
        "schema_version": "v0",
        "promote": bool(promote),
        "guardrails": {
            "topk": int(args.topk),
            "max_tx_drop_ppm": int(args.max_tx_drop_ppm),
            "max_coa_top1_drop": float(args.max_coa_top1_drop),
            "max_coa_top3_drop": float(args.max_coa_top3_drop),
            "min_overall_delta": float(args.min_overall_delta),
            "min_family_examples": int(args.min_family_examples),
            "max_family_tx_drop": float(args.max_family_tx_drop),
            "max_policy_blocked_top1_rate": float(args.max_policy_blocked_top1_rate),
            "max_policy_discouraged_top1_rate": float(args.max_policy_discouraged_top1_rate),
            "max_policy_blocked_top1_rate_delta": float(args.max_policy_blocked_top1_rate_delta),
            "max_policy_discouraged_top1_rate_delta": float(args.max_policy_discouraged_top1_rate_delta),
            "allow_period_lock_autoaccept": bool(args.allow_period_lock_autoaccept),
            "max_gst_major_fail_rate": float(args.max_gst_major_fail_rate),
            "max_gst_examples_with_issues_rate": float(args.max_gst_examples_with_issues_rate),
            "max_gst_critical_issues": int(args.max_gst_critical_issues),
            "max_gst_major_fail_rate_delta": float(args.max_gst_major_fail_rate_delta),
            "max_gst_issues_total_delta": int(args.max_gst_issues_total_delta),
            "max_gst_sale_missing_delta": int(args.max_gst_sale_missing_delta),
            "max_gst_purchase_missing_delta": int(args.max_gst_purchase_missing_delta),
            "threshold_key": str(args.threshold_key),
            "min_auto_accept_rate_ppm": int(args.min_auto_accept_rate_ppm),
            "min_auto_accept_accuracy_ppm": int(args.min_auto_accept_accuracy_ppm),
            "max_auto_accept_rate_drop_ppm": int(args.max_auto_accept_rate_drop_ppm),
            "max_auto_accept_accuracy_drop_ppm": int(args.max_auto_accept_accuracy_drop_ppm),
            "max_review_rate_increase_ppm": int(args.max_review_rate_increase_ppm),
        },
        "champion_scorecard": champion_path.as_posix() if champion_path else None,
        "challenger_scorecard": challenger_path.as_posix(),
        "champion_gst_audit": champion_gst_path.as_posix() if champion_gst_path else None,
        "challenger_gst_audit": challenger_gst_path.as_posix() if challenger_gst_path else None,
        "scores": {
            "champion": {
                "tx_accuracy": cmp_tx,
                "tx_accuracy_ppm": cmp_tx_ppm,
                "coa_top1_mean": cmp_coa_top1,
                "coa_top3_mean": cmp_coa_top3,
                "overall": cmp_overall,
                "policy_blocked_top1_rate_max": cmp_pol_blocked_max,
                "policy_discouraged_top1_rate_max": cmp_pol_disc_max,
                "period_lock_autoaccept_ppm": cmp_period_lock_autoaccept_ppm,
                "auto_accept_rate_ppm": cmp_auto_accept_rate_ppm,
                "auto_accept_accuracy_ppm": cmp_auto_accept_accuracy_ppm,
                "review_rate_ppm": cmp_review_rate_ppm,
                "gst_major_fail_rate": cmp_gst_major_fail_rate,
                "gst_examples_with_issues_rate": cmp_gst_examples_with_issues_rate,
                "gst_critical_issues": cmp_gst_critical_issues,
                "gst_issues_total": cmp_gst_issues_total,
                "gst_sale_missing": cmp_gst_sale_missing,
                "gst_purchase_missing": cmp_gst_purchase_missing,
            },
            "challenger": {
                "tx_accuracy": ch_tx,
                "tx_accuracy_ppm": ch_tx_ppm,
                "coa_top1_mean": ch_coa_top1,
                "coa_top3_mean": ch_coa_top3,
                "overall": ch_overall,
                "policy_blocked_top1_rate_max": ch_pol_blocked_max,
                "policy_discouraged_top1_rate_max": ch_pol_disc_max,
                "period_lock_autoaccept_ppm": ch_period_lock_autoaccept_ppm,
                "auto_accept_rate_ppm": ch_auto_accept_rate_ppm,
                "auto_accept_accuracy_ppm": ch_auto_accept_accuracy_ppm,
                "review_rate_ppm": ch_review_rate_ppm,
                "gst_major_fail_rate": ch_gst_major_fail_rate,
                "gst_examples_with_issues_rate": ch_gst_examples_with_issues_rate,
                "gst_critical_issues": ch_gst_critical_issues,
                "gst_issues_total": ch_gst_issues_total,
                "gst_sale_missing": ch_gst_sale_missing,
                "gst_purchase_missing": ch_gst_purchase_missing,
            },
            "delta": {
                "tx_accuracy": None if cmp_tx is None else (ch_tx - cmp_tx),
                "tx_accuracy_ppm": None if cmp_tx_ppm is None else (ch_tx_ppm - cmp_tx_ppm),
                "coa_top1_mean": None if cmp_coa_top1 is None else (ch_coa_top1 - cmp_coa_top1),
                "coa_top3_mean": None if cmp_coa_top3 is None else (ch_coa_top3 - cmp_coa_top3),
                "overall": None if cmp_overall is None else (ch_overall - cmp_overall),
                "policy_blocked_top1_rate_max": None if cmp_pol_blocked_max is None else (ch_pol_blocked_max - cmp_pol_blocked_max),
                "policy_discouraged_top1_rate_max": None if cmp_pol_disc_max is None else (ch_pol_disc_max - cmp_pol_disc_max),
                "period_lock_autoaccept_ppm": None if cmp_period_lock_autoaccept_ppm is None else (int(ch_period_lock_autoaccept_ppm or 0) - int(cmp_period_lock_autoaccept_ppm)),
                "auto_accept_rate_ppm": None if cmp_auto_accept_rate_ppm is None or ch_auto_accept_rate_ppm is None else (int(ch_auto_accept_rate_ppm) - int(cmp_auto_accept_rate_ppm)),
                "auto_accept_accuracy_ppm": None if cmp_auto_accept_accuracy_ppm is None or ch_auto_accept_accuracy_ppm is None else (int(ch_auto_accept_accuracy_ppm) - int(cmp_auto_accept_accuracy_ppm)),
                "review_rate_ppm": None if cmp_review_rate_ppm is None or ch_review_rate_ppm is None else (int(ch_review_rate_ppm) - int(cmp_review_rate_ppm)),
                "gst_major_fail_rate": None if cmp_gst_major_fail_rate is None or ch_gst_major_fail_rate is None else (ch_gst_major_fail_rate - cmp_gst_major_fail_rate),
                "gst_examples_with_issues_rate": None if cmp_gst_examples_with_issues_rate is None or ch_gst_examples_with_issues_rate is None else (ch_gst_examples_with_issues_rate - cmp_gst_examples_with_issues_rate),
                "gst_critical_issues": None if cmp_gst_critical_issues is None or ch_gst_critical_issues is None else (int(ch_gst_critical_issues) - int(cmp_gst_critical_issues)),
                "gst_issues_total": None if cmp_gst_issues_total is None or ch_gst_issues_total is None else (int(ch_gst_issues_total) - int(cmp_gst_issues_total)),
                "gst_sale_missing": None if cmp_gst_sale_missing is None or ch_gst_sale_missing is None else (int(ch_gst_sale_missing) - int(cmp_gst_sale_missing)),
                "gst_purchase_missing": None if cmp_gst_purchase_missing is None or ch_gst_purchase_missing is None else (int(ch_gst_purchase_missing) - int(cmp_gst_purchase_missing)),
            },
        },
        "violations": violations,
    }

    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.decision_out:
        out = Path(args.decision_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(payload, encoding="utf-8")

    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
