#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

LABEL_TO_FAMILY = {
    "SALE": "sale",
    "SALE_RETURN": "sale",
    "PURCHASE": "purchase",
    "PAYMENT": "payment",
    "COGS": "cogs",
    "WRITE_OFF": "write_off",
    "INVENTORY_COUNT": "inventory_count",
    "PERIOD_LOCK": "period_lock",
    "PAYROLL": "payroll",
    "PAYROLL_JOURNAL": "payroll",
    "TAX_SETTLEMENT": "tax_settlement",
}

HIGH_RISK_FAMILIES = {
    "period_lock",
    "write_off",
    "inventory_count",
    "tax_settlement",
    "payroll",
}


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _round6(value: float) -> float:
    return float(f"{value:.6f}")


def _candidate_thresholds(scorecard: dict[str, Any]) -> list[str]:
    stats = (
        scorecard.get("tx_eval_snapshot", {})
        .get("threshold_stats", {})
    )
    if not isinstance(stats, dict) or not stats:
        return ["0.950000", "0.900000", "0.800000"]

    keys: list[tuple[float, str]] = []
    for key in stats.keys():
        try:
            keys.append((float(key), str(key)))
        except Exception:
            continue
    if not keys:
        return ["0.950000", "0.900000", "0.800000"]
    return [k for _f, k in sorted(keys, key=lambda t: t[0], reverse=True)]


def _family_examples(scorecard: dict[str, Any]) -> dict[str, int]:
    tx_by_label = dict(scorecard.get("tx", {}).get("by_label") or {})
    out: dict[str, int] = {}
    for label, payload in tx_by_label.items():
        family = LABEL_TO_FAMILY.get(str(label).upper(), "other")
        n = _safe_int(dict(payload).get("examples"), 0)
        out[family] = out.get(family, 0) + n
    return out


def _family_auto_accept_rate_ppm(
    scorecard: dict[str, Any],
    *,
    threshold_key: str,
) -> int | None:
    thresholds = (
        scorecard.get("tx_eval_snapshot", {})
        .get("threshold_stats", {})
    )
    if not isinstance(thresholds, dict):
        return None
    payload = dict(thresholds.get(threshold_key) or {})
    if not payload:
        return None
    return _safe_int(payload.get("auto_accept_rate_ppm"), 0)


def _family_auto_accept_rate_ppm_by_label(
    scorecard: dict[str, Any],
    *,
    threshold_key: str,
) -> dict[str, int]:
    thresholds = (
        scorecard.get("tx_eval_snapshot", {})
        .get("threshold_stats", {})
    )
    tx_by_label = dict(scorecard.get("tx", {}).get("by_label") or {})
    payload = dict(dict(thresholds).get(threshold_key) or {})
    by_true_label = dict(payload.get("by_true_label") or {})

    family_auto: dict[str, int] = {}
    family_total: dict[str, int] = {}
    for label, m in tx_by_label.items():
        label_norm = str(label).upper()
        family = LABEL_TO_FAMILY.get(label_norm, "other")
        examples = _safe_int(dict(m).get("examples"), 0)
        if examples <= 0:
            continue
        s = dict(by_true_label.get(label_norm) or {})
        auto_accepted = _safe_int(s.get("auto_accepted"), 0)
        if auto_accepted <= 0 and examples > 0:
            rate_ppm = _safe_int(s.get("auto_accept_rate_ppm"), 0)
            auto_accepted = int(round((float(rate_ppm) / 1_000_000.0) * float(examples)))
        family_auto[family] = family_auto.get(family, 0) + auto_accepted
        family_total[family] = family_total.get(family, 0) + examples

    out: dict[str, int] = {}
    for family, total in family_total.items():
        if total <= 0:
            continue
        out[family] = int(round((float(family_auto.get(family, 0)) / float(total)) * 1_000_000.0))
    return out


def _threshold_auto_accept_accuracy_ppm(scorecard: dict[str, Any], threshold_key: str) -> int:
    thresholds = (
        scorecard.get("tx_eval_snapshot", {})
        .get("threshold_stats", {})
    )
    payload = dict(dict(thresholds).get(threshold_key) or {})
    return _safe_int(payload.get("auto_accept_accuracy_ppm"), 0)


def _choose_threshold_for_family(
    *,
    family: str,
    family_examples: int,
    candidates: list[str],
    family_auto_by_threshold: dict[str, dict[str, int]],
    scorecard: dict[str, Any],
    min_family_examples: int,
    target_auto_accept_rate_ppm: int,
    min_auto_accept_accuracy_ppm: int,
    preferred_threshold: float,
    conservative_threshold: float,
) -> tuple[str, str]:
    # Insufficient data: choose conservative threshold.
    if family_examples < min_family_examples:
        return (f"{conservative_threshold:.6f}", "insufficient_family_examples_conservative")

    feasible: list[str] = []
    for key in candidates:
        acc_ppm = _threshold_auto_accept_accuracy_ppm(scorecard, key)
        if acc_ppm < min_auto_accept_accuracy_ppm:
            continue
        feasible.append(key)
    if not feasible:
        return (f"{conservative_threshold:.6f}", "no_feasible_threshold_conservative")

    if family in HIGH_RISK_FAMILIES:
        # High-risk: choose strictest feasible threshold.
        return (feasible[0], "high_risk_family_strict_threshold")

    for key in reversed(feasible):
        fam_rate = _safe_int(family_auto_by_threshold.get(key, {}).get(family), 0)
        if fam_rate >= target_auto_accept_rate_ppm:
            return (key, "target_auto_accept_rate_satisfied")

    # Fall back to closest feasible to preferred threshold.
    preferred_key = min(
        feasible,
        key=lambda k: abs(_safe_float(k, preferred_threshold) - preferred_threshold),
    )
    return (preferred_key, "closest_to_preferred_threshold")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build workflow-specific threshold + routing policy from tx/coA scorecard.")
    p.add_argument("--scorecard", required=True)
    p.add_argument("--out-policy-json", required=True)
    p.add_argument("--out-summary-json", default=None)
    p.add_argument("--min-family-examples", type=int, default=80)
    p.add_argument("--target-auto-accept-rate-ppm", type=int, default=850000)
    p.add_argument("--min-auto-accept-accuracy-ppm", type=int, default=980000)
    p.add_argument("--preferred-threshold", type=float, default=0.90)
    p.add_argument("--conservative-threshold", type=float, default=0.95)
    p.add_argument("--always-review-label", action="append", default=["PERIOD_LOCK"])
    return p


def main() -> int:
    args = _build_parser().parse_args()

    scorecard_path = Path(args.scorecard)
    if not scorecard_path.exists() or not scorecard_path.is_file():
        raise SystemExit(f"Scorecard not found: {scorecard_path}")

    scorecard = json.loads(scorecard_path.read_text(encoding="utf-8"))
    candidates = _candidate_thresholds(scorecard)
    families = _family_examples(scorecard)

    family_auto_by_threshold: dict[str, dict[str, int]] = {}
    for key in candidates:
        family_auto_by_threshold[key] = _family_auto_accept_rate_ppm_by_label(scorecard, threshold_key=key)

    family_policy: dict[str, Any] = {}
    for family in sorted(families.keys()):
        chosen_key, rationale = _choose_threshold_for_family(
            family=family,
            family_examples=_safe_int(families.get(family), 0),
            candidates=candidates,
            family_auto_by_threshold=family_auto_by_threshold,
            scorecard=scorecard,
            min_family_examples=int(args.min_family_examples),
            target_auto_accept_rate_ppm=int(args.target_auto_accept_rate_ppm),
            min_auto_accept_accuracy_ppm=int(args.min_auto_accept_accuracy_ppm),
            preferred_threshold=float(args.preferred_threshold),
            conservative_threshold=float(args.conservative_threshold),
        )

        acc_ppm = _threshold_auto_accept_accuracy_ppm(scorecard, chosen_key)
        fam_auto_ppm = _safe_int(family_auto_by_threshold.get(chosen_key, {}).get(family), 0)
        family_policy[family] = {
            "threshold": _round6(_safe_float(chosen_key)),
            "threshold_key": chosen_key,
            "examples": _safe_int(families.get(family), 0),
            "estimated_auto_accept_rate_ppm": fam_auto_ppm,
            "threshold_auto_accept_accuracy_ppm": acc_ppm,
            "risk_profile": "high" if family in HIGH_RISK_FAMILIES else "standard",
            "rationale": rationale,
        }

    overall_preferred_rate_ppm = _family_auto_accept_rate_ppm(
        scorecard,
        threshold_key=f"{float(args.preferred_threshold):.6f}",
    )

    policy = {
        "schema_version": "v0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_scorecard": scorecard_path.as_posix(),
        "policy_kind": "workflow_threshold_routing",
        "global": {
            "min_family_examples": int(args.min_family_examples),
            "target_auto_accept_rate_ppm": int(args.target_auto_accept_rate_ppm),
            "min_auto_accept_accuracy_ppm": int(args.min_auto_accept_accuracy_ppm),
            "preferred_threshold": _round6(float(args.preferred_threshold)),
            "conservative_threshold": _round6(float(args.conservative_threshold)),
            "always_review_labels": sorted({str(v).upper() for v in list(args.always_review_label or []) if str(v).strip()}),
            "preferred_threshold_auto_accept_rate_ppm": overall_preferred_rate_ppm,
        },
        "families": family_policy,
    }

    out_policy_path = Path(args.out_policy_json)
    out_policy_path.parent.mkdir(parents=True, exist_ok=True)
    out_policy_path.write_text(json.dumps(policy, indent=2, sort_keys=True), encoding="utf-8")

    summary = {
        "schema_version": "v0",
        "source_scorecard": scorecard_path.as_posix(),
        "policy_json": out_policy_path.as_posix(),
        "families": {
            k: {
                "threshold": v["threshold"],
                "examples": v["examples"],
                "risk_profile": v["risk_profile"],
                "estimated_auto_accept_rate_ppm": v["estimated_auto_accept_rate_ppm"],
            }
            for k, v in sorted(family_policy.items(), key=lambda t: t[0])
        },
        "family_count": len(family_policy),
    }

    if args.out_summary_json:
        out_summary_path = Path(args.out_summary_json)
        out_summary_path.parent.mkdir(parents=True, exist_ok=True)
        out_summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    print(out_policy_path.as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
