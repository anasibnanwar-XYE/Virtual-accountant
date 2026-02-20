#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

CORE_TARGETS = ["revenue", "cogs", "inventory", "tax"]
ALL_TARGETS = ["revenue", "cogs", "inventory", "tax", "discount"]


def _load(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    if not path.exists() or not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_holdout_top1(metrics_obj: dict[str, Any] | None) -> dict[str, float]:
    if not metrics_obj:
        return {}
    top = metrics_obj.get("metrics", {})
    if "targets" in top:
        targets = top.get("targets", {})
    else:
        targets = top.get("metrics", {}).get("targets", {})
    out: dict[str, float] = {}
    for t in ALL_TARGETS:
        try:
            out[t] = float(targets.get(t, {}).get("holdout", {}).get("top1", 0.0))
        except Exception:
            out[t] = 0.0
    return out


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / float(len(values))


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Champion/challenger gating for product-account recommender")
    p.add_argument("--challenger-metrics", required=True)
    p.add_argument("--champion-metrics", default=None)
    p.add_argument("--decision-out", default=None)
    p.add_argument("--max-degrade-core", type=float, default=0.005)
    p.add_argument("--max-degrade-discount", type=float, default=0.02)
    p.add_argument("--min-overall-delta", type=float, default=-0.001)
    return p


def main() -> int:
    args = _build_parser().parse_args()

    challenger_path = Path(args.challenger_metrics)
    champion_path = Path(args.champion_metrics) if args.champion_metrics else None

    challenger = _load(challenger_path)
    if challenger is None:
        raise SystemExit(f"Challenger metrics not found: {challenger_path}")

    champion = _load(champion_path)

    challenger_scores = _extract_holdout_top1(challenger)
    champion_scores = _extract_holdout_top1(champion)

    challenger_overall = _mean([challenger_scores.get(t, 0.0) for t in ALL_TARGETS])
    champion_overall = _mean([champion_scores.get(t, 0.0) for t in ALL_TARGETS]) if champion else None

    per_target: dict[str, dict[str, float | None]] = {}
    violations: list[str] = []

    for t in ALL_TARGETS:
        c = champion_scores.get(t) if champion else None
        n = challenger_scores.get(t, 0.0)
        delta = (n - c) if c is not None else None
        per_target[t] = {
            "champion_holdout_top1": c,
            "challenger_holdout_top1": n,
            "delta": delta,
        }
        if c is None:
            continue
        if t in CORE_TARGETS and delta is not None and delta < -float(args.max_degrade_core):
            violations.append(
                f"core_target_drop:{t}:{delta:.6f} < -{float(args.max_degrade_core):.6f}"
            )
        if t == "discount" and delta is not None and delta < -float(args.max_degrade_discount):
            violations.append(
                f"discount_drop:{t}:{delta:.6f} < -{float(args.max_degrade_discount):.6f}"
            )

    overall_delta = None
    if champion_overall is not None:
        overall_delta = challenger_overall - champion_overall
        if overall_delta < float(args.min_overall_delta):
            violations.append(
                f"overall_drop:{overall_delta:.6f} < {float(args.min_overall_delta):.6f}"
            )

    promote = len(violations) == 0
    if champion is None:
        promote = True

    report = {
        "schema_version": "v0",
        "promote": bool(promote),
        "champion_metrics": champion_path.as_posix() if champion_path else None,
        "challenger_metrics": challenger_path.as_posix(),
        "overall": {
            "champion_mean_holdout_top1": champion_overall,
            "challenger_mean_holdout_top1": challenger_overall,
            "delta": overall_delta,
        },
        "per_target": per_target,
        "guardrails": {
            "max_degrade_core": float(args.max_degrade_core),
            "max_degrade_discount": float(args.max_degrade_discount),
            "min_overall_delta": float(args.min_overall_delta),
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
