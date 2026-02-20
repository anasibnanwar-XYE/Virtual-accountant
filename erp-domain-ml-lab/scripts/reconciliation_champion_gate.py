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


def _extract_holdout(metrics_obj: dict[str, Any] | None) -> dict[str, float] | None:
    if not metrics_obj:
        return None
    root = metrics_obj.get("metrics", metrics_obj)
    hold = dict(root.get("holdout") or {})
    if not hold:
        return None

    out: dict[str, float] = {}
    for k in ("accuracy", "precision", "recall", "f1", "fnr", "fpr", "positive_rate", "predicted_positive_rate"):
        try:
            out[k] = float(hold.get(k, 0.0))
        except Exception:
            out[k] = 0.0
    try:
        out["examples"] = float(hold.get("examples", 0))
    except Exception:
        out["examples"] = 0.0
    return out


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Champion/challenger gate for reconciliation-exception model")
    p.add_argument("--challenger-metrics", required=True)
    p.add_argument("--champion-metrics", default=None)
    p.add_argument("--decision-out", default=None)

    p.add_argument("--max-fnr-increase", type=float, default=0.02)
    p.add_argument("--max-recall-drop", type=float, default=0.02)
    p.add_argument("--min-f1-delta", type=float, default=-0.01)
    p.add_argument("--min-precision", type=float, default=0.55)
    p.add_argument("--min-accuracy", type=float, default=0.65)
    p.add_argument("--max-predicted-positive-rate-increase", type=float, default=0.08)
    p.add_argument("--max-predicted-positive-rate-drop", type=float, default=0.08)
    return p


def main() -> int:
    args = _parser().parse_args()

    challenger_path = Path(args.challenger_metrics)
    champion_path = Path(args.champion_metrics) if args.champion_metrics else None

    challenger_obj = _load(challenger_path)
    if challenger_obj is None:
        raise SystemExit(f"Challenger metrics not found: {challenger_path}")
    champion_obj = _load(champion_path)

    challenger = _extract_holdout(challenger_obj)
    if challenger is None:
        raise SystemExit("Challenger holdout metrics missing")
    champion = _extract_holdout(champion_obj)

    violations: list[str] = []

    if challenger["precision"] < float(args.min_precision):
        violations.append(f"precision_floor:{challenger['precision']:.6f} < {float(args.min_precision):.6f}")
    if challenger["accuracy"] < float(args.min_accuracy):
        violations.append(f"accuracy_floor:{challenger['accuracy']:.6f} < {float(args.min_accuracy):.6f}")

    deltas: dict[str, float | None] = {
        "recall_delta": None,
        "f1_delta": None,
        "fnr_delta": None,
        "precision_delta": None,
        "accuracy_delta": None,
        "predicted_positive_rate_delta": None,
        "positive_rate_delta": None,
    }

    if champion is not None:
        recall_delta = challenger["recall"] - champion["recall"]
        f1_delta = challenger["f1"] - champion["f1"]
        fnr_delta = challenger["fnr"] - champion["fnr"]
        precision_delta = challenger["precision"] - champion["precision"]
        accuracy_delta = challenger["accuracy"] - champion["accuracy"]
        predicted_positive_rate_delta = challenger["predicted_positive_rate"] - champion["predicted_positive_rate"]
        positive_rate_delta = challenger["positive_rate"] - champion["positive_rate"]

        deltas = {
            "recall_delta": recall_delta,
            "f1_delta": f1_delta,
            "fnr_delta": fnr_delta,
            "precision_delta": precision_delta,
            "accuracy_delta": accuracy_delta,
            "predicted_positive_rate_delta": predicted_positive_rate_delta,
            "positive_rate_delta": positive_rate_delta,
        }

        if recall_delta < -float(args.max_recall_drop):
            violations.append(f"recall_drop:{recall_delta:.6f} < -{float(args.max_recall_drop):.6f}")
        if f1_delta < float(args.min_f1_delta):
            violations.append(f"f1_drop:{f1_delta:.6f} < {float(args.min_f1_delta):.6f}")
        if fnr_delta > float(args.max_fnr_increase):
            violations.append(f"fnr_increase:{fnr_delta:.6f} > {float(args.max_fnr_increase):.6f}")
        if predicted_positive_rate_delta > float(args.max_predicted_positive_rate_increase):
            violations.append(
                f"predicted_positive_rate_increase:{predicted_positive_rate_delta:.6f} > {float(args.max_predicted_positive_rate_increase):.6f}"
            )
        if predicted_positive_rate_delta < -float(args.max_predicted_positive_rate_drop):
            violations.append(
                f"predicted_positive_rate_drop:{predicted_positive_rate_delta:.6f} < -{float(args.max_predicted_positive_rate_drop):.6f}"
            )

    promote = len(violations) == 0
    if champion is None:
        promote = True

    report = {
        "schema_version": "v0",
        "promote": bool(promote),
        "champion_metrics": champion_path.as_posix() if champion_path else None,
        "challenger_metrics": challenger_path.as_posix(),
        "guardrails": {
            "max_fnr_increase": float(args.max_fnr_increase),
            "max_recall_drop": float(args.max_recall_drop),
            "min_f1_delta": float(args.min_f1_delta),
            "min_precision": float(args.min_precision),
            "min_accuracy": float(args.min_accuracy),
            "max_predicted_positive_rate_increase": float(args.max_predicted_positive_rate_increase),
            "max_predicted_positive_rate_drop": float(args.max_predicted_positive_rate_drop),
        },
        "scores": {
            "champion_holdout": champion,
            "challenger_holdout": challenger,
            "deltas": deltas,
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
