#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

VALID_STAGES = ("permissive", "stage1", "stage2", "stage3")


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists() or not path.is_file():
        raise SystemExit(f"Missing required JSON file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.exists() or not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _default_state(initial_stage: str) -> dict[str, Any]:
    if initial_stage not in VALID_STAGES:
        initial_stage = "stage1"
    return {
        "schema_version": "v0",
        "current_stage": initial_stage,
        "consecutive_gst_passes": 0,
        "consecutive_gst_failures": 0,
        "total_cycles_seen": 0,
        "last_cycle_summary": None,
        "last_transition": None,
        "history": [],
    }


def _is_gst_pass(cycle: dict[str, Any], *, require_promoted_status: bool) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    status = str(cycle.get("status") or "").strip().lower()
    if require_promoted_status and status != "promoted":
        reasons.append("cycle_not_promoted")

    gate = dict(cycle.get("gate_decision") or {})
    violations = [str(v) for v in list(gate.get("violations") or [])]
    gst_violations = [v for v in violations if v.startswith("gst_")]
    if gst_violations:
        reasons.append("gst_violations_present")

    challenger = dict((gate.get("scores") or {}).get("challenger") or {})
    gst_major_fail_rate = challenger.get("gst_major_fail_rate")
    gst_issue_rate = challenger.get("gst_examples_with_issues_rate")
    gst_critical_issues = challenger.get("gst_critical_issues")
    if gst_major_fail_rate is None:
        reasons.append("missing_gst_major_fail_rate")
    if gst_issue_rate is None:
        reasons.append("missing_gst_issue_rate")
    if gst_critical_issues is None:
        reasons.append("missing_gst_critical_issues")

    passed = len(reasons) == 0
    return passed, reasons


def _append_history(state: dict[str, Any], row: dict[str, Any], *, max_history: int) -> None:
    history = list(state.get("history") or [])
    history.append(row)
    if max_history > 0 and len(history) > max_history:
        history = history[-max_history:]
    state["history"] = history


def _transition(
    state: dict[str, Any],
    *,
    new_stage: str,
    reason: str,
    cycle_summary_path: str,
) -> None:
    old_stage = str(state.get("current_stage") or "stage1")
    if new_stage == old_stage:
        return
    state["current_stage"] = new_stage
    state["last_transition"] = {
        "at_utc": datetime.now(timezone.utc).isoformat(),
        "from_stage": old_stage,
        "to_stage": new_stage,
        "reason": reason,
        "cycle_summary": cycle_summary_path,
    }


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Update persistent GST guardrail stage state from tx+coa cycle summary")
    p.add_argument("--cycle-summary", required=True)
    p.add_argument("--state-json", required=True)
    p.add_argument("--out-json", required=True)
    p.add_argument("--initial-stage", default="stage1", choices=list(VALID_STAGES))
    p.add_argument("--require-promoted-status", action="store_true")
    p.add_argument("--promote-stage1-after", type=int, default=2)
    p.add_argument("--promote-stage2-after", type=int, default=3)
    p.add_argument("--promote-stage3-after", type=int, default=5)
    p.add_argument("--demote-after-fails", type=int, default=2)
    p.add_argument("--allow-demote-to-permissive", action="store_true")
    p.add_argument("--max-history", type=int, default=200)
    return p


def main() -> int:
    args = _parser().parse_args()

    cycle_path = Path(args.cycle_summary)
    state_path = Path(args.state_json)
    out_path = Path(args.out_json)

    cycle = _load_json(cycle_path)
    state = _load_optional_json(state_path)
    if state is None:
        state = _default_state(initial_stage=args.initial_stage)

    current_stage = str(state.get("current_stage") or args.initial_stage)
    if current_stage not in VALID_STAGES:
        current_stage = args.initial_stage
        state["current_stage"] = current_stage

    before_passes = int(state.get("consecutive_gst_passes", 0))
    before_failures = int(state.get("consecutive_gst_failures", 0))
    before_stage = str(state.get("current_stage") or current_stage)

    gst_pass, fail_reasons = _is_gst_pass(
        cycle,
        require_promoted_status=bool(args.require_promoted_status),
    )

    if gst_pass:
        state["consecutive_gst_passes"] = int(state.get("consecutive_gst_passes", 0)) + 1
        state["consecutive_gst_failures"] = 0
    else:
        state["consecutive_gst_passes"] = 0
        state["consecutive_gst_failures"] = int(state.get("consecutive_gst_failures", 0)) + 1

    state["total_cycles_seen"] = int(state.get("total_cycles_seen", 0)) + 1
    state["last_cycle_summary"] = cycle_path.as_posix()

    # Upward progression (based on pass streak)
    pass_streak = int(state.get("consecutive_gst_passes", 0))
    if current_stage == "permissive" and pass_streak >= int(args.promote_stage1_after):
        _transition(
            state,
            new_stage="stage1",
            reason=f"pass_streak_reached:{pass_streak}",
            cycle_summary_path=cycle_path.as_posix(),
        )
    elif current_stage == "stage1" and pass_streak >= int(args.promote_stage2_after):
        _transition(
            state,
            new_stage="stage2",
            reason=f"pass_streak_reached:{pass_streak}",
            cycle_summary_path=cycle_path.as_posix(),
        )
    elif current_stage == "stage2" and pass_streak >= int(args.promote_stage3_after):
        _transition(
            state,
            new_stage="stage3",
            reason=f"pass_streak_reached:{pass_streak}",
            cycle_summary_path=cycle_path.as_posix(),
        )

    # Downward progression (based on fail streak)
    fail_streak = int(state.get("consecutive_gst_failures", 0))
    current_stage = str(state.get("current_stage") or current_stage)
    if fail_streak >= int(args.demote_after_fails):
        if current_stage == "stage3":
            _transition(
                state,
                new_stage="stage2",
                reason=f"fail_streak_reached:{fail_streak}",
                cycle_summary_path=cycle_path.as_posix(),
            )
            state["consecutive_gst_failures"] = 0
        elif current_stage == "stage2":
            _transition(
                state,
                new_stage="stage1",
                reason=f"fail_streak_reached:{fail_streak}",
                cycle_summary_path=cycle_path.as_posix(),
            )
            state["consecutive_gst_failures"] = 0
        elif current_stage == "stage1" and bool(args.allow_demote_to_permissive):
            _transition(
                state,
                new_stage="permissive",
                reason=f"fail_streak_reached:{fail_streak}",
                cycle_summary_path=cycle_path.as_posix(),
            )
            state["consecutive_gst_failures"] = 0

    update = {
        "schema_version": "v0",
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "cycle_summary": cycle_path.as_posix(),
        "gst_pass": bool(gst_pass),
        "gst_fail_reasons": fail_reasons,
        "state_before": {
            "current_stage": before_stage,
            "consecutive_gst_passes": before_passes,
            "consecutive_gst_failures": before_failures,
        },
        "state_after": {
            "current_stage": str(state.get("current_stage")),
            "consecutive_gst_passes": int(state.get("consecutive_gst_passes", 0)),
            "consecutive_gst_failures": int(state.get("consecutive_gst_failures", 0)),
            "total_cycles_seen": int(state.get("total_cycles_seen", 0)),
        },
        "last_transition": state.get("last_transition"),
        "next_profile": str(state.get("current_stage")),
    }

    _append_history(
        state,
        {
            "at_utc": update["updated_at_utc"],
            "cycle_summary": update["cycle_summary"],
            "gst_pass": update["gst_pass"],
            "gst_fail_reasons": update["gst_fail_reasons"],
            "stage_after": update["state_after"]["current_stage"],
            "transition": state.get("last_transition"),
        },
        max_history=int(args.max_history),
    )

    state_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
    out_path.write_text(json.dumps(update, indent=2, sort_keys=True), encoding="utf-8")
    print(out_path.as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
