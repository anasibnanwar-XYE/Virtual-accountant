#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


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


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True))
            f.write("\n")


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


def _first_dict(items: Any) -> dict[str, Any]:
    if not isinstance(items, list) or not items:
        return {}
    first = items[0]
    if isinstance(first, dict):
        return first
    return {}


def _top_tx_label(tx: dict[str, Any]) -> tuple[str, float]:
    suggested = _as_str(tx.get("suggested_label")).upper()
    predictions = list(tx.get("predictions") or [])
    if suggested:
        for p in predictions:
            if not isinstance(p, dict):
                continue
            label = _as_str(p.get("label")).upper()
            if label == suggested:
                return suggested, _as_float(p.get("score"), 0.0)
        return suggested, _as_float(tx.get("confidence", {}).get("score"), 0.0)
    first = _first_dict(predictions)
    return _as_str(first.get("label")).upper(), _as_float(first.get("score"), 0.0)


def _policy_severity(bands: list[str]) -> str:
    lowered = {b.lower() for b in bands if b}
    if "blocked" in lowered:
        return "blocked"
    if "discouraged" in lowered:
        return "discouraged"
    return "ok"


def _confidence_band(calibrated_score: float, threshold: float) -> str:
    hi = max(0.95, threshold + 0.04)
    med = max(threshold, 0.80)
    if calibrated_score >= hi:
        return "high"
    if calibrated_score >= med:
        return "medium"
    return "low"


def _feature_signals(numeric: dict[str, Any]) -> list[str]:
    if not isinstance(numeric, dict):
        return []
    checks = [
        ("doc_type_invoice", "Document type signal: invoice."),
        ("doc_type_supplier_payment", "Document type signal: supplier payment."),
        ("doc_type_dealer_receipt", "Document type signal: dealer receipt."),
        ("doc_type_tax_payment", "Document type signal: tax payment."),
        ("doc_type_payroll_journal", "Document type signal: payroll journal."),
        ("doc_type_period_lock", "Document type signal: period lock."),
        ("workflow_sale", "Workflow signal: sales flow."),
        ("workflow_purchase", "Workflow signal: purchase flow."),
        ("workflow_payment", "Workflow signal: payment flow."),
        ("workflow_settlement_split", "Workflow signal: split settlement."),
        ("workflow_tax_settlement", "Workflow signal: tax settlement."),
        ("workflow_payroll", "Workflow signal: payroll flow."),
        ("workflow_period_lock", "Workflow signal: period lock control."),
        ("memo_hint_settlement", "Memo hint indicates settlement intent."),
        ("memo_hint_adjustment", "Memo hint indicates adjustment intent."),
        ("memo_hint_payroll", "Memo hint indicates payroll intent."),
        ("reference_hint_lock", "Reference hint indicates lock operation."),
        ("has_ap_account", "Entry contains payable account context."),
        ("has_ar_account", "Entry contains receivable account context."),
        ("has_bank_account", "Entry contains bank account context."),
        ("has_cash_account", "Entry contains cash account context."),
        ("has_tax_line", "Entry contains tax line context."),
    ]
    out: list[str] = []
    for key, text in checks:
        if _as_float(numeric.get(key), 0.0) > 0.5:
            out.append(text)
        if len(out) >= 5:
            break
    amount = _as_float(numeric.get("amount"), 0.0)
    if amount > 0:
        out.append(f"Amount context: {amount:.2f}.")
    return out[:6]


def _alternatives(preds: list[dict[str, Any]], *, limit: int = 2) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for p in preds[1 : 1 + limit]:
        if not isinstance(p, dict):
            continue
        label = _as_str(p.get("label")).upper()
        if not label:
            continue
        out.append(
            {
                "label": label,
                "score": round(_as_float(p.get("score"), 0.0), 6),
                "margin": round(_as_float(p.get("margin"), 0.0), 6),
            }
        )
    return out


def _tx_action_hint(
    *,
    policy_severity: str,
    auto_accept: bool,
    calibrated_score: float,
    threshold: float,
) -> tuple[str, list[str]]:
    reasons: list[str] = []
    if policy_severity == "blocked":
        reasons.append("Top-1 account recommendation hits blocked policy band.")
        return "reject_or_override_required", reasons
    if policy_severity == "discouraged":
        reasons.append("Top-1 account recommendation is discouraged by policy.")
    if not auto_accept:
        reasons.append("Classifier did not meet auto-accept threshold.")
    if calibrated_score < threshold:
        reasons.append("Calibrated confidence is below threshold.")
    if reasons:
        return "manual_review_required", reasons
    return "quick_confirm", ["Policy-safe posting with threshold-cleared confidence."]


def _routing_reason_text(reason_code: str) -> str:
    mapping = {
        "always_review_label": "Workflow routing forces manual review for this label.",
        "policy_blocked": "Workflow routing marked this as policy-blocked.",
        "policy_discouraged": "Workflow routing marked this as policy-discouraged.",
        "below_family_threshold": "Workflow-family threshold requires manual review.",
        "threshold_and_policy_ok": "Workflow-family threshold and policy both passed.",
    }
    return mapping.get(reason_code.lower(), "")


def _priority_for_action(action_code: str) -> str:
    action = action_code.lower()
    if action == "reject_or_override_required":
        return "P1"
    if action == "manual_review_required":
        return "P2"
    return "P3"


def _priority_rank(priority: str) -> int:
    p = priority.strip().upper()
    if p == "P1":
        return 0
    if p == "P2":
        return 1
    if p == "P3":
        return 2
    return 3


def _review_urgency_score(
    *,
    priority: str,
    policy_severity: str,
    calibrated_score: float,
    threshold: float,
    margin: float,
) -> float:
    base = {
        "P1": 3000.0,
        "P2": 2000.0,
        "P3": 1000.0,
    }.get(priority.strip().upper(), 500.0)
    score_gap = max(0.0, threshold - calibrated_score)
    policy_boost = 0.0
    if policy_severity == "blocked":
        policy_boost = 350.0
    elif policy_severity == "discouraged":
        policy_boost = 150.0

    return base + (score_gap * 1000.0) + policy_boost + max(0.0, (0.10 - margin) * 100.0)


def _tx_coa_explanations(
    tx_rows: list[dict[str, Any]],
    coa_rows: list[dict[str, Any]],
    *,
    generated_at: str,
    routing_by_id: dict[str, dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    coa_by_id = {_as_str(r.get("example_id")): r for r in coa_rows if _as_str(r.get("example_id"))}
    routing_by_id = routing_by_id or {}

    policy_top1_counts: Counter[str] = Counter()
    action_counts: Counter[str] = Counter()
    model_action_counts: Counter[str] = Counter()
    routing_reason_counts: Counter[str] = Counter()
    routing_attached_examples = 0
    routing_override_actions = 0
    missing_coa = 0
    out: list[dict[str, Any]] = []
    review_candidates: list[dict[str, Any]] = []

    for tx in tx_rows:
        example_id = _as_str(tx.get("example_id"))
        if not example_id:
            continue
        coa = coa_by_id.get(example_id)
        if coa is None:
            missing_coa += 1

        record = dict(tx.get("record") or {})
        confidence = dict(tx.get("confidence") or {})
        predictions = [p for p in list(tx.get("predictions") or []) if isinstance(p, dict)]
        suggested_label, top_score = _top_tx_label(tx)

        calibrated_score = _as_float(confidence.get("calibrated_score"), top_score)
        threshold = _as_float(confidence.get("threshold"), 0.9)
        margin = _as_float(confidence.get("margin"), 0.0)
        auto_accept = bool(confidence.get("auto_accept"))

        debit_top = _first_dict((coa or {}).get("debit_recommendations"))
        credit_top = _first_dict((coa or {}).get("credit_recommendations"))
        debit_band = _as_str(debit_top.get("policy_band"), "unknown").lower()
        credit_band = _as_str(credit_top.get("policy_band"), "unknown").lower()
        debit_reason = _as_str(debit_top.get("policy_reason"))
        credit_reason = _as_str(credit_top.get("policy_reason"))
        policy_reason = debit_reason or credit_reason
        if debit_reason and credit_reason and debit_reason != credit_reason:
            policy_reason = f"{debit_reason}; {credit_reason}"

        if debit_band:
            policy_top1_counts[f"debit:{debit_band}"] += 1
        if credit_band:
            policy_top1_counts[f"credit:{credit_band}"] += 1
        policy_severity = _policy_severity([debit_band, credit_band])
        model_action_code, model_action_reasons = _tx_action_hint(
            policy_severity=policy_severity,
            auto_accept=auto_accept,
            calibrated_score=calibrated_score,
            threshold=threshold,
        )
        model_action_counts[model_action_code] += 1

        action_code = model_action_code
        action_reasons = list(model_action_reasons)
        routing_row = routing_by_id.get(example_id)
        routing_payload: dict[str, Any] | None = None
        routing_reason_code = ""
        routing_priority = ""
        if routing_row is not None:
            routing_attached_examples += 1
            route = dict(routing_row.get("routing") or {})
            routed_action = _as_str(route.get("action")).lower()
            routing_reason_code = _as_str(route.get("reason_code")).lower()
            routing_priority = _as_str(route.get("priority")).upper()

            if routed_action:
                action_code = routed_action
            if routing_reason_code:
                routing_reason_counts[routing_reason_code] += 1
                reason_text = _routing_reason_text(routing_reason_code)
                if reason_text and reason_text not in action_reasons:
                    action_reasons.append(reason_text)

            routed_threshold = _as_float(route.get("family_threshold"), 0.0)
            if routed_threshold > 0:
                threshold = routed_threshold

            routing_payload = {
                "applied": True,
                "action": action_code,
                "priority": routing_priority or _priority_for_action(action_code),
                "reason_code": routing_reason_code or None,
                "family_threshold": round(threshold, 6),
                "source": "workflow_routing_policy.v1",
            }
        else:
            routing_payload = {
                "applied": False,
                "action": action_code,
                "priority": _priority_for_action(action_code),
                "reason_code": None,
                "family_threshold": round(threshold, 6),
                "source": "explanation_pack_default",
            }

        if action_code != model_action_code:
            routing_override_actions += 1
        action_counts[action_code] += 1

        workflow_family = _as_str((coa or {}).get("workflow_context", {}).get("family"), "unknown").lower()
        if workflow_family == "unknown":
            workflow_family = _as_str(routing_row.get("workflow_family") if routing_row else "", "unknown").lower()
        confidence_band = _confidence_band(calibrated_score, threshold)
        evidence = _feature_signals(dict(tx.get("numeric") or {}))
        questions = [
            _as_str(q.get("prompt"))
            for q in list(tx.get("questions") or [])
            if isinstance(q, dict) and _as_str(q.get("prompt"))
        ]

        summary = (
            f"Suggest {suggested_label or 'UNKNOWN'} with "
            f"{(_as_str(debit_top.get('label')) or 'UNKNOWN')} Dr / "
            f"{(_as_str(credit_top.get('label')) or 'UNKNOWN')} Cr in {workflow_family} workflow. "
            f"Confidence {calibrated_score:.3f} vs threshold {threshold:.3f}; policy top-1 "
            f"{debit_band}/{credit_band}. Action: {action_code}."
        )

        row = {
            "schema_version": "v0",
            "template_id": "tx_coa_accountant_explanation.v1",
            "generated_at_utc": generated_at,
            "example_id": example_id,
            "record": {
                "public_id": _as_str(record.get("publicId")),
                "id": record.get("id"),
                "reference_number": _as_str(record.get("referenceNumber")),
                "entry_date": _as_str(record.get("entryDate")),
                "kind": _as_str(record.get("kind")),
            },
            "suggestion": {
                "transaction_label": suggested_label,
                "debit_account": _as_str(debit_top.get("label")),
                "credit_account": _as_str(credit_top.get("label")),
                "workflow_family": workflow_family,
            },
            "scores": {
                "tx_score": round(top_score, 6),
                "tx_calibrated_score": round(calibrated_score, 6),
                "tx_margin": round(margin, 6),
                "tx_threshold": round(threshold, 6),
                "tx_auto_accept": auto_accept,
                "confidence_band": confidence_band,
                "debit_top_score": round(_as_float(debit_top.get("score"), 0.0), 6),
                "credit_top_score": round(_as_float(credit_top.get("score"), 0.0), 6),
            },
            "policy": {
                "debit_top1_band": debit_band,
                "credit_top1_band": credit_band,
                "policy_reason": policy_reason,
                "severity": policy_severity,
            },
            "action_hint": {
                "code": action_code,
                "reasons": action_reasons,
                "model_action_code": model_action_code,
            },
            "routing": routing_payload,
            "evidence": {
                "feature_signals": evidence,
                "review_questions": questions,
                "tx_alternatives": _alternatives(predictions, limit=2),
                "debit_alternatives": _alternatives(list((coa or {}).get("debit_recommendations") or []), limit=2),
                "credit_alternatives": _alternatives(list((coa or {}).get("credit_recommendations") or []), limit=2),
            },
            "explanation_text": summary,
        }
        out.append(row)

        if action_code != "quick_confirm":
            review_candidates.append(
                {
                    "reference_number": _as_str(record.get("referenceNumber")),
                    "example_id": example_id,
                    "workflow_family": workflow_family,
                    "action": action_code,
                    "priority": _as_str(routing_payload.get("priority"), _priority_for_action(action_code)),
                    "reason_code": _as_str(routing_payload.get("reason_code")),
                    "policy_severity": policy_severity,
                    "tx_calibrated_score": round(calibrated_score, 6),
                    "tx_threshold": round(threshold, 6),
                    "tx_margin": round(margin, 6),
                    "urgency_score": round(
                        _review_urgency_score(
                            priority=_as_str(routing_payload.get("priority"), _priority_for_action(action_code)),
                            policy_severity=policy_severity,
                            calibrated_score=calibrated_score,
                            threshold=threshold,
                            margin=margin,
                        ),
                        6,
                    ),
                }
            )

    review_candidates_sorted = sorted(
        review_candidates,
        key=lambda r: (
            _priority_rank(_as_str(r.get("priority"))),
            -_as_float(r.get("urgency_score"), 0.0),
            _as_str(r.get("reference_number")),
            _as_str(r.get("example_id")),
        ),
    )
    for i, row in enumerate(review_candidates_sorted, start=1):
        row["review_rank"] = int(i)

    summary = {
        "examples": len(out),
        "missing_coa_examples": int(missing_coa),
        "action_counts": dict(sorted(action_counts.items(), key=lambda t: t[0])),
        "model_action_counts": dict(sorted(model_action_counts.items(), key=lambda t: t[0])),
        "policy_top1_counts": dict(sorted(policy_top1_counts.items(), key=lambda t: t[0])),
        "routing": {
            "attached_examples": int(routing_attached_examples),
            "override_actions": int(routing_override_actions),
            "reason_counts": dict(sorted(routing_reason_counts.items(), key=lambda t: t[0])),
        },
        "manual_review_samples": review_candidates_sorted[:8],
    }
    return out, summary, review_candidates_sorted


def _target_review_reason(target: str, combined: dict[str, Any], min_score: float) -> list[str]:
    reasons: list[str] = []
    source = _as_str(combined.get("source")).lower()
    ml_score = _as_float(combined.get("ml_score"), 0.0)
    ml_label = _as_str(combined.get("ml_label")).upper()
    final_label = _as_str(combined.get("label")).upper()
    votes = int(_as_float(combined.get("neighbor_vote_count"), 0.0))

    if ml_score < min_score:
        reasons.append(f"{target} ML confidence below floor ({ml_score:.3f} < {min_score:.3f}).")
    if source == "neighbor_history" and ml_label and final_label and ml_label != final_label:
        reasons.append(f"{target} final label switched to neighbor history from ML top-1.")
    if source == "neighbor_history" and votes < 2:
        reasons.append(f"{target} neighbor-history vote support is weak ({votes}).")
    return reasons


def _product_explanations(
    product_rows: list[dict[str, Any]],
    *,
    generated_at: str,
    min_ml_score: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    action_counts: Counter[str] = Counter()
    out: list[dict[str, Any]] = []

    for rec in product_rows:
        candidate = dict(rec.get("candidate") or {})
        recs = dict(rec.get("recommendations") or {})
        targets = sorted(recs.keys())
        target_payload: dict[str, Any] = {}
        reasons: list[str] = []

        for target in targets:
            block = dict(recs.get(target) or {})
            combined = dict(block.get("combined_best") or {})
            topk = [r for r in list(block.get("topk") or []) if isinstance(r, dict)]
            target_reasons = _target_review_reason(target, combined, min_ml_score)
            reasons.extend(target_reasons)
            target_payload[target] = {
                "final_label": _as_str(combined.get("label")).upper(),
                "source": _as_str(combined.get("source")).lower(),
                "ml_label": _as_str(combined.get("ml_label")).upper(),
                "ml_score": round(_as_float(combined.get("ml_score"), 0.0), 6),
                "neighbor_vote_label": _as_str(combined.get("neighbor_vote_label")).upper(),
                "neighbor_vote_count": int(_as_float(combined.get("neighbor_vote_count"), 0.0)),
                "alternatives": _alternatives(topk, limit=2),
                "review_notes": target_reasons,
            }

        action_code = "manual_review_required" if reasons else "quick_confirm"
        action_counts[action_code] += 1

        similar = [s for s in list(rec.get("similar_products") or []) if isinstance(s, dict)]
        similar_evidence = [
            {
                "sku": _as_str(s.get("sku")),
                "product_name": _as_str(s.get("product_name")),
                "similarity": round(_as_float(s.get("similarity"), 0.0), 6),
            }
            for s in similar[:3]
        ]

        summary = (
            f"Product {(_as_str(candidate.get('sku')) or 'UNKNOWN')} mapped across "
            f"{len(targets)} account targets. Action: {action_code}. "
            f"Similar-product evidence count: {len(similar)}."
        )

        out.append(
            {
                "schema_version": "v0",
                "template_id": "product_account_explanation.v1",
                "generated_at_utc": generated_at,
                "candidate": {
                    "sku": _as_str(candidate.get("sku")),
                    "product_name": _as_str(candidate.get("product_name")),
                    "category": _as_str(candidate.get("category")),
                    "product_kind": _as_str(candidate.get("product_kind")),
                    "uom": _as_str(candidate.get("uom")),
                    "gst_rate": _as_float(candidate.get("gst_rate"), 0.0),
                    "base_price": _as_float(candidate.get("base_price"), 0.0),
                    "avg_cost": _as_float(candidate.get("avg_cost"), 0.0),
                },
                "targets": target_payload,
                "action_hint": {
                    "code": action_code,
                    "reasons": reasons,
                },
                "evidence": {
                    "similar_products": similar_evidence,
                },
                "explanation_text": summary,
            }
        )

    summary = {
        "examples": len(out),
        "action_counts": dict(sorted(action_counts.items(), key=lambda t: t[0])),
    }
    return out, summary


def _write_markdown(
    path: Path,
    *,
    generated_at: str,
    tx_summary: dict[str, Any],
    product_summary: dict[str, Any] | None,
    review_queue: list[dict[str, Any]] | None = None,
) -> None:
    tx_actions = dict(tx_summary.get("action_counts") or {})
    tx_model_actions = dict(tx_summary.get("model_action_counts") or {})
    tx_policy = dict(tx_summary.get("policy_top1_counts") or {})
    tx_routing = dict(tx_summary.get("routing") or {})
    lines = [
        "# Accountant Explanation Pack",
        "",
        f"Generated at (UTC): {generated_at}",
        "",
        "## Transaction + CoA",
        f"- Examples: {int(tx_summary.get('examples', 0))}",
        f"- Missing CoA match: {int(tx_summary.get('missing_coa_examples', 0))}",
        f"- Quick confirm: {int(tx_actions.get('quick_confirm', 0))}",
        f"- Manual review: {int(tx_actions.get('manual_review_required', 0))}",
        f"- Reject/override required: {int(tx_actions.get('reject_or_override_required', 0))}",
        f"- Routing attached examples: {int(tx_routing.get('attached_examples', 0))}",
        f"- Routing action overrides: {int(tx_routing.get('override_actions', 0))}",
        "",
        "Top-1 policy-band counts:",
        f"- Debit blocked: {int(tx_policy.get('debit:blocked', 0))}",
        f"- Debit discouraged: {int(tx_policy.get('debit:discouraged', 0))}",
        f"- Credit blocked: {int(tx_policy.get('credit:blocked', 0))}",
        f"- Credit discouraged: {int(tx_policy.get('credit:discouraged', 0))}",
        "",
        "Model-only action counts (before routing):",
        f"- Quick confirm: {int(tx_model_actions.get('quick_confirm', 0))}",
        f"- Manual review: {int(tx_model_actions.get('manual_review_required', 0))}",
        f"- Reject/override required: {int(tx_model_actions.get('reject_or_override_required', 0))}",
    ]

    samples = list(tx_summary.get("manual_review_samples") or [])
    if samples:
        lines.append("")
        lines.append("Top prioritized non-quick-confirm items:")
        for row in samples[:8]:
            lines.append(
                f"- #{row.get('review_rank', '?')} {row.get('reference_number') or '(no-reference)'} "
                f"[{row.get('example_id')}] -> {row.get('action')} ({row.get('priority')}) "
                f"urgency={row.get('urgency_score')}"
            )

    if product_summary is not None:
        p_actions = dict(product_summary.get("action_counts") or {})
        lines.extend(
            [
                "",
                "## Product Account Mapping",
                f"- Examples: {int(product_summary.get('examples', 0))}",
                f"- Quick confirm: {int(p_actions.get('quick_confirm', 0))}",
                f"- Manual review: {int(p_actions.get('manual_review_required', 0))}",
            ]
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate accountant-facing explanation templates for ERP advisory outputs")
    p.add_argument("--tx-jsonl", required=True)
    p.add_argument("--coa-jsonl", required=True)
    p.add_argument("--routing-jsonl", default=None)
    p.add_argument("--out-tx-jsonl", required=True)
    p.add_argument("--out-review-queue-jsonl", default=None)
    p.add_argument("--summary-json", required=True)
    p.add_argument("--out-md", default=None)
    p.add_argument("--product-jsonl", default=None)
    p.add_argument("--out-product-jsonl", default=None)
    p.add_argument("--product-min-ml-score", type=float, default=0.80)
    return p


def main() -> int:
    args = _parser().parse_args()

    tx_path = Path(args.tx_jsonl)
    coa_path = Path(args.coa_jsonl)
    routing_path = Path(args.routing_jsonl) if args.routing_jsonl else None
    out_tx = Path(args.out_tx_jsonl)
    out_review_queue = Path(args.out_review_queue_jsonl) if args.out_review_queue_jsonl else None
    summary_path = Path(args.summary_json)
    out_md = Path(args.out_md) if args.out_md else None

    if not tx_path.exists() or not tx_path.is_file():
        raise SystemExit(f"Missing transaction jsonl: {tx_path}")
    if not coa_path.exists() or not coa_path.is_file():
        raise SystemExit(f"Missing coa jsonl: {coa_path}")
    if routing_path is not None and (not routing_path.exists() or not routing_path.is_file()):
        raise SystemExit(f"Missing routing jsonl: {routing_path}")

    generated_at = datetime.now(timezone.utc).isoformat()

    tx_rows = _read_jsonl(tx_path)
    coa_rows = _read_jsonl(coa_path)
    routing_by_id: dict[str, dict[str, Any]] = {}
    if routing_path is not None:
        for row in _read_jsonl(routing_path):
            example_id = _as_str(row.get("example_id"))
            if example_id:
                routing_by_id[example_id] = row

    tx_out_rows, tx_summary, review_queue = _tx_coa_explanations(
        tx_rows,
        coa_rows,
        generated_at=generated_at,
        routing_by_id=routing_by_id,
    )
    _write_jsonl(out_tx, tx_out_rows)

    if out_review_queue is None:
        out_review_queue = out_tx.parent / "transaction_review_priority_queue.jsonl"
    _write_jsonl(out_review_queue, review_queue)

    product_summary: dict[str, Any] | None = None
    out_product = Path(args.out_product_jsonl) if args.out_product_jsonl else None
    if args.product_jsonl:
        p_path = Path(args.product_jsonl)
        if not p_path.exists() or not p_path.is_file():
            raise SystemExit(f"Missing product jsonl: {p_path}")
        if out_product is None:
            out_product = out_tx.parent / "product_account_explanations.jsonl"
        p_rows = _read_jsonl(p_path)
        p_out_rows, product_summary = _product_explanations(
            p_rows,
            generated_at=generated_at,
            min_ml_score=float(args.product_min_ml_score),
        )
        _write_jsonl(out_product, p_out_rows)

    summary = {
        "schema_version": "v0",
        "generated_at_utc": generated_at,
        "inputs": {
            "tx_jsonl": tx_path.as_posix(),
            "coa_jsonl": coa_path.as_posix(),
            "routing_jsonl": routing_path.as_posix() if routing_path else None,
            "product_jsonl": Path(args.product_jsonl).as_posix() if args.product_jsonl else None,
        },
        "outputs": {
            "tx_explanations_jsonl": out_tx.as_posix(),
            "tx_review_priority_queue_jsonl": out_review_queue.as_posix() if out_review_queue else None,
            "product_explanations_jsonl": out_product.as_posix() if out_product else None,
            "markdown_brief": out_md.as_posix() if out_md else None,
        },
        "tx_coa": tx_summary,
        "product_account": product_summary,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    if out_md is not None:
        _write_markdown(
            out_md,
            generated_at=generated_at,
            tx_summary=tx_summary,
            product_summary=product_summary,
            review_queue=review_queue,
        )

    print(summary_path.as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
