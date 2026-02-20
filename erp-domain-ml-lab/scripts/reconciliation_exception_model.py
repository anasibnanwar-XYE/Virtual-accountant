#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ledgerstudio_analytics.canonical_json import dump_bytes
from ledgerstudio_analytics.errors import AnalyticsError, ExitCode
from ledgerstudio_analytics.fs import atomic_write_bytes, ensure_dir
from ledgerstudio_analytics.hashing import sha256_bytes, sha256_file
from ledgerstudio_analytics.ml import training_data_v2 as td_v2
from ledgerstudio_analytics.ml.nn_text_classifier import (
    NNTextClassifier,
    build_text_features,
    load_model,
    predict_topk,
    resolve_device,
    train_mlp_text_classifier_from_features,
)
from ledgerstudio_analytics.ml.tokenize import join_fields
from _row_source import detect_training_format, iter_rows_from_file

MODEL_ID = "orchestrator_erp.reconciliation_exception_model.v1"
BUNDLE_KIND = "reconciliation_exception_registry"
LABEL_EXCEPTION = "RECON_EXCEPTION_REQUIRED"
LABEL_MATCH = "RECON_AUTO_MATCH_CANDIDATE"
NUMERIC_FEATURES = [
    "qty_abs",
    "price_abs",
    "cost_abs",
    "amount_proxy",
    "tax_rate",
    "journal_line_count",
    "journal_debit_count",
    "journal_credit_count",
    "account_diversity",
    "drcr_imbalance",
    "is_posted",
    "is_cancelled",
    "is_payment",
    "is_receipt",
    "is_sale",
    "is_purchase",
    "is_tax_settlement",
    "is_return_like",
    "is_period_lock",
    "payment_cash",
    "payment_bank",
    "has_bank_account",
    "has_ar_account",
    "has_ap_account",
    "has_clear_like",
    "has_suspense_like",
    "has_round_account",
    "has_discount_account",
    "complexity_ratio",
]
RECON_TRAINING_COLUMNS = (
    "type",
    "reference",
    "date",
    "qty",
    "price",
    "cost",
    "tax_rate",
    "party",
    "notes",
    "doc_type",
    "doc_status",
    "memo",
    "payment_method",
    "gst_treatment",
    "gst_inclusive",
    "currency",
    "journal_lines",
)


@dataclass(frozen=True, slots=True)
class ReconExample:
    example_id: str
    text: str
    numeric: dict[str, float]
    label: str
    meta: dict[str, Any]


def _fmt(v: float) -> str:
    return f"{float(v):.6f}"


def _canon_score(v: float) -> float:
    return float(_fmt(v))


def _stable_id(obj: dict[str, Any]) -> str:
    return sha256_bytes(dump_bytes(obj))


def _parse_decimal(x: str | None) -> float:
    s = str(x or "").strip()
    if not s:
        return 0.0
    try:
        return float(s)
    except ValueError:
        return 0.0


def _hash_noise(reference: str, *, seed: int) -> float:
    h = hashlib.sha256()
    h.update(f"seed:{seed}".encode("utf-8"))
    h.update(b"|")
    h.update(reference.encode("utf-8"))
    raw = int.from_bytes(h.digest()[:4], "big")
    return raw / float(2**32)


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _safe_ratio(n: float, d: float) -> float:
    if d <= 0:
        return 0.0
    return float(n / d)


def _inject_numeric_features(model: NNTextClassifier) -> NNTextClassifier:
    return NNTextClassifier(
        schema_version=model.schema_version,
        advisory_only=model.advisory_only,
        model_id=model.model_id,
        algorithm=model.algorithm,
        vectorizer=model.vectorizer,
        numeric_features=list(NUMERIC_FEATURES),
        labels=model.labels,
        weights1=model.weights1,
        bias1=model.bias1,
        weights2=model.weights2,
        bias2=model.bias2,
        numeric_mean=model.numeric_mean,
        numeric_std=model.numeric_std,
    )


def _example_from_row(row: dict[str, Any], *, idx: int, seed: int, exception_threshold: float, source: str) -> ReconExample:
    typ = str(row.get("type") or "").strip().upper()
    ref = str(row.get("reference") or f"ROW-{idx:09d}").strip().upper()
    doc_type = str(row.get("doc_type") or "").strip().upper()
    doc_status = str(row.get("doc_status") or "").strip().upper()
    pay_method = str(row.get("payment_method") or "").strip().upper()
    gst_treatment = str(row.get("gst_treatment") or "").strip().upper()
    gst_inclusive_raw = str(row.get("gst_inclusive") or "").strip().lower()
    party = str(row.get("party") or "").strip().upper()
    notes = str(row.get("notes") or "").strip()
    memo = str(row.get("memo") or "").strip()
    currency = str(row.get("currency") or "INR").strip().upper()

    qty = _parse_decimal(row.get("qty"))
    qty_abs = abs(qty)
    price_abs = abs(_parse_decimal(row.get("price")))
    cost_abs = abs(_parse_decimal(row.get("cost")))
    tax_rate = abs(_parse_decimal(row.get("tax_rate")))
    amount_proxy = max(price_abs * max(qty_abs, 1.0), cost_abs * max(qty_abs, 1.0), price_abs, cost_abs)

    journal_entries = td_v2._parse_journal_lines(row.get("journal_lines"))
    accounts: list[str] = []
    debit_count = 0
    credit_count = 0
    for acc, direction, _amt in journal_entries:
        acc_norm = str(acc or "").strip().upper()
        if not acc_norm:
            continue
        accounts.append(acc_norm)
        if direction == "D":
            debit_count += 1
        elif direction == "C":
            credit_count += 1

    line_count = len(accounts)
    account_diversity = len(set(accounts))
    has_bank_account = 1.0 if any("BANK" in a for a in accounts) else 0.0
    has_ar_account = 1.0 if any(("AR" in a or "RECEIV" in a or "DEBTOR" in a) for a in accounts) else 0.0
    has_ap_account = 1.0 if any(("AP" in a or "PAYAB" in a or "CREDITOR" in a) for a in accounts) else 0.0
    has_clear_like = 1.0 if any(("CLEAR" in a or "SETTLE" in a) for a in accounts) else 0.0
    has_suspense_like = 1.0 if any(("SUSPENSE" in a or "UNMAPPED" in a) for a in accounts) else 0.0
    has_round_account = 1.0 if any(("ROUND" in a or "RND" in a) for a in accounts) else 0.0
    has_discount_account = 1.0 if any("DISCOUNT" in a for a in accounts) else 0.0

    is_posted = 1.0 if doc_status == "POSTED" else 0.0
    is_cancelled = 1.0 if ("CANCEL" in doc_status or "VOID" in doc_status) else 0.0
    is_payment = 1.0 if typ in {"PAYMENT", "SETTLEMENT"} else 0.0
    is_receipt = 1.0 if typ in {"RECEIPT", "PAYMENT_RECEIPT"} else 0.0
    is_sale = 1.0 if ("SALE" in typ or "INVOICE" in typ) else 0.0
    is_purchase = 1.0 if ("PURCHASE" in typ or typ == "GRN") else 0.0
    is_tax_settlement = 1.0 if ("TAX" in typ or "GSTPAY" in ref or "TAX_PAYMENT" in doc_type) else 0.0
    is_return_like = 1.0 if ("RETURN" in typ or "CREDIT_NOTE" in typ or "DEBIT_NOTE" in typ) else 0.0
    is_period_lock = 1.0 if typ == "PERIOD_LOCK" else 0.0
    payment_cash = 1.0 if pay_method in {"CASH", "PETTY_CASH"} else 0.0
    payment_bank = 1.0 if pay_method in {"NEFT", "RTGS", "IMPS", "BANK_TRANSFER", "UPI"} else 0.0

    drcr_imbalance = abs(float(debit_count - credit_count))
    complexity_ratio = _safe_ratio(float(line_count), 2.0)

    # Synthetic pseudo-label for reconciliation exception routing.
    exception_logit = -0.9
    if has_suspense_like > 0:
        exception_logit += 1.4
    if has_clear_like > 0 and (has_ar_account <= 0 and has_ap_account <= 0):
        exception_logit += 0.8
    if (is_payment > 0 or is_receipt > 0) and has_bank_account <= 0:
        exception_logit += 1.0
    if (is_payment > 0 or is_receipt > 0) and (has_ar_account <= 0 and has_ap_account <= 0):
        exception_logit += 0.8
    if line_count >= 4:
        exception_logit += 0.45
    if line_count >= 6:
        exception_logit += 0.35
    if drcr_imbalance >= 1:
        exception_logit += 0.5
    if drcr_imbalance >= 2:
        exception_logit += 0.4
    if payment_cash > 0 and amount_proxy >= 75000:
        exception_logit += 0.7
    if is_return_like > 0 and (has_ar_account <= 0 and has_ap_account <= 0):
        exception_logit += 0.6
    if is_period_lock > 0:
        exception_logit += 0.8
    if is_posted <= 0:
        exception_logit += 0.3
    if is_cancelled > 0:
        exception_logit += 0.3
    if has_round_account > 0 and has_discount_account > 0:
        exception_logit += 0.2

    if (is_payment > 0 or is_receipt > 0) and has_bank_account > 0 and (has_ar_account > 0 or has_ap_account > 0) and line_count <= 3:
        exception_logit -= 0.5
    if payment_bank > 0 and line_count <= 3:
        exception_logit -= 0.25
    if is_tax_settlement > 0 and has_bank_account > 0 and line_count <= 3:
        exception_logit -= 0.2

    noise = (_hash_noise(ref, seed=seed) - 0.5) * 0.12
    exception_prob = _sigmoid(exception_logit + noise)
    label = LABEL_EXCEPTION if exception_prob >= exception_threshold else LABEL_MATCH

    numeric = {
        "qty_abs": float(qty_abs),
        "price_abs": float(price_abs),
        "cost_abs": float(cost_abs),
        "amount_proxy": float(amount_proxy),
        "tax_rate": float(tax_rate),
        "journal_line_count": float(line_count),
        "journal_debit_count": float(debit_count),
        "journal_credit_count": float(credit_count),
        "account_diversity": float(account_diversity),
        "drcr_imbalance": float(drcr_imbalance),
        "is_posted": float(is_posted),
        "is_cancelled": float(is_cancelled),
        "is_payment": float(is_payment),
        "is_receipt": float(is_receipt),
        "is_sale": float(is_sale),
        "is_purchase": float(is_purchase),
        "is_tax_settlement": float(is_tax_settlement),
        "is_return_like": float(is_return_like),
        "is_period_lock": float(is_period_lock),
        "payment_cash": float(payment_cash),
        "payment_bank": float(payment_bank),
        "has_bank_account": float(has_bank_account),
        "has_ar_account": float(has_ar_account),
        "has_ap_account": float(has_ap_account),
        "has_clear_like": float(has_clear_like),
        "has_suspense_like": float(has_suspense_like),
        "has_round_account": float(has_round_account),
        "has_discount_account": float(has_discount_account),
        "complexity_ratio": float(complexity_ratio),
    }

    account_tokens = " ".join(sorted(set(accounts))[:10])
    text = join_fields(
        typ,
        doc_type,
        doc_status,
        pay_method,
        gst_treatment,
        currency,
        party,
        notes,
        memo,
        account_tokens,
    )

    meta = {
        "source": source,
        "reference": ref,
        "type": typ,
        "doc_type": doc_type,
        "doc_status": doc_status,
        "payment_method": pay_method,
        "exception_prob_seeded": _canon_score(exception_prob),
    }

    example_id = _stable_id(
        {
            "source": source,
            "reference": ref,
            "type": typ,
            "date": str(row.get("date") or ""),
            "idx": idx,
        }
    )
    return ReconExample(example_id=example_id, text=text, numeric=numeric, label=label, meta=meta)


def _rows_from_training_csv(
    path: Path,
    *,
    exception_threshold: float,
    seed: int,
    training_format: str = "auto",
    parquet_batch_size: int = 100000,
) -> list[ReconExample]:
    out: list[ReconExample] = []
    for idx, row in enumerate(
        iter_rows_from_file(
            path,
            requested_format=training_format,
            parquet_columns=RECON_TRAINING_COLUMNS,
            parquet_batch_size=parquet_batch_size,
        )
    ):
        out.append(_example_from_row(row, idx=idx, seed=seed, exception_threshold=exception_threshold, source="training_csv"))

    if not out:
        raise AnalyticsError(ExitCode.INVALID_ARGS, "No rows available for reconciliation-exception training")
    return sorted(out, key=lambda r: r.example_id)


def _split_rows(rows: list[ReconExample], holdout_ratio: float, seed: int) -> tuple[list[ReconExample], list[ReconExample]]:
    if holdout_ratio < 0 or holdout_ratio >= 1:
        raise AnalyticsError(ExitCode.INVALID_ARGS, "holdout_ratio must be in [0,1)")

    def split_key(row: ReconExample) -> tuple[int, str]:
        h = hashlib.sha256()
        h.update(f"seed:{seed}".encode("utf-8"))
        h.update(b"|")
        h.update(row.example_id.encode("utf-8"))
        return int.from_bytes(h.digest()[:4], "big"), row.example_id

    ordered = sorted(rows, key=split_key)
    holdout_n = int(len(ordered) * holdout_ratio)
    if holdout_ratio > 0 and holdout_n == 0 and len(ordered) >= 5:
        holdout_n = 1
    holdout = sorted(ordered[:holdout_n], key=lambda r: r.example_id)
    train = sorted(ordered[holdout_n:], key=lambda r: r.example_id)
    if not train:
        raise AnalyticsError(ExitCode.INVALID_ARGS, "Train split is empty; reduce holdout_ratio")
    return train, holdout


def _extract_probability(preds: list[dict[str, Any]], label: str) -> float:
    if not preds:
        return 0.0
    for item in preds:
        if str(item.get("label") or "") == label:
            try:
                return float(item.get("score") or 0.0)
            except Exception:
                return 0.0
    return 0.0


def _binary_metrics(tp: int, fp: int, tn: int, fn: int) -> dict[str, float | int]:
    total = tp + fp + tn + fn
    accuracy = _safe_ratio(float(tp + tn), float(total))
    precision = _safe_ratio(float(tp), float(tp + fp))
    recall = _safe_ratio(float(tp), float(tp + fn))
    f1 = _safe_ratio(2.0 * precision * recall, precision + recall)
    fnr = _safe_ratio(float(fn), float(tp + fn))
    fpr = _safe_ratio(float(fp), float(fp + tn))
    return {
        "examples": int(total),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "accuracy": _canon_score(accuracy),
        "precision": _canon_score(precision),
        "recall": _canon_score(recall),
        "f1": _canon_score(f1),
        "fnr": _canon_score(fnr),
        "fpr": _canon_score(fpr),
        "positive_rate": _canon_score(_safe_ratio(float(tp + fn), float(total))),
        "predicted_positive_rate": _canon_score(_safe_ratio(float(tp + fp), float(total))),
    }


def _eval_rows(model: NNTextClassifier, rows: list[ReconExample], *, positive_label: str) -> dict[str, float | int]:
    tp = fp = tn = fn = 0
    for row in rows:
        numeric = [float(row.numeric.get(name, 0.0)) for name in NUMERIC_FEATURES]
        preds = predict_topk(model, text=row.text, numeric=numeric, k=2)
        pred_label = str(preds[0].get("label") or LABEL_MATCH) if preds else LABEL_MATCH
        truth_positive = row.label == positive_label
        pred_positive = pred_label == positive_label

        if pred_positive and truth_positive:
            tp += 1
        elif pred_positive and not truth_positive:
            fp += 1
        elif (not pred_positive) and truth_positive:
            fn += 1
        else:
            tn += 1

    return _binary_metrics(tp, fp, tn, fn)


def _store_bundle(
    *,
    out_root: Path,
    model: NNTextClassifier,
    train_rows: list[ReconExample],
    holdout_rows: list[ReconExample],
    metrics: dict[str, Any],
    config: dict[str, Any],
) -> Path:
    ensure_dir(out_root)

    model_payload = dump_bytes(model.to_dict())
    model_sha = sha256_bytes(model_payload)
    bundle_sha = sha256_bytes(model_payload + dump_bytes(config))

    bundle_dir = out_root / BUNDLE_KIND / MODEL_ID / bundle_sha
    ensure_dir(bundle_dir)
    atomic_write_bytes(bundle_dir / "model.json", model_payload)

    index_rows = [
        {
            "example_id": row.example_id,
            "text": row.text,
            "numeric": {k: float(row.numeric.get(k, 0.0)) for k in NUMERIC_FEATURES},
            "label": row.label,
            "meta": row.meta,
        }
        for row in sorted(train_rows, key=lambda r: r.example_id)
    ]
    index_lines = b"".join(dump_bytes(r).rstrip(b"\n") + b"\n" for r in index_rows)
    atomic_write_bytes(bundle_dir / "training_index.jsonl", index_lines)

    manifest = {
        "schema_version": "v0",
        "advisory_only": True,
        "bundle_kind": BUNDLE_KIND,
        "model_id": MODEL_ID,
        "bundle_sha256": bundle_sha,
        "model_sha256": model_sha,
        "numeric_features": list(NUMERIC_FEATURES),
        "labels": list(model.labels),
        "metrics": metrics,
        "config": config,
        "index_examples": len(index_rows),
        "holdout_examples": len(holdout_rows),
    }
    atomic_write_bytes(bundle_dir / "bundle_manifest.json", dump_bytes(manifest))
    return bundle_dir


def _load_bundle(bundle_dir: Path) -> tuple[NNTextClassifier, dict[str, Any]]:
    if not bundle_dir.exists() or not bundle_dir.is_dir():
        raise AnalyticsError(ExitCode.INVALID_ARGS, f"Bundle dir not found: {bundle_dir}")

    manifest_path = bundle_dir / "bundle_manifest.json"
    model_path = bundle_dir / "model.json"
    if not manifest_path.exists() or not model_path.exists():
        raise AnalyticsError(ExitCode.INVALID_ARGS, f"Invalid reconciliation bundle: {bundle_dir}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if str(manifest.get("model_id") or "") != MODEL_ID:
        raise AnalyticsError(ExitCode.INVALID_ARGS, "Bundle model_id mismatch")

    model = load_model(model_path)
    if not model.numeric_features:
        model = _inject_numeric_features(model)
    return model, manifest


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or not path.is_file():
        raise AnalyticsError(ExitCode.INVALID_ARGS, f"Input file not found: {path}")

    if path.suffix.lower() in {".json", ".jsonl"}:
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

    out_csv: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            out_csv.append(dict(row))
    return out_csv


def _rows_from_override_memory(path: Path, *, exception_threshold: float, seed: int) -> list[ReconExample]:
    if not path.exists() or not path.is_file():
        raise AnalyticsError(ExitCode.INVALID_ARGS, f"Override memory not found: {path}")

    out: list[ReconExample] = []
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            raw = line.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except Exception:
                continue
            if not isinstance(obj, dict):
                continue

            reference = str(obj.get("reference") or f"OVERRIDE-{idx:09d}").strip().upper()
            typ = str(obj.get("approved_label") or obj.get("type") or obj.get("suggested_label") or "").strip().upper()
            reason_code = str(obj.get("reason_code") or "").strip().upper()
            reason_text = str(obj.get("reason_text") or "").strip()
            correction_scope = str(obj.get("correction_scope") or "").strip().upper()
            approved_debit = str(obj.get("approved_debit_account_code") or "").strip().upper()
            approved_credit = str(obj.get("approved_credit_account_code") or "").strip().upper()
            suggested_debit = str(obj.get("suggested_debit_account_code") or "").strip().upper()
            suggested_credit = str(obj.get("suggested_credit_account_code") or "").strip().upper()

            debit = approved_debit or suggested_debit
            credit = approved_credit or suggested_credit
            journal_lines = ""
            if debit and credit:
                journal_lines = f"{debit} | D | 1 | override-approved || {credit} | C | 1 | override-approved"
            elif debit:
                journal_lines = f"{debit} | D | 1 | override-approved"
            elif credit:
                journal_lines = f"{credit} | C | 1 | override-approved"

            note_parts = []
            if reason_code:
                note_parts.append(f"override_reason_code={reason_code}")
            if correction_scope:
                note_parts.append(f"override_scope={correction_scope}")
            if reason_text:
                note_parts.append(f"override_reason_text={reason_text}")
            notes = " ; ".join(note_parts)

            row = {
                "type": typ or "UNKNOWN",
                "reference": reference,
                "date": "",
                "qty": "1",
                "price": "1",
                "cost": "0",
                "tax_rate": "0",
                "party": str(obj.get("party") or "").strip().upper(),
                "notes": notes,
                "doc_type": str(obj.get("doc_type") or "").strip().upper(),
                "doc_status": str(obj.get("doc_status") or "POSTED").strip().upper(),
                "memo": notes,
                "payment_method": str(obj.get("payment_method") or "").strip().upper(),
                "gst_treatment": str(obj.get("gst_treatment") or "NON_GST").strip().upper(),
                "gst_inclusive": "false",
                "currency": str(obj.get("currency") or "INR").strip().upper(),
                "journal_lines": journal_lines,
            }
            out.append(
                _example_from_row(
                    row,
                    idx=idx,
                    seed=seed,
                    exception_threshold=exception_threshold,
                    source="override_memory",
                )
            )

    return sorted(out, key=lambda r: r.example_id)


def _cmd_train(args: argparse.Namespace) -> int:
    training_path = Path(args.training_csv)
    training_format = detect_training_format(training_path, requested=args.training_format)
    base_rows = _rows_from_training_csv(
        training_path,
        exception_threshold=args.exception_threshold,
        seed=args.seed,
        training_format=training_format,
        parquet_batch_size=args.parquet_batch_size,
    )
    override_rows: list[ReconExample] = []
    if args.override_memory_jsonl:
        override_rows = _rows_from_override_memory(
            Path(args.override_memory_jsonl),
            exception_threshold=args.exception_threshold,
            seed=args.seed,
        )

    dedup: dict[str, ReconExample] = {}
    for row in [*base_rows, *override_rows]:
        dedup[row.example_id] = row
    rows = sorted(dedup.values(), key=lambda r: r.example_id)
    train_rows, holdout_rows = _split_rows(rows, holdout_ratio=args.holdout_ratio, seed=args.seed)

    texts = [row.text for row in train_rows]
    numeric_matrix = np.array(
        [[float(row.numeric.get(name, 0.0)) for name in NUMERIC_FEATURES] for row in train_rows],
        dtype=np.float32,
    )

    text_features, _active = build_text_features(
        texts,
        n_features=args.n_features,
        hash_algo=args.hash_algo,
        max_hash_cache=args.max_hash_cache,
    )

    labels = [row.label for row in train_rows]
    device_info = resolve_device(args.device)
    model = train_mlp_text_classifier_from_features(
        model_id=MODEL_ID,
        text_features=text_features,
        numeric_features=numeric_matrix,
        labels=labels,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        l2=args.l2,
        hidden_size=args.hidden_size,
        batch_size=args.batch_size,
        seed=args.seed,
        hash_algo=args.hash_algo,
        device_info=device_info,
        gpu_batch_size=args.gpu_batch_size,
        loss_weighting="class_balanced",
    )
    model = _inject_numeric_features(model)

    metrics = {
        "train_examples": len(train_rows),
        "holdout_examples": len(holdout_rows),
        "positive_label": LABEL_EXCEPTION,
        "train": _eval_rows(model, train_rows, positive_label=LABEL_EXCEPTION),
        "holdout": _eval_rows(model, holdout_rows, positive_label=LABEL_EXCEPTION),
    }

    config = {
        "dataset": {
            "training_csv": training_path.name,
            "training_csv_sha256": sha256_file(training_path),
            "training_file": training_path.name,
            "training_file_sha256": sha256_file(training_path),
            "training_format": training_format,
            "exception_threshold": float(args.exception_threshold),
            "override_memory_jsonl": Path(args.override_memory_jsonl).as_posix() if args.override_memory_jsonl else None,
            "override_memory_sha256": sha256_file(Path(args.override_memory_jsonl)) if args.override_memory_jsonl else None,
            "override_examples": len(override_rows),
        },
        "training": {
            "seed": args.seed,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "l2": args.l2,
            "hidden_size": args.hidden_size,
            "n_features": args.n_features,
            "batch_size": args.batch_size,
            "gpu_batch_size": args.gpu_batch_size,
            "hash_algo": args.hash_algo,
            "max_hash_cache": args.max_hash_cache,
            "device": device_info.to_manifest(),
            "holdout_ratio": args.holdout_ratio,
            "parquet_batch_size": args.parquet_batch_size,
        },
    }

    bundle_dir = _store_bundle(
        out_root=Path(args.out),
        model=model,
        train_rows=train_rows,
        holdout_rows=holdout_rows,
        metrics=metrics,
        config=config,
    )

    summary = {
        "bundle_dir": bundle_dir.as_posix(),
        "metrics": metrics,
    }
    if args.metrics_out:
        atomic_write_bytes(Path(args.metrics_out), dump_bytes(summary))

    print(bundle_dir.as_posix())
    return 0


def _cmd_score(args: argparse.Namespace) -> int:
    model, manifest = _load_bundle(Path(args.bundle_dir))
    raw_rows = _read_rows(Path(args.input_file))
    examples = [
        _example_from_row(r, idx=i, seed=args.seed, exception_threshold=0.58, source="inference")
        for i, r in enumerate(raw_rows)
    ]

    out_records: list[dict[str, Any]] = []
    for ex in examples:
        numeric = [float(ex.numeric.get(name, 0.0)) for name in NUMERIC_FEATURES]
        preds = predict_topk(model, text=ex.text, numeric=numeric, k=2)
        pred_label = str(preds[0].get("label") or LABEL_MATCH) if preds else LABEL_MATCH
        exception_prob = _extract_probability(preds, LABEL_EXCEPTION)

        if exception_prob >= 0.80:
            priority = "P1"
        elif exception_prob >= 0.55:
            priority = "P2"
        else:
            priority = "P3"

        out_records.append(
            {
                "schema_version": "v0",
                "advisory_only": True,
                "module_id": MODEL_ID,
                "reference": ex.meta.get("reference"),
                "type": ex.meta.get("type"),
                "prediction": {
                    "label": pred_label,
                    "exception_probability": _fmt(exception_prob),
                    "reconciliation_priority": priority,
                    "topk": preds,
                },
                "meta": ex.meta,
            }
        )

    out_path = Path(args.out)
    ensure_dir(out_path.parent)
    content = b"".join(dump_bytes(r).rstrip(b"\n") + b"\n" for r in out_records)
    atomic_write_bytes(out_path, content)

    if args.manifest_out:
        manifest_out = {
            "schema_version": "v0",
            "advisory_only": True,
            "module_id": MODEL_ID,
            "bundle_dir": Path(args.bundle_dir).as_posix(),
            "bundle_sha256": manifest.get("bundle_sha256"),
            "input_file": Path(args.input_file).as_posix(),
            "records": len(out_records),
            "out": out_path.as_posix(),
        }
        atomic_write_bytes(Path(args.manifest_out), dump_bytes(manifest_out))

    print(out_path.as_posix())
    return 0


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train and run reconciliation-exception model")
    sub = p.add_subparsers(dest="command", required=True)

    t = sub.add_parser("train", help="Train reconciliation-exception model")
    t.add_argument("--training-csv", required=True, help="Training input file (CSV or Parquet)")
    t.add_argument("--training-format", choices=["auto", "csv", "parquet"], default="auto")
    t.add_argument("--parquet-batch-size", type=int, default=100000)
    t.add_argument("--out", required=True, help="Model root directory")
    t.add_argument("--metrics-out", default=None)
    t.add_argument("--override-memory-jsonl", default=None)
    t.add_argument("--exception-threshold", type=float, default=0.58)
    t.add_argument("--seed", type=int, default=131)
    t.add_argument("--epochs", type=int, default=8)
    t.add_argument("--learning-rate", type=float, default=0.08)
    t.add_argument("--l2", type=float, default=1e-4)
    t.add_argument("--hidden-size", type=int, default=64)
    t.add_argument("--n-features", type=int, default=2048)
    t.add_argument("--batch-size", type=int, default=128)
    t.add_argument("--gpu-batch-size", type=int, default=1024)
    t.add_argument("--device", default="auto")
    t.add_argument("--holdout-ratio", type=float, default=0.2)
    t.add_argument("--hash-algo", default="crc32")
    t.add_argument("--max-hash-cache", type=int, default=200000)

    s = sub.add_parser("score", help="Score reconciliation exception risk")
    s.add_argument("--bundle-dir", required=True)
    s.add_argument("--input-file", required=True, help="CSV/JSONL with transaction-like rows")
    s.add_argument("--out", required=True)
    s.add_argument("--manifest-out", default=None)
    s.add_argument("--seed", type=int, default=131)

    return p


def main() -> int:
    args = _build_parser().parse_args()
    try:
        if args.command == "train":
            return _cmd_train(args)
        if args.command == "score":
            return _cmd_score(args)
        raise AnalyticsError(ExitCode.INVALID_ARGS, f"Unknown command: {args.command}")
    except AnalyticsError as exc:
        raise SystemExit(exc.exit_code.value)


if __name__ == "__main__":
    raise SystemExit(main())
