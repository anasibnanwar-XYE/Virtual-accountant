#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ledgerstudio_analytics.canonical_json import dump_bytes
from ledgerstudio_analytics.errors import AnalyticsError, ExitCode
from ledgerstudio_analytics.fs import atomic_write_bytes, ensure_dir
from ledgerstudio_analytics.hashing import sha256_bytes, sha256_file
from ledgerstudio_analytics.ml.nn_text_classifier import (
    NNTextClassifier,
    build_text_features,
    load_model,
    predict_topk,
    resolve_device,
    train_mlp_text_classifier_from_features,
)
from ledgerstudio_analytics.ml.tokenize import join_fields
from ledgerstudio_analytics.ml import training_data_v2 as td_v2
from _row_source import detect_training_format, iter_rows_from_file

MODEL_ID = "orchestrator_erp.product_account_recommender.v1"
BUNDLE_KIND = "product_account_registry"
TARGET_TO_FIELD = {
    "revenue": "revenue_account_code",
    "cogs": "cogs_account_code",
    "inventory": "inventory_account_code",
    "tax": "tax_account_code",
    "discount": "discount_account_code",
}
NUMERIC_FEATURES = [
    "is_finished_good",
    "is_raw_material",
    "gst_rate",
    "base_price",
    "avg_cost",
    "price_cost_ratio",
    "sale_ratio",
    "purchase_ratio",
    "cogs_ratio",
    "txn_count_log",
    "party_diversity",
    "avg_qty",
    "qty_cv",
    "taxable_ratio",
    "has_discount_history",
    "history_revenue_conf",
    "history_cogs_conf",
    "history_inventory_conf",
    "history_tax_conf",
    "history_discount_conf",
]
PRODUCT_TRAINING_COLUMNS = (
    "sku",
    "type",
    "notes",
    "memo",
    "doc_type",
    "party",
    "qty",
    "price",
    "cost",
    "tax_rate",
    "journal_lines",
)


@dataclass(frozen=True, slots=True)
class ProductExample:
    example_id: str
    text: str
    numeric: dict[str, float]
    labels: dict[str, str]
    meta: dict[str, Any]


def _fmt(v: float) -> str:
    return f"{float(v):.6f}"


def _canon_score(v: float) -> float:
    return float(_fmt(v))


def _sha_seed(seed: int, *parts: str) -> bytes:
    h = hashlib.sha256()
    h.update(f"seed:{seed}".encode("utf-8"))
    for p in parts:
        h.update(b"|")
        h.update(p.encode("utf-8"))
    return h.digest()


def _rand_int(seed: int, tag: str, idx: int, low: int, high: int) -> int:
    if high <= low:
        return low
    raw = int.from_bytes(_sha_seed(seed, tag, str(idx))[:4], "big")
    return low + (raw % (high - low))


def _rand_float(seed: int, tag: str, idx: int) -> float:
    raw = int.from_bytes(_sha_seed(seed, tag, str(idx))[:4], "big")
    return raw / float(2**32)


def _stable_id(obj: dict[str, Any]) -> str:
    return sha256_bytes(dump_bytes(obj))


def _analytics_lock_hash() -> str:
    pyproject = Path(td_v2.__file__).resolve().parents[2] / "pyproject.toml"
    if not pyproject.exists():
        return ""
    return sha256_file(pyproject)


def _parse_decimal(x: str | None) -> float:
    s = str(x or "").strip()
    if not s:
        return 0.0
    return float(s)


def _normalized_account(acc: str) -> str:
    return str(acc or "").strip().upper()


def _pick_account(counter: Counter[str], *, contains_any: tuple[str, ...], fallback: str) -> tuple[str, float]:
    if not counter:
        return fallback, 0.0
    ordered = sorted(counter.items(), key=lambda t: (-t[1], t[0]))
    for label, cnt in ordered:
        if contains_any and not any(tok in label for tok in contains_any):
            continue
        total = float(sum(counter.values()))
        conf = cnt / total if total else 0.0
        return label, conf
    label, cnt = ordered[0]
    total = float(sum(counter.values()))
    conf = cnt / total if total else 0.0
    return label, conf


def _top_note_tokens(notes: list[str], k: int = 6) -> list[str]:
    tok_counter: Counter[str] = Counter()
    for note in notes:
        for tok in str(note or "").strip().lower().replace("|", " ").replace("-", " ").split():
            if len(tok) < 3:
                continue
            if tok in {"the", "and", "for", "with", "from", "this", "that", "sale", "bill", "invoice", "vendor"}:
                continue
            tok_counter[tok] += 1
    return [t for t, _ in tok_counter.most_common(k)]


def _default_label_for_kind(kind: str, target: str) -> str:
    fg = kind == "FINISHED_GOOD"
    if target == "revenue":
        return "SALES"
    if target == "cogs":
        return "COGS" if fg else "FREIGHT_IN"
    if target == "inventory":
        return "INVENTORY"
    if target == "tax":
        return "GST_OUTPUT" if fg else "GST_INPUT"
    if target == "discount":
        return "DISCOUNT_ALLOWED" if fg else "DISCOUNT_RECEIVED"
    return ""


def _infer_kind(label_counts: Counter[str]) -> str:
    sale = label_counts.get("SALE", 0) + label_counts.get("SALE_RETURN", 0)
    purchase = label_counts.get("PURCHASE", 0)
    cogs = label_counts.get("COGS", 0) + label_counts.get("WRITE_OFF", 0)
    if purchase > sale + cogs:
        return "RAW_MATERIAL"
    return "FINISHED_GOOD"


def _score_similarity(example: ProductExample, query_meta: dict[str, Any]) -> float:
    score = 0.0
    kind = str(example.meta.get("product_kind") or "")
    q_kind = str(query_meta.get("product_kind") or "")
    if kind and q_kind and kind == q_kind:
        score += 0.40

    cat = str(example.meta.get("category") or "")
    q_cat = str(query_meta.get("category") or "")
    if cat and q_cat and cat == q_cat:
        score += 0.25

    gst = float(example.meta.get("gst_rate") or 0.0)
    q_gst = float(query_meta.get("gst_rate") or 0.0)
    score += max(0.0, 0.15 - min(0.15, abs(gst - q_gst)))

    base = float(example.meta.get("base_price") or 0.0)
    q_base = float(query_meta.get("base_price") or 0.0)
    if base > 0 and q_base > 0:
        ratio = min(base, q_base) / max(base, q_base)
        score += 0.20 * ratio

    return _canon_score(score)


def _augment_with_synthetic(seed: int, count: int) -> list[ProductExample]:
    categories = [
        "DECORATIVE_INTERIOR",
        "DECORATIVE_EXTERIOR",
        "INDUSTRIAL_COATINGS",
        "WOOD_FINISH",
        "METAL_PROTECTION",
        "SOLVENTS",
        "PIGMENTS",
        "PACKAGING",
    ]
    uoms = ["L", "KG", "PCS"]
    out: list[ProductExample] = []
    for i in range(count):
        is_fg = _rand_int(seed, "kind", i, 0, 100) < 70
        kind = "FINISHED_GOOD" if is_fg else "RAW_MATERIAL"
        category = categories[_rand_int(seed, "cat", i, 0, len(categories))]
        uom = uoms[_rand_int(seed, "uom", i, 0, len(uoms))]
        gst_rate = [0.0, 0.05, 0.12, 0.18][_rand_int(seed, "gst", i, 0, 4)]
        base_price = float(_rand_int(seed, "price", i, 80, 3200))
        avg_cost = float(_rand_int(seed, "cost", i, 40, 2200))
        if avg_cost >= base_price:
            base_price = avg_cost + float(_rand_int(seed, "mark", i, 30, 900))

        sku_prefix = "FG" if is_fg else "RM"
        sku = f"{sku_prefix}-{category.split('_')[0][:3]}-{i+1:06d}"
        product_name = f"{category.title().replace('_', ' ')} {i+1}"

        if is_fg:
            revenue = "SALES"
            cogs = "WRITE_OFF" if category in {"PACKAGING"} and _rand_int(seed, "wo", i, 0, 100) < 20 else "COGS"
            inventory = "INVENTORY_ADJ" if category in {"PACKAGING"} and _rand_int(seed, "iadj", i, 0, 100) < 18 else "INVENTORY"
            tax = "GST_OUTPUT" if gst_rate > 0 else "TAX_PAYABLE"
            discount = "ROUND_OFF" if _rand_int(seed, "rdoff", i, 0, 100) < 8 else "DISCOUNT_ALLOWED"
        else:
            revenue = "SALES"
            cogs = "FREIGHT_IN" if category in {"SOLVENTS", "PIGMENTS"} else "COGS"
            inventory = "INVENTORY"
            tax = "GST_INPUT" if gst_rate > 0 else "TAX_PAYABLE"
            discount = "ROUND_OFF" if _rand_int(seed, "rdoff2", i, 0, 100) < 10 else "DISCOUNT_RECEIVED"

        numeric = {
            "is_finished_good": 1.0 if is_fg else 0.0,
            "is_raw_material": 0.0 if is_fg else 1.0,
            "gst_rate": float(gst_rate),
            "base_price": float(base_price),
            "avg_cost": float(avg_cost),
            "price_cost_ratio": float(base_price / avg_cost) if avg_cost > 0 else 0.0,
            "sale_ratio": 0.85 if is_fg else 0.12,
            "purchase_ratio": 0.20 if is_fg else 0.88,
            "cogs_ratio": 0.62 if is_fg else 0.18,
            "txn_count_log": math.log1p(float(_rand_int(seed, "txn", i, 4, 160))),
            "party_diversity": float(_rand_int(seed, "pty", i, 1, 20)) / 20.0,
            "avg_qty": float(_rand_int(seed, "qty", i, 1, 80)),
            "qty_cv": float(_rand_int(seed, "qcv", i, 5, 95)) / 100.0,
            "taxable_ratio": 1.0 if gst_rate > 0 else 0.0,
            "has_discount_history": 1.0 if "DISCOUNT" in discount or "ROUND" in discount else 0.0,
            "history_revenue_conf": 0.82,
            "history_cogs_conf": 0.76,
            "history_inventory_conf": 0.80,
            "history_tax_conf": 0.84,
            "history_discount_conf": 0.70,
        }
        text = join_fields(
            sku,
            product_name,
            category,
            kind,
            uom,
            f"gst_{gst_rate}",
            "synthetic",
        )
        labels = {
            "revenue": revenue,
            "cogs": cogs,
            "inventory": inventory,
            "tax": tax,
            "discount": discount,
        }
        meta = {
            "source": "synthetic_product_master",
            "sku": sku,
            "product_name": product_name,
            "category": category,
            "product_kind": kind,
            "uom": uom,
            "gst_rate": gst_rate,
            "base_price": base_price,
            "avg_cost": avg_cost,
        }
        out.append(
            ProductExample(
                example_id=_stable_id({"source": "synthetic", "sku": sku, "kind": kind, "seed": seed, "i": i}),
                text=text,
                numeric=numeric,
                labels=labels,
                meta=meta,
            )
        )
    return out


def _rows_from_training_csv(
    path: Path,
    *,
    training_format: str = "auto",
    parquet_batch_size: int = 100000,
) -> list[ProductExample]:
    class Agg:
        def __init__(self) -> None:
            self.notes: list[str] = []
            self.doc_types: Counter[str] = Counter()
            self.parties: set[str] = set()
            self.label_counts: Counter[str] = Counter()
            self.debit_counts: Counter[str] = Counter()
            self.credit_counts: Counter[str] = Counter()
            self.all_counts: Counter[str] = Counter()
            self.qty: list[float] = []
            self.price: list[float] = []
            self.cost: list[float] = []
            self.tax_rate: list[float] = []
            self.taxable_rows = 0
            self.total_rows = 0

    by_sku: dict[str, Agg] = {}

    for row in iter_rows_from_file(
        path,
        requested_format=training_format,
        parquet_columns=PRODUCT_TRAINING_COLUMNS,
        parquet_batch_size=parquet_batch_size,
    ):
        sku = str(row.get("sku") or "").strip().upper()
        if not sku:
            continue
        agg = by_sku.setdefault(sku, Agg())
        agg.total_rows += 1
        label = str(row.get("type") or "").strip().upper()
        if label:
            agg.label_counts[label] += 1
        note = str(row.get("notes") or row.get("memo") or "").strip()
        if note:
            agg.notes.append(note)
        doc_type = str(row.get("doc_type") or "").strip().upper()
        if doc_type:
            agg.doc_types[doc_type] += 1
        party = str(row.get("party") or "").strip().upper()
        if party:
            agg.parties.add(party)
        qty = abs(_parse_decimal(row.get("qty")))
        price = abs(_parse_decimal(row.get("price")))
        cost = abs(_parse_decimal(row.get("cost")))
        tax_rate = abs(_parse_decimal(row.get("tax_rate")))
        if qty > 0:
            agg.qty.append(qty)
        if price > 0:
            agg.price.append(price)
        if cost > 0:
            agg.cost.append(cost)
        agg.tax_rate.append(tax_rate)
        if tax_rate > 0:
            agg.taxable_rows += 1

        entries = td_v2._parse_journal_lines(row.get("journal_lines"))
        for account, direction, _amt in entries:
            acc = _normalized_account(account)
            if not acc:
                continue
            agg.all_counts[acc] += 1
            if direction == "D":
                agg.debit_counts[acc] += 1
            elif direction == "C":
                agg.credit_counts[acc] += 1

    out: list[ProductExample] = []
    for sku, agg in sorted(by_sku.items()):
        if agg.total_rows <= 0:
            continue
        kind = _infer_kind(agg.label_counts)
        category = sku.split("-")[0] if "-" in sku else "GENERIC"
        top_notes = _top_note_tokens(agg.notes)
        top_doc_types = [k for k, _ in agg.doc_types.most_common(3)]

        revenue_label, revenue_conf = _pick_account(
            agg.credit_counts,
            contains_any=("SALES", "INCOME"),
            fallback=_default_label_for_kind(kind, "revenue"),
        )
        cogs_label, cogs_conf = _pick_account(
            agg.debit_counts,
            contains_any=("COGS", "WRITE_OFF", "INVENTORY_ADJ", "FREIGHT"),
            fallback=_default_label_for_kind(kind, "cogs"),
        )
        inventory_label, inventory_conf = _pick_account(
            agg.all_counts,
            contains_any=("INVENTORY",),
            fallback=_default_label_for_kind(kind, "inventory"),
        )
        tax_label, tax_conf = _pick_account(
            agg.all_counts,
            contains_any=("GST", "TAX", "TDS"),
            fallback=_default_label_for_kind(kind, "tax"),
        )
        discount_label, discount_conf = _pick_account(
            agg.all_counts,
            contains_any=("DISCOUNT", "ROUND"),
            fallback=_default_label_for_kind(kind, "discount"),
        )

        avg_qty = sum(agg.qty) / float(len(agg.qty)) if agg.qty else 0.0
        qty_cv = 0.0
        if len(agg.qty) >= 2 and avg_qty > 0:
            mu = avg_qty
            var = sum((q - mu) ** 2 for q in agg.qty) / float(len(agg.qty))
            qty_cv = math.sqrt(var) / mu

        avg_price = sum(agg.price) / float(len(agg.price)) if agg.price else 0.0
        avg_cost = sum(agg.cost) / float(len(agg.cost)) if agg.cost else 0.0
        avg_tax_rate = sum(agg.tax_rate) / float(len(agg.tax_rate)) if agg.tax_rate else 0.0

        sale_cnt = agg.label_counts.get("SALE", 0) + agg.label_counts.get("SALE_RETURN", 0)
        purchase_cnt = agg.label_counts.get("PURCHASE", 0)
        cogs_cnt = agg.label_counts.get("COGS", 0) + agg.label_counts.get("WRITE_OFF", 0) + agg.label_counts.get("INVENTORY_COUNT", 0)
        total = float(agg.total_rows)

        numeric = {
            "is_finished_good": 1.0 if kind == "FINISHED_GOOD" else 0.0,
            "is_raw_material": 1.0 if kind == "RAW_MATERIAL" else 0.0,
            "gst_rate": float(avg_tax_rate),
            "base_price": float(avg_price),
            "avg_cost": float(avg_cost),
            "price_cost_ratio": float(avg_price / avg_cost) if avg_cost > 0 else 0.0,
            "sale_ratio": sale_cnt / total,
            "purchase_ratio": purchase_cnt / total,
            "cogs_ratio": cogs_cnt / total,
            "txn_count_log": math.log1p(total),
            "party_diversity": float(len(agg.parties)) / total,
            "avg_qty": float(avg_qty),
            "qty_cv": float(min(3.0, qty_cv)),
            "taxable_ratio": float(agg.taxable_rows) / total,
            "has_discount_history": 1.0 if discount_conf > 0 else 0.0,
            "history_revenue_conf": revenue_conf,
            "history_cogs_conf": cogs_conf,
            "history_inventory_conf": inventory_conf,
            "history_tax_conf": tax_conf,
            "history_discount_conf": discount_conf,
        }

        text = join_fields(
            sku,
            kind,
            category,
            " ".join(top_doc_types),
            " ".join(top_notes),
            "from_transactions",
        )
        labels = {
            "revenue": revenue_label,
            "cogs": cogs_label,
            "inventory": inventory_label,
            "tax": tax_label,
            "discount": discount_label,
        }
        meta = {
            "source": "transaction_aggregation",
            "sku": sku,
            "product_name": sku,
            "category": category,
            "product_kind": kind,
            "uom": "PCS",
            "gst_rate": avg_tax_rate,
            "base_price": avg_price,
            "avg_cost": avg_cost,
            "txn_rows": agg.total_rows,
        }
        out.append(
            ProductExample(
                example_id=_stable_id({"source": "tx_agg", "sku": sku, "rows": agg.total_rows}),
                text=text,
                numeric=numeric,
                labels=labels,
                meta=meta,
            )
        )

    return out


def _rows_from_feedback_jsonl(path: Path) -> list[ProductExample]:
    if not path.exists() or not path.is_file():
        raise AnalyticsError(ExitCode.INVALID_ARGS, f"Feedback JSONL not found: {path}")

    out: list[ProductExample] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except Exception as e:  # pragma: no cover
                raise AnalyticsError(ExitCode.INVALID_ARGS, f"Invalid feedback JSON at line {line_no}") from e
            if not isinstance(obj, dict):
                continue
            sku = str(obj.get("sku") or "").strip().upper()
            if not sku:
                continue
            kind = str(obj.get("product_kind") or "FINISHED_GOOD").strip().upper()
            category = str(obj.get("category") or "GENERIC").strip().upper()
            gst_rate = float(obj.get("gst_rate") or 0.0)
            base_price = float(obj.get("base_price") or 0.0)
            avg_cost = float(obj.get("avg_cost") or 0.0)
            uom = str(obj.get("uom") or "PCS").strip().upper()
            product_name = str(obj.get("product_name") or sku).strip()

            labels = {
                "revenue": _normalized_account(str(obj.get("revenue_account_code") or _default_label_for_kind(kind, "revenue"))),
                "cogs": _normalized_account(str(obj.get("cogs_account_code") or _default_label_for_kind(kind, "cogs"))),
                "inventory": _normalized_account(str(obj.get("inventory_account_code") or _default_label_for_kind(kind, "inventory"))),
                "tax": _normalized_account(str(obj.get("tax_account_code") or _default_label_for_kind(kind, "tax"))),
                "discount": _normalized_account(str(obj.get("discount_account_code") or _default_label_for_kind(kind, "discount"))),
            }
            numeric = {
                "is_finished_good": 1.0 if kind == "FINISHED_GOOD" else 0.0,
                "is_raw_material": 1.0 if kind == "RAW_MATERIAL" else 0.0,
                "gst_rate": gst_rate,
                "base_price": base_price,
                "avg_cost": avg_cost,
                "price_cost_ratio": float(base_price / avg_cost) if avg_cost > 0 else 0.0,
                "sale_ratio": float(obj.get("sale_ratio") or (0.80 if kind == "FINISHED_GOOD" else 0.10)),
                "purchase_ratio": float(obj.get("purchase_ratio") or (0.20 if kind == "FINISHED_GOOD" else 0.90)),
                "cogs_ratio": float(obj.get("cogs_ratio") or (0.60 if kind == "FINISHED_GOOD" else 0.15)),
                "txn_count_log": math.log1p(float(obj.get("txn_count") or 20.0)),
                "party_diversity": float(obj.get("party_diversity") or 0.3),
                "avg_qty": float(obj.get("avg_qty") or 8.0),
                "qty_cv": float(obj.get("qty_cv") or 0.4),
                "taxable_ratio": 1.0 if gst_rate > 0 else 0.0,
                "has_discount_history": 1.0,
                "history_revenue_conf": float(obj.get("history_revenue_conf") or 0.95),
                "history_cogs_conf": float(obj.get("history_cogs_conf") or 0.95),
                "history_inventory_conf": float(obj.get("history_inventory_conf") or 0.95),
                "history_tax_conf": float(obj.get("history_tax_conf") or 0.95),
                "history_discount_conf": float(obj.get("history_discount_conf") or 0.95),
            }
            text = join_fields(sku, product_name, category, kind, uom, f"gst_{gst_rate}", "feedback")
            meta = {
                "source": "feedback",
                "sku": sku,
                "product_name": product_name,
                "category": category,
                "product_kind": kind,
                "uom": uom,
                "gst_rate": gst_rate,
                "base_price": base_price,
                "avg_cost": avg_cost,
            }
            out.append(
                ProductExample(
                    example_id=_stable_id({"source": "feedback", "sku": sku, "line": line_no}),
                    text=text,
                    numeric=numeric,
                    labels=labels,
                    meta=meta,
                )
            )
    return out


def _dedupe_rows(rows: list[ProductExample]) -> list[ProductExample]:
    by_key: dict[tuple[str, str], ProductExample] = {}
    for row in rows:
        sku = str(row.meta.get("sku") or "")
        source = str(row.meta.get("source") or "")
        key = (sku, source)
        by_key[key] = row
    return sorted(by_key.values(), key=lambda r: r.example_id)


def _split_rows(rows: list[ProductExample], holdout_ratio: float, seed: int) -> tuple[list[ProductExample], list[ProductExample]]:
    if holdout_ratio < 0 or holdout_ratio >= 1:
        raise AnalyticsError(ExitCode.INVALID_ARGS, "holdout_ratio must be in [0,1)")
    ordered = sorted(
        rows,
        key=lambda r: (
            int.from_bytes(_sha_seed(seed, "split", r.example_id)[:4], "big"),
            r.example_id,
        ),
    )
    if not ordered:
        raise AnalyticsError(ExitCode.INVALID_ARGS, "No product-account rows available for training")

    holdout_n = int(len(ordered) * holdout_ratio)
    if holdout_ratio > 0 and holdout_n == 0 and len(ordered) >= 5:
        holdout_n = 1
    holdout = sorted(ordered[:holdout_n], key=lambda r: r.example_id)
    train = sorted(ordered[holdout_n:], key=lambda r: r.example_id)
    if not train:
        raise AnalyticsError(ExitCode.INVALID_ARGS, "Train split is empty; reduce holdout_ratio")
    return train, holdout


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


def _eval_target(model: NNTextClassifier, rows: list[ProductExample], target: str, topk: int) -> dict[str, Any]:
    total = len(rows)
    if total == 0:
        return {"examples": 0, "top1": 0.0, "topk": 0.0}

    top1 = 0
    topk_hits = 0
    for row in rows:
        numeric = [float(row.numeric.get(name, 0.0)) for name in NUMERIC_FEATURES]
        preds = predict_topk(model, text=row.text, numeric=numeric, k=topk)
        truth = row.labels[target]
        labels = [str(p.get("label") or "") for p in preds]
        if labels and labels[0] == truth:
            top1 += 1
        if truth in labels:
            topk_hits += 1

    return {
        "examples": total,
        "top1": _canon_score(top1 / float(total)),
        f"top{topk}": _canon_score(topk_hits / float(total)),
    }


def _store_bundle(
    *,
    out_root: Path,
    models: dict[str, NNTextClassifier],
    train_rows: list[ProductExample],
    holdout_rows: list[ProductExample],
    metrics: dict[str, Any],
    config: dict[str, Any],
) -> Path:
    ensure_dir(out_root)

    model_bytes_by_target: dict[str, bytes] = {}
    model_sha_by_target: dict[str, str] = {}
    for target, model in models.items():
        payload = dump_bytes(model.to_dict())
        model_bytes_by_target[target] = payload
        model_sha_by_target[target] = sha256_bytes(payload)

    bundle_material = b"".join(
        model_bytes_by_target[target]
        for target in sorted(model_bytes_by_target.keys())
    ) + dump_bytes(config)
    bundle_sha = sha256_bytes(bundle_material)

    bundle_dir = out_root / BUNDLE_KIND / MODEL_ID / bundle_sha
    ensure_dir(bundle_dir)

    for target in sorted(models.keys()):
        atomic_write_bytes(bundle_dir / f"{target}_model.json", model_bytes_by_target[target])

    index_records = [
        {
            "example_id": row.example_id,
            "text": row.text,
            "numeric": {k: float(row.numeric.get(k, 0.0)) for k in NUMERIC_FEATURES},
            "labels": row.labels,
            "meta": row.meta,
        }
        for row in sorted(train_rows, key=lambda r: r.example_id)
    ]
    index_lines = b"".join(dump_bytes(rec).rstrip(b"\n") + b"\n" for rec in index_records)
    atomic_write_bytes(bundle_dir / "product_index.jsonl", index_lines)

    manifest = {
        "schema_version": "v0",
        "advisory_only": True,
        "bundle_kind": BUNDLE_KIND,
        "model_id": MODEL_ID,
        "bundle_sha256": bundle_sha,
        "model_shas": model_sha_by_target,
        "targets": sorted(models.keys()),
        "numeric_features": list(NUMERIC_FEATURES),
        "metrics": metrics,
        "config": config,
        "index_examples": len(index_records),
        "holdout_examples": len(holdout_rows),
        "python_env_lock_hash": _analytics_lock_hash(),
    }
    atomic_write_bytes(bundle_dir / "bundle_manifest.json", dump_bytes(manifest))

    return bundle_dir


def _load_bundle(bundle_dir: Path) -> tuple[dict[str, NNTextClassifier], dict[str, Any], list[ProductExample]]:
    if not bundle_dir.exists() or not bundle_dir.is_dir():
        raise AnalyticsError(ExitCode.INVALID_ARGS, f"Bundle dir not found: {bundle_dir}")

    manifest_path = bundle_dir / "bundle_manifest.json"
    if not manifest_path.exists():
        raise AnalyticsError(ExitCode.INVALID_ARGS, f"Missing bundle manifest: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("model_id") != MODEL_ID:
        raise AnalyticsError(ExitCode.INVALID_ARGS, "Bundle model_id mismatch")

    models: dict[str, NNTextClassifier] = {}
    targets = list(manifest.get("targets") or [])
    for target in targets:
        model_path = bundle_dir / f"{target}_model.json"
        if not model_path.exists():
            raise AnalyticsError(ExitCode.INVALID_ARGS, f"Missing model file: {model_path}")
        model = load_model(model_path)
        if not model.numeric_features:
            model = _inject_numeric_features(model)
        models[target] = model

    index_rows: list[ProductExample] = []
    index_path = bundle_dir / "product_index.jsonl"
    if index_path.exists():
        with index_path.open("r", encoding="utf-8") as f:
            for line in f:
                raw = line.strip()
                if not raw:
                    continue
                obj = json.loads(raw)
                index_rows.append(
                    ProductExample(
                        example_id=str(obj.get("example_id") or ""),
                        text=str(obj.get("text") or ""),
                        numeric={k: float(v) for k, v in dict(obj.get("numeric") or {}).items()},
                        labels={k: str(v) for k, v in dict(obj.get("labels") or {}).items()},
                        meta=dict(obj.get("meta") or {}),
                    )
                )
    return models, manifest, index_rows


def _read_candidate_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or not path.is_file():
        raise AnalyticsError(ExitCode.INVALID_ARGS, f"Candidate file not found: {path}")

    if path.suffix.lower() in {".jsonl", ".json"}:
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

    rows_csv: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows_csv.append(dict(row))
    return rows_csv


def _candidate_to_example(obj: dict[str, Any], idx: int) -> ProductExample:
    sku = str(obj.get("sku") or obj.get("sku_code") or f"NEW-{idx:06d}").strip().upper()
    product_name = str(obj.get("product_name") or obj.get("name") or sku).strip()
    category = str(obj.get("category") or "GENERIC").strip().upper()
    kind = str(obj.get("product_kind") or "FINISHED_GOOD").strip().upper()
    uom = str(obj.get("uom") or obj.get("unit") or "PCS").strip().upper()
    gst_rate = float(obj.get("gst_rate") or 0.0)
    base_price = float(obj.get("base_price") or 0.0)
    avg_cost = float(obj.get("avg_cost") or obj.get("cost") or 0.0)

    numeric = {
        "is_finished_good": 1.0 if kind == "FINISHED_GOOD" else 0.0,
        "is_raw_material": 1.0 if kind == "RAW_MATERIAL" else 0.0,
        "gst_rate": gst_rate,
        "base_price": base_price,
        "avg_cost": avg_cost,
        "price_cost_ratio": float(base_price / avg_cost) if avg_cost > 0 else 0.0,
        "sale_ratio": float(obj.get("sale_ratio") or (0.75 if kind == "FINISHED_GOOD" else 0.15)),
        "purchase_ratio": float(obj.get("purchase_ratio") or (0.25 if kind == "FINISHED_GOOD" else 0.85)),
        "cogs_ratio": float(obj.get("cogs_ratio") or (0.55 if kind == "FINISHED_GOOD" else 0.20)),
        "txn_count_log": math.log1p(float(obj.get("txn_count") or 12.0)),
        "party_diversity": float(obj.get("party_diversity") or 0.35),
        "avg_qty": float(obj.get("avg_qty") or 6.0),
        "qty_cv": float(obj.get("qty_cv") or 0.45),
        "taxable_ratio": 1.0 if gst_rate > 0 else 0.0,
        "has_discount_history": float(obj.get("has_discount_history") or 0.0),
        "history_revenue_conf": 0.0,
        "history_cogs_conf": 0.0,
        "history_inventory_conf": 0.0,
        "history_tax_conf": 0.0,
        "history_discount_conf": 0.0,
    }

    text = join_fields(sku, product_name, category, kind, uom, f"gst_{gst_rate}", "inference")
    meta = {
        "source": "candidate",
        "sku": sku,
        "product_name": product_name,
        "category": category,
        "product_kind": kind,
        "uom": uom,
        "gst_rate": gst_rate,
        "base_price": base_price,
        "avg_cost": avg_cost,
    }
    labels = {t: "" for t in TARGET_TO_FIELD.keys()}
    return ProductExample(
        example_id=_stable_id({"source": "candidate", "sku": sku, "i": idx}),
        text=text,
        numeric=numeric,
        labels=labels,
        meta=meta,
    )


def _suggest(
    *,
    models: dict[str, NNTextClassifier],
    index_rows: list[ProductExample],
    candidates: list[ProductExample],
    topk: int,
    neighbors: int,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for cand in candidates:
        numeric = [float(cand.numeric.get(name, 0.0)) for name in NUMERIC_FEATURES]
        target_preds: dict[str, list[dict[str, Any]]] = {}
        for target, model in models.items():
            target_preds[target] = predict_topk(model, text=cand.text, numeric=numeric, k=topk)

        scored = []
        for row in index_rows:
            sim = _score_similarity(row, cand.meta)
            if sim <= 0:
                continue
            scored.append((sim, row))
        scored.sort(key=lambda t: (-t[0], t[1].example_id))
        top_neighbors = scored[:neighbors]

        neighbor_votes: dict[str, Counter[str]] = {t: Counter() for t in TARGET_TO_FIELD.keys()}
        neighbor_payload: list[dict[str, Any]] = []
        for sim, row in top_neighbors:
            for target in TARGET_TO_FIELD.keys():
                label = str(row.labels.get(target) or "")
                if label:
                    neighbor_votes[target][label] += 1
            neighbor_payload.append(
                {
                    "similarity": _fmt(sim),
                    "sku": row.meta.get("sku"),
                    "product_name": row.meta.get("product_name"),
                    "category": row.meta.get("category"),
                    "product_kind": row.meta.get("product_kind"),
                    "mapped_accounts": {k: row.labels.get(k) for k in TARGET_TO_FIELD.keys()},
                }
            )

        combined: dict[str, dict[str, Any]] = {}
        for target in TARGET_TO_FIELD.keys():
            ml_top = target_preds.get(target, [])
            best_ml = ml_top[0] if ml_top else {"label": "", "score": "0.000000"}
            vote_counter = neighbor_votes[target]
            vote_label = ""
            vote_count = 0
            if vote_counter:
                vote_label, vote_count = sorted(vote_counter.items(), key=lambda t: (-t[1], t[0]))[0]

            use_neighbor = False
            ml_score = float(best_ml.get("score") or 0.0)
            if vote_label and vote_count >= 2 and ml_score < 0.88:
                use_neighbor = True

            combined[target] = {
                "label": vote_label if use_neighbor else str(best_ml.get("label") or ""),
                "source": "neighbor_history" if use_neighbor else "ml_model",
                "ml_label": str(best_ml.get("label") or ""),
                "ml_score": _fmt(ml_score),
                "neighbor_vote_label": vote_label,
                "neighbor_vote_count": int(vote_count),
            }

        out.append(
            {
                "schema_version": "v0",
                "advisory_only": True,
                "module_id": MODEL_ID,
                "candidate": {
                    "sku": cand.meta.get("sku"),
                    "product_name": cand.meta.get("product_name"),
                    "category": cand.meta.get("category"),
                    "product_kind": cand.meta.get("product_kind"),
                    "uom": cand.meta.get("uom"),
                    "gst_rate": cand.meta.get("gst_rate"),
                    "base_price": cand.meta.get("base_price"),
                    "avg_cost": cand.meta.get("avg_cost"),
                },
                "recommendations": {
                    target: {
                        "topk": target_preds.get(target, []),
                        "combined_best": combined[target],
                    }
                    for target in TARGET_TO_FIELD.keys()
                },
                "similar_products": neighbor_payload,
            }
        )
    return out


def _cmd_train(args: argparse.Namespace) -> int:
    training_path = Path(args.training_csv)
    training_format = detect_training_format(training_path, requested=args.training_format)
    tx_rows = _rows_from_training_csv(
        training_path,
        training_format=training_format,
        parquet_batch_size=args.parquet_batch_size,
    )
    synth_rows = _augment_with_synthetic(seed=args.seed, count=args.synthetic_products)

    feedback_rows: list[ProductExample] = []
    if args.feedback_jsonl:
        feedback_rows = _rows_from_feedback_jsonl(Path(args.feedback_jsonl))

    rows = _dedupe_rows(tx_rows + synth_rows + feedback_rows)
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

    models: dict[str, NNTextClassifier] = {}
    device_info = resolve_device(args.device)
    for target in TARGET_TO_FIELD.keys():
        labels = [row.labels[target] for row in train_rows]
        model = train_mlp_text_classifier_from_features(
            model_id=f"{MODEL_ID}.{target}",
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
        models[target] = _inject_numeric_features(model)

    metrics = {
        "train_examples": len(train_rows),
        "holdout_examples": len(holdout_rows),
        "targets": {
            target: {
                "train": _eval_target(models[target], train_rows, target=target, topk=args.topk),
                "holdout": _eval_target(models[target], holdout_rows, target=target, topk=args.topk),
            }
            for target in TARGET_TO_FIELD.keys()
        },
    }

    config = {
        "dataset": {
            "training_csv": training_path.name,
            "training_csv_sha256": sha256_file(training_path),
            "training_file": training_path.name,
            "training_file_sha256": sha256_file(training_path),
            "training_format": training_format,
            "synthetic_products": args.synthetic_products,
            "feedback_rows": len(feedback_rows),
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
            "topk_eval": args.topk,
            "parquet_batch_size": args.parquet_batch_size,
        },
    }

    bundle_dir = _store_bundle(
        out_root=Path(args.out),
        models=models,
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


def _cmd_suggest(args: argparse.Namespace) -> int:
    models, manifest, index_rows = _load_bundle(Path(args.bundle_dir))
    raw_candidates = _read_candidate_records(Path(args.input_file))
    candidates = [_candidate_to_example(obj, idx=i) for i, obj in enumerate(raw_candidates)]

    rows = _suggest(
        models=models,
        index_rows=index_rows,
        candidates=candidates,
        topk=args.topk,
        neighbors=args.neighbors,
    )

    out_path = Path(args.out)
    ensure_dir(out_path.parent)
    content = b"".join(dump_bytes(r).rstrip(b"\n") + b"\n" for r in rows)
    atomic_write_bytes(out_path, content)

    manifest_out = {
        "schema_version": "v0",
        "advisory_only": True,
        "module_id": MODEL_ID,
        "bundle_dir": Path(args.bundle_dir).as_posix(),
        "bundle_sha256": manifest.get("bundle_sha256"),
        "input_file": Path(args.input_file).as_posix(),
        "records": len(rows),
        "out": out_path.as_posix(),
    }
    if args.manifest_out:
        atomic_write_bytes(Path(args.manifest_out), dump_bytes(manifest_out))

    print(out_path.as_posix())
    return 0


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train and run product-account recommender for ERP advisory workflows.")
    sub = p.add_subparsers(dest="command", required=True)

    t = sub.add_parser("train", help="Train product-account recommender bundle")
    t.add_argument("--training-csv", required=True, help="Training input file (CSV or Parquet)")
    t.add_argument("--training-format", choices=["auto", "csv", "parquet"], default="auto")
    t.add_argument("--parquet-batch-size", type=int, default=100000)
    t.add_argument("--out", required=True, help="Model root directory")
    t.add_argument("--metrics-out", default=None)
    t.add_argument("--feedback-jsonl", default=None)
    t.add_argument("--synthetic-products", type=int, default=4000)
    t.add_argument("--seed", type=int, default=41)
    t.add_argument("--epochs", type=int, default=8)
    t.add_argument("--learning-rate", type=float, default=0.1)
    t.add_argument("--l2", type=float, default=0.0001)
    t.add_argument("--hidden-size", type=int, default=64)
    t.add_argument("--n-features", type=int, default=2048)
    t.add_argument("--batch-size", type=int, default=128)
    t.add_argument("--gpu-batch-size", type=int, default=None)
    t.add_argument("--hash-algo", choices=["crc32", "sha256"], default="crc32")
    t.add_argument("--max-hash-cache", type=int, default=200000)
    t.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    t.add_argument("--holdout-ratio", type=float, default=0.2)
    t.add_argument("--topk", type=int, default=3)

    r = sub.add_parser("suggest", help="Suggest accounts for candidate products")
    r.add_argument("--bundle-dir", required=True)
    r.add_argument("--input-file", required=True, help="CSV/JSONL with candidate product records")
    r.add_argument("--out", required=True, help="Output JSONL suggestions")
    r.add_argument("--manifest-out", default=None)
    r.add_argument("--topk", type=int, default=3)
    r.add_argument("--neighbors", type=int, default=5)

    return p


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    try:
        if args.command == "train":
            return _cmd_train(args)
        if args.command == "suggest":
            return _cmd_suggest(args)
        raise AnalyticsError(ExitCode.INVALID_ARGS, f"Unknown command: {args.command}")
    except AnalyticsError as e:
        print(f"ERROR[{e.code.name}]: {e.message}")
        return int(e.code.value)


if __name__ == "__main__":
    raise SystemExit(main())
