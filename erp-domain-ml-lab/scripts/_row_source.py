from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable, Iterator

from ledgerstudio_analytics.errors import AnalyticsError, ExitCode

_PARQUET_SUFFIXES = {".parquet", ".pq"}


def detect_training_format(path: Path, *, requested: str = "auto") -> str:
    req = str(requested or "auto").strip().lower()
    if req not in {"auto", "csv", "parquet"}:
        raise AnalyticsError(ExitCode.INVALID_ARGS, f"Invalid training format: {requested} (expected auto|csv|parquet)")
    if req != "auto":
        return req
    if path.suffix.lower() in _PARQUET_SUFFIXES:
        return "parquet"
    return "csv"


def _iter_csv_rows(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise AnalyticsError(ExitCode.INVALID_ARGS, f"CSV has no header: {path}")
        for row in reader:
            yield dict(row)


def _coerce_arrow_value(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _iter_parquet_rows(path: Path, *, columns: Iterable[str] | None, batch_size: int) -> Iterator[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except Exception as e:
        raise AnalyticsError(
            ExitCode.INVALID_ARGS,
            "Parquet input requires pyarrow. Install it with: pip install pyarrow",
        ) from e

    try:
        parquet = pq.ParquetFile(path.as_posix())
    except Exception as e:
        raise AnalyticsError(ExitCode.INVALID_ARGS, f"Failed to read Parquet file: {path}") from e

    selected_cols = list(columns) if columns is not None else None
    try:
        batches = parquet.iter_batches(batch_size=max(1, int(batch_size)), columns=selected_cols)
    except Exception as e:
        raise AnalyticsError(ExitCode.INVALID_ARGS, f"Failed to iterate Parquet batches: {path}") from e

    for batch in batches:
        names = list(batch.schema.names)
        cols = [batch.column(i).to_pylist() for i in range(len(names))]
        n_rows = int(batch.num_rows)
        for r in range(n_rows):
            row: dict[str, Any] = {}
            for c_idx, c_name in enumerate(names):
                row[c_name] = _coerce_arrow_value(cols[c_idx][r])
            yield row


def iter_rows_from_file(
    path: Path,
    *,
    requested_format: str = "auto",
    parquet_columns: Iterable[str] | None = None,
    parquet_batch_size: int = 100000,
) -> Iterator[dict[str, Any]]:
    if not path.exists() or not path.is_file():
        raise AnalyticsError(ExitCode.INVALID_ARGS, f"Training file not found: {path}")

    resolved = detect_training_format(path, requested=requested_format)
    if resolved == "csv":
        yield from _iter_csv_rows(path)
        return
    if resolved == "parquet":
        yield from _iter_parquet_rows(path, columns=parquet_columns, batch_size=parquet_batch_size)
        return
    raise AnalyticsError(ExitCode.INVALID_ARGS, f"Unsupported resolved training format: {resolved}")
