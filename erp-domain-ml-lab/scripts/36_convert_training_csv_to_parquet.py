#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from ledgerstudio_analytics.errors import AnalyticsError, ExitCode


def _parse_columns(raw: str | None) -> list[str] | None:
    if raw is None:
        return None
    out = [c.strip() for c in str(raw).split(",") if c.strip()]
    return out or None


def _run(args: argparse.Namespace) -> int:
    try:
        import pyarrow as pa
        import pyarrow.csv as pacsv
        import pyarrow.parquet as pq
    except Exception as e:
        raise AnalyticsError(ExitCode.INVALID_ARGS, "pyarrow is required. Install with: pip install pyarrow") from e

    in_csv = Path(args.input_csv)
    out_parquet = Path(args.output_parquet)

    if not in_csv.exists() or not in_csv.is_file():
        raise AnalyticsError(ExitCode.INVALID_ARGS, f"Input CSV not found: {in_csv}")
    if out_parquet.exists() and not args.force:
        raise AnalyticsError(ExitCode.INVALID_ARGS, f"Output already exists: {out_parquet} (use --force to overwrite)")

    out_parquet.parent.mkdir(parents=True, exist_ok=True)

    include_columns = _parse_columns(args.columns)
    read_opts = pacsv.ReadOptions(block_size=max(1, int(args.block_size_bytes)))
    parse_opts = pacsv.ParseOptions(delimiter=str(args.delimiter))
    convert_opts = pacsv.ConvertOptions(strings_can_be_null=True, include_columns=include_columns)

    reader = pacsv.open_csv(
        in_csv.as_posix(),
        read_options=read_opts,
        parse_options=parse_opts,
        convert_options=convert_opts,
    )

    writer: pq.ParquetWriter | None = None
    rows = 0
    batches = 0
    try:
        for batch in reader:
            table = pa.Table.from_batches([batch])
            if writer is None:
                writer = pq.ParquetWriter(
                    out_parquet.as_posix(),
                    table.schema,
                    compression=args.compression,
                    use_dictionary=bool(args.use_dictionary),
                )
            writer.write_table(table)
            rows += int(batch.num_rows)
            batches += 1

        if writer is None:
            empty = pa.Table.from_batches([], schema=reader.schema)
            writer = pq.ParquetWriter(
                out_parquet.as_posix(),
                empty.schema,
                compression=args.compression,
                use_dictionary=bool(args.use_dictionary),
            )
            writer.write_table(empty)
    finally:
        if writer is not None:
            writer.close()

    print(
        f"{out_parquet.as_posix()} rows={rows} batches={batches} compression={args.compression}"
    )
    return 0


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Convert ERP training CSV to Parquet for faster/lower-RAM ingestion.")
    p.add_argument("--input-csv", required=True)
    p.add_argument("--output-parquet", required=True)
    p.add_argument("--compression", choices=["zstd", "snappy", "gzip", "brotli", "lz4", "none"], default="zstd")
    p.add_argument("--delimiter", default=",")
    p.add_argument("--columns", default=None, help="Optional comma-separated subset of columns to keep")
    p.add_argument("--block-size-bytes", type=int, default=8 * 1024 * 1024)
    p.add_argument("--use-dictionary", action="store_true", default=True)
    p.add_argument("--force", action="store_true")
    return p


def main() -> int:
    args = _build_parser().parse_args()
    if args.compression == "none":
        args.compression = None
    try:
        return _run(args)
    except AnalyticsError as e:
        print(f"ERROR[{e.code.name}]: {e.message}")
        raise SystemExit(int(e.code.value))


if __name__ == "__main__":
    raise SystemExit(main())
