#!/usr/bin/env python3
"""Summarize DeltaNetAttention NVTX ranges from an Nsight Systems sqlite DB.

The NVTX range duration itself is a CPU enqueue duration.  For GPU timing this
script finds CUDA runtime launches issued inside each NVTX range, maps them to
their CUPTI kernel records by correlationId, and summarizes the GPU kernel span
for each range.
"""

from __future__ import annotations

import argparse
import sqlite3
import statistics
from collections import defaultdict
from dataclasses import dataclass


DEFAULT_TEXTS = (
    "DeltaNetAttention.forward",
    "DeltaNetAttention.stateful_middle",
    "DeltaNetAttention.delta_rule_core",
)


@dataclass(frozen=True)
class Row:
    text: str
    global_tid: int
    device_id: int
    nvtx_start: int
    cpu_ms: float
    gpu_span_ms: float
    gpu_active_ms: float
    kernels: int


def table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return float("nan")
    values = sorted(values)
    idx = round((len(values) - 1) * pct)
    return values[idx]


def summarize_values(values: list[float]) -> str:
    if not values:
        return "n=0"
    return (
        f"n={len(values)} avg={statistics.fmean(values):.4f} "
        f"p50={statistics.median(values):.4f} "
        f"p90={percentile(values, 0.90):.4f} "
        f"min={min(values):.4f} max={max(values):.4f}"
    )


def read_rows(path: str, texts: tuple[str, ...], skip_first_frac: float) -> list[Row]:
    conn = sqlite3.connect(path)
    try:
        nvtx_cols = table_columns(conn, "NVTX_EVENTS")
        runtime_cols = table_columns(conn, "CUPTI_ACTIVITY_KIND_RUNTIME")
        kernel_cols = table_columns(conn, "CUPTI_ACTIVITY_KIND_KERNEL")
        required_nvtx = {"start", "end", "text", "globalTid"}
        required_runtime = {"start", "end", "correlationId"}
        required_kernel = {"start", "end", "deviceId", "correlationId"}
        missing = (
            (required_nvtx - nvtx_cols)
            | (required_runtime - runtime_cols)
            | (required_kernel - kernel_cols)
        )
        if missing:
            raise RuntimeError(f"{path}: missing expected sqlite columns: {sorted(missing)}")

        runtime_thread_join = (
            "AND r.globalTid = n.globalTid" if "globalTid" in runtime_cols else ""
        )
        placeholders = ",".join("?" for _ in texts)
        query = f"""
        SELECT
            n.rowid AS range_id,
            n.text AS text,
            n.globalTid AS global_tid,
            k.deviceId AS device_id,
            n.start AS nvtx_start,
            (n.end - n.start) / 1e6 AS cpu_ms,
            (MAX(k.end) - MIN(k.start)) / 1e6 AS gpu_span_ms,
            SUM(k.end - k.start) / 1e6 AS gpu_active_ms,
            COUNT(*) AS kernels
        FROM NVTX_EVENTS n
        JOIN CUPTI_ACTIVITY_KIND_RUNTIME r
          ON r.start >= n.start
         AND r.start <= n.end
         {runtime_thread_join}
        JOIN CUPTI_ACTIVITY_KIND_KERNEL k
          ON k.correlationId = r.correlationId
        WHERE n.text IN ({placeholders})
          AND n.end > n.start
        GROUP BY n.rowid, n.text, n.globalTid, k.deviceId, n.start, n.end
        HAVING kernels > 0
        ORDER BY n.text, n.globalTid, k.deviceId, n.start
        """
        raw_rows = [
            Row(
                text=row[1],
                global_tid=int(row[2]),
                device_id=int(row[3]),
                nvtx_start=int(row[4]),
                cpu_ms=float(row[5]),
                gpu_span_ms=float(row[6]),
                gpu_active_ms=float(row[7]),
                kernels=int(row[8]),
            )
            for row in conn.execute(query, texts)
        ]
    finally:
        conn.close()

    if skip_first_frac <= 0:
        return raw_rows

    grouped: dict[tuple[str, int, int], list[Row]] = defaultdict(list)
    for row in raw_rows:
        grouped[(row.text, row.global_tid, row.device_id)].append(row)

    kept: list[Row] = []
    for rows in grouped.values():
        rows.sort(key=lambda row: row.nvtx_start)
        skip = int(len(rows) * skip_first_frac)
        kept.extend(rows[skip:])
    kept.sort(key=lambda row: (row.text, row.global_tid, row.device_id, row.nvtx_start))
    return kept


def print_one(name: str, rows: list[Row], texts: tuple[str, ...]) -> dict[str, float]:
    print(f"===== {name} =====")
    by_text: dict[str, list[Row]] = defaultdict(list)
    for row in rows:
        by_text[row.text].append(row)

    avg_span: dict[str, float] = {}
    for text in texts:
        values = by_text.get(text, [])
        span = [row.gpu_span_ms for row in values]
        active = [row.gpu_active_ms for row in values]
        cpu = [row.cpu_ms for row in values]
        kernels = [row.kernels for row in values]
        if span:
            avg_span[text] = statistics.fmean(span)
        print(text)
        print(f"  gpu_span_ms:  {summarize_values(span)}")
        print(f"  gpu_active_ms: {summarize_values(active)}")
        print(f"  cpu_nvtx_ms:   {summarize_values(cpu)}")
        if kernels:
            print(f"  kernels/range: avg={statistics.fmean(kernels):.1f} min={min(kernels)} max={max(kernels)}")
    return avg_span


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", help="Base sqlite path")
    parser.add_argument("--grid", help="Grid/overlap sqlite path")
    parser.add_argument("--sqlite", action="append", default=[], help="Additional sqlite path to summarize")
    parser.add_argument("--skip-first-frac", type=float, default=0.0)
    parser.add_argument("--texts", nargs="+", default=list(DEFAULT_TEXTS))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    texts = tuple(args.texts)
    summaries: dict[str, dict[str, float]] = {}

    for label, path in (("base", args.base), ("grid", args.grid)):
        if not path:
            continue
        rows = read_rows(path, texts, args.skip_first_frac)
        summaries[label] = print_one(label, rows, texts)

    for idx, path in enumerate(args.sqlite):
        rows = read_rows(path, texts, args.skip_first_frac)
        print_one(path if path else f"sqlite{idx}", rows, texts)

    if "base" in summaries and "grid" in summaries:
        print("===== speedup: base_gpu_span_avg / grid_gpu_span_avg =====")
        for text in texts:
            base = summaries["base"].get(text)
            grid = summaries["grid"].get(text)
            if base and grid:
                print(f"{text}: {base / grid:.4f}x  ({base:.4f} ms -> {grid:.4f} ms)")


if __name__ == "__main__":
    main()
