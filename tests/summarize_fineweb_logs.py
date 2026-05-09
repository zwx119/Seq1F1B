#!/usr/bin/env python3
"""Summarize FineWeb Seq1F1B benchmark logs.

The training runner prints compact summary lines at the end of each log:

    time:  9.20+-0.09
    toks:  42736.99+-410.93
    tflops:  154.05+-1.48
    mem_arr:  31.9/27.6/...

This helper collects those blocks recursively from an experiment root. If a log
was interrupted before the compact summary was printed, it falls back to parsing
per-iteration timing lines and computes an approximate summary after skipping
the first few warmup iterations.
"""

from __future__ import annotations

import argparse
import csv
import re
import statistics
from dataclasses import dataclass
from pathlib import Path


SUMMARY_KEYS = ("time", "toks", "tflops", "mem_arr")
ITER_RE = re.compile(
    r"iteration\s+(?P<iter>\d+)/\s*(?P<total>\d+).*?"
    r"elapsed time per iteration \(ms\):\s*(?P<time_ms>[0-9.]+).*?"
    r"toks/s:\s*(?P<toks>[0-9.]+).*?"
    r"TFlops/s:\s*(?P<tflops>[0-9.]+).*?"
    r"mem_each_stage:\s*(?P<mem>[0-9.,]+)"
)
REPEAT_SUFFIX_RE = re.compile(r"(?P<base>.+?)(?:_run\d+|_[a-z])$")


@dataclass
class LogSummary:
    label: str
    path: Path
    time: str | None = None
    toks: str | None = None
    tflops: str | None = None
    mem_arr: str | None = None
    status: str = "OK"

    def as_row(self) -> dict[str, str]:
        return {
            "label": self.label,
            "status": self.status,
            "time": self.time or "",
            "toks": self.toks or "",
            "tflops": self.tflops or "",
            "mem_arr": self.mem_arr or "",
            "path": str(self.path),
        }


def mean_pm(values: list[float], digits: int = 2) -> str:
    if not values:
        return ""
    mean = statistics.fmean(values)
    std = statistics.pstdev(values) if len(values) > 1 else 0.0
    return f"{mean:.{digits}f}+-{std:.{digits}f}"


def mean_value(value: str | None) -> float | None:
    if not value:
        return None
    return float(value.split("+-", 1)[0])


def base_repeat_label(label: str) -> str:
    match = REPEAT_SUFFIX_RE.fullmatch(label)
    return match.group("base") if match else label


def aggregate_repeats(summaries: list[LogSummary]) -> list[LogSummary]:
    grouped: dict[str, list[LogSummary]] = {}
    order: list[str] = []
    for item in summaries:
        base = base_repeat_label(item.label)
        if base not in grouped:
            grouped[base] = []
            order.append(base)
        grouped[base].append(item)

    aggregated: list[LogSummary] = []
    for base in order:
        items = grouped[base]
        if len(items) == 1:
            aggregated.append(items[0])
            continue

        ok_items = [item for item in items if item.status == "OK"]
        status = "OK" if len(ok_items) == len(items) else f"PARTIAL_OK_{len(ok_items)}_OF_{len(items)}"
        metric_values = {
            key: [value for value in (mean_value(getattr(item, key)) for item in ok_items) if value is not None]
            for key in ("time", "toks", "tflops")
        }

        mem_rows: list[list[float]] = []
        for item in ok_items:
            if not item.mem_arr:
                continue
            mem_rows.append([float(value) for value in item.mem_arr.split("/") if value])
        mem_arr = None
        if mem_rows and len({len(row) for row in mem_rows}) == 1:
            mem_arr = "/".join(f"{statistics.fmean(stage):.1f}" for stage in zip(*mem_rows))

        aggregated.append(
            LogSummary(
                label=base,
                path=Path(f"{len(ok_items)}/{len(items)} repeats"),
                time=mean_pm(metric_values["time"]) if metric_values["time"] else None,
                toks=mean_pm(metric_values["toks"]) if metric_values["toks"] else None,
                tflops=mean_pm(metric_values["tflops"]) if metric_values["tflops"] else None,
                mem_arr=mem_arr,
                status=status,
            )
        )
    return aggregated


def parse_compact_summary(lines: list[str]) -> dict[str, str]:
    found: dict[str, str] = {}
    for line in lines:
        stripped = line.strip()
        for key in SUMMARY_KEYS:
            prefix = f"{key}:"
            if stripped.startswith(prefix):
                found[key] = stripped[len(prefix) :].strip()
    return found


def parse_iter_fallback(lines: list[str], skip_iters: int) -> dict[str, str]:
    time_s: list[float] = []
    toks: list[float] = []
    tflops: list[float] = []
    mem_arr = ""
    for line in lines:
        match = ITER_RE.search(line)
        if not match:
            continue
        iteration = int(match.group("iter"))
        if iteration <= skip_iters:
            continue
        time_s.append(float(match.group("time_ms")) / 1000.0)
        toks.append(float(match.group("toks")))
        tflops.append(float(match.group("tflops")))
        mem_arr = "/".join(match.group("mem").split(","))
    if not time_s:
        return {}
    return {
        "time": mean_pm(time_s),
        "toks": mean_pm(toks),
        "tflops": mean_pm(tflops),
        "mem_arr": mem_arr,
    }


def detect_failure(lines: list[str]) -> str:
    text = "\n".join(lines[-200:]).lower()
    if "out of memory" in text or "cuda error: out of memory" in text:
        return "OOM"
    if "traceback" in text or "childfailederror" in text:
        return "FAILED"
    return "NO_SUMMARY"


def summarize_log(root: Path, log: Path, label: str, skip_iters: int) -> LogSummary:
    lines = log.read_text(errors="replace").splitlines()
    compact = parse_compact_summary(lines)
    fallback = {} if all(k in compact for k in SUMMARY_KEYS) else parse_iter_fallback(lines, skip_iters)
    data = {**fallback, **compact}
    status = "OK" if all(k in data for k in SUMMARY_KEYS) else detect_failure(lines)
    return LogSummary(
        label=label,
        path=log.relative_to(root),
        time=data.get("time"),
        toks=data.get("toks"),
        tflops=data.get("tflops"),
        mem_arr=data.get("mem_arr"),
        status=status,
    )


def candidate_dirs(root: Path) -> list[Path]:
    dirs = {p.parent for p in root.rglob("log_fineweb_sp*.txt")}
    if any(root.glob("log_fineweb_sp*.txt")):
        dirs.add(root)
    for child in root.iterdir() if root.exists() else []:
        if child.is_dir() and not child.name.startswith(("tb", "ckpt")):
            dirs.add(child)
    return sorted(dirs, key=lambda p: str(p.relative_to(root)))


def collect(root: Path, skip_iters: int) -> list[LogSummary]:
    summaries: list[LogSummary] = []
    for directory in candidate_dirs(root):
        logs = sorted(directory.glob("log_fineweb_sp*.txt"))
        rel = "." if directory == root else str(directory.relative_to(root))
        if not logs:
            summaries.append(LogSummary(label=rel, path=directory.relative_to(root), status="MISSING_LOG"))
            continue
        for log in logs:
            label = rel if len(logs) == 1 else f"{rel}/{log.stem}"
            summaries.append(summarize_log(root, log, label, skip_iters))
    return summaries


def print_summary(summaries: list[LogSummary]) -> None:
    for item in summaries:
        print(f"===== {item.label} =====")
        if item.status != "OK":
            print(f"{item.status}: {item.path}")
        if item.time:
            print(f"time:  {item.time}")
        if item.toks:
            print(f"toks:  {item.toks}")
        if item.tflops:
            print(f"tflops:  {item.tflops}")
        if item.mem_arr:
            print(f"mem_arr:  {item.mem_arr}")
        print()


def write_csv(path: Path, summaries: list[LogSummary]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["label", "status", "time", "toks", "tflops", "mem_arr", "path"])
        writer.writeheader()
        for item in summaries:
            writer.writerow(item.as_row())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Experiment output root")
    parser.add_argument("--skip-iters", type=int, default=2, help="Fallback parser warmup iterations to skip")
    parser.add_argument("--csv", type=Path, default=None, help="Optional CSV output path")
    args = parser.parse_args()

    if not args.root.exists():
        print(f"MISSING_ROOT: {args.root}")
        return

    summaries = collect(args.root, args.skip_iters)
    if not summaries:
        print(f"NO_LOGS_FOUND: {args.root}")
        return

    summaries = aggregate_repeats(summaries)
    print_summary(summaries)
    if args.csv is not None:
        write_csv(args.csv, summaries)


if __name__ == "__main__":
    main()
