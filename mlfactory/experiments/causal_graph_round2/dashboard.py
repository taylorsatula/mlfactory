"""Standalone Rich dashboard for a direct Round 2 run directory."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

from .progress import latest


def _alive(pid_file: Path) -> tuple[int | None, bool]:
    try:
        pid = int(pid_file.read_text(encoding="utf-8").strip())
        os.kill(pid, 0)
        return pid, True
    except (OSError, ValueError):
        return None, False


def _gpu_rows() -> list[list[str]]:
    command = ["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu", "--format=csv,noheader,nounits"]
    try:
        output = subprocess.run(command, capture_output=True, text=True, timeout=4, check=False).stdout
        return [[part.strip() for part in line.split(",")] for line in output.splitlines() if line.strip()]
    except Exception:
        return []


def _events(path: Path, count: int = 8) -> list[dict[str, Any]]:
    rows: deque[dict[str, Any]] = deque(maxlen=count)
    if not path.exists():
        return []
    with path.open(encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            try:
                rows.append(json.loads(line))
            except (json.JSONDecodeError, TypeError):
                continue
    return list(rows)


def _render(run_dir: Path) -> Panel:
    progress_path = run_dir / "dashboard.jsonl"
    record = latest(progress_path) or {}
    pid, alive = _alive(run_dir / "run.pid")
    current = float(record.get("current", 0) or 0)
    total = float(record.get("total", 0) or 0)
    percent = current / total * 100 if total else 0

    summary = Table.grid(padding=(0, 2))
    summary.add_column(style="bold cyan")
    summary.add_column()
    summary.add_row("Run", str(run_dir))
    summary.add_row("Process", f"PID {pid} running" if alive else "not running")
    summary.add_row("Stage", str(record.get("stage", "waiting")))
    summary.add_row("Event", str(record.get("event", "none")))
    summary.add_row("Stage progress", f"{current:.0f}/{total:.0f} ({percent:.1f}%)" if total else "n/a")
    if "loss" in record or "final_loss" in record:
        summary.add_row("Loss", str(record.get("loss", record.get("final_loss"))))
    if "accuracy" in record:
        summary.add_row("Accuracy", f"{float(record['accuracy']):.4f}")
    for key in ("parse_failures", "budget_exhausted", "thinking_markers"):
        if key in record:
            summary.add_row(key.replace("_", " ").title(), str(record[key]))

    gpu = Table(title="GPU", show_header=True)
    for column in ("GPU", "Name", "Used MiB", "Total MiB", "Util %", "Temp C"):
        gpu.add_column(column)
    gpu_rows = _gpu_rows()
    for row in gpu_rows:
        gpu.add_row(*row[:6])
    if not gpu_rows:
        gpu.add_row("-", "unavailable", "-", "-", "-", "-")

    events = Table(title="Recent events", show_header=True)
    events.add_column("Time")
    events.add_column("Stage")
    events.add_column("Event")
    events.add_column("Progress", justify="right")
    for event in _events(progress_path):
        stamp = datetime.fromtimestamp(float(event.get("timestamp", 0))).strftime("%H:%M:%S")
        value = f"{event.get('current', '-')}/{event.get('total', '-')}"
        events.add_row(stamp, str(event.get("stage", "")), str(event.get("event", "")), value)

    artifacts = Table(title="Artifacts", show_header=False)
    for name in ("sealed_eval.jsonl", "summary.json", "final_report.md"):
        path = run_dir / name
        artifacts.add_row(name, "present" if path.exists() else "pending")

    outer = Table.grid(expand=True)
    outer.add_row(Panel(summary, title="Round 2 status"), Panel(gpu))
    outer.add_row(Panel(events), Panel(artifacts))
    return Panel(outer, title="CausalGraph Round 2", border_style="cyan")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default="runs/causal-graph-round2-contract")
    parser.add_argument("--refresh", type=float, default=2.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    run_dir = Path(args.run_dir).resolve()
    console = Console()
    if args.once:
        console.print(_render(run_dir))
        return
    with Live(_render(run_dir), console=console, refresh_per_second=max(0.2, 1.0 / args.refresh), screen=True) as live:
        while True:
            time.sleep(args.refresh)
            live.update(_render(run_dir))


if __name__ == "__main__":
    main()
