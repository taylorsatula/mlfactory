#!/usr/bin/env python3
"""Read-only Rich Live dashboard for the mlfactory registry.

Usage:
    mlfactory dashboard
    mlfactory dashboard --watch-run <run_id>
    mlfactory dashboard --stage train
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import time
from pathlib import Path

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from mlfactory.core.registry import Registry


GPU_QUERY = [
    "nvidia-smi",
    "--query-gpu=index,name,temperature.gpu,memory.used,memory.total,utilization.gpu,power.draw",
    "--format=csv,noheader",
]


def _gpu_stats() -> list[dict] | None:
    try:
        out = subprocess.check_output(GPU_QUERY, text=True, stderr=subprocess.DEVNULL)
    except Exception:
        return None
    gpus = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 7:
            gpus.append(
                {
                    "index": parts[0],
                    "name": parts[1],
                    "temp": parts[2],
                    "mem_used": parts[3],
                    "mem_total": parts[4],
                    "util": parts[5].replace(" %", ""),
                    "power": parts[6],
                }
            )
    return gpus


def _disk_usage() -> str:
    try:
        u = shutil.disk_usage(".")
        free_gb = u.free / (1024**3)
        total_gb = u.total / (1024**3)
        return f"{free_gb:.1f} / {total_gb:.1f} GB free"
    except Exception as e:
        return f"unknown ({e})"


def _stage_counts(registry: Registry) -> dict[str, int]:
    counts: dict[str, int] = {}
    for stage in ["collect", "classify", "train", "eval"]:
        counts[stage] = len(registry.find(stage=stage, limit=10000))
    return counts


def _recent_runs_table(registry: Registry, stage: str | None, limit: int) -> Table:
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Run ID")
    table.add_column("Stage", width=10)
    table.add_column("Status", width=10)
    table.add_column("Created")
    for r in registry.find(stage=stage, limit=limit):
        status_style = {
            "completed": "green",
            "running": "blue",
            "failed": "red",
            "guarded": "yellow",
            "aborted": "dim",
            "pending": "dim",
        }.get(r.status, "white")
        table.add_row(
            r.run_id,
            r.stage,
            Text(r.status, style=status_style),
            r.created_at[:19],
        )
    return table


def _detail_panel(registry: Registry, run_id: str | None) -> Panel:
    if run_id is None:
        return Panel("No run selected. Use --watch-run or press n/p to cycle.", title="Run Detail")
    manifest = registry.get(run_id)
    if manifest is None:
        return Panel(f"Run {run_id} not found in registry.", title="Run Detail")

    text = Text()
    text.append(f"{manifest.run_id}\n", style="bold cyan")
    text.append(f"Stage: {manifest.stage}  Status: {manifest.status}\n")
    text.append(f"Created: {manifest.created_at[:19]}\n")
    if manifest.git.commit:
        dirty = " (dirty)" if manifest.git.dirty else ""
        text.append(f"Git: {manifest.git.commit[:12]}{dirty}  {manifest.git.branch or ''}\n")
    text.append(f"Source: {manifest.source.path if manifest.source else 'n/a'}\n")
    text.append(f"Inputs: {len(manifest.inputs)}\n")
    for inp in manifest.inputs[:3]:
        text.append(f"  • {Path(inp.path).name}: {inp.sha256[:16]}...\n", style="dim")
    text.append(f"Artifacts: {len(manifest.artifacts)}\n")
    if manifest.summary:
        text.append("Summary:\n")
        for k, v in list(manifest.summary.items())[:8]:
            text.append(f"  {k}: {v}\n", style="dim")
    return Panel(text, title=f"Run Detail: {run_id[:40]}")


def _gpu_panel() -> Panel:
    gpus = _gpu_stats()
    if not gpus:
        return Panel("GPU data unavailable", title="GPU Status")
    table = Table(show_header=True, header_style="bold green")
    table.add_column("GPU")
    table.add_column("Temp")
    table.add_column("Mem")
    table.add_column("Util")
    table.add_column("Power")
    for g in gpus:
        try:
            temp = int(g["temp"])
            temp_style = "red" if temp >= 80 else "yellow" if temp >= 70 else "green"
        except ValueError:
            temp_style = "white"
        util = g["util"]
        try:
            util_i = int(util)
            util_bar = "█" * (util_i // 10) + "░" * (10 - util_i // 10)
        except ValueError:
            util_bar = "?"
        table.add_row(
            f"{g['index']}: {g['name'][:30]}",
            Text(f"{g['temp']}°C", style=temp_style),
            f"{g['mem_used']} / {g['mem_total']}",
            f"{util_bar} {util}%",
            g["power"],
        )
    return Panel(table, title="GPU Status")


def _stats_panel(registry: Registry, stage: str | None) -> Panel:
    counts = _stage_counts(registry)
    table = Table(show_header=False)
    table.add_column("Metric", style="bold")
    table.add_column("Value")
    for st, c in counts.items():
        table.add_row(f"{st} runs", str(c))
    table.add_row("Total", str(len(registry.find(limit=100000))))
    table.add_row("Disk", _disk_usage())
    if stage:
        table.add_row("Filter", stage)
    return Panel(table, title="Registry")


def build_layout(registry: Registry, stage: str | None, watch_run: str | None, recent_limit: int) -> Layout:
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="body"),
        Layout(name="footer", size=10),
    )
    layout["body"].split_row(
        Layout(name="runs", ratio=2),
        Layout(name="detail", ratio=3),
    )
    layout["footer"].split_row(
        Layout(name="stats", ratio=1),
        Layout(name="gpu", ratio=2),
    )

    header = Text()
    header.append("mlfactory dashboard", style="bold cyan")
    header.append(f"  •  registry: {registry.db_path}  •  ")
    header.append("Ctrl-C to quit", style="dim")
    layout["header"].update(Panel(header, border_style="cyan"))

    layout["runs"].update(Panel(_recent_runs_table(registry, stage, recent_limit), title="Recent Runs"))
    layout["detail"].update(_detail_panel(registry, watch_run))
    layout["stats"].update(_stats_panel(registry, stage))
    layout["gpu"].update(_gpu_panel())
    return layout


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="mlfactory read-only dashboard")
    parser.add_argument("--registry", "-r", default="data/registry.db")
    parser.add_argument("--stage", default=None)
    parser.add_argument("--watch-run", default=None)
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--refresh", type=float, default=2.0)
    args = parser.parse_args()

    registry = Registry(args.registry)
    console = Console()

    watch_run = args.watch_run
    if watch_run is None:
        running = registry.find(status="running", limit=1)
        if running:
            watch_run = running[0].run_id
        else:
            latest = registry.find(stage=args.stage, limit=1)
            if latest:
                watch_run = latest[0].run_id

    with Live(
        build_layout(registry, args.stage, watch_run, args.limit),
        console=console,
        refresh_per_second=1,
        screen=True,
    ) as live:
        try:
            while True:
                live.update(build_layout(registry, args.stage, watch_run, args.limit))
                time.sleep(args.refresh)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
