#!/usr/bin/env python3
"""Config-driven Rich Live dashboard for the mlfactory registry.

Each experiment can expose a ``dashboard.json`` (or ``dashboard.yaml``) that
declares which probes to run and how to lay out the TUI. The dashboard itself
remains read-only and reusable.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from mlfactory.core.dashboard_config import (
    ExperimentDashboardConfig,
    Pane,
    Probe,
    generic_config,
)
from mlfactory.core.manifest import RunManifest
from mlfactory.core.registry import Registry


GPU_QUERY = [
    "nvidia-smi",
    "--query-gpu=index,name,temperature.gpu,memory.used,memory.total,utilization.gpu,power.draw",
    "--format=csv,noheader",
]


@dataclass
class ProbeResult:
    value: Any
    display: str
    style: str = ""
    healthy: bool | None = None


# ---------------------------------------------------------------------------
# probe execution
# ---------------------------------------------------------------------------

def _resolve_path(template: str | None, run_dir: Path | None) -> Path | None:
    if not template:
        return None
    t = template
    if run_dir is not None:
        t = t.replace("{run_dir}", str(run_dir))
    t = os.path.expanduser(t)
    return Path(t)


def _get_pid(probe: Probe, run_dir: Path | None) -> int | None:
    if probe.pid_file:
        path = _resolve_path(probe.pid_file, run_dir)
        if path and path.exists():
            try:
                return int(path.read_text(encoding="utf-8").strip().split()[0])
            except Exception:
                return None
    if probe.pid_env:
        val = os.environ.get(probe.pid_env)
        if val:
            try:
                return int(val)
            except Exception:
                return None
    pid = probe.params.get("pid")
    if pid is not None:
        try:
            return int(pid)
        except Exception:
            return None
    return None


def _style_for_numeric(result: ProbeResult, probe: Probe) -> str:
    try:
        v = float(result.value) if result.value is not None else None
    except (TypeError, ValueError):
        return result.style
    if v is None:
        return result.style
    if probe.danger_if_above is not None and v > probe.danger_if_above:
        return "red"
    if probe.danger_if_below is not None and v < probe.danger_if_below:
        return "red"
    if probe.warn_if_above is not None and v > probe.warn_if_above:
        return "yellow"
    if probe.warn_if_below is not None and v < probe.warn_if_below:
        return "yellow"
    return "green"


def _run_probe(probe: Probe, run_dir: Path | None, manifest: RunManifest | None) -> ProbeResult:
    if not probe.enabled:
        return ProbeResult(value=None, display="disabled", style="dim")

    try:
        if probe.type == "const":
            val = probe.params.get("value")
            return ProbeResult(value=val, display=str(val), style="")

        if probe.type == "spec_value":
            if manifest is None:
                return ProbeResult(value=None, display="no run", style="dim")
            val = manifest.spec
            path = probe.path or probe.params.get("path")
            if path:
                for key in path.split("."):
                    if isinstance(val, dict):
                        val = val.get(key)
                    else:
                        val = None
                        break
            display = str(val) if val is not None else "n/a"
            r = ProbeResult(value=val, display=display)
            try:
                r.style = _style_for_numeric(r, probe)
            except Exception:
                pass
            return r

        if probe.type == "file_line_count":
            path = _resolve_path(probe.file, run_dir)
            if not path or not path.exists():
                return ProbeResult(value=0, display="0", style="dim")
            count = sum(1 for _ in open(path, "rb"))
            r = ProbeResult(value=count, display=str(count))
            r.style = _style_for_numeric(r, probe)
            return r

        if probe.type == "regex_count":
            path = _resolve_path(probe.file, run_dir)
            if not path or not path.exists() or not probe.regex:
                return ProbeResult(value=0, display="0", style="dim")
            pattern = re.compile(probe.regex)
            count = sum(1 for line in open(path, "r", encoding="utf-8", errors="ignore") if pattern.search(line))
            r = ProbeResult(value=count, display=str(count))
            r.style = _style_for_numeric(r, probe)
            return r

        if probe.type == "regex_last_match":
            path = _resolve_path(probe.file, run_dir)
            if not path or not path.exists() or not probe.regex:
                return ProbeResult(value=None, display="n/a", style="dim")
            pattern = re.compile(probe.regex)
            last_match: str | None = None
            for line in open(path, "r", encoding="utf-8", errors="ignore"):
                m = pattern.search(line)
                if m:
                    last_match = m.group(1) if m.groups() else m.group(0)
            return ProbeResult(value=last_match, display=last_match or "n/a", style="" if last_match else "dim")

        if probe.type == "jsonl_last_record":
            path = _resolve_path(probe.file, run_dir)
            if not path or not path.exists():
                return ProbeResult(value=None, display="n/a", style="dim")
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    lines = [l.strip() for l in f if l.strip()]
                if not lines:
                    return ProbeResult(value=None, display="empty", style="dim")
                rec = json.loads(lines[-1])
                val = rec
                if probe.path:
                    for key in probe.path.split("."):
                        if isinstance(val, dict):
                            val = val.get(key)
                        else:
                            val = None
                            break
                display = str(val) if val is not None else "n/a"
                r = ProbeResult(value=val, display=display)
                try:
                    r.style = _style_for_numeric(r, probe)
                except Exception:
                    pass
                return r
            except Exception as e:
                return ProbeResult(value=None, display=f"error: {e}", style="red")

        if probe.type == "process_alive":
            pid = _get_pid(probe, run_dir)
            if pid is None:
                return ProbeResult(value=False, display="no pid", style="yellow")
            try:
                os.kill(pid, 0)
                return ProbeResult(value=True, display="running", style="green", healthy=True)
            except Exception:
                return ProbeResult(value=False, display="not found", style="red", healthy=False)

        if probe.type == "http_status":
            url = probe.url or probe.params.get("url")
            if not url:
                return ProbeResult(value=None, display="no url", style="dim")
            try:
                out = subprocess.check_output(
                    ["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}", "-m", "5", url],
                    stderr=subprocess.DEVNULL,
                    text=True,
                )
                code = int(out.strip())
                healthy = 200 <= code < 300
                return ProbeResult(value=code, display=str(code), style="green" if healthy else "red", healthy=healthy)
            except Exception as e:
                return ProbeResult(value=None, display=f"down ({e})", style="red", healthy=False)

        if probe.type == "http_json":
            url = probe.url or probe.params.get("url")
            if not url:
                return ProbeResult(value=None, display="no url", style="dim")
            try:
                out = subprocess.check_output(
                    ["curl", "-s", "-m", "5", url],
                    stderr=subprocess.DEVNULL,
                    text=True,
                )
                data = json.loads(out)
                val = data
                if probe.path:
                    for key in probe.path.split("."):
                        if isinstance(val, dict):
                            val = val.get(key)
                        else:
                            val = None
                            break
                display = str(val) if val is not None else out[:30]
                return ProbeResult(value=val, display=display, style="green")
            except Exception as e:
                return ProbeResult(value=None, display=f"error ({e})", style="red")

        if probe.type == "gpu_status":
            try:
                out = subprocess.check_output(GPU_QUERY, text=True, stderr=subprocess.DEVNULL)
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
                return ProbeResult(value=gpus, display=f"{len(gpus)} GPUs", style="")
            except Exception as e:
                return ProbeResult(value=[], display=f"unavailable ({e})", style="red")

        if probe.type == "disk_usage":
            path = _resolve_path(probe.file or probe.params.get("path") or ".", run_dir)
            if path is None:
                path = Path(".")
            try:
                u = shutil.disk_usage(path)
                free_gb = u.free / (1024**3)
                total_gb = u.total / (1024**3)
                pct = (u.used / u.total) * 100
                display = f"{free_gb:.1f}/{total_gb:.1f} GB free ({pct:.1f}% used)"
                r = ProbeResult(value=pct, display=display)
                r.style = _style_for_numeric(r, probe) or ""
                return r
            except Exception as e:
                return ProbeResult(value=None, display=f"unknown ({e})", style="red")

        if probe.type == "file_exists":
            path = _resolve_path(probe.file, run_dir)
            exists = bool(path and path.exists())
            return ProbeResult(value=exists, display="yes" if exists else "no", style="green" if exists else "red")

        if probe.type == "shell_command":
            cmd = probe.command or probe.params.get("command")
            if not cmd:
                return ProbeResult(value=None, display="no command", style="dim")
            try:
                out = subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.DEVNULL).strip()
                if probe.parser == "json":
                    val = json.loads(out)
                elif probe.parser == "float":
                    val = float(out)
                elif probe.parser == "int":
                    val = int(out)
                else:
                    val = out
                display = str(val)
                r = ProbeResult(value=val, display=display)
                try:
                    r.style = _style_for_numeric(r, probe)
                except Exception:
                    pass
                return r
            except Exception as e:
                return ProbeResult(value=None, display=f"error ({e})", style="red")

        return ProbeResult(value=None, display=f"unknown probe type {probe.type}", style="red")
    except Exception as e:
        return ProbeResult(value=None, display=f"probe error: {e}", style="red")


# ---------------------------------------------------------------------------
# pane rendering
# ---------------------------------------------------------------------------

def _run_all_probes(config: ExperimentDashboardConfig, run_dir: Path | None, manifest: RunManifest | None) -> dict[str, ProbeResult]:
    results: dict[str, ProbeResult] = {}
    for probe in config.probes:
        results[probe.id] = _run_probe(probe, run_dir, manifest)
    return results


def _ascii_bar(value: float, width: int = 30, max_val: float = 100.0) -> str:
    if max_val == 0:
        return "░" * width
    filled = int(round((max(value, 0) / max_val) * width))
    filled = max(0, min(width, filled))
    return "█" * filled + "░" * (width - filled)


def _render_overview(config: ExperimentDashboardConfig, results: dict[str, ProbeResult], manifest: RunManifest | None) -> Panel:
    text = Text()
    if manifest is None:
        text.append("No run selected\n", style="bold yellow")
    else:
        text.append(f"{manifest.run_id}\n", style="bold cyan")
        text.append(f"Stage: {manifest.stage}  Status: ")
        status_style = {
            "completed": "green",
            "running": "blue",
            "failed": "red",
            "guarded": "yellow",
            "aborted": "dim",
            "pending": "dim",
        }.get(manifest.status, "white")
        text.append(f"{manifest.status}\n", style=status_style)
        if manifest.git.commit:
            dirty = " (dirty)" if manifest.git.dirty else ""
            text.append(f"Git: {manifest.git.commit[:12]}{dirty}  {manifest.git.branch or ''}\n", style="dim")

    if config.progress:
        num_res = results.get(config.progress.numerator_probe)
        den_res = results.get(config.progress.denominator_probe)
        num = float(num_res.value) if num_res and num_res.value is not None else 0
        den = float(den_res.value) if den_res and den_res.value is not None else 0
        pct = (num / den * 100) if den else 0
        bar = _ascii_bar(pct, width=40, max_val=100)
        text.append(f"\nProgress: {num:.0f}/{den:.0f} ({pct:.1f}%)\n")
        text.append(f"[{bar}]\n")

    for probe in config.probes:
        if probe.id in (config.progress.numerator_probe if config.progress else None,
                        config.progress.denominator_probe if config.progress else None):
            continue
        res = results.get(probe.id)
        if res is None:
            continue
        label = probe.label or probe.id
        text.append(f"{label}: ", style="bold")
        text.append(f"{res.display} {probe.unit}\n", style=res.style or "")

    return Panel(text, title=config.description or "Overview", border_style="cyan")


def _render_metrics_table(config: ExperimentDashboardConfig, results: dict[str, ProbeResult], pane: Pane) -> Panel:
    table = Table(show_header=True, header_style="bold")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    for pid in pane.probes:
        probe = config.get_probe(pid)
        res = results.get(pid)
        if probe is None or res is None:
            continue
        label = probe.label or probe.id
        style = res.style or ""
        table.add_row(label, Text(f"{res.display} {probe.unit}".strip(), style=style))
    return Panel(table, title=pane.title, border_style="blue")


def _render_bars(config: ExperimentDashboardConfig, results: dict[str, ProbeResult], pane: Pane) -> Panel:
    table = Table(show_header=False)
    table.add_column("Label")
    table.add_column("Bar")
    table.add_column("Value", justify="right")
    max_val = pane.config.get("max_value", 1.0)
    for pid in pane.probes:
        probe = config.get_probe(pid)
        res = results.get(pid)
        if probe is None or res is None:
            continue
        try:
            val = float(res.value)
        except (TypeError, ValueError):
            continue
        label = probe.label or probe.id
        bar = _ascii_bar(val, width=25, max_val=max_val)
        table.add_row(label, bar, f"{val:.{probe.precision}f} {probe.unit}".strip())
    return Panel(table, title=pane.title, border_style="magenta")


def _render_recent_log(pane: Pane, run_dir: Path | None, config: ExperimentDashboardConfig) -> Panel:
    log_file = pane.config.get("file") or config.log_file
    path = _resolve_path(log_file, run_dir)
    if not path or not path.exists():
        return Panel("(log file not found)", title=pane.title, border_style="blue")
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        tail = lines[-int(pane.config.get("lines", 8)) :]
        return Panel("\n".join(tail), title=pane.title, border_style="blue")
    except Exception as e:
        return Panel(f"error reading log: {e}", title=pane.title, border_style="red")


def _render_gpu_table(results: dict[str, ProbeResult]) -> Panel:
    res = results.get("gpu_status")
    gpus = res.value if res and isinstance(res.value, list) else []
    if not gpus:
        return Panel(res.display if res else "GPU data unavailable", title="GPU Status", border_style="red")
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
        except (ValueError, TypeError):
            temp_style = "white"
        try:
            util = int(g["util"])
            util_bar = "█" * (util // 10) + "░" * (10 - util // 10)
        except (ValueError, TypeError):
            util_bar = "?"
        table.add_row(
            f"{g['index']}: {g['name'][:30]}",
            Text(f"{g['temp']}°C", style=temp_style),
            f"{g['mem_used']} / {g['mem_total']}",
            f"{util_bar} {g['util']}%",
            g["power"],
        )
    return Panel(table, title="GPU Status", border_style="green")


def _render_run_info(manifest: RunManifest | None, results: dict[str, ProbeResult]) -> Panel:
    if manifest is None:
        return Panel("No run selected", title="Run Info", border_style="yellow")
    text = Text()
    text.append(f"{manifest.run_id}\n", style="bold cyan")
    text.append(f"Stage: {manifest.stage}\n")
    text.append(f"Status: {manifest.status}\n")
    text.append(f"Created: {manifest.created_at[:19]}\n", style="dim")
    text.append(f"Inputs: {len(manifest.inputs)}\n")
    text.append(f"Artifacts: {len(manifest.artifacts)}\n")
    text.append(f"Source archive: {manifest.source.path if manifest.source else 'n/a'}\n", style="dim")
    if manifest.git.commit:
        text.append(f"Commit: {manifest.git.commit[:12]} {'(dirty)' if manifest.git.dirty else ''}\n", style="dim")
    return Panel(text, title="Run Info", border_style="cyan")


def _render_lineage(registry: Registry, manifest: RunManifest | None) -> Panel:
    if manifest is None:
        return Panel("No run selected", title="Lineage", border_style="yellow")
    text = Text()
    parents = registry.parents(manifest.run_id)
    children = registry.children(manifest.run_id)
    text.append("Parents:\n", style="bold")
    for pid, rel in parents:
        text.append(f"  {pid} ({rel})\n", style="dim")
    text.append("Children:\n", style="bold")
    for cid, rel in children:
        text.append(f"  {cid} ({rel})\n", style="dim")
    return Panel(text, title="Lineage", border_style="cyan")


def _render_text(results: dict[str, ProbeResult], pane: Pane, config: ExperimentDashboardConfig) -> Panel:
    text = Text()
    for pid in pane.probes:
        res = results.get(pid)
        if res is None:
            continue
        text.append(f"{pid}: ", style="bold")
        text.append(f"{res.display}\n", style=res.style or "")
    return Panel(text, title=pane.title, border_style="blue")


def _load_experiment_config(manifest: RunManifest | None) -> ExperimentDashboardConfig:
    if manifest is None:
        return generic_config()
    experiment = manifest.spec.get("experiment")
    stage = manifest.stage
    if not experiment:
        return generic_config()
    candidates: list[Path] = []
    if stage:
        candidates.extend([
            Path(f"mlfactory/experiments/{experiment}/dashboard_{stage}.json"),
            Path(f"mlfactory/experiments/{experiment}/dashboard_{stage}.yaml"),
            Path(f"mlfactory/experiments/{experiment}/dashboard_{stage}.yml"),
        ])
    candidates.extend([
        Path(f"mlfactory/experiments/{experiment}/dashboard.json"),
        Path(f"mlfactory/experiments/{experiment}/dashboard.yaml"),
        Path(f"mlfactory/experiments/{experiment}/dashboard.yml"),
    ])
    for c in candidates:
        if c.exists():
            return ExperimentDashboardConfig.load(c)
    return generic_config()


def build_layout(
    registry: Registry,
    config: ExperimentDashboardConfig,
    watch_run: str | None,
    stage: str | None,
    recent_limit: int,
) -> Layout:
    manifest = registry.get(watch_run) if watch_run else None
    if manifest is None:
        latest = registry.find(stage=stage, limit=1)
        if latest:
            manifest = latest[0]
            watch_run = manifest.run_id

    run_dir = Path(manifest.source.path).parent if manifest and manifest.source else None
    results = _run_all_probes(config, run_dir, manifest)

    # Build body rows from panes.
    body = Layout(name="body")

    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        body,
    )

    header = Text()
    header.append("mlfactory dashboard", style="bold cyan")
    header.append(f"  •  registry: {registry.db_path}  •  ")
    if watch_run:
        header.append(f"watching: {watch_run[:50]}  •  ", style="dim")
    header.append("Ctrl-C to quit", style="dim")
    layout["header"].update(Panel(header, border_style="cyan"))

    # Split body into rows, one per pane group? Simpler: one row with all panes.
    # If many panes, split into multiple rows of roughly equal count.
    panes = list(config.panes)
    if not panes:
        panes = [Pane(title="No panes", type="text", probes=[])]

    # Group panes into rows of up to 3.
    rows: list[list[Pane]] = []
    row_size = 3
    for i in range(0, len(panes), row_size):
        rows.append(panes[i : i + row_size])

    body.split_column(*[Layout(name=f"row_{i}") for i in range(len(rows))])
    for i, row_panes in enumerate(rows):
        row_layout = body[f"row_{i}"]
        if len(row_panes) == 1:
            row_layout.update(_render_pane(row_panes[0], config, results, registry, manifest, run_dir))
        else:
            row_layout.split_row(
                *[Layout(_render_pane(p, config, results, registry, manifest, run_dir), ratio=p.ratio) for p in row_panes]
            )

    return layout


def _render_pane(
    pane: Pane,
    config: ExperimentDashboardConfig,
    results: dict[str, ProbeResult],
    registry: Registry,
    manifest: RunManifest | None,
    run_dir: Path | None,
) -> Panel:
    if pane.type == "overview":
        return _render_overview(config, results, manifest)
    if pane.type == "metrics_table":
        return _render_metrics_table(config, results, pane)
    if pane.type == "bars":
        return _render_bars(config, results, pane)
    if pane.type == "recent_log":
        return _render_recent_log(pane, run_dir, config)
    if pane.type == "gpu_table":
        return _render_gpu_table(results)
    if pane.type == "run_info":
        return _render_run_info(manifest, results)
    if pane.type == "lineage":
        return _render_lineage(registry, manifest)
    if pane.type == "text":
        return _render_text(results, pane, config)
    return Panel(f"unknown pane type {pane.type}", title=pane.title, border_style="red")


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="mlfactory read-only dashboard")
    parser.add_argument("--registry", "-r", default=".mlfactory/registry.db")
    parser.add_argument("--stage", default=None)
    parser.add_argument("--watch-run", default=None)
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--refresh", type=float, default=None)
    args = parser.parse_args()

    registry = Registry(args.registry)
    manifest: RunManifest | None = None
    if args.watch_run:
        manifest = registry.get(args.watch_run)
    if manifest is None:
        latest = registry.find(stage=args.stage, limit=1)
        if latest:
            manifest = latest[0]

    config = _load_experiment_config(manifest)
    refresh = args.refresh if args.refresh is not None else config.refresh

    console = Console()
    with Live(
        build_layout(registry, config, args.watch_run or (manifest.run_id if manifest else None), args.stage, args.limit),
        console=console,
        refresh_per_second=1,
        screen=True,
    ) as live:
        try:
            while True:
                live.update(
                    build_layout(
                        registry,
                        config,
                        args.watch_run or (manifest.run_id if manifest else None),
                        args.stage,
                        args.limit,
                    )
                )
                time.sleep(refresh)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
