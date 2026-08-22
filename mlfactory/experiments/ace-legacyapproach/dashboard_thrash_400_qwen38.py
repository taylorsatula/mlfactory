#!/usr/bin/env python3
"""Live Rich dashboard for the ACE Qwen3.8 trace-capture worker.

Run from any directory:
    python3 mlfactory/experiments/ace/dashboard_thrash_400_qwen38.py
"""
from __future__ import annotations

import json
import os
import re
import time
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path

import requests

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

from mlfactory.core.dashboard import (
    _render_gpu_table,
    _render_metrics_table,
    _render_overview,
    _render_recent_log,
    _run_all_probes,
    ProbeResult,
)
from mlfactory.core.dashboard_config import ExperimentDashboardConfig

ACE_DIR = Path(__file__).parent
CONFIG_PATH = ACE_DIR / "dashboard_thrash_400_qwen38.json"
METRICS_URL = "http://127.0.0.1:3090/metrics"
_progress_samples: deque[tuple[float, float]] = deque(maxlen=120)


def _api_key() -> str | None:
    key = os.environ.get("LLAMA_API_KEY")
    if key:
        return key
    key_file = Path("/etc/llama-server/api-keys")
    try:
        for line in key_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                return line
    except OSError:
        return None
    return None


def _server_tokens_per_second_1m() -> ProbeResult:
    """Return a rolling one-minute rate from llama.cpp Prometheus counters."""
    try:
        headers = {}
        key = _api_key()
        if key:
            headers["Authorization"] = f"Bearer {key}"
        response = requests.get(METRICS_URL, headers=headers, timeout=2)
        response.raise_for_status()
        match = re.search(
            r"^llamacpp:tokens_predicted_total\s+([0-9.eE+-]+)$",
            response.text,
            re.MULTILINE,
        )
        if not match:
            return ProbeResult(value=None, display="n/a", style="dim")
        now = time.monotonic()
        total = float(match.group(1))
        _progress_samples.append((now, total))
        cutoff = now - 60.0
        baseline = next((sample for sample in _progress_samples if sample[0] <= cutoff), _progress_samples[0])
        elapsed = now - baseline[0]
        if elapsed < 2.0:
            return ProbeResult(value=None, display="warming", style="dim")
        rate = max(0.0, (total - baseline[1]) / elapsed)
        return ProbeResult(value=rate, display=f"{rate:.1f} tok/s", style="green")
    except Exception as exc:
        return ProbeResult(value=None, display=f"unavailable ({type(exc).__name__})", style="red")


def _eta_probe() -> ProbeResult:
    progress_path = ACE_DIR / "data" / "thrash_400_qwen38_progress.json"
    log_path = ACE_DIR / "data" / "thrash_400_qwen38.log"
    try:
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        completed = int(progress.get("completed", 0))
        total = int(progress.get("total", 0))
        remaining = max(0, total - completed)
        if remaining == 0:
            return ProbeResult(value=0.0, display="complete", style="green")

        # Read only the tail: the log is append-only and may become large.
        with log_path.open("rb") as stream:
            stream.seek(0, 2)
            stream.seek(max(0, stream.tell() - 65536))
            tail = stream.read().decode("utf-8", "ignore")
        samples = [float(value) for value in re.findall(r"TRACE OK .*?elapsed=([0-9.]+)s", tail)]
        if samples:
            seconds_per_item = sum(samples[-20:]) / len(samples[-20:])
        else:
            seconds_per_item = float(progress.get("estimated_seconds_per_prompt", 0) or 0)
        if seconds_per_item <= 0:
            return ProbeResult(value=None, display="estimating", style="dim")
        eta_seconds = remaining * seconds_per_item
        finish = datetime.now() + timedelta(seconds=eta_seconds)
        hours, remainder = divmod(int(eta_seconds), 3600)
        minutes = remainder // 60
        duration = f"{hours}h {minutes:02d}m" if hours else f"{minutes}m"
        return ProbeResult(
            value=eta_seconds,
            display=f"{finish:%Y-%m-%d %H:%M} ({duration})",
            style="yellow",
        )
    except Exception as exc:
        return ProbeResult(value=None, display=f"unavailable ({type(exc).__name__})", style="red")


def build_layout(config: ExperimentDashboardConfig) -> Layout:
    results = _run_all_probes(config, ACE_DIR, None)
    results["eta"] = _eta_probe()
    results["tokens_per_second_1m"] = _server_tokens_per_second_1m()
    layout = Layout()
    layout.split_column(Layout(name="header", size=3), Layout(name="body"))

    header = Text()
    header.append("ACE thrash-400 Qwen3.8 trace capture", style="bold cyan")
    header.append("  •  Ctrl-C to quit", style="dim")
    layout["header"].update(Panel(header, border_style="cyan"))

    rows: list[list] = []
    pending: list = []
    for pane in config.panes:
        if pane.type == "gpu_table":
            if pending:
                rows.append(pending)
                pending = []
            rows.append([pane])
        else:
            pending.append(pane)
            if len(pending) == 2:
                rows.append(pending)
                pending = []
    if pending:
        rows.append(pending)

    body = layout["body"]
    body.split_column(*[Layout(name=f"row_{i}") for i in range(len(rows))])
    for i, pane_row in enumerate(rows):
        row_layout = body[f"row_{i}"]
        rendered = []
        for pane in pane_row:
            if pane.type == "overview":
                rendered.append(_render_overview(config, results, None, pane))
            elif pane.type == "metrics_table":
                rendered.append(_render_metrics_table(config, results, pane))
            elif pane.type == "recent_log":
                rendered.append(_render_recent_log(pane, ACE_DIR, config))
            elif pane.type == "gpu_table":
                rendered.append(_render_gpu_table(results))
        if len(rendered) == 1:
            row_layout.update(rendered[0])
        else:
            row_layout.split_row(*rendered)
    return layout


def main() -> None:
    config = ExperimentDashboardConfig.load(CONFIG_PATH)
    console = Console()
    with Live(build_layout(config), console=console, refresh_per_second=1, screen=True) as live:
        try:
            while True:
                live.update(build_layout(config))
                time.sleep(config.refresh)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
