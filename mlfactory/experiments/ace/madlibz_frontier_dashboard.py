#!/usr/bin/env python3
"""Screen-friendly live dashboard for the detached Madlibz frontier authoring run."""
from __future__ import annotations

import time
from pathlib import Path

from rich.console import Console
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.text import Text

from mlfactory.core.dashboard import (
    _render_metrics_table,
    _render_overview,
    _render_recent_log,
    _run_all_probes,
)
from mlfactory.core.dashboard_config import ExperimentDashboardConfig

HERE = Path(__file__).resolve().parent
CONFIG_PATH = HERE / "madlibz_frontier_dashboard.json"


def build_layout(config: ExperimentDashboardConfig) -> Layout:
    results = _run_all_probes(config, HERE.parent.parent.parent, None)
    layout = Layout()
    layout.split_column(Layout(name="header", size=3), Layout(name="body"))

    header = Text()
    header.append("madlibz verifiable frontier • live authoring", style="bold cyan")
    header.append("  •  remote kimi-k3-flex  •  Ctrl-C to quit", style="dim")
    layout["header"].update(Panel(header, border_style="cyan"))

    overview = next(p for p in config.panes if p.type == "overview")
    recent = next(p for p in config.panes if p.type == "recent_log")
    metrics = next(p for p in config.panes if p.type == "metrics_table")
    body = layout["body"]
    body.split_column(Layout(name="top"), Layout(name="bottom", size=12))
    body["top"].split_row(
        Layout(_render_overview(config, results, None, overview), ratio=2),
        Layout(_render_recent_log(recent, HERE.parent.parent.parent, config), ratio=3),
    )
    body["bottom"].update(_render_metrics_table(config, results, metrics))
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
