#!/usr/bin/env python3
"""Standalone dashboard for corpus generation (not tied to registry)."""
import time
from pathlib import Path
from rich.console import Console
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.text import Text

from mlfactory.core.dashboard import (
    _run_all_probes,
    _render_overview,
    _render_recent_log,
    _render_gpu_table,
    ProbeResult,
)
from mlfactory.core.dashboard_config import ExperimentDashboardConfig

ACE_DIR = Path(__file__).parent


def build_layout(config: ExperimentDashboardConfig) -> Layout:
    results = _run_all_probes(config, ACE_DIR, None)

    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="body"),
    )

    header = Text()
    header.append("corpus generation dashboard", style="bold cyan")
    header.append("  •  Ctrl-C to quit", style="dim")
    layout["header"].update(Panel(header, border_style="cyan"))

    # Split body into rows
    panes = list(config.panes)
    rows: list[list] = []
    pending: list = []
    for pane in panes:
        if pane.type == "gpu_table":
            if pending:
                rows.append(pending)
                pending = []
            rows.append([pane])
            continue
        pending.append(pane)
        if len(pending) == 2:
            rows.append(pending)
            pending = []
    if pending:
        rows.append(pending)

    body = layout["body"]
    body.split_column(*[Layout(name=f"row_{i}") for i in range(len(rows))])

    for i, row_panes in enumerate(rows):
        row_layout = body[f"row_{i}"]
        if len(row_panes) == 1:
            p = row_panes[0]
            if p.type == "overview":
                row_layout.update(_render_overview(config, results, None, p))
            elif p.type == "recent_log":
                row_layout.update(_render_recent_log(p, ACE_DIR, config))
            elif p.type == "gpu_table":
                row_layout.update(_render_gpu_table(results))
        else:
            panels = []
            for p in row_panes:
                if p.type == "overview":
                    panels.append(_render_overview(config, results, None, p))
                elif p.type == "recent_log":
                    panels.append(_render_recent_log(p, ACE_DIR, config))
            row_layout.split_row(*panels)

    return layout


def main():
    config = ExperimentDashboardConfig.load(ACE_DIR / "dashboard_corpus.json")
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
