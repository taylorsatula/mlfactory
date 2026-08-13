---
name: monitoring-dashboard
description: Use when watching a live run, building a dashboard config for a new experiment, or deciding which probes and panes to expose. **At every extended run (batch generation, long training, multi-hour pipelines, overnight jobs) a dashboard must be configured and presented to the user before or immediately after launch.**
---

# Monitoring with the dashboard

## Overview

The dashboard is a read-only Rich Live TUI. It polls probes declared in an experiment's `dashboard.json` and renders panes. It cannot start or stop runs; it only surfaces state.

## When to use

- Watching a training run's loss curve or sample count.
- Building a live view for a new experiment stage.
- You need GPU temperature, disk usage, or process health at a glance.

## Core pattern

Each experiment domain may provide:

- `mlfactory/experiments/<domain>/dashboard.json`
- `mlfactory/experiments/<domain>/dashboard_<stage>.json` (stage-specific)

Launch with:

```python
import subprocess
import sys

subprocess.run([
    sys.executable, "-m", "mlfactory.core.dashboard",
    "--registry", ".mlfactory/registry.db",
    "--watch-run", run_id,
])
```

Or from the shell: `mlfactory dashboard --watch-run <run_id>`.

## Dashboard config structure

```json
{
  "experiment": "dft",
  "description": "DFT training monitor",
  "refresh": 2.0,
  "probes": [
    {"id": "loss", "type": "jsonl_last_record", "file": "{run_dir}/dashboard.jsonl", "path": "loss"}
  ],
  "panes": [
    {"title": "Loss", "type": "bars", "probes": ["loss"], "config": {"max_value": 1.0}}
  ]
}
```

## Wireframe template

Copy this skeleton and adapt to your run. Most extended runs need progress tracking, recent log tail, and GPU status at minimum.

```json
{
  "experiment": "<domain>",
  "description": "<what this dashboard monitors>",
  "refresh": 5.0,
  "log_file": "<path-to-log-file>",
  "probes": [
    {
      "id": "completed",
      "type": "file_line_count",
      "file": "{run_dir}/outputs.jsonl",
      "label": "Completed"
    },
    {
      "id": "process",
      "type": "shell_command",
      "command": "pgrep -f <your-script-name> | head -1",
      "parser": "int",
      "label": "PID"
    },
    {
      "id": "last_metric",
      "type": "regex_last_match",
      "file": "{run_dir}/run.log",
      "regex": "metric=([\\d.]+)",
      "label": "Latest"
    },
    {
      "id": "gpu_status",
      "type": "gpu_status"
    },
    {
      "id": "const_total",
      "type": "const",
      "value": 100
    }
  ],
  "progress": {
    "numerator_probe": "completed",
    "denominator_probe": "const_total"
  },
  "panes": [
    {
      "title": "Progress",
      "type": "overview",
      "probes": ["completed", "process", "last_metric"],
      "ratio": 1
    },
    {
      "title": "Recent Log",
      "type": "recent_log",
      "config": {"lines": 12},
      "ratio": 1
    },
    {
      "title": "GPU",
      "type": "gpu_table",
      "ratio": 1
    }
  ]
}
```

**Standalone launcher** (for runs not tied to the registry):

```python
#!/usr/bin/env python3
from pathlib import Path
from rich.console import Console
from rich.live import Live
from mlfactory.core.dashboard import (
    _run_all_probes, _render_overview, _render_recent_log, _render_gpu_table
)
from mlfactory.core.dashboard_config import ExperimentDashboardConfig

def build_layout(config):
    results = _run_all_probes(config, Path("."), None)
    # Build layout with header + panes
    # See dashboard_corpus.py for full example
    pass

config = ExperimentDashboardConfig.load(Path("dashboard.json"))
with Live(build_layout(config), refresh_per_second=1, screen=True) as live:
    while True:
        live.update(build_layout(config))
        import time; time.sleep(config.refresh)
```

## Useful probe types

| Type | Measures |
|------|----------|
| `file_line_count` | Number of lines in a file (e.g., generated samples). |
| `jsonl_last_record` | Last JSON row in `dashboard.jsonl`. |
| `regex_last_match` | Last match of a regex in a log file. |
| `process_alive` | Whether a PID is still running. |
| `http_status` / `http_json` | Health of a local HTTP service. |
| `gpu_status` | GPU temp, memory, utilization. |
| `disk_usage` | Free disk space. |
| `file_exists` | Presence of a marker file. |

## Common mistakes

- **Trying to control runs from the dashboard.** It is read-only; use `mlfactory run` or plugin code to start/stop.
- **Forgetting `{run_dir}` expansion.** Use it in probe `file` paths so the dashboard finds the active run's files.
- **Writing heavy shell probes.** Probes run every refresh; keep them cheap.
- **Not providing a stage-specific config.** Name it `dashboard_<stage>.json` when the generic config is not enough.
