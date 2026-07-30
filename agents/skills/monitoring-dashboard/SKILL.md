---
name: monitoring-dashboard
description: Use when watching a live run, building a dashboard config for a new experiment, or deciding which probes and panes to expose.
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
