---
name: logging-mlfactory-metrics
description: Use when an experiment produces per-step or per-event measurements that should appear in the live dashboard and persist in the registry.
---

# Logging mlfactory metrics

## Overview

`MetricsLogger` writes append-only `dashboard.jsonl` in the run directory, inserts into the SQLite registry metrics table, and optionally echoes to stdout. Use it in plugins and experiment scripts so the dashboard can surface progress without parsing arbitrary log files.

## When to use

- Training loops emitting loss, reward, KL, or eval scores.
- Long-running generation jobs tracking sample counts.
- Events like "checkpoint saved" or "guard triggered".
- You are writing raw `print()` statements for metrics.

## Core pattern

```python
from mlfactory.core.metrics import MetricsLogger

metrics = MetricsLogger(run_dir, run_id=manifest.run_id, registry=registry)

for step, batch in enumerate(batches):
    loss = train(batch)
    metrics.step(step, loss=loss, lr=scheduler.get_last_lr()[0])

metrics.event("checkpoint_saved", {"step": step, "path": str(ckpt_dir)})
```

## Quick reference

| Method | Use for |
|--------|---------|
| `log(key, value, step=...)` | One scalar metric. |
| `step(step, **metrics)` | A bundle of metrics at the same step; writes one combined dashboard row plus individual rows. |
| `event(name, detail={})` | Discrete events (checkpoints, guards, phase transitions). |

## Registry integration

Pass `registry=Registry(...)` to also store metrics in `.mlfactory/registry.db`. This lets `mlfactory show` and the dashboard query historical series.

## Common mistakes

- **Writing metrics only to stdout.** stdout is lost; `dashboard.jsonl` and the registry are the durable sources.
- **Inventing ad-hoc JSON files.** Use `MetricsLogger` so probes in `dashboard.json` can read the data consistently.
- **Forgetting `step` on scalar logs.** Steps make time-series queries meaningful.
