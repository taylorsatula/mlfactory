---
name: operating-locally
description: Use when running or debugging experiments on the local host (192.168.1.9) via the Python API or CLI.
---

# Operating locally

## Overview

The preferred way to operate the factory is through the Python API in `mlfactory.core.runner` and `mlfactory.core.registry`. Specs drive plugins; plugins create run directories and update the manifest. Use the CLI only when a human explicitly asks for it. Avoid calling experiment scripts directly.

## When to use

- Running ACE collect, DFT train, or any future stage programmatically.
- Inspecting past runs or lineage from code.
- Launching the dashboard as a subprocess for a human.
- You are tempted to run `python mlfactory/experiments/.../train_dft.py` by hand.

## Core pattern (Python API)

```python
from pathlib import Path
from mlfactory.core.registry import Registry
from mlfactory.core.runner import create_run, run_from_spec

registry = Registry(".mlfactory/registry.db")

# Create a run directory without executing.
manifest = create_run(Path("mlfactory/experiments/ace/specs/ace_collect_qwen35.yaml"))

# Execute it.
manifest = run_from_spec(
    spec_path=Path("mlfactory/experiments/ace/specs/ace_collect_qwen35.yaml"),
    registry=registry,
)

# Inspect.
print(registry.get(manifest.run_id).model_dump_json(indent=2))
print(registry.parents(manifest.run_id))
print(registry.children(manifest.run_id))
```

## Quick reference

| Goal | Python API | CLI fallback |
|------|------------|--------------|
| List runs | `registry.find(stage=..., status=..., limit=...)` | `mlfactory ls` |
| Create run dir | `create_run(spec_path)` | `mlfactory init <spec>` |
| Run a spec | `run_from_spec(spec_path, registry=registry)` | `mlfactory run <spec>` |
| Show manifest | `registry.get(run_id)` | `mlfactory show <run_id>` |
| Lineage | `registry.parents/children(run_id)` | `mlfactory lineage <run_id>` |
| Ingest legacy | `registry.ingest_manifest(manifest, parents)` | `mlfactory ingest <manifest>` |
| Dashboard | `subprocess.run([sys.executable, "-m", "mlfactory.core.dashboard", ...])` | `mlfactory dashboard` |

## Specs and stages

Specs live in `mlfactory/experiments/<domain>/specs/`. A spec must contain `stage` and `experiment`. Known stages are registered by plugins in `mlfactory/experiments/*/`*`_plugin.py`.

## Common mistakes

- **Running experiment scripts directly.** Always go through `run_from_spec` so the manifest, source snapshot, and registry are written.
- **Forgetting `registry=registry`.** Without it, lineage and metrics are not persisted.
- **Expecting the dashboard to control runs.** It is read-only. Start/stop runs from code.
- **Auto-generating run ids when stability matters.** Pass `run_id=` when another run or a human will reference it.
