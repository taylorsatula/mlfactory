---
name: writing-plugins
description: Use when adding a new experiment stage, porting a legacy script into the factory, or writing a plugin for collect/classify/train/eval.
---

# Writing plugins

## Overview

A plugin is a `StagePlugin` subclass registered in `PLUGINS`. It receives a `RunManifest`, validates inputs in `prepare()`, runs the workload in `execute()`, and records artifacts in `finalize()`. The runner handles manifest creation and registry updates; the plugin focuses on the experiment logic.

## When to use

- Porting `collect.py`, `classify.py`, `train_dft.py`, or `eval.py`.
- Adding a new stage such as `transform` or `analyze`.
- You need a reusable wrapper around a domain-specific script.

## Core pattern

```python
from mlfactory.core.manifest import FileRecord, sha256_file
from mlfactory.plugins.base import PLUGINS, StagePlugin

class AnalyzePlugin(StagePlugin):
    stage = "analyze"

    def prepare(self) -> None:
        # Validate inputs, create subdirs.
        pass

    def execute(self) -> None:
        # Run the workload.
        pass

    def finalize(self) -> None:
        # Hash artifacts and update manifest.summary.
        for p in (self.run_dir / "artifacts").glob("*.json"):
            self.manifest.artifacts.append(
                FileRecord(
                    path=str(p.resolve()),
                    sha256=sha256_file(p),
                    role=f"artifact:{p.name}",
                    size_bytes=p.stat().st_size,
                )
            )
        self.manifest.write(self.run_dir / "manifest.json")

PLUGINS.register(AnalyzePlugin)
```

## Spec structure

```yaml
stage: analyze
name: my-analysis
experiment: mydomain
input_run: <run_id>
# stage-specific fields...
```

## Quick reference

| Hook | Purpose |
|------|---------|
| `prepare()` | Validate inputs, start services, create dirs. |
| `execute()` | Run the actual workload; may raise on failure. |
| `finalize()` | Hash artifacts, write summary, stop services. Always called. |
| `self.manifest` | Read/write run metadata; persists to `manifest.json`. |
| `self.run_dir` | `runs/<run_id>` directory. |

## Common mistakes

- **Not hashing artifacts.** Add every output file as a `FileRecord` so the manifest can verify integrity.
- **Writing outputs outside `self.run_dir`.** Keep artifacts under `self.run_dir/artifacts` and logs under `self.run_dir/logs`.
- **Forgetting `self.manifest.write(...)`.** `finalize()` must persist the updated manifest.
- **Throwing away the traceback.** The runner catches exceptions and marks the run `failed`, but include the error in `self.manifest.summary`.
