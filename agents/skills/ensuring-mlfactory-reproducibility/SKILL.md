---
name: ensuring-mlfactory-reproducibility
description: Use when declaring a run complete, comparing runs, sharing results, or migrating legacy data into the factory without losing provenance.
---

# Ensuring mlfactory reproducibility

## Overview

Every run gets a `manifest.json` that is the source of truth for provenance. It records git state, a byte-stable source archive, input hashes, environment freeze, hardware snapshot, artifact hashes, and parent/child lineage. Treat the manifest as the contract for reproducibility.

## When to use

- Before claiming a result is final.
- When you need to compare two runs.
- Ingesting a legacy run into the registry.
- You are tempted to move/delete run directories by hand.

## What the manifest contains

| Field | Source |
|-------|--------|
| `git.commit` / `dirty` / `branch` | `git rev-parse HEAD`, `git status --porcelain` |
| `source` | `git archive --format=tar.gz HEAD` |
| `inputs` | Hashed files referenced by the spec |
| `env` | `pip freeze`, platform, selected env vars |
| `hardware` | `nvidia-smi`, `/proc/meminfo` |
| `artifacts` | Hashed output files written by the plugin |
| `parent_runs` / `child_runs` | Lineage links in the registry |

## Core pattern

```python
from pathlib import Path
from mlfactory.core.manifest import RunManifest
from mlfactory.core.runner import create_run
from mlfactory.core.registry import Registry

# Create a run without executing and inspect its manifest.
manifest = create_run(Path("mlfactory/experiments/ace/specs/ace_collect_qwen35.yaml"))
print(manifest.model_dump_json(indent=2))

# Ingest a legacy run while preserving the original data.
m = RunManifest.read(Path("/path/to/old_run/manifest.json"))
Registry(".mlfactory/registry.db").ingest_manifest(m, parents=[parent_run_id])
```

## Non-destructive migration

Never delete or overwrite original experiment data when migrating. The registry can reference a manifest that points to files outside `runs/`. Use `mlfactory ingest` or `Registry.ingest_manifest()`.

## Common mistakes

- **Running code outside a git repo.** `git archive` needs a clean commit tree; commit work-in-progress before running.
- **Ignoring `git.dirty`.** A dirty commit in the manifest warns that the source archive may not match the working tree.
- **Moving artifact files after hashing.** The manifest hashes are invalid if files are moved or edited.
- **Deleting run directories to save space.** Keep at least the manifest and key artifacts; move large artifacts to cold storage if needed, but update the manifest paths.
