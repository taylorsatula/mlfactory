---
name: building-an-mlfactory-experiment
description: Use when an agent is asked to add a new experiment domain, port a standalone training/evaluation script, or wire a new stage into the factory from scratch.
---

# Building an mlfactory experiment

## Overview

An experiment domain lives under `mlfactory/experiments/<name>/` and contains data, specs, processing code, a stage plugin, and an optional dashboard config. The factory driver (`mlfactory.core.runner.run_from_spec`) creates the run directory, source snapshot, and manifest; the plugin runs the domain-specific workload. Build experiments so other agents can run them without reading the internals.

## When to use

- Porting a standalone script (ACE collect, DFT train, a new eval harness) into the factory.
- Adding a new stage such as `transform`, `analyze`, or `synthesize`.
- A human asks for an experiment that can run locally and on Vast.ai.

## Design checklist

1. **Pick a stage.** Existing stages: `collect`, `classify`, `train`, `eval`, `transform`, `analyze`.
2. **Create the directory layout:**
   ```
   mlfactory/experiments/<domain>/
     __init__.py
     <stage>.py
     <stage>_plugin.py
     dashboard.json
     data/
     specs/<spec>.yaml
   ```
3. **Write a spec** that captures every tunable.
4. **Write the plugin** with `prepare()`, `execute()`, `finalize()`.
5. **Use core utilities** for inference, metrics, env guards, embeddings, checkpoints.
6. **Add a dashboard config** for long runs.
7. **Test with `create_run` and `run_from_spec`.**

## Core pattern: the plugin

```python
from pathlib import Path
from mlfactory.core.env import training_env
from mlfactory.core.manifest import FileRecord, sha256_file
from mlfactory.core.metrics import MetricsLogger
from mlfactory.plugins.base import PLUGINS, StagePlugin

class MyStagePlugin(StagePlugin):
    stage = "my_stage"

    def prepare(self) -> None:
        (self.run_dir / "artifacts").mkdir(parents=True, exist_ok=True)
        (self.run_dir / "logs").mkdir(parents=True, exist_ok=True)

    def execute(self) -> None:
        spec = self.manifest.spec
        metrics = MetricsLogger(self.run_dir, run_id=self.manifest.run_id)

        with training_env(hf_home=spec.get("env", {}).get("HF_HOME")):
            # call domain script or run workload directly
            pass

        metrics.event("done")

    def finalize(self) -> None:
        for p in (self.run_dir / "artifacts").rglob("*"):
            if p.is_file():
                self.manifest.artifacts.append(
                    FileRecord(path=str(p.resolve()), sha256=sha256_file(p),
                               role=f"artifact:{p.relative_to(self.run_dir/'artifacts')}",
                               size_bytes=p.stat().st_size)
                )
        self.manifest.write(self.run_dir / "manifest.json")

PLUGINS.register(MyStagePlugin)
```

## Core utilities to reuse

| Need | Use |
|------|-----|
| Model server | `mlfactory.core.model_server.model(alias, gpu=...)` |
| API / judge | `mlfactory.core.api.APIClient`, `Judge` |
| Embeddings | `mlfactory.core.embeddings.embedder(name, device=...)` |
| Metrics | `mlfactory.core.metrics.MetricsLogger` |
| Env guards | `mlfactory.core.env.training_env()`, `inference_env()` |
| Checkpoints / summary | `mlfactory.core.artifacts.save_checkpoint()`, `save_summary()` |

## Dashboard config

Create `mlfactory/experiments/<domain>/dashboard.json` with probes and panes. The dashboard loads it by `experiment` and `stage`. See `mlfactory/experiments/ace/dashboard.json` and `mlfactory/experiments/dft/dashboard.json`.

## Testing the experiment

```python
from pathlib import Path
from mlfactory.core.runner import create_run, run_from_spec
from mlfactory.core.registry import Registry

spec = Path("mlfactory/experiments/<domain>/specs/<spec>.yaml")
registry = Registry(".mlfactory/registry.db")

# Dry-run: create directory and manifest without executing.
manifest = create_run(spec)

# Full run.
manifest = run_from_spec(spec, registry=registry)
```

For heavy compute, run the same spec on Vast from Python:

```python
from mlfactory.remote.vast import VastRunner

runner = VastRunner.from_search(api_key=os.environ["VAST_API_KEY"])
runner.provision()
runner.setup(setup_script="mlfactory/experiments/<domain>/setup_h100.sh")
run_id = runner.run_spec("mlfactory/experiments/<domain>/specs/<spec>.yaml")
runner.pull_outputs(run_id)
runner.stop()
```

## Common mistakes

- **Calling the domain script directly.** Route through `run_from_spec` so the manifest and registry are populated.
- **Skipping artifact hashing.** The manifest must list every output file with a SHA-256.
- **Writing outputs outside `self.run_dir`.** All artifacts and logs belong under the run directory.
- **Not registering the plugin.** Import the plugin module from `mlfactory.core.runner` or the stage will not resolve.
