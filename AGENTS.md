# AGENTS.md — mlfactory

> Agent instructions for working in the mlfactory codebase.

## What is mlfactory

A reproducibility-first experiment factory for ML research. Every invocation of `mlfactory run <spec>` produces a versioned run directory with a source snapshot, hashed inputs/outputs, frozen environment, hardware provenance, and a SQLite registry entry. Experiments live in self-contained domains under `mlfactory/experiments/<name>/`.

## Project layout

```
mlfactory/                      # reusable harness (DO NOT modify experiment internals)
  core/
    manifest.py                 # RunManifest schema, git_info, archive_source, sha256_file
    registry.py                 # SQLite API: register, find, lineage, metrics, merge
    runner.py                   # spec -> manifest -> plugin execution (create_run, run_from_spec)
    model_server.py             # disposable llama-server context manager: model("qwen3.5:4b")
    metrics.py                  # MetricsLogger (dashboard.jsonl + registry)
    env.py                      # training_env(), inference_env() context managers
    api.py                      # APIClient, Judge, extract_json
    artifacts.py                # save_checkpoint, save_summary, save_config
    dashboard.py                # Rich Live TUI
    dashboard_config.py         # Probe/pane schema
    embeddings.py               # embedder() helper
    prompts.py                  # Prompt loading utilities
    secrets.py                  # ${secrets.KEY} expansion
  plugins/
    base.py                     # StagePlugin ABC, PluginRegistry, PLUGINS singleton
  remote/
    ssh_runner.py               # Generic SSH remote runner
    vast.py                     # Vast.ai provision/run/pull wrapper
  experiments/                  # experiment domains (each is self-contained)
    sample/                     # ★ REFERENCE: transform→classify with train/eval branches
    ace/                        # ACE trajectory collection/classification/stratification
    dft/                        # DFT training/evaluation/build-pilot
    voice/                      # Voice review corpus analysis

agents/skills/                  # pi agent skill docs (one per topic)
runs/                           # per-run output directories (gitignored)
.mlfactory/                     # registry.db and factory state (gitignored)
models.yaml                     # local GGUF alias registry
pyproject.toml
```

## Start here: the sample experiment

`mlfactory/experiments/sample/` is a **full-featured 4-stage pipeline** that demonstrates every pattern:

| Stage | What it does | Patterns demonstrated |
|---|---|---|
| `transform` | Chunk text, compute statistics | Plugin lifecycle, MetricsLogger, save_summary, artifact hashing |
| `classify` | Topic classification via LLM | Model server (`model()`), APIClient, `inference_env()`, multi-stage lineage |
| `train` | Supervised fine-tuning | NumPy smoke backend, optional Hugging Face full/LoRA backend, checkpoints, loss metrics |
| `eval` | Judge classification quality | APIClient, `Judge.compare()`, **guard logic**, save_config |

Read `mlfactory/experiments/sample/README.md` for the full walkthrough.

## How to run an experiment

```python
from pathlib import Path
from mlfactory.core.registry import Registry
from mlfactory.core.runner import create_run, run_from_spec

registry = Registry(".mlfactory/registry.db")

# Dry-run: create directory and manifest without executing.
manifest = create_run(Path("mlfactory/experiments/sample/specs/sample_transform.yaml"))

# Full run.
manifest = run_from_spec(
    spec_path=Path("mlfactory/experiments/sample/specs/sample_transform.yaml"),
    registry=registry,
)
```

CLI equivalent:

```bash
mlfactory run mlfactory/experiments/sample/specs/sample_transform.yaml
mlfactory ls
mlfactory show <run_id>
mlfactory dashboard --watch-run <run_id>
```

## How to create a new experiment

1. **Read the sample experiment first**: `mlfactory/experiments/sample/`
2. **Pick a stage** from `STAGES` in `mlfactory/core/manifest.py` or add a new one
3. **Create the directory**: `mlfactory/experiments/<domain>/`
4. **Write these files**:
   - `__init__.py` (empty)
   - `<stage>.py` — domain logic (the actual computation, mlfactory-agnostic)
   - `<stage>_plugin.py` — `StagePlugin` subclass that wraps the domain logic
   - `specs/<name>.yaml` — run spec with `stage`, `name`, `experiment`, and stage-specific fields
   - `dashboard.json` — probe/pane config for the live TUI
   - `README.md` — what this experiment does, how to run it
5. **Register the plugin** by adding an import in `mlfactory/core/runner.py`
6. **Test** with `create_run()` then `run_from_spec()`

## Multi-stage pipelines with lineage

Experiments often chain stages where one stage consumes another's output. The pattern:

```python
# Stage 1: produce artifacts
m1 = run_from_spec(Path("experiments/myexp/specs/stage1.yaml"), registry=registry)

# Stage 2: consume stage 1's artifacts via input_run in the spec
# The spec has: input_run: <stage1-run-id>
# The plugin resolves it via Registry.get(input_run) → finds artifacts/chunks.jsonl
m2 = run_from_spec(
    Path("experiments/myexp/specs/stage2.yaml"),
    registry=registry,
    parent_runs=[m1.run_id],  # establishes lineage in the registry
)
```

The sample experiment demonstrates this: transform → classify, followed by sibling train and eval stages.

## Plugin contract

Every plugin extends `StagePlugin` and implements three methods:

```python
from mlfactory.plugins.base import PLUGINS, StagePlugin

class MyPlugin(StagePlugin):
    stage = "transform"  # must match manifest.STAGES

    def prepare(self) -> None:
        """Validate inputs, create subdirs, start services."""

    def execute(self) -> None:
        """Run the workload. May raise on failure."""

    def finalize(self) -> None:
        """Hash artifacts, write summary, stop services. ALWAYS called."""

PLUGINS.register(MyPlugin)
```

Key rules:
- All outputs go under `self.run_dir` (artifacts in `artifacts/`, logs in `logs/`)
- Hash every artifact with `sha256_file()` and add as `FileRecord`
- Call `self.manifest.write(self.run_dir / "manifest.json")` in `finalize()`
- Use `MetricsLogger` for any per-step measurements
- Use `save_summary()` / `save_config()` from `core.artifacts`
- Use `inference_env()` / `training_env()` for environment management
- Set `self.manifest.status = "guarded"` for quality gates

## Core utilities cheat sheet

| Need | Import | Usage |
|---|---|---|
| Model server | `from mlfactory.core.model_server import model` | `with model("qwen3.5:4b", gpu=0) as srv:` |
| API client | `from mlfactory.core.api import APIClient, APIConfig` | `client.chat_completion(messages=[...])` |
| Judge | `from mlfactory.core.api import Judge` | `judge.compare(prompt, a, b)` |
| JSON extraction | `from mlfactory.core.api import extract_json` | `extract_json(model_output)` |
| Metrics | `from mlfactory.core.metrics import MetricsLogger` | `m.step(step, loss=loss)` / `m.event("done")` |
| Env guards | `from mlfactory.core.env import training_env, inference_env` | `with training_env(hf_home=...):` |
| Embeddings | `from mlfactory.core.embeddings import embedder` | `emb.encode(texts, normalize=True)` |
| File hashing | `from mlfactory.core.manifest import sha256_file` | `sha256_file(path)` |
| Checkpoints | `from mlfactory.core.artifacts import save_checkpoint` | `save_checkpoint(run_dir, step, model, tok, manifest=m)` |
| Summary | `from mlfactory.core.artifacts import save_summary` | `save_summary(run_dir, data, manifest=m)` |
| Config | `from mlfactory.core.artifacts import save_config` | `save_config(run_dir, config_dict)` |

## Spec format

YAML files with at minimum `stage`, `name`, and `experiment`:

```yaml
stage: transform
name: my-analysis
experiment: mydomain
# ... stage-specific fields ...
```

The runner resolves file-path values in the spec as inputs (hashes them automatically). For multi-stage pipelines, use `input_run: <run_id>` and resolve in the plugin via `Registry.get()`.

## Guard logic

Experiments can gate quality by setting the manifest status to `"guarded"`:

```python
if quality_score < threshold:
    self.manifest.status = "guarded"
    self.manifest.guard_report = {
        "reason": f"accuracy {quality_score:.4f} below threshold {threshold}",
        "threshold": threshold,
        "accuracy": quality_score,
    }
```

The sample `eval` plugin demonstrates this pattern.

## Dashboard config

Each experiment provides `dashboard.json` (or `dashboard_<stage>.json`). Probe types: `file_line_count`, `regex_count`, `regex_last_match`, `jsonl_last_record`, `process_alive`, `http_status`, `http_json`, `gpu_status`, `disk_usage`, `shell_command`, `spec_value`, `file_exists`, `const`.

## Remote execution (Vast.ai)

```python
from mlfactory.remote.vast import VastRunner

runner = VastRunner.from_search(api_key=os.environ["VAST_API_KEY"])
runner.provision()
runner.setup(setup_script="mlfactory/experiments/<domain>/setup_h100.sh")
run_id = runner.run_spec("mlfactory/experiments/<domain>/specs/<spec>.yaml")
runner.pull_outputs(run_id)
runner.pull_registry(".mlfactory/registry-remote.db")
runner.stop()
```

Merge remote registry: `mlfactory registry merge .mlfactory/registry-remote.db`

## Secrets

Store API keys in `.mlfactory/secrets.yaml` and reference them in specs:

```yaml
api_key: ${secrets.OPENROUTER_API_KEY}
```

The runner expands secrets at execution time; they are never written into manifests.

## Provider preferences

- **Lunaroute:** always prefer the `-ballast` model variant when one is available (e.g., `glm-5.2-vision-ballast`). Fall back to the plain model only if no ballast variant exists.

## Common mistakes

- **Running experiment scripts directly** — always go through `run_from_spec()`
- **Writing outputs outside `self.run_dir`** — artifacts and logs belong under the run dir
- **Skipping artifact hashing** — every output file needs a `FileRecord` with SHA-256
- **Not registering the plugin** — add the import in `mlfactory/core/runner.py`
- **Hardcoding model paths** — use `models.yaml` aliases and the `model()` context manager
- **Forgetting `self.manifest.write()`** in `finalize()`
- **Running code outside a git repo** — `git archive` needs a committed tree
- **Using `Registry()` default path in plugins** — use `Registry(".mlfactory/registry.db")` to match the CLI

## Testing

```bash
python3.14 -m pytest tests/
```

## Install

```bash
python3.14 -m pip install -e . --break-system-packages
# or with remote deps:
python3.14 -m pip install -e ".[remote]" --break-system-packages
```
