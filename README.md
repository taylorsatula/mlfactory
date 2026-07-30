# mlfactory

A reproducibility-first experiment factory for machine-learning research.

`mlfactory` turns every experiment invocation into a first-class artifact: a versioned source snapshot, hashed inputs and outputs, a frozen environment, hardware provenance, and a queryable SQLite registry. Experiments live in self-contained domains under `mlfactory/experiments/<name>/`; the top-level package provides the reusable harness, CLI, dashboard, and remote execution layer.

---

## Why mlfactory

Research code tends to outgrow ad-hoc shell scripts and notebook cells. `mlfactory` exists to make every run:

- **Reproducible**: each run archives the git tree, records the commit, hashes every input and artifact, and freezes the Python environment.
- **Traceable**: the SQLite registry indexes runs, metrics, and lineage so you can answer "what produced this file?" without guessing directory names.
- **Portable**: the same spec runs locally on a dual-RTX-3090 box or remotely on a Vast.ai H100 instance with no code changes.
- **Observable**: a config-driven Rich dashboard shows live run state, GPU telemetry, log tails, and experiment-specific probes.

---

## High-level architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         CLI / Dashboard                      │
│              mlfactory run | init | ls | dashboard          │
└─────────────────────────────┬───────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────┐
│                     Run driver (runner.py)                   │
│   spec ──► manifest ──► plugin ──► artifacts + registry      │
└─────────────────────────────┬───────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌───────────────┐    ┌────────────────┐    ┌────────────────┐
│  Manifest &   │    │   Plugins      │    │    Remote      │
│  Registry     │◄──►│   (per stage)  │    │  (Vast.ai/SSH) │
│  (core)       │    │                │    │                │
└───────────────┘    │ • collect      │    └────────────────┘
                     │ • classify     │
                     │ • stratify     │
                     │ • train        │
                     │ • eval         │
                     │ • build-pilot  │
                     └────────────────┘
```

---

## Core concepts

### Run manifest

Every run produces a `manifest.json` containing:

| Field | Purpose |
|-------|---------|
| `run_id` | Timestamped, human-readable identifier |
| `stage` | Experiment stage (`collect`, `train`, `eval`, ...) |
| `status` | `pending`, `running`, `completed`, `failed`, `guarded`, `aborted` |
| `spec` | The YAML/JSON spec that drove the run |
| `git` | Commit, branch, dirty flag, remote URL |
| `source` | `git archive` tar.gz and its SHA-256 |
| `inputs` | Resolved input files with SHA-256 and size |
| `artifacts` | Output files with SHA-256 and size |
| `env` | Python version, platform, `pip freeze`, selected env vars |
| `hardware` | GPU list from `nvidia-smi`, CPU count, RAM |
| `parent_runs` / `child_runs` | Provenance lineage |
| `summary` | Stage-specific summary (loss curves, counts, guard reports) |

The manifest is the source of truth; the registry is an index over it.

### Registry

A SQLite database (default `.mlfactory/registry.db`) stores runs, lineage edges, and time-series metrics. It is updated atomically so dashboards and notebooks always see a consistent view.

```python
from mlfactory.core.registry import Registry

registry = Registry(".mlfactory/registry.db")
runs = registry.find(stage="train", status="completed", limit=10)
```

### Plugins

A plugin is a Python class extending `StagePlugin` that implements `prepare()`, `execute()`, and `finalize()`. Plugins are registered by stage name and discovered automatically when `runner.py` is imported.

```python
from mlfactory.plugins.base import PLUGINS, StagePlugin

class MyStagePlugin(StagePlugin):
    stage = "my_stage"

    def prepare(self) -> None: ...
    def execute(self) -> None: ...
    def finalize(self) -> None: ...

PLUGINS.register(MyStagePlugin)
```

### Model server

Local inference is handled by the `model()` context manager in `mlfactory.core.model_server`. It resolves a model alias from `models.yaml`, starts a disposable `llama-server` on a free port, waits for health, and stops the server on exit.

```python
from mlfactory.core.model_server import model

with model("qwen3.5:4b", gpu=0) as srv:
    client = srv.client()
    response = client.chat.completions.create(...)
```

---

## Project layout

```
mlfactory/                      # reusable harness
  __init__.py
  __main__.py
  cli.py                        # Click CLI entry point
  core/
    manifest.py                 # RunManifest schema + snapshot/hashing helpers
    registry.py                 # SQLite API
    runner.py                   # spec -> run driver
    model_server.py             # disposable llama-server resource
    dashboard.py                # Rich Live read-only dashboard
    dashboard_config.py         # Probe/pane schema
    api.py                      # OpenAI-compatible client + judge utilities
    metrics.py                  # MetricsLogger
    env.py                      # Training/inference env guards
    embeddings.py               # Embedding helpers
    artifacts.py                # Checkpoint/summary helpers
  plugins/
    base.py                     # StagePlugin base class + registry
  remote/
    ssh_runner.py               # Generic SSH remote runner
    vast.py                     # Vast.ai provision/run/pull wrapper
  experiments/                  # experiment domains
    ace/                        # ACE baseline trajectory collection
      collect.py
      collect_plugin.py
      classify.py
      classify_plugin.py
      stratify.py
      stratify_plugin.py
      generate_prompts.py
      generate_prompts_plugin.py
      prompts/                  # system prompts as markdown
      specs/                    # YAML run specs
      data/                     # experiment-specific data (gitignored)
      dashboard.json
    dft/                        # DFT training / evaluation
      train_dft.py
      train_plugin.py
      eval.py
      eval_plugin.py
      eval_local_pair.py
      build_pilot.py
      build_pilot_plugin.py
      prompts/
      specs/
      data/
      setup_h100.sh
      dashboard.json
  utils/                        # small shared helpers

agents/skills/                  # pi agent skill docs
  building-an-experiment/
  ensuring-reproducibility/
  logging-metrics/
  monitoring-dashboard/
  operating-locally/
  running-on-vast/
  using-core-utilities/
  using-model-server/
  writing-plugins/

runs/                           # per-run output directories (gitignored)
.mlfactory/                     # registry.db and factory state (gitignored)
migrations/                     # legacy-run ingestion scripts
tests/                          # smoke tests
models.yaml                     # local GGUF alias registry
pyproject.toml
```

---

## Quick start

### Local run

```bash
# Create a run directory and manifest without executing.
mlfactory init mlfactory/experiments/ace/specs/ace_collect_qwen35.yaml

# Execute the run and register it.
mlfactory run mlfactory/experiments/ace/specs/ace_collect_qwen35.yaml

# List completed runs.
mlfactory ls

# Inspect a manifest.
mlfactory show <run_id>

# Watch the registry and GPU telemetry.
mlfactory dashboard
```

### Remote run on Vast.ai

```bash
# Provision an H100 instance and run a spec.
mlfactory remote provision
mlfactory remote run \
  --host <host> --port <port> \
  --spec mlfactory/experiments/dft/specs/dft_train_h100_validation.yaml

# Manage instances.
mlfactory remote list
mlfactory remote stop --instance-id <id>
mlfactory remote destroy --instance-id <id>
```

Python API for the same flow:

```python
import os
from mlfactory.remote.vast import VastRunner

runner = VastRunner.from_search(api_key=os.environ["VAST_API_KEY"])
runner.provision()
runner.setup(setup_script="mlfactory/experiments/dft/setup_h100.sh")
run_id = runner.run_spec("mlfactory/experiments/dft/specs/dft_train_h100_validation.yaml")
runner.pull_outputs(run_id)
runner.pull_registry(".mlfactory/registry-remote.db")
runner.stop()
```

---

## Writing a spec

Specs are YAML files with at least a `stage` and a `name`. Everything else is stage-specific.

```yaml
stage: collect
name: qwen35-4b-baseline
experiment: ace

model: qwen3.5:4b
model_name: Qwen/Qwen3.5-4B
provider: llama
prompts: mlfactory/experiments/ace/data/prompts.jsonl
gpu: 0

max_model_len: 32768
max_output_tokens: 16000
request_timeout: 1200
time_budget_seconds: 900
seed_offset: 0
stratified_extras: 3

env:
  HF_HOME: /home/admin/.cache/huggingface
```

See `mlfactory/experiments/ace/specs/` and `mlfactory/experiments/dft/specs/` for more examples.

---

## Dashboard

The dashboard is read-only and experiment-configurable. Each domain may provide a `dashboard.json` (or `dashboard.yaml`) that declares probes and panes. Built-in probe types include:

- `file_line_count`, `regex_count`, `regex_last_match`
- `jsonl_last_record`
- `process_alive`
- `http_status`, `http_json`
- `gpu_status`
- `disk_usage`
- `shell_command`
- `spec_value`

Pane types include `overview`, `metrics_table`, `bars`, `recent_log`, `gpu_table`, `run_info`, `lineage`, and `text`.

```bash
mlfactory dashboard --watch-run <run_id> --refresh 2.0
```

---

## Design principles

1. **Experiment domains are self-contained.** Each experiment owns its data, specs, prompts, and bespoke code. The factory does not assume it knows what is inside an experiment's black box.
2. **The manifest is the source of truth.** Every run gets one, and it is immutable once finalized.
3. **One driver for all stages.** `mlfactory run` reads a spec, picks the plugin, executes it, and registers the result.
4. **Source snapshot by default.** Every run archives the repo with `git archive` before execution.
5. **Non-destructive migration.** Existing legacy runs can be ingested into the registry without moving or modifying originals.
6. **Read-only dashboard.** Monitoring never controls runs, only observes them.

---

## Development

### Install

```bash
python -m pip install -e .
```

Optional remote dependencies:

```bash
python -m pip install -e ".[remote]"
```

### Run tests

```bash
python -m pytest tests/
```

Current smoke tests cover manifest round-trip, registry CRUD, and lineage.

### Local environment assumptions

This repo is developed on a host with:

- Ubuntu 26.04, AMD Ryzen 7 9800X3D, 30 GiB RAM
- 2× NVIDIA RTX 3090 (24 GB VRAM each)
- `llama.cpp` at `/home/admin/llama.cpp/build/bin/llama-server`
- Local GGUFs under `/home/admin/models/`
- DFT training uses a separate CUDA/torch venv at `/home/admin/dft-eval-harness/.venv312/bin/python`

These paths are reflected in `models.yaml` and the DFT specs. Override them in your own specs as needed.

---

## Status

Implemented:

- [x] Git-backed manifest schema with source snapshot, input/artifact hashing, env freeze, and hardware capture
- [x] SQLite registry with runs, lineage, and metrics
- [x] CLI: `init`, `run`, `ls`, `show`, `lineage`, `ingest`, `dashboard`
- [x] Rich Live config-driven dashboard
- [x] Disposable `llama-server` model resource and alias registry
- [x] ACE collect, classify, stratify, and generate-prompts plugins
- [x] DFT train, eval, and build-pilot plugins
- [x] Vast.ai remote runner: provision, run spec, pull outputs/registry, stop/destroy
- [x] Non-destructive legacy-run migration
- [x] Smoke tests for manifest and registry

Still tightening:

- [ ] Telemetry streaming into `dashboard.jsonl` during runs
- [ ] More stage-specific dashboard panes
- [ ] Vast.ai-specific provisioning helpers (templates, spot/preemptible handling)
- [ ] Regression tests for plugin execution paths
- [ ] Remote registry merge utility

---

## License

This is a personal research project. No license is granted unless explicitly stated in a `LICENSE` file.
