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
    notes.py                    # run-attached lab notes (append/read/search)
    env.py                      # training_env(), inference_env() context managers
    api.py                      # APIClient, Judge, extract_json
    artifacts.py                # save_checkpoint (with metadata), save_summary, save_config
    datasave.py                 # datasave()/DataSaver — save data with lab-notebook metadata
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
| `transform` | Chunk text, compute statistics | Plugin lifecycle, MetricsLogger, **datasave** (title+description), `finalize_artifacts` |
| `classify` | Topic classification via LLM | Model server (`model()`), APIClient, `inference_env()`, multi-stage lineage |
| `train` | Supervised fine-tuning | NumPy smoke backend, optional Hugging Face full/LoRA backend, labeled checkpoints, loss metrics |
| `eval` | Judge classification quality | APIClient, `Judge.compare()`, **guard logic**, datasave |

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
- **Save every datum through `datasave()`** (or `save_checkpoint()` for model checkpoints) — it writes the file, attaches a lab-notebook `title` + two-sentence `description`, drops a `<file>.meta.json` label card, and hashes+registers the `FileRecord` into the manifest automatically. Never hand-write data with `open()`/`json.dump`/`np.save`/`to_csv`.
- In `finalize()`, call `finalize_artifacts(self.manifest, self.run_dir)` — one line that hashes any files not already registered (logs, stragglers) and persists the manifest. It de-dups files `datasave` already labeled and skips `.meta.json` sidecars.
- When you save the run summary via `datasave`, set `self.manifest.summary = <dict>` yourself (this is the special behavior `save_summary()` performed).
- Use `MetricsLogger` for any per-step measurements
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
| **Save data (datasets, predictions, reports, configs, arrays, plots)** | `from mlfactory.core.datasave import datasave` | `datasave("x.jsonl", rows, title=..., description=..., tags=[...], manifest=m)` — **required** for all generated data |
| Save data (plugin-bound) | `from mlfactory.core.datasave import DataSaver` | `saver = DataSaver(run_dir, m); saver.save(...); saver.finalize()` |
| Finalize artifacts | `from mlfactory.core.datasave import finalize_artifacts` | `finalize_artifacts(m, run_dir)` — replaces the rglob+sha256+FileRecord finalize loop |
| Checkpoints | `from mlfactory.core.artifacts import save_checkpoint` | `save_checkpoint(run_dir, step, model, tok, manifest=m, title=..., description=...)` |
| Label an existing checkpoint dir | `from mlfactory.core.datasave import register_checkpoint_dir` | `register_checkpoint_dir(m, run_dir, ckpt_dir, title=..., description=...)` |
| Find artifacts across runs | `Registry(...).find_artifacts(...)` | `registry.find_artifacts(tag=..., title=..., stage=..., format=...)` |
| Summary (legacy) | `from mlfactory.core.artifacts import save_summary` | `save_summary(run_dir, data, manifest=m)` — prefer `datasave("summary.json", ...)` + set `manifest.summary` |
| Config (legacy) | `from mlfactory.core.artifacts import save_config` | `save_config(run_dir, config_dict)` — prefer `datasave("config.json", ..., title=..., description=...)` |

## Saving data with metadata (datasave)

**Every datum an experiment writes must go through `datasave()`** (or `save_checkpoint()` for model checkpoints). This is non-negotiable — do not hand-write files with `open()` + `json.dump`, `np.save`, pandas `to_csv`, etc.

`datasave` implements the laboratory model: it captures **provenance** automatically (git commit, environment, hardware, parent runs — already on the run manifest) and requires the caller to supply only the **meaning** a scientist would write on the sample label. Provenance that does not vary per artifact is never re-entered.

Required arguments (the only things the code cannot derive on its own):
- `title` — short human name (e.g. `"Chunked corpus"`, `"DFT policy checkpoint"`).
- `description` — ~two sentences: what the data is, and how it was made / what it measures.

Optional lab metadata:
- `tags` — findability keywords (queried via `Registry.find_artifacts(tag=...)`).
- `caveats` — known-issue warnings (e.g. `"batch 3 had a calibration drift — do not use for training"`).
- `sensitivity` — `public` | `internal` | `restricted` (privacy / human-subjects).
- `schema` — how to load it: columns/keys, dtypes, units.
- `name` — stable slug key (defaults to the file stem).
- `format` — `auto` (default; inferred from suffix) or `json|jsonl|csv|tsv|text|yaml|numpy|npz|parquet|bytes`.

`datasave` writes the file, drops a sidecar `<file>.meta.json` label card next to it (so a scientist browsing the folder can read what each file is), and registers a hashed `FileRecord` carrying the metadata into the manifest — so `Registry.find_artifacts()` can discover it across all runs.

```python
from mlfactory.core.datasave import DataSaver, finalize_artifacts

# in execute():
saver = DataSaver(self.run_dir, self.manifest)
saver.save("chunks.jsonl", chunk_records,
           title="Chunked corpus",
           description="Input text split into fixed-size chunks for classification. "
                       "Each row is one chunk with its word/sentence statistics.",
           tags=["corpus", "sample"], format="jsonl")

# the run summary is a normal datum too — mirror it onto manifest.summary
self.manifest.summary = summary
saver.save("summary.json", summary,
           title="Transform summary",
           description="Aggregate statistics for the chunked corpus. Computed from chunks.jsonl.",
           format="json")

# in finalize(): replaces the manual rglob + sha256_file + FileRecord loop
finalize_artifacts(self.manifest, self.run_dir)
```

Model checkpoints use `save_checkpoint(run_dir, step, model, tokenizer, manifest=m, title=..., description=...)` — it attaches the same metadata to every file in the checkpoint and writes a `<ckpt_dir>.meta.json` label. For a checkpoint directory you wrote yourself (e.g. a numpy model with no `save_pretrained`), use `register_checkpoint_dir(manifest, run_dir, ckpt_dir, title=..., description=...)`.

Standalone scripts with no plugin/manifest can still call `datasave(path, data, title=..., description=...)` — it writes the file + sidecar label even without a manifest (no `FileRecord` is created).

Discover data across runs (the lab master catalog):

```python
registry.find_artifacts(tag="corpus")              # by tag
registry.find_artifacts(title="statistics")        # by title substring
registry.find_artifacts(stage="train", format="checkpoint")
```

The sample experiment (`mlfactory/experiments/sample/`) is the reference: all four stages save through `DataSaver` and finalize with `finalize_artifacts`.

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

- **Lunaroute:** always prefer the `-ballast` model variant when one is available (e.g., `glm-5.2-vision-ballast`). Fall back to the plain model only if no ballast variant exists. On first use in a session, query `GET /v1/models` to get the list of currently active model names — the available set can change between sessions.

## Common mistakes

- **Running experiment scripts directly** — always go through `run_from_spec()`
- **Writing outputs outside `self.run_dir`** — artifacts and logs belong under the run dir
- **Manually writing data files** (`open()`/`json.dump`/`np.save`/`to_csv`) — use `datasave()` so every datum gets a title + two-sentence description, a `.meta.json` label card, and is hashed+registered into the manifest automatically
- **Saving data without a title/description** — `datasave()` requires both; they are the only provenance the code cannot derive
- **Hand-rolling the `finalize()` rglob+sha256+`FileRecord` loop** — use `finalize_artifacts(self.manifest, self.run_dir)`
- **Not registering the plugin** — add the import in `mlfactory/core/runner.py`
- **Hardcoding model paths** — use `models.yaml` aliases and the `model()` context manager
- **Forgetting `self.manifest.write()`** in `finalize()` — `finalize_artifacts()` persists it for you
- **Running code outside a git repo** — `git archive` needs a committed tree
- **Using `Registry()` default path in plugins** — use `Registry(".mlfactory/registry.db")` to match the CLI

## Lab notes

Lab notes are the frequent, cheap, mid-flight end of the notebook axis. They live in `runs/<run_id>/notes.jsonl`, appended one line at a time via `mlfactory note <run_id> <text>`. One line, no structure required — cheaper than a sticky note. They are the *thinking* layer on top of the manifest: the manifest records *what* a run did, a lab note records *why you changed something*, *what you expected*, or *what surprised you*.

Write a lab note at:
- a **hypothesis change** — "trying lr 3e-5 because 1e-4 plateaued, expect lower floor"
- an **unexpected result** — "loss spiked at step 800 then recovered; not the usual divergence shape"
- a **parameter change with rationale** — "warmup 500→1000 steps to avoid the early instability from run X"
- a **dead end** (the negative result that prevents a retry) — "mixture-of-depths worse than dense at equal FLOPs, don't revisit without a new reason"
- a **resumption point** — "pausing here; next step is to re-run eval with the fixed judge prompt"

Do not write a lab note for:
- routine execution (the manifest already captures it)
- anything the metrics/logs/summary already record
- a full session's worth of thinking (that is `session_notes/`, below)

Read with `mlfactory notes <run_id>`; search across all runs with `mlfactory notes --grep <term>` (the "what have I tried" query). `mlfactory show <run_id>` surfaces notes inline after the manifest. Notes are hashed as `FileRecord(role="note")` and provenance-linked like any other artifact.

Lab notes are deliberately low-ceremony. If writing one feels like work, you're writing the wrong thing — write a session note instead, or nothing.

## Session notes

End-of-session notes live in `session_notes/YYYY-MM-DD-<slug>.md`. Write one when a session produced durable knowledge — findings, decisions, environment traps — that the repo doesn't record on its own. Do not write one for routine execution.

Required structure:

```
# Session notes — <date> — <slug>
Scope: <one or two lines>

## What exists now that didn't before     # new artifacts, paths, commits (hash only)
## Findings (the durable knowledge)       # numbered; each = claim + the evidence that established it
## Decisions with rationale               # rulings made + why; include rejected alternatives that look plausible
## Environment traps encountered          # failure -> working fix (debugging time is the expensive artifact)
## State at note time                     # running jobs, progress counters, immediate next step
```

Rules:
- Findings are claims with evidence, not observations ("X is true because measured Y", not "we saw Y").
- Decisions record the final causal model, not the chronology of reaching it.
- Reference committed work by hash; never restate what git history carries.
- Session notes and lab notes together form the lab notebook for this project. Session notes are the rare, high-cost, end-of-session record for a successor with zero context; lab notes are the frequent, one-line, run-attached record for future-you skimming your own past (see "Lab notes" above). They are distinct artifacts with distinct protocols and must not collapse into one — a single notebook that tries to serve both audiences serves neither.
- Keep them dense. A future reader with zero session context is the audience.

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
