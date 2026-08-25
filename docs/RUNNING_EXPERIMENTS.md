# Spec format, multi-stage lineage, and guard logic

> Update when: the spec schema, the lineage protocol, or the guard mechanism
> changes. Sources of truth: `mlfactory/core/runner.py` (spec resolution,
> `run_from_spec`, `parent_runs`), `mlfactory/core/manifest.py` (RunManifest
> schema, `guard_report`), `mlfactory/core/registry.py` (`link_runs`).

## Spec format

YAML files with at minimum `stage`, `name`, and `experiment`:

```yaml
stage: transform
name: my-analysis
experiment: mydomain
# ... stage-specific fields ...
```

- `stage` must be a name in `manifest.STAGES` (see `docs/PLUGIN_CONTRACT.md`).
- The runner resolves file-path values in the spec as inputs (hashes them
  automatically into `manifest.inputs`).
- For multi-stage pipelines, use `input_run: <run_id>` and resolve it in
  the plugin via `Registry.get(input_run)`.
- Secret references `${secrets.KEY}` are expanded at execution time;
  see `docs/SECRETS.md`.

## Running an experiment

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

## Multi-stage pipelines with lineage

Experiments often chain stages where one stage consumes another's output.
The pattern:

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

`parent_runs` calls `registry.link_runs(parent, child, "input")` under the
hood, queryable via `registry.parents(run_id)` / `registry.children(run_id)`.
The sample experiment demonstrates this: transform → classify, followed by
sibling train and eval stages.

## Guard logic (quality gates)

Experiments gate quality by setting the manifest status to `"guarded"`. The
status is a fixed set in `manifest.py`:
`pending|running|completed|failed|guarded|aborted`. The runner leaves the
status as `completed` unless your plugin sets `guarded`; on exception it
sets `failed`.

```python
if quality_score < threshold:
    self.manifest.status = "guarded"
    self.manifest.guard_report = {
        "reason": f"accuracy {quality_score:.4f} below threshold {threshold}",
        "threshold": threshold,
        "accuracy": quality_score,
    }
```

`guard_report` is a free-form dict on the manifest. The sample `eval` plugin
demonstrates this pattern. A `guarded` run is still `completed` from the
runner's perspective (no exception), but its status marks it as not passing
the quality gate — distinguishable from `completed` in `mlfactory ls` and
in the registry.
