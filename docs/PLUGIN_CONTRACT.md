# Plugin contract

> Update when: `mlfactory/plugins/base.py` or the runner protocol changes.
> Source of truth: `mlfactory/plugins/base.py` (read the `StagePlugin`
> docstring). This doc carries the *why* and the *when*; the module carries
> the *what*.

## The shape

Every plugin extends `StagePlugin` and implements three methods. The
runner calls `run()`, which calls `prepare()` → `execute()` → `finalize()`
in a try/finally, so `finalize()` is **always** called (even on failure).

```python
from mlfactory.plugins.base import PLUGINS, StagePlugin

class MyPlugin(StagePlugin):
    stage = "transform"  # must match a name in manifest.STAGES

    def prepare(self) -> None:
        """Validate inputs, create subdirs, start services."""

    def execute(self) -> None:
        """Run the workload. May raise on failure."""

    def finalize(self) -> None:
        """Hash artifacts, write summary, stop services. ALWAYS called."""

    @classmethod
    def dashboard(cls) -> "ExperimentDashboardConfig | None":
        """Optional: return this stage's dashboard config. None opts out."""

PLUGINS.register(MyPlugin)
```

## The available stages

`STAGES` is a fixed set in `mlfactory/core/manifest.py` (validation
rejects others). Current set:

```
collect, classify, stratify, generate, train, eval, build-pilot,
transform, analyze, review, voice-train, voice-synthetic-train,
voice-robust-train, voice-dpo-train, causal-graph
```

Adding a stage means editing `STAGES` in `manifest.py` (it is not
open-ended). Pick an existing one if it fits.

## Key rules (non-negotiable)

- All outputs go under `self.run_dir` (artifacts in `artifacts/`, logs in `logs/`).
- **Save every datum through `datasave()`** (or `save_checkpoint()` for
  checkpoints) — see `docs/DATASAVE.md`. Never hand-write data with
  `open()`/`json.dump`/`np.save`/`to_csv`.
- In `finalize()`, call `finalize_artifacts(self.manifest, self.run_dir)` —
  one line that hashes files not already registered (logs, stragglers) and
  persists the manifest. It de-dups files `datasave` already labeled and
  skips `.meta.json` sidecars.
- When you save the run summary via `datasave`, set
  `self.manifest.summary = <dict>` yourself (this is the special behavior
  the legacy `save_summary()` performed).
- Use `MetricsLogger` for any per-step measurements.
- Use `inference_env()` / `training_env()` for environment management.
- Set `self.manifest.status = "guarded"` for quality gates
  (see `docs/GUARD_AND_SPEC.md`).

## The `run()` method (do not override)

`StagePlugin.run()` (in `base.py`) wraps your three methods:

```
status = "running"; started_at = now
try:
    prepare(); execute()
    if status != "guarded": status = "completed"
except Exception:
    status = "failed"; summary["error"] = ...; raise
finally:
    completed_at = now; finalize()    # always
return manifest
```

You implement the three phases; the runner handles status transitions and
the guarantee that `finalize()` runs even on failure.

## Dashboard (optional)

`dashboard()` is a `@classmethod` — a dashboard is a property of the stage
*type*, not of a run instance. The runner calls it at create time and
persists the result to `run_dir/dashboard.json` so the viewer resolves a
run's dashboard from the run itself (reproducible), not from the source
tree checked out at view time. Returning `None` (the default) opts out;
the viewer falls back to a generic status/GPU/disk view.

Two common implementations:
- load a sibling static file: `ExperimentDashboardConfig.load(Path(__file__).with_name("dashboard_<stage>.json"))`
- build the config programmatically.

See `docs/DASHBOARD.md` for the probe/pane schema.

## Registering the plugin

Add an import in `mlfactory/core/runner.py` so the registry loads your
plugin. The `PLUGINS.register(MyPlugin)` call at module level runs on
import; the runner imports the module to populate the registry.

## The reference implementation

`mlfactory/experiments/sample/` is a full-featured 4-stage pipeline that
demonstrates every pattern (plugin lifecycle, MetricsLogger, datasave,
`finalize_artifacts`, model server, APIClient, `inference_env`,
multi-stage lineage, guard logic). Read its README and its four plugins
before writing a new one.
