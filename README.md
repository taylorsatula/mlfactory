# mlfactory

A reproducibility-first experiment factory for ML research.

Every run gets:
- a git commit + source archive,
- hashed inputs and artifacts,
- a frozen environment,
- a SQLite registry,
- a read-only Rich Live dashboard.

Experiments live in their own directories (`mlfactory/experiments/<name>/`)
with their own data, specs, and processing code. The top-level `mlfactory/`
package holds the reusable harness.

## Quick start

```bash
# Use the same virtual environment that already has pydantic/rich/click.
source /home/admin/ace-baseline-trajectories/.venv/bin/activate

# List ingested legacy runs.
python -m mlfactory.cli ls

# Create a run directory from a spec without executing.
python -m mlfactory.cli init mlfactory/experiments/ace/specs/ace_collect_qwen35.yaml

# Execute a run.
python -m mlfactory.cli run mlfactory/experiments/ace/specs/ace_collect_qwen35.yaml

# Watch the registry.
python -m mlfactory.cli dashboard
```

## Layout

```
mlfactory/                  # reusable harness
  core/
    manifest.py             # RunManifest schema + source snapshot helpers
    registry.py             # SQLite API
    runner.py               # spec -> run driver
  plugins/
    base.py                 # StagePlugin base class + registry
  dashboard.py              # Rich Live read-only dashboard
  cli.py                    # command-line entry point
  remote/
    ssh_runner.py           # generic SSH remote runner (Vast.ai-first)

mlfactory/experiments/      # experiment domains
  ace/                      # ACE baseline trajectory collection
    data/                   # experiment-specific data (gitignored/symlinked)
    specs/                  # experiment-specific run specs
    collect.py
    classify.py
    collect_plugin.py
    classify_plugin.py
    generate_prompts.py
    launch_llama_qwen35_4b.sh
  dft/                      # DFT training / evaluation
    data/                   # experiment-specific data (gitignored/symlinked)
    specs/                  # experiment-specific run specs
    train_dft.py
    eval.py
    eval_local_pair.py
    train_plugin.py
    eval_plugin.py
    setup_h100.sh

runs/                       # per-run output directories (gitignored)
data/                       # registry.db (gitignored)
migrations/                 # non-destructive legacy-run ingestion
tests/                      # smoke tests
```

## Design principles

1. **Experiment domains are self-contained.** Each experiment owns its data,
   specs, and bespoke code. The factory does not assume it knows what is inside
   an experiment's black box.
2. **Manifest is the source of truth.** `manifest.json` contains the spec,
   git commit, source archive hash, input hashes, environment freeze, hardware
   info, and artifact hashes.
3. **One driver for all stages.** `mlfactory run` reads a spec, picks the
   plugin, executes it, and registers the result.
4. **Source snapshot by default.** Every run archives the repo with
   `git archive` before execution.
5. **Non-destructive migration.** Existing `ace-baseline-trajectories` and
   `dft-eval-harness` runs were ingested into `data/registry.db` without moving
   or modifying the originals.
6. **Read-only dashboard.** `mlfactory dashboard` queries the registry and
   shows run state, lineage, and local GPU status. It does not control runs.

## Writing a spec

Specs are YAML (or JSON) files with at least:

```yaml
stage: collect
name: my-run
# stage-specific fields...
```

See `mlfactory/experiments/ace/specs/ace_collect_qwen35.yaml` and
`mlfactory/experiments/dft/specs/dft_train_h100_validation.yaml` for examples.

## Legacy run migration

```bash
python migrations/ingest_existing_runs.py
```

This populates `data/registry.db` with manifests derived from existing output
directories in `ace-baseline-trajectories/` and `dft-eval-harness/`.

## Remote execution (SSH / Vast.ai)

```python
from mlfactory.remote.ssh_runner import RemoteRunner, vast_config

runner = RemoteRunner(vast_config(host="1.2.3.4", port=12345, key="~/.ssh/id_vast"))
runner.setup()
run_id = runner.run("dft_train_h100_validation")
```

The runner syncs the repo, creates the run remotely, executes it, and pulls
outputs back.

## Status

This is the greenfield scaffold. The following are in place:
- [x] git repo + manifest schema
- [x] source snapshot + input/artifact hashing
- [x] SQLite registry + lineage
- [x] CLI (`init`, `run`, `ls`, `show`, `lineage`, `ingest`, `dashboard`)
- [x] Rich Live dashboard
- [x] non-destructive migration of legacy runs
- [x] per-experiment `data/` and `specs/` directories
- [x] ACE collect plugin + DFT train plugin
- [x] generic SSH remote runner skeleton

Still to tighten:
- [ ] ACE classify plugin
- [ ] DFT eval plugin
- [ ] telemetry streaming into `dashboard.jsonl` during runs
- [ ] dashboard stage-specific panes
- [ ] remote runner Vast.ai-specific provisioning helpers
- [ ] regression tests for manifest round-trip and registry queries
