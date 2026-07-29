# mlfactory

A reproducibility-first experiment factory for ML research.

Every run gets:
- a git commit + source archive,
- hashed inputs and artifacts,
- a frozen environment,
- a SQLite registry,
- a read-only Rich Live dashboard.

## Quick start

```bash
# Use the same virtual environment that already has pydantic/rich/click.
source /home/admin/ace-baseline-trajectories/.venv/bin/activate

# List ingested legacy runs.
python -m mlfactory.cli ls

# Create a run directory from a spec without executing.
python -m mlfactory.cli init specs/ace_collect_qwen35.yaml

# Execute a run.
python -m mlfactory.cli run specs/ace_collect_qwen35.yaml

# Watch the registry.
python -m mlfactory.cli dashboard
```

## Layout

```
mlfactory/
  core/
    manifest.py      # RunManifest schema + source snapshot helpers
    registry.py      # SQLite API
    runner.py        # spec -> run driver
  plugins/
    base.py          # StagePlugin base class + registry
  experiments/
    ace/             # ported collect/classify/generate_prompts
    dft/             # ported train_dft/eval
  remote/
    ssh_runner.py    # generic SSH remote runner (Vast.ai-first)
  dashboard.py       # Rich Live read-only dashboard
  cli.py             # command-line entry point
specs/               # YAML run specifications
runs/                # per-run output directories (gitignored)
data/                # registry.db + symlinks to input corpora
migrations/          # non-destructive legacy-run ingestion
```

## Design principles

1. **Manifest is the source of truth.** `manifest.json` contains the spec,
   git commit, source archive hash, input hashes, environment freeze, hardware
   info, and artifact hashes.
2. **One driver for all stages.** `mlfactory run` reads a spec, picks the
   plugin, executes it, and registers the result.
3. **Source snapshot by default.** Every run archives the repo with
   `git archive` before execution.
4. **Non-destructive migration.** Existing `ace-baseline-trajectories` and
   `dft-eval-harness` runs were ingested into `data/registry.db` without moving
   or modifying the originals.
5. **Read-only dashboard.** `mlfactory dashboard` queries the registry and
   shows run state, lineage, and local GPU status. It does not control runs.

## Writing a spec

Specs are YAML (or JSON) files with at least:

```yaml
stage: collect
name: my-run
# stage-specific fields...
```

See `specs/ace_collect_qwen35.yaml` and `specs/dft_train_h100_validation.yaml`
for examples.

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
- [x] ACE collect plugin + DFT train plugin
- [x] generic SSH remote runner skeleton

Still to tighten:
- [ ] ACE classify plugin
- [ ] DFT eval plugin
- [ ] telemetry streaming into `dashboard.jsonl` during runs
- [ ] dashboard stage-specific panes
- [ ] remote runner Vast.ai-specific provisioning helpers
- [ ] regression tests for manifest round-trip and registry queries
