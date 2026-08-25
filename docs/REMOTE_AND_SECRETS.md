# Remote execution (Vast.ai) and secrets

> Update when: the Vast runner API or the secrets mechanism changes. Sources
> of truth: `mlfactory/remote/vast.py`, `mlfactory/remote/ssh_runner.py`,
> `mlfactory/core/secrets.py`.

## Vast.ai runbook (read before any Vast work)

Before provisioning, configuring, debugging, or destroying a Vast training
instance, read the three docs in this directory:

- `VAST_REMOTE.md` — provisioning, connecting, launch packages, Supervisor,
  monitoring, artifact preservation, teardown.
- `TRAINING_STACK.md` — GPU memory, the smoke-test ladder, batch
  benchmarking, the OOM/hang decision tree, objective safety checks.
- `DEBUGGING_METHOD.md` — the universal investigation discipline (hypothesis
  ranking, smallest faithful reproducer, one variable at a time).

DFT/Qwen3.5-specific version pins, H100 setup, and the B200/FLA/TileLang
specifics live in `/home/admin/facktry/BABYS_FIRST_VAST_ML_ENGINEER.md`.
Do not repeat the B200/Qwen3.5 kernel investigation or reduce batches until
the runbook's hidden-process and software-path checks have been completed.

## Vast.ai runner

Manages Vast instances and runs experiments on them. Requires the `vastai`
CLI installed and a Vast API key (env var `VAST_API_KEY` or `~/.config/vastai/vast_api_key`).

```python
import os
from mlfactory.remote.vast import VastRunner

# Provision from the default H100 search, or pass a custom query.
runner = VastRunner.from_search(api_key=os.environ["VAST_API_KEY"])
runner.provision()                                   # search offer → create instance → wait for SSH
runner.setup(experiment="dft")                       # pip install -e . + optional setup_h100.sh
runner.setup(setup_script="mlfactory/experiments/<domain>/setup_h100.sh")  # or explicit
run_id = runner.run_spec("mlfactory/experiments/<domain>/specs/<spec>.yaml")
runner.pull_outputs(run_id)                          # rsync runs/<run_id> back
runner.pull_registry(".mlfactory/registry-remote.db")
runner.stop()                                        # stop the instance (keeps it)
runner.destroy()                                    # destroy the instance
```

- `from_search(query=...)` runs a custom Vast query; default is `find_h100_offer` (H100, ≥80 GB VRAM, ≥2 GPUs, ≥300 GB disk).
- `setup(experiment=...)` runs a known experiment's setup script (e.g. `dft` → `setup_h100.sh`); `setup(setup_script=...)` runs an explicit path.
- `run_spec` syncs code (rsync, excludes `.venv`/`runs`/`.mlfactory`/etc), runs `mlfactory init` then `mlfactory run` remotely, and pulls outputs in a `finally`.
- `pull_registry` pulls the remote `.mlfactory/registry.db` for merging.

Merge remote registry into local:

```bash
mlfactory registry merge .mlfactory/registry-remote.db
```

`Registry.merge_from(source_db_path, on_conflict="skip"|"replace")` copies
runs, lineage, and metrics; `skip` keeps existing runs, `replace`
overwrites. Only runs imported by the merge get their lineage/metrics
copied (both ends must be imported).

## Generic SSH runner

`mlfactory/remote/ssh_runner.py` provides `SSHRunner` (run commands,
rsync to/from remote, stream output) and `SSHConfig`. `VastRunner`
subclasses it; use `SSHRunner` directly for a pre-provisioned box.

## Secrets

Store API keys in `.mlfactory/secrets.yaml` (chmod 0o600) and reference
them in specs:

```yaml
api_key: ${secrets.OPENROUTER_API_KEY}
```

The runner expands `${secrets.KEY}` at execution time via
`mlfactory.core.secrets.expand_secrets`; they are **never** written into
run manifests. `SecretsStore` values may themselves reference environment
variables (`$VAR` / `${VAR}`), resolved at load time.

Programmatic use:

```python
from mlfactory.core.secrets import SecretsStore, expand_secrets
store = SecretsStore()                    # .mlfactory/secrets.yaml
store.set("KEY", "value"); store.get("KEY")
expand_secrets({"api_key": "${secrets.KEY}"})   # recursive over dict/list/str
```
