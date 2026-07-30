---
name: running-mlfactory-on-vast
description: Use when offloading heavy compute to a Vast.ai instance, or when a local run needs more GPUs/VRAM than the host has.
---

# Running mlfactory on Vast.ai

## Overview

Use `VastRunner` from `mlfactory.remote.vast` to provision or attach to a Vast.ai instance, sync the repo, install dependencies, run a spec, and pull outputs and registry state back. Prefer the Python API; the CLI is for one-off human commands.

## When to use

- DFT training on H100s.
- Any workload that needs more than the local dual-3090 host can provide.
- You have a Vast API key and an SSH key for the instances.

## Core pattern (Python API)

```python
import os
from mlfactory.remote.vast import VastRunner
from mlfactory.core.registry import Registry

runner = VastRunner.from_search(api_key=os.environ["VAST_API_KEY"])
runner.provision()                    # finds H100 offer, creates instance
runner.setup(setup_script="mlfactory/experiments/dft/setup_h100.sh")

run_id = runner.run_spec(
    "mlfactory/experiments/dft/specs/dft_train_h100_validation.yaml"
)
runner.pull_outputs(run_id)
runner.pull_registry(".mlfactory/registry-remote.db")
runner.stop()                         # halt charges, keep storage
```

## Quick reference

| Goal | Python API | CLI fallback |
|------|------------|--------------|
| List instances | `list_instances(api_key=...)` | `mlfactory remote list` |
| Provision | `VastRunner.from_search(...).provision()` | `mlfactory remote provision` |
| Attach | `VastRunner.from_instance_id(id)` | — |
| Sync code | `runner.sync_code()` | — |
| Install deps | `runner.setup(setup_script=...)` | — |
| Run spec | `runner.run_spec(spec_path)` | `mlfactory remote run` |
| Pull outputs | `runner.pull_outputs(run_id)` | — |
| Pull remote registry | `runner.pull_registry(local_path)` | — |
| Stop (keep disk) | `runner.stop()` | `mlfactory remote stop` |
| Destroy (irreversible) | `runner.destroy()` | `mlfactory remote destroy` |

## Generic SSH fallback

If you are not on Vast, use `mlfactory.remote.ssh_runner.SSHConfig` + `SSHRunner` directly. `VastRunner` is a thin subclass on top of it.

## Common mistakes

- **Running long training inline over SSH.** For multi-hour training, wrap the remote `mlfactory run` in Supervisor or `tmux` so a disconnect does not kill the job; `run_spec` streams synchronously and pulls outputs on completion.
- **Destroying before pulling outputs.** `run_spec` pulls outputs in a `finally` block, but verify `runs/<run_id>` locally before destroying the instance.
- **Hardcoding the API key in files.** Read `VAST_API_KEY` from the environment or the vastai config file.
- **Forgetting to stop the instance.** Stopped instances do not incur GPU charges but keep disk.
- **Ignoring the remote registry.** Pull it and merge into the local registry so lineage and metrics survive instance teardown.
