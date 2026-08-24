---
name: administering-runs
description: Use when starting, launching, monitoring, stopping, resuming, or finalizing any mlfactory or legacy experiment run. Provides a generic, experiment-independent lifecycle with duration estimation, dashboard setup, GPU power management, provenance checks, failure handling, and completion reporting.
---

# Starting and administering runs

## Mission

Operate a run as a managed job, not as an ad-hoc command. The run must have:

- a clear input/spec and immutable run identity;
- a durable log, PID or process handle, and output location;
- a registry/manifest record when the run is an mlfactory experiment;
- a live dashboard for extended work;
- an explicit resource policy, including GPU power handling;
- a final verification and handoff report.

This skill is deliberately experiment-independent. Never assume ACE, DFT, a particular model, a fixed output filename, or a particular stage. Discover those from the request, spec, plugin, manifest, and repository instructions.

## Run request wireframe

Before touching a process, normalize the request into this record:

```text
experiment/domain: <domain or standalone job>
spec or entrypoint: <path>
stage: <spec stage, if applicable>
run_id: <explicit id or generated id>
inputs: <paths, parent run ids, datasets>
expected work: <examples, steps, epochs, or unknown>
resource class: <CPU / GPU / remote GPU>
model/service: <alias, endpoint, or none>
estimated duration: <hours, method, confidence>
dashboard: <experiment-specific / generic / required custom>
long-run GPU policy: threshold=2h, limit=340W, restore=exact previous limits
stop/resume policy: <graceful signal, checkpoint/resume support>
secrets: <names only; never values>
```

If a field is unknown, investigate it or state the uncertainty. Do not silently invent an output path, run count, model name, or resume behavior.

## Lifecycle wireframe

| Phase | Questions | Required action | Stop condition |
|---|---|---|---|
| 1. Scope | What exactly is being run? | Identify spec/entrypoint, inputs, stage, parents, resources, and completion criterion. | Ambiguous target or destructive side effect. |
| 2. Preflight | Can it run safely here? | Read applicable `AGENTS.md`; inspect spec/plugin; check dependencies, secrets, services, disk, GPU/process occupancy, and git state. | Missing input, unavailable service, conflicting job, or unsafe resource change. |
| 3. Estimate | Will it exceed two hours? | Use a pilot, prior run metrics, workload size, or a conservative per-item estimate. Record method and uncertainty. | Estimate cannot be defended for a high-cost run. |
| 4. Prepare | Can it be reproduced and watched? | Create/validate the run identity, manifest/registry entry, launcher log, PID handle, and dashboard config. | No durable handle or no dashboard for an extended run. |
| 5. Launch | Is the exact command recorded? | Start through `run_from_spec()`/`mlfactory run` for standard experiments; use a documented wrapper for legacy jobs. | Process does not become healthy or registry state is wrong. |
| 6. Administer | Is it progressing and healthy? | Watch dashboard, log, process, registry, service health, GPU thermals/power, disk, and output growth. | Hang, OOM, repeated failures, unsafe temperature, or resource conflict. |
| 7. Stop/resume | How do we interrupt safely? | Prefer graceful termination; verify checkpoints and status; resume only if the workload explicitly supports it. | Resume semantics are unknown or artifacts are inconsistent. |
| 8. Finalize | Is the result real and complete? | Verify terminal status, hashes, outputs, metrics, logs, process exit, resource restoration, and registry consistency. | Any missing artifact, stale process, or unreconciled resource state. |

## 1. Scope and repository discovery

1. Read the nearest `AGENTS.md` files before operating the run. Follow repository-specific instructions over this document.
2. Locate the experiment directory, spec, plugin, entrypoint, and dashboard config:

```bash
find mlfactory/experiments -path '*<domain>*' -maxdepth 4 -type f | sort
rg -n 'stage:|experiment:|input_run:|model:|epochs:|steps:|batch|count|output' <spec> <experiment-dir>
```

3. For a standard mlfactory stage, confirm the plugin is imported by `mlfactory/core/runner.py`. If it is not registered, stop and fix registration before running.
4. For a legacy or standalone script, do not pretend it is registry-managed. Define an output directory, log, PID file, progress marker, and dashboard before launch; preserve the original input data.
5. Identify parent runs and establish lineage from the spec or request. Do not guess parent IDs.

## 2. Preflight checklist

Run cheap checks before allocating GPUs or starting a long process:

```bash
python3 - <<'PY'
from pathlib import Path
import yaml
p = Path('<spec>')
print(yaml.safe_load(p.read_text()))
PY

python3 -m py_compile <entrypoint-or-changed-files>

git status --short
nvidia-smi
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
```

Also check:

- input files exist and have plausible line/record counts;
- model aliases resolve and the intended local/remote endpoint is healthy;
- no mutually exclusive local model services are running together;
- enough disk remains for checkpoints, logs, and source snapshots;
- secrets are referenced by name and loaded through the repository secret mechanism;
- the working tree state is understood. Do not commit or discard unrelated changes;
- a pilot or smoke test exists when the main job is uncertain.

For a run that could touch shared services or all GPUs, identify the owning process first. Never lower power, stop a service, or kill a process that belongs to an unrelated run.

## 3. Duration and GPU power policy

The default long-run policy is:

```text
LONG_RUN_THRESHOLD = 2 hours
LONG_RUN_POWER_LIMIT = 340 W
RESTORE = each GPU's exact pre-run power limit
```

The policy is configurable per host/run. It must not be hardcoded to an experiment.

### Estimate

Prefer, in order:

1. measured wall time from a representative pilot;
2. recent completed records from the same workload;
3. prior runs of the same stage/model/hardware;
4. `item_count × conservative seconds_per_item`, including retry/checkpoint overhead.

Estimate the **remaining** work when resuming. If the estimate is above two hours, classify the run as extended even if the first request is currently fast.

Record the estimate and method in the launcher log or run metadata, for example:

```text
estimate: 6.4h remaining; basis: median of last 20 items (31.0s/item), 742 items
```

### Lower and restore limits

Only do this when the run owns the GPUs and the host policy permits it. Capture the per-device state first:

```bash
nvidia-smi --query-gpu=index,power.limit --format=csv,noheader,nounits
```

For each GPU used by the run:

```bash
sudo nvidia-smi -i <index> -pl 340
```

The run wrapper must restore each captured value in a `finally`/cleanup path, including graceful interruption and ordinary exceptions. Do not restore all devices to one guessed value when their original limits differ.

For detached automation, use passwordless sudo for the narrowly required `nvidia-smi` command or an explicitly injected secret such as `MLFACTORY_GPU_SUDO_PASSWORD`/host-equivalent. Never put a password in source, a spec, a committed shell command, or a report. If lowering cannot be guaranteed and the policy requires it, fail preflight rather than silently running at the wrong level.

A separate watchdog may be used for a manually started run, but it must:

- save the original limits;
- identify the exact worker PID;
- restore on normal exit and graceful stop;
- log restoration failure for manual intervention;
- never restore based only on a broad process-name match.

## 4. Prepare the run

### Standard mlfactory run

Use the API or CLI, not an experiment module directly:

```python
from pathlib import Path
from mlfactory.core.registry import Registry
from mlfactory.core.runner import run_from_spec

manifest = run_from_spec(
    spec_path=Path('<spec>'),
    registry=Registry('.mlfactory/registry.db'),
    run_id='<explicit-run-id-if-needed>',
    parent_runs=['<parent-id>'],
)
print(manifest.run_id, manifest.status)
```

Before execution, use `create_run()` when you need to inspect the generated manifest/source snapshot without running the plugin. Do not call `create_run()` and then call `run_from_spec()` with the same run ID: `run_from_spec()` creates its own run directory.

For a synchronous interactive run:

```bash
mlfactory run <spec> --run-id <run-id>
```

For a detached run, record the command and PID in a durable launcher location. Use an explicit run ID so the dashboard can be attached immediately:

```bash
mkdir -p .mlfactory/launchers
nohup mlfactory run <spec> --run-id <run-id> \
  > .mlfactory/launchers/<run-id>.log 2>&1 &
echo $! > .mlfactory/launchers/<run-id>.pid
```

The exact command, environment, start time, estimate, and resource policy belong in the launcher log. If power management is implemented by a wrapper, the wrapper owns cleanup and must not daemonize without a reliable restoration path.

### Legacy/standalone run

Use a thin wrapper rather than editing the experiment logic. The wrapper should:

- create a unique run directory;
- copy or hash the input specification;
- write `run.log`, `run.pid`, and `progress.json`;
- emit append-only output/checkpoint records;
- trap `SIGTERM`/`SIGINT` for graceful cleanup;
- lower/restore GPU power when required;
- expose a generic dashboard or a stage-specific dashboard;
- write a terminal marker such as `completed`, `failed`, or `aborted`.

## 5. Dashboard requirement

Any run estimated above two hours, any overnight run, and any multi-stage pipeline must have a dashboard before or immediately after launch.

Prefer, in order:

1. `mlfactory dashboard --watch-run <run-id>` with the experiment's `dashboard.json` or `dashboard_<stage>.json`;
2. a generic registry dashboard if no custom config exists;
3. a standalone Rich dashboard for a legacy run.

At minimum expose:

- completed versus planned work and ETA, including an absolute estimated completion time;
- process/worker health;
- latest step/item and last error;
- recent log tail;
- GPU utilization, memory, temperature, and power draw;
- server/service health when applicable;
- one-minute-smoothed generated tokens/sec for model runs, clearly scoped as run-specific or server-wide;
- disk usage;
- checkpoint/output existence.

The dashboard is read-only. It must not be the only control path for stopping or changing a run.

### Mandatory user handoff

Once the run is fully configured and launched, immediately give the user a single copy-paste dashboard command—do not make them infer it from paths or logs:

```text
Dashboard: `mlfactory dashboard --watch-run <run-id>`
```

For a standalone run, provide the exact launcher instead:

```text
Dashboard: `python3 <path-to-standalone-dashboard.py>`
```

If the dashboard is already running in another terminal, say so and still provide the command or location to reconnect. Include the run ID, dashboard path/command, and whether it is registry-backed or standalone.

## 6. Administration loop

During an active run, check the following rather than relying on process existence alone:

```bash
mlfactory ls --status running
mlfactory show <run-id>
ps -p "$(cat .mlfactory/launchers/<run-id>.pid)" -o pid,etime,stat,cmd
 tail -40 .mlfactory/launchers/<run-id>.log
nvidia-smi
```

Use the dashboard for repeated polling. Investigate when:

- output/metrics stop growing for longer than the expected item time;
- GPU utilization is zero while the worker claims to be active;
- memory approaches capacity, temperatures become unsafe, or power behavior is unexpected;
- the local model/service health endpoint fails;
- retries, OOMs, or malformed outputs accumulate;
- disk usage crosses the configured warning threshold;
- the registry says `running` but the PID is gone.

Do not reduce batch size or change kernels as a first reaction. Establish whether the process is actually alive, blocked on I/O/service startup, duplicated, or using the intended software path.

## 7. Stop, failure, and resume procedure

1. Stop gracefully through the launcher PID or supervisor:

```bash
kill -TERM "$(cat .mlfactory/launchers/<run-id>.pid)"
```

2. Wait and inspect logs, manifest, checkpoints, and GPU processes. Use `SIGKILL` only when the process is unresponsive or unsafe.
3. Confirm the cleanup path restored GPU limits and stopped disposable services.
4. If the process died without finalizing, mark the run `aborted` only after verifying it is not still running:

```python
from mlfactory.core.registry import Registry
Registry('.mlfactory/registry.db').update_status('<run-id>', 'aborted')
```

5. Never assume a generic run can resume. Read its plugin/script contract and inspect checkpoint compatibility, input ordering, and output atomicity. If it cannot resume safely, start a new run with a new ID and link the prior run as an ancestor or record it in the run notes.
6. Never overwrite partial outputs to make a run look complete. Preserve them and report the failure boundary.

## 8. Completion and handoff

A run is complete only when all are true:

- worker and child processes are gone or intentionally handed off;
- registry and `manifest.json` agree on terminal status;
- expected artifacts exist and are hashed/registered;
- output counts and metrics meet the requested completion criterion;
- no unreported errors remain in logs;
- checkpoints and summaries are readable;
- GPU power limits are restored exactly, or a restoration failure is prominently reported;
- disposable model/server contexts are stopped when no longer needed;
- the final dashboard snapshot/status is captured or the dashboard command is supplied;
- parent/child lineage is present for pipelines.

Report in this format:

```text
status: completed | failed | aborted | guarded
run_id: <id>
spec: <path>
worker: exited | PID <pid> | supervisor <name>
progress: <completed>/<planned>
artifacts: <key paths>
manifest: <path>
registry: <db path>
dashboard: <command/path>
gpu policy: <not applicable | lowered to X W | restored to [per-device values] | restoration failed>
notes: <errors, guard decision, resume instructions>
```

## Hard rules

- Do not run a registered experiment by directly invoking its internal script.
- Do not start an extended run without a live dashboard or a documented reason it is impossible.
- Do not hardcode credentials, API keys, or sudo passwords.
- Do not change models, services, GPU allocation, or power limits without checking ownership and recording the change.
- Do not call a run successful because its process exited zero; verify manifest, artifacts, counts, and hashes.
- Do not claim resume support without reading the implementation.
- Do not delete or overwrite partial outputs, source snapshots, or manifests during recovery.
