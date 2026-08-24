# General-Purpose Run Administration Design

## Purpose

Make long-running experiments and jobs easy to start, monitor, stop, resume, verify, and hand off without putting experiment-specific administration logic into individual domains.

ACE is one intended consumer, but this design applies equally to DFT, voice, causal-graph, remote jobs, and future experiments.

## Design principles

- **Core owns administration; plugins own computation.**
- **Optional capabilities stay optional.** A run may use no GPU, model server, checkpoint, or external service.
- **Run state is durable and inspectable.** A detached process must not be the only record of progress.
- **Resume is identity-based.** Do not rely on raw output line counts as a universal checkpoint.
- **Resource ownership is explicit.** Never stop or reconfigure an unrelated process or service.
- **Artifacts belong to the run directory.** The registry stores metadata, lineage, and metrics; large outputs remain files.
- **Every run is reproducible and verifiable.** Record resolved inputs, configuration, environment, resources, and final artifact hashes.

## Proposed core components

### `RunController`

Add a reusable controller in `mlfactory/core/run_control.py`:

```python
from mlfactory.core.run_control import RunController

controller = RunController(registry)
run = controller.start(spec_path)
controller.status(run.run_id)
controller.stop(run.run_id)
controller.resume(run.run_id)
```

The controller owns:

- lifecycle state transitions;
- PID, heartbeat, and lease management;
- graceful stop and signal handling;
- checkpoint coordination;
- resource and service ownership;
- finalization and verification.

Plugins should implement the workload, not duplicate this process-management logic.

### Managed plugin context

Extend the plugin contract so plugins receive a managed context:

```python
class MyPlugin(StagePlugin):
    def execute(self, ctx):
        ...

    def resume(self, ctx, checkpoint):
        ...
```

The context should expose:

- `run_dir`;
- the registry;
- structured events and metrics;
- a checkpoint store;
- an artifact manager;
- a resource manager;
- a service manager.

Existing `prepare()`, `execute()`, and `finalize()` methods should remain compatible during migration.

### Checkpoints and resume

Add `mlfactory/core/checkpoints.py` with a common checkpoint abstraction supporting:

- atomic writes;
- schema and input/spec hash validation;
- completed-item identities;
- retrying failed items;
- interruption-safe append-only outputs.

A checkpoint should contain information such as:

```json
{
  "schema_version": 1,
  "run_id": "...",
  "input_set_hash": "...",
  "completed": 217,
  "successful": 214,
  "failed": 3,
  "last_completed_id": "...",
  "updated_at": "...",
  "attempt": 1
}
```

Resuming should:

1. validate the materialized input plan;
2. validate input and spec hashes;
3. scan outputs for malformed or duplicate records;
4. skip completed identities;
5. continue from the first incomplete item;
6. update the checkpoint atomically after each committed result.

A failed item should normally produce a durable error record, so resuming does not silently repeat requests. A separate retry operation can create a new attempt or continuation run for failed items only.

### Resources

Add `mlfactory/core/resources.py` for optional resource policies, including:

- GPU discovery and locking;
- CPU and memory limits;
- GPU power policies;
- resource acquisition and release;
- exact restoration after failure or interruption.

GPU power limiting is one policy implementation, not an ACE-specific behavior and not a silent default. A policy must capture the original per-device state before changing anything and must restore it in all cleanup paths.

### Services

Add `mlfactory/core/services.py` with a generic service interface supporting:

- no service;
- externally managed services;
- subprocess-owned services;
- systemd or Docker-backed services.

Every service should record ownership, PID, command/configuration, health checks, and shutdown behavior. A service manager must never kill an arbitrary process merely because it occupies a port.

## Run layout and registry model

A managed run should use a layout like:

```text
runs/<run_id>/
  manifest.json
  spec.resolved.json
  inputs/
  artifacts/
    plan.json
    checkpoint.json
    summary.json
    service.json
    resource_policy.json
  logs/
    run.log
    service.log
  events.jsonl
```

The registry should track:

- lifecycle state: `pending`, `running`, `paused`, `completed`, `failed`, or `aborted`;
- attempts and heartbeats;
- checkpoint state;
- resource claims;
- service metadata;
- typed lineage;
- metrics and artifact metadata.

Large outputs should remain in the run directory rather than being copied into SQLite.

Run identity should be immutable. If a configuration or input changes, create a new run linked to the previous one rather than mutating the old run. A resumable run may have multiple execution attempts while retaining its logical run identity.

## Generic CLI

The administration surface should work for every experiment:

```bash
mlfactory run <spec>
mlfactory status <run_id>
mlfactory logs <run_id>
mlfactory dashboard --watch-run <run_id>
mlfactory stop <run_id>
mlfactory resume <run_id>
mlfactory verify <run_id>
```

Useful future additions include:

```bash
mlfactory retry <run_id> --failed-only
mlfactory checkpoint <run_id>
mlfactory artifacts <run_id>
mlfactory export <run_id> --output <path>
```

## Dashboard

The generic dashboard should read from the manifest, registry, events, and run directory. It should provide:

- status and completed/planned work;
- current item and last success/failure;
- retry counts, latency, throughput, and ETA;
- worker and service health;
- GPU/CPU/memory/disk status when relevant;
- checkpoint age and artifact validation state;
- resource policy and restoration state;
- parent and child lineage.

Experiment-specific dashboards should become declarative configurations or optional probes on top of the generic dashboard, rather than copied standalone programs.

## Spec shape

The common spec should provide optional execution, resource, service, and resume sections without imposing them on every experiment:

```yaml
stage: <plugin stage>
name: <run name>
experiment: <domain>

execution:
  timeout_seconds: 900
  retries: 3

resume:
  enabled: true
  checkpoint_interval: 1

resources:
  gpus: [0, 1]
  power_policy: observe

service:
  ownership: external  # none | external | managed
  base_url: http://127.0.0.1:3090
  health_url: http://127.0.0.1:3090/health
```

Domain plugins add only fields specific to their workload.

## What should not go into core

- ACE-specific stages or data paths;
- model names, ports, or prompt schemas tied to one experiment;
- a universal GPU requirement;
- a universal service requirement;
- hard-coded power limits;
- one-off dashboard implementations;
- line-count-based resume semantics;
- credentials in source, specs, manifests, or command logs.

ACE would become a normal plugin consumer: it would define prompt preparation, request logic, and its trace schema, while mlfactory would provide lifecycle, checkpointing, monitoring, resource ownership, provenance, and verification.

## Migration plan

1. Add lifecycle states, heartbeats, durable run metadata, and generic status commands.
2. Add checkpoint and resume interfaces while preserving existing plugins.
3. Add resource and service managers with explicit ownership.
4. Move dashboard data sources to registry/run-directory metadata.
5. Migrate the sample experiment first as the reference implementation.
6. Migrate ACE using an ACE-specific plugin and specs, without adding ACE logic to core.
7. Migrate DFT, voice, remote runs, and other domains incrementally.

## Acceptance criteria

A run-administration implementation is ready when it can:

- launch a detached run with a durable identity;
- show live progress without reading an experiment-specific hard-coded path;
- stop gracefully and record the terminal state;
- resume after interruption without duplicate or silently skipped work;
- verify input and output integrity;
- report worker, service, and resource ownership;
- restore changed resources exactly;
- preserve lineage and final artifact hashes;
- support runs with no GPU, no service, and no checkpoint as well as complex jobs that use all three.
