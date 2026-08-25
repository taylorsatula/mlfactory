# Vast.ai remote training operations

> Update when: the Vast provisioning/ops workflow changes. Source of truth
> for the Vast runner API: `mlfactory/remote/vast.py` (see
> `mlfactory/docs/REMOTE_AND_SECRETS.md`). This doc covers the *operational
> discipline* of running a training job on a rented GPU box — what you're
> renting, how to set it up, how to run safely, how to get artifacts home.
> GPU memory, the smoke ladder, OOM, and objective safety live in
> `TRAINING_STACK.md`; the general debugging method in `DEBUGGING_METHOD.md`.

## Mental model: what you are actually renting

A Vast.ai instance is a remote container on somebody else's GPU host. Four
separate things to keep straight:

1. **Local machine** — where the durable copy of code and results should live.
2. **Vast host** — the physical machine with the GPUs.
3. **Container filesystem** — the environment visible after SSH. May disappear.
4. **Workspace volume** — optional persistent storage mounted into the container.

`/workspace` is only persistent when Vast says it is backed by a volume —
its name alone does not make it durable. Check on a Vast base image:

```bash
vast-capabilities | jq '.instance.workspace_is_volume'
```
- `true`: `/workspace` is backed by a persistent host volume.
- `false`: recycling or destroying the instance erases it.

A normal **stop/start** usually preserves the container filesystem. A
**recycle** or **destroy** may erase it. **Treat all remote storage as
temporary until an off-box checksum matches.**

Commands marked **LOCAL** run from this machine; **REMOTE** run after SSHing
in. Do not paste local paths into the remote shell or vice versa.

## Choosing a Vast offer

### GPU architecture

The newest GPU is not automatically the easiest training GPU. For ordinary
dense transformers, B200 may be excellent. For hybrid linear-attention
stacks (Qwen3.5's Gated DeltaNet), the surrounding Triton/FLA kernels may
be much more mature on Hopper. **Architecture compatibility beats
advertised VRAM** — see `TRAINING_STACK.md` for the failure mode.

Do not conclude a newer GPU is safe because the model loads, a forward pass
works, it has more VRAM, or another standard-attention model trains on it.
A tiny real backward-pass smoke must have already proven the exact package
versions (see `TRAINING_STACK.md` §smoke-ladder).

### Number and size of GPUs

For the known 4B configuration: 2× H100 80 GB, prefer NVLink/SXM when
price/availability are sensible. Two GPUs are useful even when the trainable
model fits on one — the second can hold a frozen reference, reward model,
embedder, or verifier. For a different experiment, make a memory inventory
first (`TRAINING_STACK.md` §where-memory-goes); don't assume parameter
counts give peak memory.

### Host RAM

Model loading and quantization can temporarily use far more CPU RAM than
steady-state GPU training. Guidelines: 4B/8B ≥ 64 GB; 27B QLoRA ≥ 128 GB;
very large loading/optimizer offload/multiple models → 256 GB may be
prudent. `bitsandbytes` stages high-precision weights in CPU RAM before
quantization — a 27B model can briefly need ~60+ GB just for staging.

### Disk

Disk must cover more than the final model: base + reference + embedding/reward
model caches, venv + compiled extensions, datasets, checkpoints, failed-run
archives, temp files. Recommendations: 4B + 8B reward/embed ≥ 200 GB
minimum, 300 GB comfortable; 27B + multiple checkpoints 400–600 GB.

### Host reliability and networking

Prefer offers with high reliability, good download bandwidth, adequate disk
throughput, sufficient contract duration or on-demand availability, and SSH
through a mapped port. A cheaper unstable host can cost more after repeated
setup and interrupted runs.

### Image choice

A minimal CUDA/PyTorch image without a preloaded inference model is
simplest. The proven instance used Vast's `llama.cpp` CUDA 12.9 image — it
worked, but it started a preinstalled llama-server that occupied ~19 GB on
**each** H100; stopping it is mandatory before training. **Do not install
an NVIDIA driver inside the container** — the host injects the driver;
installing `cuda`, `cuda-drivers`, `nvidia-driver-*`, or replacement
`libcuda*` packages can break the container's connection to the host kernel
driver.

## Connect from the local machine

Set temporary shell variables so every command uses one source of truth.

```bash
# LOCAL
export VAST_HOST='REPLACE_WITH_PUBLIC_IP'
export VAST_PORT='REPLACE_WITH_SSH_PORT'
export VAST_KEY="$HOME/.ssh/id_vast"
ssh -i "$VAST_KEY" -p "$VAST_PORT" root@"$VAST_HOST"
# First connection only, if needed:
ssh -o StrictHostKeyChecking=accept-new -i "$VAST_KEY" -p "$VAST_PORT" root@"$VAST_HOST"
```

Avoid disabling host-key checking globally. If Vast reassigns the same
IP/port and SSH warns the host key changed, remove only the stale entry:
`ssh-keygen -R "[$VAST_HOST]:$VAST_PORT"`.

Read the image's own operating guide before changing services:
```bash
# REMOTE
[ -f /etc/vast-agents-guide.md ] && less /etc/vast-agents-guide.md
```
It tells you about the container's service manager, persistence, CUDA
inventory, and image-specific daemons.

## Inventory the instance before installing anything

```bash
# REMOTE
date; uname -a; python3 --version; free -h
df -h / /workspace
nvidia-smi; nvidia-smi topo -m
nvcc --version || true
vast-capabilities | jq '{image, workspace_is_volume: .instance.workspace_is_volume, cuda: .hardware.gpu.cuda, services}'
```
Record results in experiment notes — future debugging is much easier when
you know which driver/toolkit/image/architecture produced a result.

## Clear hidden GPU consumers first

Before interpreting any OOM, inspect GPU processes:

```bash
# REMOTE
nvidia-smi
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
ps aux --sort=-rss | head -20
supervisorctl status 2>/dev/null || true
# On the proven Vast llama.cpp image:
supervisorctl stop llama
# Verify both GPUs near 0 MiB before training:
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader
```

A preinstalled inference process can consume 10–20 GB per GPU and make a
configuration that otherwise fits appear to OOM. **Do not run inference and
training simultaneously unless planned.** A local API server, notebook
kernel, forgotten test process, or stale distributed worker can consume
VRAM. If the experiment needs separate inference/training phases, make the
service transition explicit.

## Copy the project to the instance

Do not copy a virtual environment between machines — compiled wheels,
interpreter paths, and CUDA extensions are machine-specific.

```bash
# LOCAL
PROJECT="$HOME/my-experiment"
REMOTE_PROJECT='/workspace/my-experiment'
rsync -az --info=progress2 \
  --exclude '.venv' --exclude '.venv312' --exclude '__pycache__' --exclude '.git/objects' \
  -e "ssh -i $VAST_KEY -p $VAST_PORT" \
  "$PROJECT/" root@"$VAST_HOST":"$REMOTE_PROJECT/"
# Verify remotely:
cd /workspace/my-experiment && find . -maxdepth 2 -type f | sort | head -100
```
If Git history matters, don't exclude `.git/objects`. **Never transfer
secrets into the repository** — keep API/HF tokens outside tracked files.

## Configure HF cache and authentication

Use one explicit cache location so model downloads don't scatter.

```bash
# REMOTE
mkdir -p /workspace/.hf_home && chmod 700 /workspace/.hf_home
export HF_HOME=/workspace/.hf_home
source /workspace/my-experiment/.venv/bin/activate
hf auth login   # interactive
# Or place a token in a root-readable file without printing it into logs:
install -m 600 /dev/null /workspace/.hf_home/token   # edit-and-paste, don't echo
# Launch scripts then use:
export HF_HOME=/workspace/.hf_home
export HF_TOKEN="$(cat /workspace/.hf_home/token)"
```
Before a long run, verify access to every gated model.

## CUDA and PyTorch compatibility (without folklore)

**Driver vs toolkit:** the NVIDIA driver comes from the host; the CUDA
toolkit and PyTorch wheel live in the container/venv. Exact minor versions
don't always have to match — a host driver supporting CUDA 13 can run many
CUDA 12.x wheels. Do not reinstall the host driver to make `12.9` visually
match.

**Architecture matters:** a wheel can install successfully but lack kernels
for a new GPU architecture. This often manifests only on the first CUDA
operation: `no kernel image is available for execution on the device`.
```bash
python - <<'PY'
import torch
print(torch.__version__, torch.version.cuda)
print(torch.cuda.get_arch_list())
print(torch.cuda.get_device_capability())
PY
```

**PTX JIT caveat:** code shipped as PTX from a toolkit newer than the host
driver's JIT compiler may fail with `CUDA_ERROR_UNSUPPORTED_PTX_VERSION`.
Prefer a known-compatible wheel or native-cubin extension build. Do not
begin by replacing the host driver from inside the container.

## Build experiments as reproducible launch packages

Every experiment should contain at least:
```
project/
├── README.md
├── requirements.txt or lock/freeze file
├── train.py
├── test_train.py
├── launch_smoke.sh
├── launch_validation.sh
├── launch_full.sh
├── data or data manifest
└── outputs/  (not committed)
```

A launch script should: use `set -euo pipefail`; `cd` to an absolute project
dir; activate the exact venv; set cache + allocator variables **before**
Python starts; write to a new descriptive output dir; include all important
hyperparameters explicitly; avoid embedding tokens/passwords; run Python in
the foreground so Supervisor owns it. Template:

```bash
#!/bin/bash
set -euo pipefail
cd /workspace/my-experiment
source .venv/bin/activate
export HF_HOME=/workspace/.hf_home
export HF_TOKEN="$(cat /workspace/.hf_home/token)"
export PYTORCH_ALLOC_CONF='expandable_segments:True,roundup_power2_divisions:[32:256,64:128,256:64,>:32]'
export PYTORCH_CUDA_ALLOC_CONF="$PYTORCH_ALLOC_CONF"
export TRITON_DISABLE_AUTOTUNING=1
exec python train.py --model-name exact/model-id --out-dir outputs/run_name --seed 42
```
Use `exec` so signals from Supervisor reach Python directly. At startup the
program should save: parsed args, model IDs/revisions, package versions,
seed, data manifest/hash, git commit/source checksum, GPU names/count,
torch/CUDA versions, start timestamp. Without this, a checkpoint may be
impossible to interpret later.

## Run long jobs under Supervisor

SSH sessions disconnect; shell background jobs are easy to lose.

```ini
# REMOTE: /etc/supervisor/conf.d/my-training.conf
[program:my-training]
command=/workspace/my-experiment/launch_full.sh
directory=/workspace/my-experiment
autostart=false
autorestart=false
startsecs=1
stopasgroup=true
killasgroup=true
stdout_logfile=/tmp/my_training.log
stderr_logfile=/tmp/my_training.err
stdout_logfile_maxbytes=0
stderr_logfile_maxbytes=0
environment=PROC_NAME="my-training"
```
Why: `autostart=false` — rebooting the container doesn't unexpectedly
consume credits. `autorestart=false` — a bad objective or OOM doesn't loop
forever. `stopasgroup=true`/`killasgroup=true` — stopping Supervisor kills
child workers too. Unlimited log size is fine for a short experiment; use
rotation for very long jobs.

```bash
supervisorctl reread && supervisorctl update
supervisorctl start my-training
supervisorctl status my-training
supervisorctl stop my-training   # if needed
```

**Interpret states correctly:** `RUNNING` — process exists (may still be
hung/unhealthy). `STOPPED` — intentionally not running. `EXITED` — process
ended (expected for a completed one-shot with `autorestart=false`; inspect
exit logs and final artifacts). `FATAL` — repeated startup failure or bad
config. Completion requires: expected final log message, exit without
traceback, final checkpoint/summary present, metrics file parseable,
artifacts copied off-box.

Use distinct names/logs for smoke vs full (`myexp-smoke`,
`myexp-validation`, `myexp-full`). Do not overwrite forensic logs from a
failed run with a new launch.

## Monitoring without disturbing training

Safe monitoring does not call the model — it reads process, logs, device
counters.

```bash
supervisorctl status my-training
tail -50 /tmp/my_training.log /tmp/my_training.err
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu,power.draw,temperature.gpu --format=csv,noheader
ps -eo pid,ppid,stat,etime,%cpu,%mem,cmd --sort=-%cpu | head -30
df -h /workspace
du -sh /workspace/my-experiment/outputs/* 2>/dev/null
```

Healthy often looks like: step logs at consistent intervals after warmup;
peak reserved memory stabilizes; GPU utilization alternates appropriately
when work is split between GPUs; loss/reward diagnostics remain finite;
output length and entropy don't run away; checkpoints/evals appear on
schedule. One GPU at 0% in a single snapshot can be normal (the other is
embedding/scoring); **repeated 0% on every GPU with no new logs is not
normal.** Do not monitor by generating extra samples from the training
model — parallel inference can consume VRAM, change caches, compete for
kernels, or deadlock a server. Use saved periodic eval outputs or pause
training deliberately.

## Multi-GPU placement

Two GPUs do not automatically combine into one pool. Decide explicitly
which device owns each component (e.g. GPU 0: policy + LoRA + backward;
GPU 1: frozen reference + embedder/reward). **Avoid copying full logits
between GPUs** — a `[batch, time, vocab]` tensor can be gigabytes; compute
selected token log-probs on the source GPU and transfer only
`[batch, response_time]`. After loading each model, verify placement:
`print(next(p).device for ...)` and inspect `hf_device_map`. Pipeline
utilization can alternate (GPU 0 idle while GPU 1 embeds; GPU 1 idle during
policy backward) — a single snapshot isn't enough to diagnose underuse.

## Preserve artifacts before rerunning or destroying

**Archive failed runs; do not overwrite them.** A failed checkpoint can
reveal exactly when collapse began. `mv outputs/current outputs/failed_<sig>`
and preserve checkpoints, stdout/stderr, per-step metrics, eval metrics,
parsed config, source code, environment freeze, failure note.

**Sync home with resumable rsync:**
```bash
# LOCAL
mkdir -p "$HOME/experiment-archives/run_name"
rsync -az --partial --info=progress2 \
  -e "ssh -i $VAST_KEY -p $VAST_PORT" \
  root@"$VAST_HOST":/workspace/my-experiment/outputs/run_name/ \
  "$HOME/experiment-archives/run_name/"
# Copy logs outside the output dir too:
scp -i "$VAST_KEY" -P "$VAST_PORT" root@"$VAST_HOST":/tmp/my_training.{log,err} "$HOME/experiment-archives/run_name/"
```

**Verify checksums:**
```bash
# REMOTE
sha256sum /workspace/my-experiment/outputs/run_name/checkpoint-*/adapter_model.safetensors
# LOCAL
sha256sum "$HOME"/experiment-archives/run_name/checkpoint-*/adapter_model.safetensors
# Manifest:
find "$HOME/experiment-archives/run_name" -type f -print0 | sort -z | xargs -0 sha256sum > "$HOME/experiment-archives/run_name/SHA256SUMS"
```
File sizes alone are not sufficient.

**Do not destroy immediately after transfer.** Order: stop training → sync
artifacts → verify checksums → open/parse summary + metrics locally →
confirm final adapter exists → stop the instance (halts GPU charges) →
destroy only after the local archive is proven complete. Storage charges
may continue while stopped, but that's cheaper than discovering a missing
checkpoint after destruction.

## Starting a fresh run after a failed run

1. Stop the Supervisor service. 2. Verify no worker children remain.
3. Copy failed artifacts home. 4. Compare checksums. 5. Rename the remote
output directory descriptively. 6. Fix code and add a regression test for
the discovered failure. 7. Sync corrected code to the instance. 8. Run
pure unit/syntax tests both locally and remotely. 9. Run a three-step
real-model smoke from the pristine base. 10. Run a guarded 10–20-step
validation from the pristine base. 11. Only then start the full run.

**Do not continue from a checkpoint trained under a broken objective.**
It confounds the corrected experiment and may preserve hidden damage even
when sample outputs still look normal.

## Teardown checklist

```text
[ ] Training service stopped or completed
[ ] Final checkpoint exists
[ ] Intermediate best checkpoint exists
[ ] Metrics and summary copied locally
[ ] stdout/stderr copied locally
[ ] Launch scripts copied locally
[ ] Exact training source copied locally
[ ] Environment freeze copied locally
[ ] Data manifest/hash copied locally
[ ] Remote and local checkpoint SHA-256 match
[ ] Local summary JSON parses
[ ] Important adapter can be loaded locally or on another test machine
[ ] Vast instance stopped before eventual destruction
```
If any box is unchecked, the experiment is not safely preserved.
