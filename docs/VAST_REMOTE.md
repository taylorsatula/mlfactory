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

## Choosing a Vast offer — agent-driven search, never the Web UI picker

> **Directive:** the agent selects the server with structured
> `vastai search offers` sweeps over the full market. Do not ask the
> principal to pick from the Web UI, and do not pick from it yourself:
> the picker shows one page of offers with no per-GPU math, no
> architecture provenance, and no workload-fit reasoning. The Web UI is
> a sanity check that a selected offer exists, not a selection tool.
> Measured cost of the old habit: a 2× H200 box rented at ~$9/hr ran at
> ~30% utilization on a workload a single $0.43/hr RTX A6000 serves
> (§measured reference).

### Step 1 — derive the minimum from the workload, not from the last phase

Three independent questions; each can shrink the box by an order of
magnitude:

- **Memory inventory — training and generation peaks differ.**
  Gradient-checkpointed full-trace replay (GRPO) needed the 140 GB card;
  generation-only regimes (rollout collection, R4 forks: weights +
  batch KV + activations + hooks) fit a 40–48 GB card. Do the inventory
  for the phase being rented (`TRAINING_STACK.md` §where-memory-goes);
  never carry the previous phase's box forward by reflex.
- **Throughput model — bandwidth-bound or latency-bound?** Don't pay
  for bandwidth the workload doesn't use. Measured: HF decode at
  batch ≤ 4 is latency-bound — ~37 tok/s per stream on a 3090
  (936 GB/s) and ~37.5 tok/s per stream on an H200 (4,800 GB/s). 5×
  the bandwidth bought nothing; that is the ~30% utilization. For
  latency-bound serving, pick for $/hr and VRAM, not GB/s.
- **Architecture provenance.** Proven torch/FLA classes: sm_86 Ampere
  (local 3090 stack), sm_90 Hopper (GRPO H200 box). sm_89 Ada:
  flash-attn/Triton mature but unproven with this repo's FLA kernels —
  rentable only behind a shakedown first. sm_120 Blackwell: verified
  for llama.cpp serving only (§fast path), not the torch stack. sm_75
  Turing: excluded (no flash-attn). The standing warning holds inside
  any proven class: a GPU is not safe because the model loads and a
  forward pass works — a tiny real backward-pass smoke must have proven
  the exact package versions before a training run (`TRAINING_STACK.md`
  §smoke-ladder).

### Step 2 — sweep the market with the CLI

Query-field gotchas (each one has already cost a wasted search):

- `gpu_ram` is in **GB in queries**, **MB in `--raw` output**, and is
  the **total across all of the machine's GPUs** in both. Per-GPU VRAM
  is `gpu_ram / num_gpus` — post-filter on that, never on the raw
  field. A 2×24 GB box passes `gpu_ram>=40`.
- `gpu_name` takes underscores: `gpu_name=RTX_4090`.
- Default query filters `external=false rentable=true verified=true`;
  `-n` disables, `--type bid` searches interruptible pricing.
- Always sweep with `--raw` (parse it), `--limit 500` (one page is not
  the market), `--storage <GB>` (prices in the storage allocation),
  `-o dph_total`.

Canonical sweep (≥40 GB total VRAM; adjust the floor per Step 1):

```bash
vastai search offers 'gpu_ram>=40 rentable=true' -o 'dph_total' \
  --storage 120 --limit 500 --raw
```

Post-filter: per-GPU VRAM ≥ requirement; `compute_cap` in the
proven/acceptable set (Step 1); `reliability >= 0.98`; `disk_space` ≥
allocation + headroom; `cpu_ram` ≥ 32 GB (loading/quantization staging
spikes — `bitsandbytes` can briefly need 60+ GB for large models);
`inet_down` adequate for the model pull (prefer ≥ 500 Mbps for an 18 GB
checkpoint even with `hf_xet`, or price the download time).

### Step 3 — interpret the specs

- `compute_cap`: 750 Turing (exclude) · 800/860 Ampere (proven) ·
  890 Ada (shakedown first) · 900 Hopper (proven) · 1200 Blackwell
  (llama.cpp only).
- `dph_total_adj`: $/hr **including the requested storage** — rank on
  this, not `dph_total`.
- `gpu_ids`: count physical GPUs when a listing looks wrong. The market
  carries real modded cards (48 GB RTX 4090 — sm_89, one physical GPU,
  common on CN hosts). Verify with `gpu_ids`/`num_gpus` before
  dismissing or trusting a listing; a strange VRAM figure is a fact to
  check, not a filter artifact.
- `gpu_mem_bw`: only ranks offers for bandwidth-bound workloads.
- `reliability`/`reliability2`, `geolocation`, `duration` (offer
  expiry): a cheap unstable host costs more after interrupted runs.
- Bid/interruptible offers: cheaper but preemptible (instance →
  `stopped`, storage still bills; resume by raising `--bid_price`).
  Acceptable for resumable collection, not for a run that must finish
  on schedule. If renting a bid offer, always pass `--bid_price` —
  `create instance` defaults to on-demand pricing without it.

### Step 4 — shortlist, shakedown, then rent

Present a shortlist — cheapest ~3 offers per acceptable class with
$/hr (disk-adjusted), arch, per-GPU VRAM, reliability, `inet_down`,
region — and a one-line cost/wall estimate for the planned workload
(tokens ÷ measured throughput; `TERMINAL_FORK_COMPUTE.md` §7 for fork
shapes). The winner gets the smoke ladder before the real run launches:
the shakedown is part of selection, not a step after it.

### Measured reference (2026-08-27, ACE R4 right-sizing)

- Full-market sweep: 172 offers ≥ 40 GB total VRAM; 65 passed
  per-GPU ≥ 40 GB + acceptable arch.
- Cheapest suitable: 1× RTX A6000 48 GB $0.43/hr (sm_86 — same family
  as the locally-proven stack); 48 GB Ada-class from $0.40/hr;
  A100-40 from $0.70/hr; cheapest 2× H200 box $8.45/hr.
- R4 (fork regime, generation-only): ~105 GPU-h on one 48 GB GPU at
  m = 32 → ~$45 on the A6000; two parallel single-GPU boxes halve wall
  at ~$90; vs ~$363 and 1.8 days on 2× H200.
- Superseded box: 2× H200 at ~$9/hr, ~30% utilization — bandwidth
  premium unused (latency-bound decode), memory premium unused (no
  replay in the fork phase). Re-rent H200-class only for a phase whose
  inventory actually needs it.

### Image choice

A minimal CUDA/PyTorch image without a preloaded inference model is
simplest. The proven instance used Vast's `llama.cpp` CUDA 12.9 image — it
worked, but it started a preinstalled llama-server that occupied ~19 GB on
**each** H100; stopping it is mandatory before training. **Do not install
an NVIDIA driver inside the container** — the host injects the driver;
installing `cuda`, `cuda-drivers`, `nvidia-driver-*`, or replacement
`libcuda*` packages can break the container's connection to the host kernel
driver.

**Image tags drift — verify before renting.** The house default
`nvidia/cuda:12.9.0-devel-ubuntu26.04` 404'd (manifest unknown) on a
2026-08-26 host. Verified working then: `vastai/llama-cpp:b10182-cuda-12.9`
(see §fast-path below). If creation fails on the image, destroy and
recreate — `update instance --image` on a still-loading instance 404s.

## Fast path: llama.cpp serving template (collection, not training)

For rollout collection / API-endpoint serving (no backward pass), the
official `vastai/llama-cpp` template is the fastest bootstrap. Measured
2026-08-26 on 2× RTX PRO 6000 Blackwell (sm_120), Qwen3.5-9B:

- **The Hopper kernel-maturity warning (§GPU architecture) applies to the
  torch/FLA training stack, not llama.cpp serving.** The prebuilt binaries
  cover the full set of compute capabilities; BF16-GGUF + draft-mtp served
  ~125 tok/s on Blackwell with zero kernel drama. A collection job on a
  new-architecture box skips the torch smoke ladder entirely.
- **Recipe:** (1) `supervisorctl stop llama` (the template service idles
  without a model; stop it to keep the GPUs clean). (2) Download the GGUF
  with `hf_xet` (~18 GB in ~40 s — HF CDN throttles single-stream curl to
  15–22 MB/s). (3) Launch one `llama-server` per GPU via
  `CUDA_VISIBLE_DEVICES`, ports 3091/3092, `--parallel 1`,
  `--spec-type draft-mtp` for MTP variants. (4) The collector side needs
  only a light venv — `torch==<ver>+cpu`, `transformers`, `numpy` satisfy
  `collect_rollouts_api`'s import chain; no GPU torch required. (5) rsync
  the project and run with `PYTHONPATH=<project>` — no `pip install -e`
  needed.
- **llama-server gotcha (measured):** `--parallel N` partitions
  `--ctx-size` across slots — ctx 32768 with `--parallel 4` gives
  8192-token slots and silently truncates long traces at the slot ceiling
  (collector rows show `truncated=True` at n_new ≈ slot size). Use
  `--parallel 1` for sequential collection; always verify `n_ctx_slot` in
  the startup log. Local /opt builds may need `LD_LIBRARY_PATH` pointing
  at the build's bin dir — read the systemd unit before launching by hand.
- **CLI syntax:** `vastai create instance <offer_id> --image <tag> --disk
  <GB>` (verb before noun); `vastai ssh-url <id>` for the address.

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
**Download with `hf_xet`** (`pip install hf_xet`, then `hf download`):
parallel chunked fetch, measured ~450 MB/s where single-stream curl from
the same HF CDN measured 15–22 MB/s. See `AGENTS.md` §model-downloads
for the standing directive.

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
