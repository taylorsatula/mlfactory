# Environment (`.venv`)

> Update when: the pinned stack changes, or a dependency is added/removed.
> Setup commands only — rationale is kept minimal and dated.

## Creation

Python 3.12 venv created with `uv` (stdlib `venv` is broken for the
uv-managed python3.12 on this host — do not recreate with `python -m venv`):

```bash
cd mlfactory/experiments/ace
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python \
    --index-url https://download.pytorch.org/whl/cu128 "torch==2.11.0+cu128"
uv pip install --python .venv/bin/python \
    "transformers==5.14.1" "accelerate==1.14.0" "safetensors==0.8.0" \
    "numpy==2.5.1" pytest
uv pip install --python .venv/bin/python -e /home/admin/mlfactory
```

## Pinned stack

| Package | Version | Why pinned |
|---|---|---|
| torch | 2.11.0+cu128 | proven by the recon probes |
| transformers | 5.14.1 | proven by the recon probes; layer/hook API surface this code uses |
| accelerate | 1.14.0 | device_map load path |
| safetensors | 0.8.0 | controller checkpoint save/load |
| numpy | 2.5.1 | array work in analysis |
| pytest | (any) | test runner |
| mlfactory | editable (`-e /home/admin/mlfactory`) | package-qualified imports resolve from anywhere |

## Deliberately NOT installed

| Package | Reason |
|---|---|
| bitsandbytes | bf16 full-precision loads only |
| flash-linear-attention / causal-conv1d | Installed **on the remote training stack** 2026-08-25 after throughput demanded the revisit; measured **zero gain** (~150 tok/s batch-4 unchanged) — transformers may be silently falling back; see `STATUS.md` Q11. Not installed locally (local venv does no generation at scale) |
| peft | custom controller instead (`core/steering_controller.py`) |
| flash-attn | Installed **on the remote training stack** 2026-08-25 (2.8.3.post1, sm_90 source build — no prebuilt wheel matches torch 2.13); measured **zero gain** vs SDPA at short and long context — the bottleneck is not the full-attention blocks (`STATUS.md` Q11) |

## Remote training stack (Vast H200, 2026-08-25)

GRPO runs on Vast instance #48673764: 2× H200 140 GB (PCIe 5.0
interconnect — no NVLink), driver 590.48.01, vLLM-template image.
Repo at `/workspace/mlfactory` (git archive + rsync, `pip install -e .`),
weights at `/workspace/models/hub/models--Qwen--Qwen3.5-9B` (bf16,
`HF_HOME=/workspace/models`). **`workspace_is_volume=false`: nothing
survives recycle/destroy — results rsync home.**

**Access and lifecycle (2026-08-25):** ssh
`root@198.145.108.59:30854` with `~/.ssh/id_vast` (`-i`,
`IdentitiesOnly`). Instance stop/start is controlled **from local**
(192.168.1.9): `~/.local/bin/vastai {stop,start} instance 48673764`
with the account key stored in mlfactory secrets (`mlfactory secrets
get VAST_API_KEY`) and mirrored at `~/.vast_api_key` (chmod 600, picked
up by the CLI automatically). Stop/start preserves the whole container
filesystem. The in-container `CONTAINER_API_KEY` is SELF-scoped only —
it 401s on external API calls and cannot restart a stopped box, so all
lifecycle control goes through local with the account key.

| Package | Version | Note |
|---|---|---|
| python | 3.12 | `/venv/main` |
| torch | 2.13.0+cu130 | image-provided (local ace venv: 2.11.0+cu128 — recorded for provenance) |
| transformers | 5.15.0 | image-provided (local: 5.14.1) |
| flash-attn | 2.8.3.post1 | sm_90 source build; zero measured gain (Q11) |
| flash-linear-attention | 0.5.2 | zero measured gain pending invocation check (Q11) |
| causal-conv1d | 1.7.0 | TORCH_CUDA_ARCH_LIST=9.0 build |

Launch convention: `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`;
tmux sessions `smoke`/`flashbuild`-style with named logs under
`/workspace/`. The vLLM/model-ui/ray template services must be stopped
before training (they hold the GPUs); `supervisorctl start vllm`
restores them.

### Remote-stack traps (measured, dated)

- **Default-SDPA generation is call-to-call non-deterministic (2026-08-26,
  R11):** two identical `generate()` calls (same seed/weights/code,
  proven-equal RNG draw counts) diverge mid-trace at 422–1764 tokens.
  Rollout generation must force a deterministic backend:
  `sdpa_kernel([SDPBackend.MATH])` verified bit-stable (0/8 flips);
  FLASH backend untested. Root cause of the Step-1 identity-gate failure
  and of `probe_determinism`'s cross-process flips.
- **Hybrid-cache continuation corrupts boundary tokens (2026-08-25,
  R10):** up to 11.8 nats for ~50 tokens after every split point under a
  zero-init no-op (likelihood ratios off >1e5); the cache object is
  bit-exact — the split/continuation is the poison — and the FLA chunk
  backward crashes on the continued cache state. Windowed replay on cache
  continuation is dead; do not rebuild it.
- **Naive float32 vocab-chunked logprob extraction under grad** retains
  ~52 GB of autograd intermediates at 26k tokens and OOMs the 140 GB
  card. `completion_logprobs` uses `_TokenLogprobs` (custom autograd fn:
  saves the bf16 logits reference + targets, recomputes softmax in
  backward). Evidence: `lab_notes/2026-08-25-step1-replay-engine-windowed-killed.md`.

## Validation

Validated 2026-08-24: `scratch/waypoint_alignment.py` (then
`probe_5046_waypoints.py`) reproduces bit-identical metrics under this
venv (waypoint entropy means 1.56/1.61 bits, matched-waypoint cosine
+0.992).

## Run convention

Run scripts as modules from the repo root (e.g.
`.venv/bin/python -m mlfactory.experiments.ace.train.grpo`); `mlfactory` is
pip-installed editable so package-qualified imports resolve from anywhere
and scripts stay runnable whether invoked as a module or by path. See
`README.md` for the full run-command list.
