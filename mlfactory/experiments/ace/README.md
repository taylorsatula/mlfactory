# ACE — Causal Reasoning Steering (rebuilt experiment)

> **Pivot, 2026-08-24.** The original ACE experiment (post-hoc trace rewriting:
> collect → classify → stratify → LLM-editor rewrite) is archived in
> `mlfactory/experiments/ace-legacyapproach/`. Its plugin stages are no longer
> registered in `mlfactory/core/runner.py`. The research goal is unchanged —
> treat reasoning as *search* and find machinery that distinguishes productive
> exploration from thrashing — but the approach moves from editing traces
> after the fact to **steering generation causally** on a full-precision model.
>
> Working hypothesis (unproven; test, don't assume): productive reasoning
> expands the search space and then prunes it; thrashing revisits/expands
> without durable pruning. A small prefix-causal controller operating on the
> residual stream during generation may learn explore→prune dynamics from
> objective task reward alone.

## This directory

Rebuilt from scratch. Current contents:

- `probe_teacher_forced.py` — recon probe 1 (inherited): teacher-forced replay
  of a legacy trace under full-precision Qwen3.5-9B; per-position next-token
  entropy from full-vocab logits + final-layer hidden snapshots.
- `probe_5046_waypoints.py` — recon probe 2 (inherited): semantically aligned
  raw/rewrite waypoints (seed 5046); showed matched latent states coincide
  (cos ≥ 0.95, argmax identity 29/29) and waypoint entropy co-varies
  (Pearson r = 0.993) between a raw trace and its ACE rewrite.
- `inspect_model.py` — structural inspection of Qwen3.5-9B under
  transformers 5.14.1 (module tree, decoder-layer return type, hook firing
  pattern during prefill vs cached decode, per-depth residual norms).
- `steering_controller.py` — causal residual steering controller (plumbing
  stage, 2026-08-24). `SteeringController`: normalized 4096→512→4096
  bottleneck adapter with scalar per-token gate, zero-init output (bit-exact
  no-op), bounded intervention (`||Δh|| < alpha·||h||`, alpha=0.1 default),
  4,195,329 params, bf16, separately saveable/loadable via safetensors + JSON
  sidecar. `ResidualSteering`: forward-hook context manager on
  `model.model.layers[15]` output (mid-depth full_attention block; the
  residual stream is the only hook point shared by both hybrid block types).
  `generate()` / `teacher_forced_logits()`: the generation/measurement paths.
  Base model frozen via `freeze_base_model()`. Run the demo:
  `CUDA_VISIBLE_DEVICES=1 .venv/bin/python steering_controller.py`
- `test_steering_controller.py` — 6 smoke tests: (1) zero-init bit-exact vs
  untouched Qwen (greedy + seeded sampling + teacher-forced logits, max
  |logit diff| = 0.0); (2) Qwen params receive no grads and don't change
  under a controller-only step; (3) nonzero intervention shifts logits (max
  2.875) and flips greedy generation; (4) prefix-causality (truncation rel.
  Δdiff 1.5e-2 within bf16 tiling noise; continuation-swap exactly 0);
  (5) save/load bit-exact reproduction; (6) bounded relative to residual.
  Run: `CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m pytest
  test_steering_controller.py -s -v`. Passing tests validate the control
  surface only — NOT that steering improves reasoning.
- `specs/` — run specs (empty until stages are implemented).
- `.venv/` — dedicated environment (below), gitignored.

Both probes read **immutable source data** from
`mlfactory/experiments/ace-legacyapproach/data/` (see path constants at top of
each file). Legacy data is read-only: never write into the archive.

## Environment (`.venv`)

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

Rationale: this pins the stack already proven by the reconnaissance probes
(torch 2.11.0+cu128, transformers 5.14.1) and adds the `mlfactory` package
itself (absent from the mtp-lab venv the probes originally ran under).
Deliberately NOT installed: `bitsandbytes` (bf16 full-precision loads only),
`flash-linear-attention`/`causal-conv1d` (Qwen3.5 hybrid linear-attention fast
kernels — transformers falls back to correct torch implementations; revisit
only if generation throughput demands it), `peft` (custom controller instead).

Validated 2026-08-24: `probe_5046_waypoints.py` reproduces bit-identical
metrics under this venv (waypoint entropy means 1.56/1.61 bits, matched-waypoint
cosine +0.992).

## Model

- Measurement/generation model: `/home/admin/models/hf/Qwen3.5-9B` (bf16
  safetensors, 8.95B params). Text config: hidden 4096, 32 blocks, vocab
  248,320, hybrid arch — `layer_types` alternate 3× `linear_attention` + 1×
  `full_attention` (`full_attention_interval: 4`); full-attention blocks at
  indices 3, 7, 11, 15, 19, 23, 27, 31. EOS 248044. Max positions 262,144.
- Fits on one RTX 3090 (24 GB) with room for training-scale activations.
- The GGUF/vLLM HTTP serving paths (llama-server, `models.yaml`) remain the
  legacy/ops path — unsuitable for activation intervention; do not use them
  for steering.

## Conventions inherited from legacy ACE

- Trace records: frozen-envelope fields (`envelope_hash`, `surface_hash`,
  `seed`, `domain`, `prose`, `surface_question`) + `trace` + `provenance`.
- Provenance fields on every produced record: model id/path, sampling
  params, seed, corpus name, source pointers.
- Immutable inputs; append-safe JSONL artifacts; sidecar files for bulky
  per-token data.
- New artifacts for this experiment live under `mlfactory/experiments/ace/data/`
  (gitignored) and must not be written into the legacy archive.

## Known stale references (legacy tooling, not yet updated)

- `/home/admin/mlfactory/run_ace_rewrite_lunaroute.py` — untracked legacy
  script; its default trace path still points at `mlfactory/experiments/ace/data/`.
- `agents/skills/*/SKILL.md` and repo `README.md` examples reference legacy
  spec paths (e.g. `mlfactory/experiments/ace/specs/ace_collect_qwen35.yaml`),
  which now live under `ace-legacyapproach/specs/`.
- `mlfactory/core/prompts.py` docstring example references a legacy prompt path.
