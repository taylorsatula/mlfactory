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
| flash-linear-attention / causal-conv1d | Qwen3.5 hybrid linear-attention fast kernels — transformers falls back to correct torch implementations; revisit only if generation throughput demands it |
| peft | custom controller instead (`core/steering_controller.py`) |

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
