# ACE — Causal Reasoning Steering

> Working hypothesis (unproven; test, don't assume): productive reasoning
> expands the search space and then prunes it; thrashing revisits/expands
> without durable pruning. A small prefix-causal controller operating on
> the residual stream during generation may learn explore→prune dynamics
> from objective task reward alone. Full statement: `HYPOTHESIS.md`.
>
> The program in one line: **outcome defines direction; counterfactuals
> solve attribution; calibration selects the substrate; anything that
> rewards how a trace looks instead of where it lands is poison.**

**Pivot, 2026-08-24:** the original ACE experiment (post-hoc trace
rewriting) is archived in `mlfactory/experiments/ace-legacyapproach/`.
This directory rebuilds the approach as prospective causal steering on a
full-precision model. Why: `APPROACH_HISTORY.md`.

## Start here

- **`AGENTS.md`** — ambient orientation: directory map, run commands, the
  documentation index, binding context (what breaks the experiment), and the
  patterns that keep the experiment orderly across long runs.
- The **documentation map** (4 maintenance tiers: concept / living /
  reference / evidence) and the **how-the-docs-stay-in-sync** contract live
  in `AGENTS.md`. Each doc opens with `Update when:` — find the doc by the
  event, not by guessing.
- **`STATUS.md`** — the open-questions ledger / change-router; the live
  state of what's resolved and what's open.

## Install

See `ENVIRONMENT.md` for the pinned `.venv` setup. Quick form:

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

Run scripts as modules from the repo root (e.g.
`.venv/bin/python -m mlfactory.experiments.ace.train.grpo`); `mlfactory`
is pip-installed editable so package-qualified imports resolve from
anywhere. See `AGENTS.md` §directory-map for the full run-command list.
