# CausalGraph closed-loop competence-frontier MVP

This experiment tests student-aware synthetic acquisition over a deterministic
Boolean dependency-graph task. The graph is authoritative; renderers,
canonical traces, and `FINAL: YES/NO` labels are generated programmatically.
No LLM judge, embedding, semantic description, or self-consistency is used.

## Staged commands

```bash
# Validate generator invariants and deterministic replay.
python3.14 -m mlfactory.experiments.causal_graph.experiment validate \
  --count 10000 --output runs/causal-graph-generator-validation/sample.jsonl

# Probe the running local f16-jackrongds4qwen server.
python3.14 -m mlfactory.experiments.causal_graph.experiment coarse \
  --output runs/causal-graph-baseline --examples-per-depth 64

# Fit logistic/isotonic/raw curves and stratified bootstrap.
python3.14 -m mlfactory.experiments.causal_graph.experiment analyze \
  runs/causal-graph-baseline/per_example_metrics.jsonl \
  --output runs/causal-graph-analysis --bootstrap 2000
```

The local train/evaluate environment is `/home/admin/.venvs/causal-graph`.
It contains the Qwen3.5-9B HF checkpoint, QLoRA trainer, deterministic HF
evaluator, and plotting dependencies.

```bash
HF_HUB_OFFLINE=1 /home/admin/.venvs/causal-graph/bin/python \
  -m mlfactory.experiments.causal_graph.full_run \
  --output runs/causal-graph-mvp-full \
  --dev-per-depth 32 --sealed-per-depth 128 \
  --candidate-pool 4000 --batch-size 1000 --rounds 3

HF_HUB_OFFLINE=1 /home/admin/.venvs/causal-graph/bin/python \
  -m mlfactory.experiments.causal_graph.plots runs/causal-graph-mvp-full
```

`full_run` evaluates a fixed sealed corpus before training, uses only fresh
development probes for acquisition, runs TARGETED and RANDOM branches from the
same Qwen3.5-9B base, stores every raw output, and performs paired,
difficulty-stratified bootstrap comparisons after the final sealed evaluation.
`--depth-matched` adds the optional secondary control.

## Generator contract

`generator.py` exposes `generate_task`, `verify_task`, `canonical_trace`, and
`regenerate_matches`. Graph metadata includes depth, relevant and distractor
counts, binary gates, negations, source updates, world, templates, source
values, topological order, canonical states, rendered prompt, and answer.
The current graph shape supports `relevant_nodes <= depth + 1 +
binary_gate_count`, with extra relevant source nodes attached at binary gates.
This is intentional MVP scope; invalid combinations are rejected.

## Calibration caveat

The first calibration runs found a clear easy-to-failing transition but some
non-monotonicity at larger depths. The full run preserves raw depth accuracy,
Wilson intervals, isotonic fits, and logistic diagnostics rather than hiding
that behavior. The final classification is based on sealed paired statistics,
not on completion of the implementation.
