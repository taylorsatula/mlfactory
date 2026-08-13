# CausalGraph Round 2

Round 2 tests student-aware acquisition under one fixed behavior contract.

## Contract

Qwen3.5 is rendered with:

- system prompt `/nothink`;
- `enable_thinking=False`;
- the tokenizer's expected closed empty thinking scaffold.

Every response has exactly two lines:

```text
TRACE: T,F,...
FINAL: YES
```

The trace contains one Boolean for every query-relevant node in topological order: sources first, then derived nodes through the query. The primary score requires both the complete trace and terminal answer to be exact. Terminal-answer and per-bit accuracy are retained as secondary diagnostics.

The exact trace removes the 50% guessing floor that made Round 1's terminal-answer `d50` poorly identified.

## Files

- `contract.py` — prompt, target, parser, and scorer.
- `generator.py` — thin wrapper around the validated Round 1 symbolic generator.
- `acquisition.py` — TARGETED, RANDOM, and DEPTH_MATCHED selection.
- `analysis.py` — reuses generic frontier utilities.
- `train_adapter.py` — QLoRA with hard contract preflight.
- `evaluate_hf.py` — deterministic fixed-contract evaluation.
- `full_run.py` — orchestration and sealed comparison.
- `dashboard.py` — standalone progress dashboard.

## Dashboard

```bash
PYTHONPATH=. python3.14 -m mlfactory.experiments.causal_graph_round2.dashboard \
  --run-dir runs/causal-graph-round2-contract
```

## Tests

```bash
PYTHONPATH=. python3.14 -m pytest tests/test_causal_graph_round2.py -q
```

No full run should be launched until explicitly requested.
