# Sample experiment — full-featured reference for mlfactory

A **four-stage pipeline** that exercises every major mlfactory pattern, including real supervised parameter updates.
New contributors and agents should read this experiment first before building
their own.

## Pipeline overview

```
transform ──► classify ──┬──► train
 chunks        labels    │     checkpoint + loss metrics
                        └──► eval
                              quality scores + guard
```

The dependency-free `numpy_softmax` train backend validates the complete training lifecycle. The optional `transformers_causal_lm` backend performs full-weight or LoRA fine-tuning of a Hugging Face causal LM.

## Stages and what they demonstrate

### Stage 1: `transform` — text chunking + statistics

| Feature demonstrated | Where |
|---|---|
| Plugin lifecycle (`prepare`/`execute`/`finalize`) | `transform_plugin.py` |
| `MetricsLogger` → `dashboard.jsonl` | Per-chunk `metrics.step()` |
| `save_summary()` from `core.artifacts` | Aggregate stats → `artifacts/summary.json` |
| Artifact hashing | `finalize()` hashes `chunks.jsonl` + `summary.json` |
| Spec-driven tunables | `chunk_size`, `num_paragraphs`, `input_text` |

**No external dependencies** — runs on any machine with Python 3.10+.

### Stage 2: `classify` — topic classification via LLM

| Feature demonstrated | Where |
|---|---|
| Model server via `model()` context manager | Local GGUF inference |
| `APIClient` for external endpoints | OpenRouter, remote llama-server |
| `inference_env()` environment guard | CUDA/HF env management |
| Multi-stage lineage (`input_run`) | Reads parent transform's `chunks.jsonl` |
| Registry lookup | Resolves parent run artifacts via `Registry.get()` |
| Per-item error handling | Failed chunks logged as events, not crashes |

**Requires**: a model server (local GGUF + llama-server) OR an OpenAI-compatible API endpoint.

### Stage 3: `train` — supervised fine-tuning

| Feature demonstrated | Where |
|---|---|
| Actual gradient-based optimization | `numpy_softmax` smoke backend |
| Full-model or LoRA causal-LM tuning | Optional `transformers_causal_lm` backend |
| Completion-only loss masking | Prompt tokens receive label `-100` |
| Reloadable checkpoint | `artifacts/checkpoint-final/` |
| Reproducible metrics and config | Loss dashboard + `training_config.json` |
| Multi-stage lineage | Reads a classify run's `classifications.jsonl` |

The NumPy backend is deliberately small and tests orchestration rather than model quality. Install `mlfactory[train]` to use the Hugging Face backend.

### Stage 4: `eval` — quality evaluation via LLM-as-judge

| Feature demonstrated | Where |
|---|---|
| `APIClient` for LLM-as-judge | Rate classification correctness |
| `Judge.compare()` for pairwise A/B | Optional pairwise comparisons |
| **Guard logic** | `manifest.status = "guarded"` when quality < threshold |
| `save_config()` for reproducibility | Eval config persisted as artifact |
| Multi-stage lineage | Reads parent classify's `classifications.jsonl` |
| `compute_quality_report()` | Aggregate accuracy, per-topic breakdown |

**Requires**: an OpenAI-compatible API endpoint for the judge model.

## Quick start

### Stage 1 (no dependencies)

```bash
# From the mlfactory repo root:
mlfactory run mlfactory/experiments/sample/specs/sample_transform.yaml

# Smoke test with smaller corpus:
mlfactory run mlfactory/experiments/sample/specs/sample_transform_small.yaml
```

### Stage 2 (needs LLM endpoint)

Edit `specs/sample_classify.yaml` to configure your backend:

```yaml
# Option A: local model server
model: qwen3.5:4b
gpu: 0

# Option B: external endpoint
base_url: https://openrouter.ai/api/v1
api_key: ${secrets.OPENROUTER_API_KEY}
model_name: qwen/qwen3.6-27b
```

Then set `input_run` or `input_file` to point at a transform run:

```yaml
input_run: 20260804-195559.transform.sample-text-analysis-smoke
```

```bash
mlfactory run mlfactory/experiments/sample/specs/sample_classify.yaml
```

### Stage 3 (fine-tuning smoke test)

Set `input_run` in `specs/sample_train.yaml` to a classify run, then execute:

```bash
mlfactory run mlfactory/experiments/sample/specs/sample_train.yaml
```

For causal-LM LoRA/full fine-tuning, install the optional dependencies and select `transformers_causal_lm` using the documented block in the spec:

```bash
python -m pip install -e '.[train]'
```

### Stage 4 (needs judge endpoint)

Edit `specs/sample_eval.yaml`:

```yaml
judge_base_url: https://openrouter.ai/api/v1
judge_api_key: ${secrets.OPENROUTER_API_KEY}
judge_model: qwen/qwen3.6-27b
input_run: <classify-run-id>
```

```bash
mlfactory run mlfactory/experiments/sample/specs/sample_eval.yaml
```

### Python API (full pipeline)

```python
from pathlib import Path
from mlfactory.core.registry import Registry
from mlfactory.core.runner import run_from_spec

registry = Registry(".mlfactory/registry.db")

# Stage 1: transform
m1 = run_from_spec(
    Path("mlfactory/experiments/sample/specs/sample_transform.yaml"),
    registry=registry,
)

# Stage 2: classify (reference the transform run)
# Edit the spec to set input_run: m1.run_id, or pass it programmatically:
import yaml, tempfile
spec = yaml.safe_load(Path("mlfactory/experiments/sample/specs/sample_classify.yaml").read_text())
spec["input_run"] = m1.run_id
spec["base_url"] = "http://localhost:3090/v1"  # or your endpoint
spec_path = Path(tempfile.mktemp(suffix=".yaml"))
spec_path.write_text(yaml.dump(spec))
m2 = run_from_spec(spec_path, registry=registry, parent_runs=[m1.run_id])

# Stage 3: fine-tune on the classification records
train_spec = yaml.safe_load(Path("mlfactory/experiments/sample/specs/sample_train.yaml").read_text())
train_spec["input_run"] = m2.run_id
train_spec_path = Path(tempfile.mktemp(suffix=".yaml"))
train_spec_path.write_text(yaml.dump(train_spec))
m_train = run_from_spec(train_spec_path, registry=registry)

# Stage 4: evaluate the original classifications (a sibling of train)
spec3 = yaml.safe_load(Path("mlfactory/experiments/sample/specs/sample_eval.yaml").read_text())
spec3["input_run"] = m2.run_id
spec3_path = Path(tempfile.mktemp(suffix=".yaml"))
spec3_path.write_text(yaml.dump(spec3))
m3 = run_from_spec(spec3_path, registry=registry, parent_runs=[m2.run_id])

print(f"Eval result: {m3.summary}")
if m3.status == "guarded":
    print(f"Guard: {m3.guard_report}")
```

## Files

| File | Purpose |
|---|---|
| `transform.py` | Domain logic: chunking, statistics, classification, evaluation |
| `transform_plugin.py` | Stage 1 plugin — MetricsLogger, save_summary |
| `classify_plugin.py` | Stage 2 plugin — model server, APIClient, inference_env |
| `train.py` | NumPy fine-tuning, checkpoint loading, prediction |
| `train_plugin.py` | Stage 3 plugin — training backends, metrics, checkpoints |
| `eval_plugin.py` | Stage 4 plugin — Judge, guard logic, save_config |
| `specs/sample_transform.yaml` | Stage 1 spec |
| `specs/sample_transform_small.yaml` | Stage 1 smoke test |
| `specs/sample_classify.yaml` | Stage 2 spec (documented with all options) |
| `specs/sample_train.yaml` | Stage 3 fine-tuning spec (both backends documented) |
| `specs/sample_eval.yaml` | Stage 4 spec (documented with all options) |
| `specs/sample_smoke.yaml` | Minimal smoke test spec |
| `dashboard.json` | Stage 1 dashboard config |
| `dashboard_classify.json` | Stage 2 dashboard config |
| `dashboard_train.json` | Stage 3 dashboard config |
| `dashboard_eval.json` | Stage 4 dashboard config |
| `README.md` | This file |

## Adapting this for your own experiment

1. **Copy this directory**: `cp -r mlfactory/experiments/sample mlfactory/experiments/myexp`
2. **Replace domain logic** in `transform.py` with your computation
3. **Update the plugin** to call your code; keep the `prepare`/`execute`/`finalize` structure
4. **Add your stage-specific fields** to the spec
5. **Update `dashboard.json`** probes for your outputs
6. **Register the plugin** in `mlfactory/core/runner.py`
7. **Add your stage** to `STAGES` in `mlfactory/core/manifest.py` if it's new
8. **For multi-stage pipelines**: use `input_run` in downstream specs and resolve via `Registry.get()`
