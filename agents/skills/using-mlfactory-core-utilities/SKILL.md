---
name: using-mlfactory-core-utilities
description: Use when writing experiment code that needs model inference, API judging, embeddings, metrics, environment guards, or checkpoint saving.
---

# Using mlfactory core utilities

## Overview

The top-level `mlfactory.core` package provides reusable resources so experiments do not reinvent inference clients, retry loops, embedding loading, or environment setup.

## When to use

- You are about to import `openai` or `requests` and write a retry loop.
- You are loading a sentence-transformer model by hand.
- You are setting `PYTORCH_CUDA_ALLOC_CONF` at the top of a script.
- You need to save a PEFT adapter or summary JSON.

## Core utilities

### Model server

```python
from mlfactory.core.model_server import model

with model("qwen3.5:4b", gpu=0) as srv:
    client = srv.client()
    r = client.chat.completions.create(model=srv.spec.alias, messages=[...])
```

### API client and judge

```python
from mlfactory.core.api import APIClient, APIConfig, Judge, extract_json

cfg = APIConfig(base_url="http://127.0.0.1:3092/v1", model="qwen3.5:4b")
client = APIClient(cfg)
text = client.chat_completion(messages=[...])
data = extract_json(text)

judge = Judge(cfg)
winner = judge.compare(prompt, candidate_a, candidate_b)
```

### Embeddings

```python
from mlfactory.core.embeddings import embedder

emb = embedder("nvidia/llama-embed-nemotron-8b", device="cuda:1")
vectors = emb.encode(texts, normalize=True)
```

### Metrics

```python
from mlfactory.core.metrics import MetricsLogger

m = MetricsLogger(run_dir, run_id=run_id, registry=registry)
m.step(step, loss=loss, reward=reward)
m.event("eval_done", {"score": score})
```

### Environment guards

```python
from mlfactory.core.env import training_env, inference_env

with training_env(hf_home="/workspace/.hf_home"):
    train(...)
```

### Artifacts

```python
from mlfactory.core.artifacts import save_checkpoint, save_summary

save_checkpoint(run_dir, step, model, tokenizer, manifest=manifest)
save_summary(run_dir, {"jmq": jmq}, manifest=manifest)
```

## Common mistakes

- **Writing raw HTTP retries.** `APIClient` already retries with exponential backoff.
- **Loading embeddings on every call.** `embedder()` returns a cached instance.
- **Mutating global env vars permanently.** Use `training_env()`/`inference_env()` so values restore on exit.
- **Saving checkpoints without updating the manifest.** Pass `manifest=` to `save_checkpoint` so artifact hashes are recorded.
