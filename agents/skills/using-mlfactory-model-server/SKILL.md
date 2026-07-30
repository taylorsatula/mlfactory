---
name: using-mlfactory-model-server
description: Use when an experiment needs a disposable llama-server for local inference, or when you see code launching llama-server by hand.
---

# Using the mlfactory model server

## Overview

The `model()` context manager resolves an alias to a GGUF, starts a disposable llama-server, waits for `/v1/models`, yields a server handle, and stops the server on exit. This replaces manual launch scripts and port management.

## When to use

- ACE collect, classify, or any stage that calls a local model.
- You see `launch_llama_*.sh`, `subprocess.Popen(["llama-server", ...])`, or hardcoded ports.
- You need to pick a specific GPU or port for a model.

## Core pattern

```python
from mlfactory.core.model_server import model

with model("qwen3.5:4b", gpu=0) as srv:
    client = srv.client()
    response = client.chat.completions.create(
        model=srv.spec.alias,
        messages=[{"role": "user", "content": "hello"}],
    )
```

## Aliases

Aliases are defined in `models.yaml` at the repo root. Each entry maps to a GGUF path and llama-server defaults. Override per-call with `port=`, `gpu=`, or `overrides={...}`.

## Quick reference

| Alias | Use case |
|-------|----------|
| `qwen3.5:4b` | Fast smoke tests, ACE collect |
| `qwopus3.6:27b` | Strong local judge/model |
| `gemma4:26b`, `glm4.7:flash`, `gpt-oss:20b` | Alternative local models |
| `ornith1:35b`, `nemotron3:30b`, `laguna:xs` | Specialty local models |

## Common mistakes

- **Starting llama-server manually.** Use `model()` so port allocation, health checks, and cleanup are handled.
- **Hardcoding ports.** The server scans `3090–3100` by default; pass `port=` only when you must share a port.
- **Ignoring the model alias registry.** Add new GGUFs to `models.yaml` instead of embedding paths in specs.
- **Forgetting the context manager scope.** The server stops on exit; keep the `with` block around all inference.
