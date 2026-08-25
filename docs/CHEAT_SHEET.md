# Cheat sheet: mlfactory core utilities

> Update when: a core module's public API changes. The source of truth is the
> module itself (`mlfactory/core/*.py`); this is an index of what exists and
> the import line, so you can look up the signature when the task triggers.
> An agent that doesn't know these exist will reinvent them — read this once
> on arrival, then come back when a task names one.

## The index

| Need | Import | Source of truth |
|---|---|---|
| Disposable llama-server (OpenAI-compatible client) | `from mlfactory.core.model_server import model` | `core/model_server.py` |
| OpenAI-compatible API client (retries, backoff) | `from mlfactory.core.api import APIClient, APIConfig` | `core/api.py` |
| A/B judge client | `from mlfactory.core.api import Judge` | `core/api.py` |
| Pairwise judge runner (JMQ-style aggregate) | `from mlfactory.core.api import run_judge_pairwise, JudgeResult` | `core/api.py` |
| JSON extraction from model output | `from mlfactory.core.api import extract_json` | `core/api.py` |
| Per-step metrics (dashboard + registry) | `from mlfactory.core.metrics import MetricsLogger` | `core/metrics.py` |
| Training/inference env guards | `from mlfactory.core.env import training_env, inference_env, env_guard` | `core/env.py` |
| Embeddings helper | `from mlfactory.core.embeddings import embedder` | `core/embeddings.py` |
| File hashing | `from mlfactory.core.manifest import sha256_file` | `core/manifest.py` |
| Save data with lab metadata (the required path) | `from mlfactory.core.datasave import datasave, DataSaver, finalize_artifacts` | `core/datasave.py` |
| Save a model checkpoint | `from mlfactory.core.artifacts import save_checkpoint` | `core/artifacts.py` |
| Label an existing checkpoint dir | `from mlfactory.core.datasave import register_checkpoint_dir` | `core/datasave.py` |
| Browse a run's labeled artifacts | `from mlfactory.core.datasave import read_catalog` | `core/datasave.py` |
| Find artifacts across runs (lab catalog) | `Registry(...).find_artifacts(...)` | `core/registry.py` |
| Run manifest (schema, git/env/hardware provenance) | `from mlfactory.core.manifest import RunManifest, FileRecord, STAGES` | `core/manifest.py` |
| SQLite registry (runs, lineage, metrics, merge) | `from mlfactory.core.registry import Registry` | `core/registry.py` |
| Dashboard config schema (probes, panes) | `from mlfactory.core.dashboard_config import ExperimentDashboardConfig, Probe, Pane` | `core/dashboard_config.py` |
| Live dashboard (Rich TUI) | `from mlfactory.core.dashboard import ...` | `core/dashboard.py` |
| Prompt loading | `from mlfactory.core.prompts import ...` | `core/prompts.py` |
| `${secrets.KEY}` expansion | `from mlfactory.core.secrets import SecretsStore, expand_secrets` | `core/secrets.py` |
| Vast.ai remote runner | `from mlfactory.remote.vast import VastRunner` | `remote/vast.py` |
| Generic SSH remote runner | `from mlfactory.remote.ssh_runner import SSHRunner, SSHConfig` | `remote/ssh_runner.py` |

## Canonical usage shapes

```python
# model server: context-managed disposable llama-server, OpenAI client inside
from mlfactory.core.model_server import model
with model("qwen3.5:4b", gpu=0) as srv:
    client = srv.client()
    resp = client.chat.completions.create(messages=[...])

# API client (OpenAI-compatible endpoint)
from mlfactory.core.api import APIClient, APIConfig
client = APIClient(APIConfig(base_url="https://...", api_key=..., model="..."))
text = client.chat_completion(messages=[...], temperature=0.0)

# judge
from mlfactory.core.api import Judge
judge = Judge(APIConfig(base_url=..., api_key=..., model="..."))
verdict = judge.compare(prompt, candidate_a, candidate_b, criterion="...")  # "A"|"B"|"TIE"

# structured JSON from a model
obj = client.structured_json(messages=[...])  # parses; falls back to regex extraction

# registry — use .mlfactory/registry.db to match the CLI (the code default is data/registry.db)
from mlfactory.core.registry import Registry
registry = Registry(".mlfactory/registry.db")
registry.find_artifacts(tag="corpus", stage="train", format="checkpoint")
```

## Drift caught vs the prior AGENTS.md (on record)

- `Registry`'s actual code default is `data/registry.db`, but the CLI uses
  `.mlfactory/registry.db`. Plugins must pass `Registry(".mlfactory/registry.db")`
  explicitly to match the CLI. (This was already in "Common mistakes"; the
  cheat sheet now points to `core/registry.py` as source of truth so future
  drift is caught by reading the module.)
- `save_checkpoint` accepts both the legacy `label=` kwarg and the lab-metadata
  `title=`/`description=`/`tags=`/`caveats=`/`sensitivity=`/`schema=` kwargs,
  with sensible defaults so existing callers keep working. It is **not** an
  error to omit `title`/`description` here (unlike `datasave`, where they are
  required).
