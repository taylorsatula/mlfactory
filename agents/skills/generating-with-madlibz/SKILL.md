---
name: generating-with-madlibz
description: Use when asked to generate, seed, or batch-produce a corpus of input prompts or problems for an experiment — especially diverse, realistic, open-ended, or anomaly-bearing prompts authored by an LLM. Operates the madlibz generator: seed levers, mixtures, authoring calls, and corpus freezing.
---

# Generating with madlibz

## Overview

`mlfactory/core/madlibz` generates prompts in three stages with a strict division of authority. What it produces: **mundane first-person situations that each carry one engineered anomaly** (a contradiction, a tempting irrelevance, a missing piece) planted deliberately and declared as ground-truth metadata. It is not a general text generator and it does not produce neutral prompts — every output has an anomaly by design.

1. **Envelope sampling (deterministic)** — seeds steer the distribution.
2. **Authoring (probabilistic, external model)** — an open task invents the actual prompt.
3. **Freeze (deterministic)** — accepted output is recorded with provenance. Culling happens downstream with a batch judge, never at generation time.

The philosophy in one line: **steer, don't specify.** The model's imagination under temperature is the diversity engine; seeds aim its distribution. Do not add levers that enumerate problem structure — every layer of specification collapses the output toward template exercises. If you find yourself wanting a lever for "how the problem works," stop; that is the model's job.

## When to use

- Generating a corpus of input prompts for an experiment.
- Building a seed mixture across domains, anomaly genuses, or detectability levels.
- Extending the generator: new domain, genus, or detectability granular.

## The control surface

```python
from mlfactory.core.madlibz import sample_envelope, authoring_messages, freeze_authored

env = sample_envelope(seed: int, domain: str, *, genus=None, detectability=None)
```

| Lever | Values | Behavior |
|---|---|---|
| `seed` | int | Fully deterministic: same (seed, domain) → same envelope, always. |
| `domain` | key of `DOMAIN_PROFILES` | Selects the persona/stakes/notes pools. |
| `genus` | one of `ANOMALY_GENUSES` | Optional override; blind draw by default. |
| `detectability` | one of `DETECTABILITY_GRANULARS` | Optional override; blind draw by default. |

**Anomaly genuses** — what kind of wrong the planted anomaly is:

| Genus | Meaning |
|---|---|
| `temporal_conflict` | Timeline or ordering that contradicts itself |
| `contradictory_goals` | Two stated wants that cannot both hold |
| `ambiguous_referent` | An unclear what/who/which the answer depends on |
| `hidden_assumption` | A premise the prompt silently needs |
| `red_herring` | A tempting detail that provably does not matter |
| `underspecified_question` | A clearly missing piece; the correct move is to flag it |
| `inconsistent_constraints` | Stated facts that physically/numerically cannot coexist |

**Detectability granulars** — where the anomaly lives relative to the surface (placement, not subtlety):

- `blatant` — the conflict sits nearly adjacent in the text
- `hidden` — present but buried among mundane detail
- `indirection` — never stated; only derivable by composing separate facts

**Naming rule for all levers:** every lever is a *classification of the thing*, never a numeric scale or a self-rating. LLMs classify reliably but cannot calibrate 0.0–1.0 intensities. When extending, name categories of the territory ("what kind" / "where placed"), not magnitudes ("how much").

## Workflow

```python
env = sample_envelope(seed=42, domain="household")
messages = authoring_messages(env)
# -> send to the authoring model with thinking/reasoning ENABLED and a
#    generous token budget. The problem is invented in the reasoning block;
#    disabling thinking starves the stage that does the work.
authored = extract_json(response_text)   # mlfactory.core.api.extract_json
record = freeze_authored(env, authored, model="<model-id>", run="<corpus-name>")
# records are plain dicts; persist them yourself (JSONL, one per line) —
# the batch runner that owns persistence at scale does not exist yet.
```

The authoring model returns JSON:

```json
{
  "prose": "<the person's message>",
  "surface_question": "<the mundane question they think they are asking>",
  "anomaly": {
    "genus": "...", "detectability": "...",
    "what_is_wrong": "...", "where_it_lives": "...", "why_it_trips_reasoning": "..."
  }
}
```

The `anomaly` block is ground truth declared by the author. Preserve it — it is free evaluation signal for any downstream judge, classifier, or auditor: you know where the bodies are buried, so you can measure detection rather than trust judgment.

**Do not verify answers at generation.** These prompts have no canonical answer by design. Structural validation only: prose present, anomaly metadata present, JSON parses. Quality culling is a separate batch-judge stage downstream.

**Know the limitation before reaching for this tool:** every prompt carries an engineered anomaly — `genus` is mandatory (blind-drawn if unspecified). A use case that wants neutral, anomaly-free prompts needs a `"none"` genus added to the generator first, or a different tool entirely.

## Building a mixture

Blind draws give the natural joint distribution over persona × stakes × genus × detectability. To hit a target mix, override levers per draw:

```python
plan = [("temporal_conflict", "hidden"), ("red_herring", "indirection"), ...]
for i, (genus, det) in enumerate(plan):
    env = sample_envelope(seed=base_seed + i, domain=domain, genus=genus, detectability=det)
```

Persona and stakes still come from the blind draw even when genus/detectability are fixed, so surface variety survives inside a fixed mixture.

## Extending

- **New domain**: add an entry to `DOMAIN_PROFILES` in `envelope.py` (`notes`, `personas`, `stakes`). Nothing else changes.
- **New genus**: append to `ANOMALY_GENUSES`. Name a category of wrongness, in the style of the existing entries.
- **New granular**: append to `DETECTABILITY_GRANULARS`. Describe placement relative to the surface, then teach the authoring prompt what the placement means (the system prompt enumerates granulars inline).

## Caveats (learned the hard way)

- **Sampling profiles are per-model compatibility settings, not global constants.** A profile that elicits rich traces from one model can induce degeneration (repetition avalanches, vocabulary collapse) in another at identical settings. Detect degeneration structurally — sliding-window lexical diversity, n-gram repetition rate, output-length anomalies — never by keyword lists; the next avalanche uses different words.
- **Detectability is per-model.** An anomaly that is `hidden` to one model is blatant to another and invisible to a third. Calibrate against the model that will consume the corpus before locking a mixture, and treat the granular as a request, not a guarantee.
- **Determinism holds per (prompt, seed, sampling).** llama.cpp reproduces output byte-identically for a fixed triple; provider APIs generally do not. Freeze records are the replay mechanism for anything probabilistic.
- **Provider preference:** on Lunaroute, prefer the `-ballast` model variant when available (see AGENTS.md).

## Reference

- Generator: `mlfactory/core/madlibz/envelope.py` (the whole tier is one module; read it before extending)
- Tests: `tests/test_madlibz.py`
- A retired deterministic-answer tier (typed slots, computed answers) exists in git history at commit `26654b0` for use cases that need programmatically verifiable answers. Do not mix the two philosophies: answer-verification at generation time is what collapses a corpus into worksheet exercises.
