---
title: Shared Knowledge
description: This is a file where we append machine learning, engineering, and related learnings and insights to carry forward in future sessions.
append_only: true
scope: Durable, broadly stated knowledge that remains useful beyond the session in which it was discovered.
---

# Shared Knowledge

This document is an append-only record of durable machine-learning, engineering, and operational knowledge. Write entries broadly enough that a future reader can understand and apply them without knowing the original session, experiment, repository state, or incident.

## How to add knowledge

- Record the general principle, not a session diary.
- Separate observations, hypotheses, decisions, and conclusions.
- Include the conditions under which a lesson applies and any important limits.
- Prefer causal explanations over unexplained fixes or workarounds.
- Capture the smallest reliable reproducer, validation method, or regression guard when relevant.
- Preserve failed approaches when they reveal a reusable failure mode.
- Do not include secrets, credentials, raw personal data, or unnecessary identifying details.
- Append new entries rather than rewriting prior knowledge; correct obsolete entries explicitly.

## Durable principles

### Reproducibility

A result is not reproducible unless the code, data definition, model and dependency versions, hardware and runtime details, configuration, random seeds, and evaluation procedure are recorded. Save this information with the result rather than relying on memory or mutable environment state.

### Validate the real path incrementally

A successful import or model load does not establish that training works. Validate progressively: environment and device access, model loading, generation, one real forward/backward step, a short target-sized run, guarded validation, and only then a long run. Each stage should exercise the same architecture and critical backend as the intended workload.

### Diagnose before reducing scope

When a workload fails, first inspect hidden processes, device placement, actual tensor shapes, memory allocation, logs, and the exact failing stage. Do not blindly reduce batch size, change the model, or upgrade dependencies before testing the most likely causes. A workaround that avoids a symptom may conceal the root cause.

### Change one meaningful variable at a time

Keep a control configuration and use a new output location for each diagnostic run. Changing hardware, dependencies, model, batch size, and algorithm simultaneously makes a successful result difficult to interpret and a failure difficult to explain.

### Distinguish allocated resources from reserved resources

Resource dashboards often report different quantities. In GPU workloads, distinguish live tensor allocation, framework-reserved memory, driver-visible usage, and unrelated processes. Apparent capacity can be consumed by caches, workspaces, compilation, activation graphs, optimizer state, or other services.

### Architecture and software compatibility matter

Advertised hardware capacity is not sufficient evidence that a model will train reliably. Kernel maturity, compiler support, attention or recurrent backends, device architecture, and exact package combinations can determine correctness and throughput. Prove compatibility with a small faithful backward test before committing to a long run.

### Monitor behavior, not just process liveness

A running process may be compiling, stalled, leaking memory, or optimizing the wrong objective. Monitor progress, throughput, memory trends, losses, gradients, evaluation metrics, output lengths, entropy, KL or trust-region diagnostics, error rates, and checkpoint creation. Add automatic stop conditions for NaNs, runaway metrics, collapse, and resource leaks.

### Protect experiment integrity

Never overwrite the only copy of a checkpoint, log, dataset manifest, or failed run. Use descriptive immutable run directories. Before stopping or destroying temporary infrastructure, copy artifacts to durable storage, verify checksums, and confirm that summaries and checkpoints can be parsed or loaded.

### Treat objectives and evaluations as first-class code

A technically healthy training loop can still optimize the wrong quantity. Check signs, scales, gradient paths, detachments, clipping, reference comparisons, and the interpretation of every metric. Ensure evaluation measures the intended behavior, uses independent data where appropriate, and tests realistic multi-turn or task-specific trajectories rather than relying only on aggregate loss.

### Preserve privacy by design

Keep sensitive source data at the narrowest boundary needed for the task. Prefer synthetic or pseudonymized derivatives for artifacts and remote execution. Logs, prompts, manifests, and debugging snapshots should not accidentally reproduce private data or credentials.

### Turn lessons into safeguards

A resolved failure should become a test, startup assertion, runtime guard, launch-script check, documentation update, or monitoring rule. Knowledge is durable when the system can detect or prevent recurrence without relying on the original person remembering the incident.

## Prompt-corpus generation

### Prompt diversity must exist before trajectory collection

A corpus can claim hundreds of prompts while containing only a small number of effective problem shapes. Fixed-text generators that randomly select from a short list create byte-identical prompts under different IDs; parameterized variants can still be near-duplicates. This produces efficient, memorized solutions and starves downstream annotators of the pathologies they are supposed to detect. Measure semantic/template diversity before spending model calls, and deduplicate at the problem-meaning level rather than relying on surface text.

### Engineer anomalies, not answers, for open-ended reasoning corpora

For open-ended trajectory editing, requiring a programmatically verifiable answer can collapse the prompt space toward textbook exercises, even when the prose is dressed in realistic settings. A productive alternative is a mundane, authentic situation carrying an engineered anomaly: temporal conflict, contradictory goals, ambiguous referent, hidden assumption, red herring, underspecification, or inconsistent constraints. The anomaly is the payload; the surface question can be ordinary and open-ended. The authoring model should be trusted to induce failure modes naturally rather than being forced to construct a predetermined solution path.

### Use deterministic envelopes to steer probabilistic authoring

The useful separation is: deterministic seeds steer broad dimensions (domain, persona, stakes, anomaly genus, anomaly placement), an open-ended LLM task invents the situation and prose, and a later batch judge culls the results. Do not turn the envelope into a detailed problem specification; that merely relocates the worksheet failure. In `mlfactory/core/madlibz`, anomaly genuses classify what is wrong and detectability granulars classify where it lives (`blatant`, `hidden`, `indirection`). These are categorical descriptions, not 0–1 subtlety or confidence scores. LLMs are better at classifying observable territory than self-rating numeric confidence.

### Preserve anomaly ground truth for later measurement

The authoring response should include the prose, surface question, anomaly genus, detectability, what is wrong, where it appears, and why it should provoke over-deliberation. Freeze that metadata with the prose. It provides a reference for evaluating batch judges, classifiers, and stratifiers: measure whether planted anomalies were detected instead of treating model judgment as ground truth. Generation-time validation should remain structural (parseable output, required fields); quality selection belongs to a separate batch-judge stage.

### Over-deliberation is model- and pathology-specific

A single planted anomaly can produce distinct behaviors across models: immediate detection followed by redundant verification; correct detection followed by dismissal; reversed inference and false closure; complete surface trust; parroting a requirement as if it were satisfied; re-litigation of whether an observation belongs in the answer; and coherent reasoning followed by answer degeneration. Surface competence and anomaly detection are independent. Multi-model collection is therefore a deliberate way to widen pathology coverage, not merely redundancy.

When reviewing trajectories, look for `repeated_state_reconstruction`, `branch_reopening`, `correction_spiral`, `redundant_verification`, `premature_commitment`, `state_inconsistency`, `unresolved_material_error`, `overextended_closure`, and `incomplete_arc`. Correct-but-redundant traces are often better editing candidates than simply incorrect traces because their substantive arc can be preserved while spans, state transitions, or closure are calibrated.

### Sampling profiles are model-specific

The same sampling profile can produce useful traces for one model and degeneration for another. Repeating an exact prompt and seed may reproduce a failure, while surface rewording may produce a different failure or change the answer. Treat model × sampling configuration as an experimental variable. Detect degeneration structurally with sliding-window lexical diversity, n-gram repetition, malformed-token rates, and output-length anomalies; keyword lists will miss novel avalanches.

### Truncation is a hard corpus filter

A generation with `truncated=true`, `finish_reason="length"`, or a time-budget hit must not enter a reasoning corpus or stratifier. Completion claims can be incorrectly assigned to traces that ended at the output limit, so they must be checked against generation metadata rather than annotation summaries. Enforce this filter at the input boundary before annotation or selection.

### Keep runtime trace capture format-agnostic

Different model families expose reasoning differently: scaffolded numbered sections, continuous prose, planning skeletons, meta-drafting, and model-specific degeneration. Local llama-server tests consistently exposed the reasoning through `reasoning_content`, but collection code should also handle think-tagged content and absent reasoning fields. Classifiers and stratifiers must judge observable trajectory behavior, not assume a particular formatting style.

## Madlibz operation

`mlfactory/core/madlibz` implements an anomaly-envelope generation pattern. The operational contract is: `sample_envelope` (blind draws by default, optional genus/detectability overrides) → `authoring_messages` (thinking enabled; open task) → `freeze_authored` (prose, envelope, anomaly ground truth, model, and provenance). Persist frozen records as JSONL or another run artifact; culling is downstream. Use the pattern for open-ended anomaly-bearing prompts, and use a separate construction when a use case requires deterministic answer verification.

The madlibz operating skill is `agents/skills/generating-with-madlibz/SKILL.md`. When using a provider with routing tiers, follow the repository's current provider preference. Sensitive source data must remain excluded by repository ignore rules and must not be committed merely because an experiment directory is being staged.
