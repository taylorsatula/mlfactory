# Lab notes and session notes

> Update when: the notes protocol changes. Sources of truth:
> `mlfactory/core/notes.py` (lab notes), `mlfactory/AGENTS.md` (session-note
> protocol, kept there because it is a writing protocol, not an API). This doc
> carries both protocols together since they are the lab notebook's two ends.

The project keeps two kinds of notes, on opposite ends of a frequency/cost
axis. They are distinct artifacts with distinct protocols and must not
collapse into one — a single notebook that tries to serve both audiences
serves neither.

## Lab notes (frequent, cheap, run-attached)

A lab note is one timestamped line appended to `runs/<run_id>/notes.jsonl`.
Cheaper than a sticky note. Structured only by being a JSONL record. The
manifest records *what* a run did; a lab note records *why you changed
something*, *what you expected*, or *what surprised you*.

Write a lab note at:
- a **hypothesis change** — "trying lr 3e-5 because 1e-4 plateaued, expect lower floor"
- an **unexpected result** — "loss spiked at step 800 then recovered; not the usual divergence shape"
- a **parameter change with rationale** — "warmup 500→1000 steps to avoid the early instability from run X"
- a **dead end** (the negative result that prevents a retry) — "mixture-of-depths worse than dense at equal FLOPs, don't revisit without a new reason"
- a **resumption point** — "pausing here; next step is to re-run eval with the fixed judge prompt"

Do not write a lab note for:
- routine execution (the manifest already captures it)
- anything the metrics/logs/summary already record
- a full session's worth of thinking (that is a session note, below)

CLI:

```bash
mlfactory note <run_id> <text>          # append one note
mlfactory notes <run_id>               # read a run's notes
mlfactory notes --grep <term>          # search across all runs (the "what have I tried" query)
mlfactory show <run_id>                # surfaces notes inline after the manifest
```

Programmatic: `mlfactory.core.notes.append_note(registry, run_id, text)`,
`read_notes(run_dir)`, `search_notes(pattern)`. Notes are hashed as
`FileRecord(role="note")` and provenance-linked like any other artifact;
the notes file gets a `.meta.json` sidecar label matching `datasave`'s
convention.

Lab notes are deliberately low-ceremony. If writing one feels like work,
you're writing the wrong thing — write a session note instead, or nothing.

## Session notes (rare, high-cost, end-of-session)

End-of-session notes live in `session_notes/YYYY-MM-DD-<slug>.md`. Write one
when a session produced durable knowledge — findings, decisions,
environment traps — that the repo doesn't record on its own. Do not write
one for routine execution.

Required structure:

```
# Session notes — <date> — <slug>
Scope: <one or two lines>

## What exists now that didn't before     # new artifacts, paths, commits (hash only)
## Findings (the durable knowledge)       # numbered; each = claim + the evidence that established it
## Decisions with rationale               # rulings made + why; include rejected alternatives that look plausible
## Environment traps encountered          # failure -> working fix (debugging time is the expensive artifact)
## State at note time                     # running jobs, progress counters, immediate next step
```

Rules:
- Findings are claims with evidence, not observations ("X is true because measured Y", not "we saw Y").
- Decisions record the final causal model, not the chronology of reaching it.
- Reference committed work by hash; never restate what git history carries.
- Keep them dense. A future reader with zero session context is the audience.
