# AGENTS.md — ACE experiment

> Ambient context for any agent working in `mlfactory/experiments/ace/`.
> The parent `mlfactory/AGENTS.md` covers framework concerns (lab-note and
> session-note protocols, `datasave`, plugin contract, registry) — not
> repeated here. This is the ace-specific layer: orientation, binding
> context, and the patterns that keep the experiment orderly across long
> runs. Depth lives in the `*.md` docs; this is the map and the method.

## What ACE is

Autoregressive Context Engineering, rebuilt 2026-08-24 as prospective
causal steering. **Hypothesis (unproven):** productive reasoning expands
the search space then durably prunes it; thrashing revisits without durable
pruning; a small prefix-causal controller on the residual stream may learn
explore→prune dynamics from terminal reward alone (`HYPOTHESIS.md`; why the
approach changed: `APPROACH_HISTORY.md`).

The program in one line: **outcome defines direction; counterfactuals
solve attribution; calibration selects the substrate; anything that
rewards how a trace looks instead of where it lands is poison.**

## Directory map

| Path | Role | Run |
|---|---|---|
| `core/` | shared foundation: steering controller, problem set, trace diagnostics | — |
| `frontier/` | author + collect the verifiable-30 problem set | `.venv/bin/python -m mlfactory.experiments.ace.frontier.collect_rollouts` |
| `analysis/` | teacher-forced mapping + Phase-1 kill tests | — |
| `annotate/` | annotation sidestep: LLM span annotation → teacher-forced capture → position probes → steering directions (`ANNOTATION_SIDESTEP.md`) | `.venv/bin/python -m mlfactory.experiments.ace.annotate.run_batch --pass pass1` |
| `train/` | controller training (GRPO) | `.venv/bin/python -m mlfactory.experiments.ace.train.grpo` |
| `tests/` | smoke tests | `.venv/bin/python -m pytest tests/test_steering.py -s -v` |
| `scratch/` | one-off recon, kept for provenance — **nothing imports these** | — |
| `gen/` | solver-built problem families with strict verifiers + calibration | `.venv/bin/python -m mlfactory.experiments.ace.gen.generate --self-test` |
| `data/` `lab_notes/` `specs/` `.venv/` | outputs / evidence / specs / env | — |

Run scripts as **modules from the repo root**. `mlfactory` is pip-installed
editable, so `from mlfactory.experiments.ace.core.X import …` resolves from
anywhere — no `sys.path` hacks.

## Documentation map (4 maintenance tiers)

Each doc opens with `Update when:` — find the doc by the event, not by
guessing. `STATUS.md` is the change-router: each open question names the
doc(s) to write back to on resolve.

| Tier | Docs | Update when |
|---|---|---|
| **Concept** | `HYPOTHESIS` `APPROACH_HISTORY` `COUNTERFACTUAL_FRAMEWORK` `REWARD_POLICY` `PHASES` `CALIBRATION` `ANNOTATION_SIDESTEP` `TERMINAL_FORK_COMPUTE` | methodology shifts; a lab note sharpens a claim (write back with `Refined by:`) |
| **Living** | `FAILURE_MODES` `OBSERVABLES` `LAYER_HYPOTHESES` `STATUS` | evidence lands — failure modes, kill-test results, layer maps, resolved questions |
| **Reference** | `QWEN35_ARCHITECTURE` `OPERATIONS` `ENVIRONMENT` `LEGACY` `TRAJECTORY_VOCABULARY` | config changes only |
| **Evidence** | `lab_notes/*.md` | a session produces durable knowledge (parent protocol) |

Living docs carry a **status column** (`candidate`/`supported`/`killed`/`underpowered` for observables and layers; `observed`/`replicating`/`stable` for phenomena) so currency is visible in-place.

## Binding context — read before acting

| Ruling | Value | Why | Doc |
|---|---|---|---|
| Reward | **terminal verified outcome only** | rewards how a trace looks instead of where it lands is poison | `REWARD_POLICY.md` |
| Forbidden as rewards | entropy, tortuosity, recurrence, length, PRM step scorers | local-proxy trap; destroys productive exploration | `REWARD_POLICY.md` |
| Precision | **bf16** | q8 is a different model; all seeds are bf16-calibrated | `OPERATIONS.md` |
| Backstop cap | **26000 tokens** | terminal loops contribute nothing to identical-conditions comparison | `OPERATIONS.md` |
| Thinking | **enabled** | the phenomenon of interest | `OPERATIONS.md` |
| Concurrency | **one sample at a time** | 24 GB VRAM; 8-wide 32k batch OOMs | `OPERATIONS.md` |
| Existing rows | **never regenerate or truncate** | artifacts are immutable evidence; resume is bit-stable | `OPERATIONS.md` |
| llama-server | **off** | user ruling; frees GPU0 | `OPERATIONS.md` |
| Steering layer | **not chosen yet** | readability ≠ causal leverage; L15 killed as measurement site, the hook is a Phase-3 decision | `LAYER_HYPOTHESES.md` |

## Keeping ACE orderly — patterns as reasoning

These generalize: apply the reasoning to a new file, doc, or failure even
if it doesn't resemble anything that exists today.

### 1. Separation of concerns is a maintenance question

The organizing question is not "does this look messy?" but **"what event
triggers this file/doc to update, and is that event distinct from its
neighbors'?"** If two files update on the same event, one doesn't have a
clear job; if you can't name the trigger, it doesn't have one either. This
is why `REWARD_POLICY` (the binding list) is separate from
`COUNTERFACTUAL_FRAMEWORK` (the method) — both touch reward, but one
changes when a term is banned and the other when the fork design changes.

**Apply:** before creating a file, name its update trigger. If the
trigger is "whenever X changes" and X already owns a doc, your file is a
section of that doc, not a new file.

### 2. The folder carries shared context; the leaf carries only what distinguishes

A name is noise if it repeats what the location already tells you. The
`probe_` prefix in `probes/`, `teacherforced_` on every file in `analysis/`,
leaked counts (`_30`) and seeds (`5046`) — all told you nothing that
disambiguates the file from its siblings. The test: **would a reader
inside the folder need this prefix to disambiguate the file?** If not,
drop it.

**Apply:** name what makes this file *different from its neighbors*, not
what category it belongs to. The category is the folder.

### 3. Scratch is a real tier, decided by "does anything import it?"

The test is structural: **does anything import or invoke it?** No →
scratch. Config that changes only when the config changes → reference.
Tracks evolving evidence → living. Stable theory → concept. A file that
produced a great result but is never called again belongs in `scratch/`
with a note in the relevant living doc pointing to its finding. Don't
promote scratch to reference because the result was good; promote it
only if something starts importing it.

### 4. No lossy compression of concepts — port in two passes

Relocating content silently drops load-bearing distinctions because they
read like flavor. "Domain averages concealed 0/8 prompts carried by
successful siblings" looks like color but carries *why* per-prompt bands
matter. **Port the full content first; massage readability as a separate
pass.** Compression-while-porting is where detail dies — the compressing
agent can't tell decoration from load-bearing reasoning.

**Apply:** faithful port then readability massage, never both at once.
Grouping, ordering, and promoting buried points are safe; deleting
sentences is where you lose things. Before shortening, ask: does this
carry a *why* the surrounding sentences depend on?

### 5. Artifacts are immutable evidence

Collected traces, `.npz` captures, and run outputs are evidence. **Never
regenerate, truncate, or rewrite an existing row.** Resume skips done
`(proposal_id, sample_i)` pairs; in-flight samples discarded on restart
redone bit-identically (deterministic seeds). Chop analytically at
analysis time (treat cap as a per-row covariate), never retroactively on
the artifact — why a mixed-cap file stays mixed-cap rather than being
"cleaned up."

**Apply:** the artifact records what happened. If it's wrong, the fix is
a new artifact or a sidecar annotation, never a rewrite.

### 6. Lab notes are evidence records — never rewrite history

A lab note records what was known at a moment. When a rename breaks its
pointer, fix the pointer in-place with a `(now <path>)` annotation; **do
not rewrite the note's content.** The note's `Decisions` section is the
manifest of what needs write-back — written when evidence is fresh, and
falsifying it breaks the evidence chain `STATUS.md` resolves from.

**Apply:** history is read-only. Pointers can be annotated; findings,
decisions, and rationale cannot be back-edited. If understanding changed,
write a new lab note.

### 7. The change-router makes "what to update" mechanical

`STATUS.md` carries an `on resolve → update` column naming the doc(s) for
each open question. When a question resolves, the row itself tells you
what to touch — no memory, no archaeology. Verdict *content* goes in the
named topical doc; STATUS records only "RESOLVED → see <doc>". This is
what stops concept docs from drifting: the write-back is triggered by an
explicit row, not by someone noticing the doc is stale.

**Apply:** when a lab-note decision changes a claim in another doc, the
note's `Decisions` names that doc, and `STATUS.md` gets a resolved row
pointing to it. Lab note = manifest; STATUS = index; topical doc =
content. Three roles, never collapsed.

### 8. Trust but verify your own work

An agent's summary describes intent, not outcome. After a multi-file
move: `py_compile` everything, import-smoke every module that doesn't
load a model, run the CPU tests, grep for stale imports. After a doc
restructure: coverage-check every old section has a home, grep for stale
references. After relaunching a run: confirm the banner's `already_done`
count, check for duplicate keys, verify the worker probes recover. **The
verification is the work; the edit is the first draft.**

**Apply:** every structural change has a verification step that would
catch its own failure. Name it before you start, run it after.

## Writing code for the experiment

The patterns above are mostly about docs, but the code is the experiment.
Same shape — reasoning, not rules, with `Apply:` lines for cases that
don't exist today.

### C1. Package-qualified imports, not `sys.path` hacks

`mlfactory` is pip-installed editable, so
`from mlfactory.experiments.ace.core.steering_controller import X` resolves
from any run mode (module, script, test). The old
`sys.path.insert(0, str(Path(__file__).resolve().parent))` was a
workaround for not trusting the editable install — it breaks the moment a
file moves and hides import errors behind a path manipulation that works
for one layout only.

**Apply:** every cross-module import is package-qualified from the
`mlfactory` root. Never reach for `sys.path`. If an import fails under
package-qualification, make the package importable (install `-e .`, add
the missing `__init__.py`) — don't patch `sys.path`.

### C2. The rename-then-verify loop

A move is a graph edit, not a file edit — dependencies don't update
themselves, and "I updated all the imports" describes intent. The
verification that catches a broken move is cheap and mechanical:
`py_compile` every moved file (syntax), import-smoke every module that
doesn't load a model on import (`importlib.import_module` loop), grep for
stale bare imports of the old names, then run the CPU tests. Files that
*do* load a model on import (probes, collectors) get `py_compile` only.

**Apply:** after any move/rename, run the four-step verification before
declaring done. Name the step *before* you start so you don't forget it
after. The grep is the one that finds the function-local `from problems
import` you missed because it wasn't at the top of the file.

### C3. `Path(__file__)` depth changes when you move a file down a dir

`Path(__file__).resolve().parent / "data"` breaks silently when a file
moves into a subdirectory — `parent` now points at the subdirectory, not
the experiment root. This is the most common silent break from a reorg:
the path resolves to a real (usually-empty) directory rather than raising.

**Apply:** when moving a file N levels deeper, audit every
`Path(__file__).resolve().parent` and add N `.parent`s. Then confirm it
finds the data — a silent wrong-path is worse than a crash because it
produces empty output that looks like a real run.

### C4. Generators (`gen/`) are solver-built; verifiers are strict

Each `gen/` family has `make(rng, knobs) -> Problem` that builds the
instance *with an internal solver* (reference answer exact by
construction), and `check(completion, reference, knobs) -> bool` — the
**strict** verifier. The collector's soft substring match is advisory
only; `gen/calibrate.py` re-scores with `check()` and is the sole
authority on band classification (`CALIBRATION.md`). A `check()` looser
than the `make()` solver admits false positives; a `check()` tighter than
the prompt specifies (e.g. requiring a canonical serialization the prompt
never asked for) admits false negatives. Both corrupt the calibration
loop silently.

**Apply:** the invariant is `check("Answer: " + problem.answer,
problem.answer, problem.knobs)` passing for every instance
(`gen/generate.py --self-test`). But that tests solver-output, not
model-output — the model writes its own serialization, which the
verifier must tolerate. Loosen *extraction* (case-insensitive,
synonymous field labels), not *semantics* (keep full pair-set, palette
membership, per-edge constraint, six-field tuple). See
`lab_notes/2026-08-24-verifier-fix-and-teacherforced-scan.md`.

### C5. Capture and analysis are separate scripts

Teacher-forced work splits into a **capture** script
(`analysis/entropy_scan.py`, `analysis/residual_map.py`) that loads the
model and writes one `.npz` per trace; and an **analysis** script
(`analysis/analyze_scan.py`, `analysis/analyze_map.py`) that reads the
`.npz` files on CPU, no GPU. Capture is GPU-bound and expensive (~10
s/trace, 32 GB on disk); analysis is cheap and re-runnable. A new metric
over the same traces is a CPU-only re-run, not another GPU pass.

**Apply:** when adding a measurement, first ask whether it needs model
hidden states (→ extend capture's hooks, re-run capture) or only the
already-captured arrays (→ add to analysis only). Most new metrics
belong in the analysis script. Capture changes are expensive — batch
them: capture once with all layers/channels you might want.

### C6. GPU work: one sample at a time, respect the desktop overhead

A 24 GB RTX 3090 cannot batch 8-wide 32k-token reasoning traces. The
collectors run one sample at a time, per-sample seed, deterministic and
resume-safe. GPU0 is desktop-resident (~1.9 GB used), effectively a
~22 GB card; 20k+ token forwards on GPU0 sit right at the edge. Route
long traces to GPU1, or free the desktop. (Once misdiagnosed as a
"leak" — it was the desktop at the edge all along.)

**Apply:** before launching a GPU job, check `nvidia-smi` for what's
already on the card; a free index doesn't mean a free card — the desktop
uses GPU0 continuously. Match the shard's `--candidate-range` to the GPU
that can hold it. When a GPU job OOMs, the first hypothesis is
fragmentation (`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` +
`empty_cache()` between rows), not capacity.

### C7. Resume is a property of the output file, not the script

Collectors write append-safe JSONL and call `already_done()` on startup,
which reads the output and skips any `(proposal_id, sample_i)` already
present. Resume is bit-stable: an interrupted run picks up at the first
missing sample, and seeds are deterministic
(`seed_base + 17*proposal_id + sample_i`), so an in-flight sample cut off
mid-generation regenerates bit-identically. The output file *is* the
resume state.

**Apply:** when writing a collector, make the output append-safe and key
each row by a deterministic identity. On startup, read the output and skip
done keys. Never regenerate or truncate existing rows — they are
evidence (`OPERATIONS.md`). If a row is wrong, leave it and annotate.

### C8. Data filenames carry identity, not description

Every datum has **two names** when written through `datasave()`: the on-disk
filename (`path`) and the registry metadata (`name` + `title` + `description`
+ `tags`). They serve different lookups. The filename is for browsing
`data/`; the registry fields are for `Registry.find_artifacts()` across all
runs. Don't stuff the description into the filename — that's what `title` and
`tags` are for.

The filename carries **identity** (what distinguishes this file from its
siblings in `data/`), not category, method, or counts:
- **No category/experiment prefix** (`acegen_`, `madlibz_`) — the directory
  is already under `ace/data/`.
- **No method prefix** (`teacherforced_`) — the extension + the script that
  reads it carry the method.
- **No model name** unless `data/` holds outputs from multiple models (then
  it's identity; until then it's noise).
- **No leaked counts** (`_30`, `_x8`) — the row count is in the file.
  **Stable batch identity stays** (`b1`, `pass1`/`pass2`, `gpu0`/`gpu1`) —
  those distinguish siblings.
- **What stays**: role (`frontier` / `scan` / `map` / `rollouts` /
  `candidates`) + stable identity.

The test: **would a reader inside `data/` need this token to disambiguate
the file from its siblings?** If not, drop it.

When you write through `datasave`, pass a clean `path` AND a `title` +
`description` + `tags`:
```python
datasave("frontier_rollouts_pass1.jsonl", rows,
         title="Frontier rollouts pass 1",
         description="8 samples × 30 authored frontier problems, Qwen3.5-9B bf16, thinking on",
         tags=["frontier", "rollouts", "pass1"], manifest=m)
```
For data written before `datasave` existed (hand-written JSONL with
`open()`), write a `.meta.json` sidecar by hand matching `datasave`'s schema
so a browser still sees what each file is. Don't re-write the data through
`datasave` retroactively — that changes the bytes of an evidence artifact.

## When something breaks

**Gather state before acting.** The cheap hypothesis is usually wrong,
and 30 seconds of `ps`/`nvidia-smi`/`grep` falsifies it. This session:
the dashboard showed 0 workers; the cheap hypothesis was "the rename
killed the runs"; the evidence (processes at 98% util, both alive)
falsified it in seconds — the actual cause was a `pgrep` pattern in
`gen/probe_stat.py` updated to match the new filename while the running
processes still had the old one. The workers never died; the *detector*
broke. **When the symptom is "X stopped working," distinguish "X broke"
from "the thing that reports X broke."** Probes, dashboards, and
detectors fail more often than the work they report on.

## Writing evidence (ace-specific)

Lab-note and session-note **protocols** are in the parent
`mlfactory/AGENTS.md` — follow them. The ace-specific additions:

- A lab note's `Decisions` section **must name every doc it changes** —
  the write-back manifest (pattern 7).
- Verdict content goes in the topical doc (`OBSERVABLES.md`,
  `LAYER_HYPOTHESES.md`, `FAILURE_MODES.md`, `CALIBRATION.md`); `STATUS.md`
  records only the resolution + pointer.
- Resolved questions can re-open (the map's loop-onset probe went
  `RESOLVED: infeasible at n=4` → re-opened needing more data). When new
  evidence re-opens a question, move it back to Open and note why; don't
  delete the prior resolution.

## Writing documentation — voice and frame

These are technical documents inside an ML experiment. Write them as
reference material for a future engineer who needs to act, not as a
blog post for a reader who needs to be persuaded. The audience is
technical; treat it as such. Two recurring failures to avoid, drawn from
this session's own drafts:

**Verbosity — restating what a heading or neighbor already says.** The
recurring move is introducing a section with a sentence that paraphrases
its own heading, then writing the content. In a doc with a descriptive
heading, the intro sentence is overhead. Test: *if the heading and the
first content line already convey the point, the intro sentence is the
defect.* The same applies to closing sentences that restate the section's
lesson, and to bridge paragraphs between two tables that say the same
thing in different words. Cut the restatement; keep the content.

**Jargon without merit — words that sound technical but add no meaning
over plain English.** Distinct from *earned* domain language
(rank-biserial, Frobenius norm, residual stream, recurrent state, bf16,
fork, passenger test, band, KL k3, AUROC), which carries precise technical
meaning and stays. The test: *would a plain-English replacement lose
meaning?* If not, the term is decoration. This session's own offenders:
"phenomenology" for "failure modes," "meta-discourse momentum" for
(defined-then-cut, nothing), "lensing" for "perspective/vocabulary," "the
error this pass exists to correct" for "the error this pass corrects."
Anthropomorphism is a subcase — "the pooled view has already lied once"
means "pooled confounds with prompt difficulty" and the latter is the
usable claim.

**The frame, not just the voice:** write as if the reader will *act* on
the next sentence, not grade the prose. State the claim, the evidence, the
constraint — then stop. A reader who needs motivation can read the
hypothesis doc; a reader scanning `OPERATIONS.md` for the bf16 ruling
doesn't need a paragraph on why bf16 matters, they need the ruling and a
one-line reason. Depth lives in the concept docs; the reference and
living docs are lookup material. When a concept genuinely needs setup,
that setup goes in the concept doc once and is linked, not re-motivated
in every doc that touches it.

**Apply:** before saving a doc, scan it for (1) sentences that restate
the heading above them, (2) terms a plain-English word would replace
without loss, (3) anthropomorphisms that dramatize a claim instead of
stating it, (4) paragraphs that motivate what the next paragraph says.
Cut all four. The earned domain language stays; the decoration goes.

## The one place you must think

Concept-doc refinement (`HYPOTHESIS.md` especially) is not mechanical. The
`Refined by:` pointers make write-backs visible and auditable, but
deciding whether a sharpening from one family is durable enough to absorb
into the core claim is judgment. A lab note sharpening "thrashing =
revisits without pruning" into "expansion into states disjoint from the
live trajectory" (machine, one family) is provisional; absorbing it into
`HYPOTHESIS.md` as if established would be premature. The patterns make the
*living* docs mechanically maintainable; concept docs still require a
reader to weigh evidence. When in doubt, leave a `Refined by:` pointer and
let the claim stay as-is until the sharpening replicates.
