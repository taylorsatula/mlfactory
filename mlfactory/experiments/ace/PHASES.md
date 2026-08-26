# Phases: falsification-first staging

> Update when: a phase is completed and emits its decision, or a kill
> criterion is added. Every phase emits a *decision*, not just data. Kill
> criteria are pre-registered so a null result terminates the line instead
> of inviting more tuning.

## Staging principle

Cheap hardware kills ideas; expensive hardware exploits survivors. A null
result at any phase gate terminates that line. Phases are sequential; later
phases assume earlier ones emitted a "proceed" decision.

```
Phase 0  calibrate a living LIVE pool
   ↓   (output: accepted pool with per-prompt band provenance)
Phase 1  teacher-forced diagnostics on collected traces
   ↓   (output: surviving observables; kill every metric that fails)
Phase 2  replay memory engineering FIRST (doubles as Phase 3 fork machinery)
   ↓   (output: fork-from-prefix capable)
Phase 3  fork causality gate — the passenger test
   ↓   (output: controller value proven on forked distributions, or kill)
```

## Phase 0 — calibrate the substrate (current)

**Output:** an accepted LIVE pool with per-prompt band provenance, or a
finding that a family cannot be tuned into band (kill or transform that
family per `CALIBRATION.md` §transformation-catalog).

**Gate:** within-group variance is the acceptance unit (`CALIBRATION.md`).
All-wrong and all-right groups both yield zero advantage — dead weight
regardless of how interesting the prompts look.

**Kill:** a family that cannot be tuned into band at any preset is dead;
either transform it per the catalog or drop it.

## Phase 1 — teacher-forced diagnostics

**Output:** surviving observables. Test whether recurrence / tortuosity /
entropy / loop-onset features correlate with terminal outcome. **Kill every
metric that does not.** Survivors become candidate intervention-state
features — diagnostics, never rewards (`REWARD_POLICY.md`).

**Gate (the honest test):** within-prompt separation. Does the observable
separate correct from wrong samples of the **same** prompt? Pooled numbers
confound with prompt difficulty. The pooled view has already lied once
(`OBSERVABLES.md`, `ent_late`). Within-prompt sign-consistency across mixed
prompts is the keeper test.

## Phase 2 — replay memory engineering FIRST

**Output:** fork-from-prefix capable. Long-context replay is the
prerequisite for Phase 3 and is engineered before it. Leaning
fork-from-prefix, which doubles as Phase 3 fork machinery.

**2026-08-25 measurement (H200, R9):** teacher-forced replay with
gradients fits a ≤8k-token window (111.6 GB peak at cap 8192; OOM at
16384; memory linear in window). Pool traces median 22.3k tokens, and
the hypothesis locates the learnable decision points (reheat, durable
pruning) mid-to-late trace (`HYPOTHESIS.md`) — so a single-prefix
window is structurally insufficient.

**2026-08-25, refined same day (Step 1, R10/R11):** gradient-checkpointed
**full-trace replay** replaces segmented (windowed) replay. Windowed
cache-continuation replay is KILLED, two independent reasons: (1)
numerical — with a zero-init controller (exact no-op), continuing a
cached prefix corrupts the first ~50 tokens after every split boundary by
up to 11.8 nats (likelihood ratios off >1e5; drift mean grows with split
depth 0.14 → 0.46); the cache object itself is bit-exact, the split is
the poison; (2) mechanical — the FLA gated-delta-rule chunk backward
crashes on the continued cache state, so gradients cannot complete. Full
replay (checkpointed layers + `_TokenLogprobs` custom autograd fn) is
bit-exact vs a single-pass reference at 3k AND 26k completion tokens,
backward finite, peak well inside the 140 GB card. Full-mode OOM is
guard trip exit 8 — there is no windowed fallback to degrade to.
Rollout generation additionally requires a deterministic SDPA backend
(default backend is call-to-call non-deterministic — R11).
Refined by: `lab_notes/2026-08-25-step1-replay-engine-windowed-killed.md`,
`...-2026-08-26-step1-sdpa-generation-nondeterminism-identity-gate.md`

## Phase 3 — fork causality gate

**Output:** steering value proven on forked outcome distributions, or the
controller line is killed.

**2026-08-25 attempt prep:** first serious attempt standing up on the
Vast H200 — thinking-on (ruling: the earlier thinking-off/short-cap
script shape was the wrong regime for this gate), b2 46-prompt pool,
gradient-checkpointed full-trace replay (windowed killed, R10), rollout
generation pinned to a deterministic SDPA backend (R11), 2-GPU rollout
parallelism. The bar is Q10's: beat a well-tuned constant-λ baseline —
one that steers along the pre-allocated (length-derived) fixed
direction — judged on forked terminal-verified outcomes (length is the
baseline's intervention axis, never a reward or metric —
`REWARD_POLICY.md`).

**Gate:** the passenger test (`COUNTERFACTUAL_FRAMEWORK.md`). Same prefix,
steered vs no-op, both run to terminal verification. Correlation between
intervention and outcome proves nothing; only forked outcome
distributions separate "controller causes" from "controller recognizes."

**Kill:** null result here terminates the controller line. No further
tuning.

## Cross-phase rules

- Diagnostics are never rewards, at any phase (`REWARD_POLICY.md`).
- Existing rows are never regenerated or truncated (`OPERATIONS.md`).
- Base-model probing is valid calibration because the controller is
  zero-init at step 0 (`CALIBRATION.md` §zero-init-transfer).
- Evidence artifacts: lab notes under `lab_notes/`; the verdict ledger is
  `STATUS.md`.
