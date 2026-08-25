# Observed failure modes

> Update when: a failure mode is newly observed, replicates cross-batch, or
> its status changes. Each row carries a status. Evidence: lab notes under
> `lab_notes/`. These are *observations*; the observables used to measure
> them live in `OBSERVABLES.md`.

## Status key

`observed` (once) → `replicating` (seen again) → `stable` (cross-batch,
characterized). A phenomenon being stable does not make it a reward
(`REWARD_POLICY.md`) — it makes it a measurement target.

## Failure modes

| Phenomenon | Status | Evidence |
|---|---|---|
| **Emission paralysis / terminal verbatim loops** | stable | frozen-30 + b1 |
| **Loop onset gives no prospective entropy warning** | stable | b1 scan |
| **Premature commitment (low early entropy, falling trend)** | replicating | b1 machine kill-test |
| **Counterfactual escape (off-path expansion)** | observed | b1 machine trace-reading |
| **Length asymmetry (correct shorter than wrong)** | stable | frozen-30 |
| **Degeneration avalanche (coherent trace, collapsed surface)** | observed | legacy multi-model collection |

## What does NOT characterize success (cross-cutting)

Success is not characterized by less churn. Correct clean traces verify
as often as wrong ones, re-verify *more* within assign, and show comparable
"wait" rates within machine. **The distinguishing feature of success is
the exit, not the absence of the loop.** Any metric premised on "good
reasoning churns less" is measuring the wrong thing
(`lab_notes/2026-08-24-branch-dynamics-elimination-species.md`).

## Emission paralysis / terminal verbatim loops

The model can consolidate a correct solution in the working state and then
**fail to emit it**, falling into a self-reinforcing verbatim loop.
Observed directly: p26 s2 solved at ~8k tokens, then looped 24k of "I'll
write the response now".

- Roughly **20% of cap-truncated traces** enter such loops; ~8% of
  truncated tokens are post-novelty-death.
- The failure lives in the dynamics, not the reasoning — driven by
  autoregressive self-reinforcement.
- **Loop-onset states are the prime fork candidates.** Solution present
  in working state, closure blocked — this is the highest-headroom
  intervention class identified so far. A closure-nudge that recovers even
  part of this mass is pure terminal-reward gain with no reasoning-quality
  change needed.

**b1 refinement (2026-08-24, corrected scoring):** the loop replicates
**cross-family** (machine, assign, certify, grid, hypothesis, adversary —
not just frozen-30/machine), but two nuances cut the headroom estimate:
(1) **truncation is not failure** — 24/63 truncated rows still emitted a
*correct* answer before the loop/cap; the loop frequently sits on top of a
completed answer rather than blocking it. The pure emission-blocked mass
(truncated loop, no answer emitted) is ~13/384. (2) On the one genuinely
hard family (adversary), the dominant failure is **search-budget
exhaustion** (truncated mid-reasoning, still exploring at the cap — 13 of
18 wrongs), not closure-blocking. Q3's "closure-nudge is the
highest-headroom single intervention" bet is **weakened at default knobs**:
the bigger b1 failure mass is genuine hard search hitting the token cap.
Caveat: this is measured at default knobs where most families are too easy;
the failure mix may shift at hard presets. Evidence:
`lab_notes/2026-08-24-b1-calibration-verifier-repair-judge-audit.md`.

## Loop onset gives no prospective entropy warning

Post-onset entropy collapses to 0.01–0.04 in all 9 loop traces (verbatim
repetition is near-deterministic), but **pre-onset entropy is bimodal**:
some traces decay into onset (p01_s1: 0.065 pre vs 0.43 baseline), others
arrive from healthy entropy (p08_s1: 0.64 pre, above baseline).

> Phase-3 fork placement **cannot use an entropy tripwire**. Candidate
> states must come from the controller's intervention magnitude or a
> trained state detector.

## Premature commitment (provisional, machine family)

Wrong machine traces start more confident and get more confident; correct
traces start less certain and stay open or get more uncertain. `ent_early`
negative (wrong = higher starting entropy = premature commitment) +
`ent_trend` positive (correct = entropy rises, wrong = entropy falls),
sign-consistent across 3 mixed machine prompts. The failure mode is the
model locking in a trajectory before it has earned the right to, then
gaining false confidence.

- **Status:** replicating (3 prompts, 1 family — thin). Must replicate on
  grid/hypothesis/adversary before treated as general. See
  `OBSERVABLES.md` and `lab_notes/2026-08-24-machine-kill-test-premature-commitment-survives.md`.

## Counterfactual escape (observed, machine)

Reading wrong machine traces: the model escapes into counterfactuals about
states the problem never reaches ("what if n had been initialized to 1?",
"if we were in ACTIVE, could PAUSE fire?"). This is not "opens too many
branches" in general — on scheduling problems, correct traces opened MORE
branches. It's opening the **wrong** branches: excursions into a different
state machine than the one being replayed. Expansions into states disjoint
from the live trajectory cannot produce durable pruning.

- **Honest scope:** observed by reading traces; NOT proven to *cause* the
  failure. Might be a symptom — the model already lost the thread, and
  hypotheticals are what it does once lost. Telling those apart is the
  Phase-3 fork test's job.
- **Refined-by pointer:** this sharpened the core thrash hypothesis
  (`HYPOTHESIS.md` §refined-by-evidence).

## Length asymmetry

Correct traces are shorter than wrong ones (frozen-30: 22k mean vs wrong
median = 32k cap). Length is an *outcome* of search quality, **never a
target**. Rewarding shortness directly would reward premature convergence
(`REWARD_POLICY.md`).

## Degeneration avalanche (distinct from terminal verbatim loops)

A separate failure genus observed in the legacy multi-model collection
(`ace-legacyapproach/AFTER_ACTION_2026-08-12_first_test_prompt.md`):
**coherent deliberation, collapsed surface.** Qwopus3.6-27B reasoned
correctly through the trace, then the final answer derailed mid-timeline
into a synonym avalanche ("…seamlessly integrating smoothly transitioning
forwardward progression onward onwards ever upward climb…") through lab
equipment, geological strata, forest biomes, geometric shapes, and finally
pseudo-inflected word salad. The model emitted meta-apologies mid-collapse
("Sorry, my internal monologue drifted off there!") and could not stop the
recurrence — **self-detected degeneration without recovery capacity.**

This is distinct from the terminal verbatim loops above: the loop is a
*near-deterministic* repetition of a fixed unit at near-zero entropy; the
avalanche is a *drift* through semantically-adjacent vocabulary at
non-trivial entropy, ending in non-word salad. The loop is autoregressive
self-reinforcement of a fixed state; the avalanche is autoregressive
runaway through a degenerate manifold.

- Deterministic under (prompt, seed, sampling) — byte-identical on rerun.
- Not a cursed prompt: a reworded prompt with the same facts produced a
  *different* avalanche. The sampling profile (temp 1.0 +
  presence_penalty 1.5) was hostile to this model.
- **Degeneration detection must be structural, not keyword-based.** A
  marker list tuned to the first avalanche's vocabulary scored 0 on the
  second. Proper detectors: sliding-window lexical diversity, n-gram
  repetition rate, output-length anomalies. This is an operations ruling,
  on record in `OPERATIONS.md`.
- Status: `observed` (legacy collection, one model family); not yet seen in
  the b1 Qwen-only run, but the detector principle stands regardless.
