# Phase 0 Doctrine: Calibration Substrate and Operative Rulings

> Companion to `NEW_UNDERSTANDING_OF_WHAT_ACE_IS.md`. That document states the
> ACE hypothesis and the counterfactual-credit architecture. This document
> records the operative concepts that govern Phase 0 (LIVE-pool calibration)
> and constrain every later phase. If a decision in this program cannot be
> traced to one of these rulings, it is ad hoc and should be re-examined.

## 0. The shortest version

Outcome defines direction. Counterfactuals solve attribution. Calibration
selects the substrate. Anything that rewards how a trace looks instead of
where it lands is poison.

## 1. Reward and credit (decided — not open for re-litigation)

- **Terminal verifiable reward is the only aligned signal.** Value is defined
  ONLY by terminal outcome under a strict verifier.
- **Variance over 20k+ token traces is solved by counterfactual forks**, not
  proxy rewards: `A(s,a) = E[R|s,a] − E[R|s,no-op]`.
- **GRPO group baselines** normalize away prompt difficulty in the broad
  phase. This is essential, not optional: the pool is deliberately
  heterogeneous in hardness, so an unnormalized baseline conflates problem
  difficulty with intervention value.
- **Amortized counterfactual critic later**: the legitimate dense signal is a
  value function trained on fork outcomes. It is downstream of fork machinery
  and does not exist before it.
- **FORBIDDEN as rewards** (diagnostics only, forever): entropy, tortuosity,
  recurrence, length, RLAIF/judge scores, any PRM-style step scorer.
- **The local-proxy trap**: any step-level scorer trained on what good
  reasoning *looks like* will penalize productive struggle and reward
  premature convergence. This is the failure mode the fork architecture
  exists to avoid.

### The fork test is the passenger test

Correlation between intervention and outcome proves nothing. A controller
that fires preferentially in states that were already going to succeed is a
passenger. Only forked outcome distributions — same prefix, steered vs.
no-op, both run to terminal verification — separate "controller causes" from
"controller recognizes." Any evidence offered for controller value that is
not a fork comparison is advisory.

## 2. Acceptance unit (Phase 0 gate)

- **Within-group variance is the acceptance unit.** All-wrong and all-right
  prompt groups both yield zero advantage under group-relative RL — dead
  weight regardless of how interesting the prompts look.
- **Per-prompt band membership, not domain averages.** Frozen-30 taught this
  directly: domain averages concealed 0/8 prompts carried by successful
  siblings. Band = 1..6 of 8 correct; **prefer 2..5 of 8**.
- **The collector's soft `correct` field is advisory only.**
  `gen/calibrate.py` re-scores every completion against the strict per-family
  `check()` verifier and is the sole authority on band classification
  (DEAD-HARD / LIVE / DEAD-EASY).

## 3. Substrate: reasoning-shaped vs. computation-shaped

A 9B can do constraint satisfaction, guarded replay, bounded search, and
small-hypothesis elimination. It cannot enumerate combinatorial objects,
simulate machines at length, or do exponential optimization. Dead domain
shapes are 0% possible — computation, not reasoning — and no knob tuning
rescues them. The transformation catalog preserves search topology while
removing computational impossibility:

| Dead shape | Transformation |
|---|---|
| Enumeration ("count all X") | Certification ("verify/complete this X") |
| Exponential optimization | Bounded construction with a feasibility budget |
| VM/code simulation | Decisive-discrepancy replay (short trace, one guarded trap) |
| Oversized bookkeeping | Small-hypothesis elimination (3–5 candidates) |

The generator (`gen/`) embodies this: solver-built exact answers by
construction, strict structural verifiers, scalar difficulty knobs per
family. Knobs + measured outcomes form the closed calibration loop:
generate → probe → strict re-score → per-prompt band verdict → regenerate
dead families at adjusted presets.

## 4. Zero-init transfer argument (why base-model probing is valid calibration)

The controller is zero-initialized, so at training step 0 it *is* the base
model. Probing the base model's per-prompt success distribution therefore
calibrates the exact substrate the controller starts from. Controller drift
under terminal reward is one-directional (upward — toward more solvable),
so **calibrate to the hard side of the band**: a prompt at the LIVE boundary
for the base model drifts into the productive zone; a prompt already easy
drifts out to DEAD-EASY.

## 5. Phenomenology (empirical, frozen-30 + b1 observations)

- **Emission paralysis / terminal verbatim loops.** The model can
  consolidate a correct solution in the working state and then fail to emit
  it, falling into a self-reinforcing verbatim loop (observed: p26 s2 solved
  at ~8k tokens, then looped 24k of "I'll write the response now").
  Roughly 20% of cap-truncated traces enter such loops; ~8% of truncated
  tokens are post-novelty-death. The failure lives in the dynamics, not the
  reasoning — driven by autoregressive self-reinforcement, closure
  undertraining, and meta-discourse momentum.
- **Loop onset states are the prime fork candidates.** Solution present in
  working state, closure blocked — this is the highest-headroom intervention
  class identified so far. A closure-nudge that recovers even part of this
  mass is pure terminal-reward gain with no reasoning-quality change needed.
- **Length asymmetry.** Correct traces are shorter than wrong ones (frozen-30:
  22k mean vs. wrong median = 32k cap). Length is an *outcome* of search
  quality, never a target. Rewarding shortness directly would reward
  premature convergence.

## 6. Standing operational rulings (binding)

- **Generation protocol**: one sample at a time (24 GB VRAM), thinking
  enabled, **bf16** (q8 rejected — different model; all seeds are
  bf16-calibrated). Per-sample seed = `seed_base + 17*proposal_id + sample_i`.
- **Backstop cap: 26000 tokens** (reduced from 32000). Rationale: terminal
  loops do not contribute to identical-conditions comparison; a trace that
  needs >26k to be right is a recorded model flaw. **NEVER regenerate or
  truncate existing rows** — chop analytically at analysis time if needed.
- **Hardware staging**: local 3090s are the falsification engine through the
  Phase 3 fork-causality gate. H200 (Vast.ai, **Hopper, NOT Blackwell**) only
  after fork delta is proven. Runbook: `/home/admin/BABYS_FIRST_VAST_ML_ENGINEER.md`.
- **llama-server stays off** (user ruling; frees GPU0 desktop contention).
- The generator is fresh task-specific code (`gen/`), not a madlibz rework.

## 7. Method: falsification-first staging

Cheap hardware kills ideas; expensive hardware exploits survivors. Every
phase emits a *decision*, not just data. Kill criteria are pre-registered so
that a null result terminates the line instead of inviting more tuning:

- **Phase 0** (current): calibrate a living LIVE pool per §2. Output: an
  accepted pool with per-prompt band provenance, or a finding that a family
  cannot be tuned into band (kill or transform that family per §3).
- **Phase 1**: teacher-forced diagnostics on collected traces. Test whether
  recurrence / tortuosity / entropy / loop-onset features correlate with
  terminal outcome. **Kill every metric that does not.** Survivors become
  candidate intervention-state features — diagnostics, never rewards.
- **Phase 2**: replay memory engineering comes FIRST — leaning
  fork-from-prefix, which doubles as Phase 3 fork machinery.
- **Phase 3**: fork causality gate. Steering value must survive the
  passenger test (§1) on forked outcome distributions. Null result here
  terminates the controller line.

## 8. Open questions carried by this phase

| Question | Current bet | Confidence |
|---|---|---|
| Do machine/adversary/grid/hypothesis land in band at default knobs? | grid and machine live; adversary at depth 4 hard | genuinely uncertain |
| certify at 2/24 soft-correct — dead-hard or per-prompt variance? | **RESOLVED: verifier artifact.** certify is DEAD-EASY at default knobs (44/48 strict after extraction fix); needs hard preset | high |
| Terminal-loop early-stop in the collector (`stopped_reason=terminal_loop` + onset offset)? | yes for future batches (H200 harvest); do not retrofit b1 | high |
| Is emission paralysis common enough that a closure-nudge recovers meaningful reward mass? | yes — possibly highest-headroom single intervention class | moderate |
| Which layer carries the readable outcome signal? | **RESOLVED: not L15.** Signal lives at L6/L17/L25 (linear-attn) + L23 (full). Recurrent channel rec_2 (+0.759) stronger than any residual layer | high |
| Does the recurrent state carry information the residual does not? | **RESOLVED: yes.** rec_2/rec_9/rec_12 Frobenius-norm separation exceeds all residual layers; v1 chunked-trajectory capture warranted | high |
| Is the loop-onset state linearly separable from matched healthy states? | **INFEASIBLE at n=4.** Only 4 usable onset traces vs 128 healthy; degenerate probe (AUROC≈0.25). Needs more loop traces (hard-preset regeneration) | n/a — underpowered |
| Which Phase-1 observables survive the kill test? | **ent_late KILLED within-prompt** (pooled +0.63 was composition; sign-flips in machine). **Survivors (machine, 3 prompts):** ent_early (−), ent_trend (+), tortuosity (−), step_L6 (−), step_L23/L25 (+), frob_rec_14/18 (−). Provisional — 3 prompts, 1 family | moderate |

Resolved verdicts get written back into this document as they land, with the
b1 calibration output (`data/acegen_live_b1.jsonl`) as the first evidence
artifact. Evidence artifacts for the resolved rows above:
`lab_notes/2026-08-24-verifier-fix-and-teacherforced-scan.md`,
`lab_notes/2026-08-24-branch-dynamics-elimination-species.md`,
`lab_notes/2026-08-24-multi-layer-map-where-signal-lives.md`.
