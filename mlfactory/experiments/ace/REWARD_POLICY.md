# Reward Policy

> Update when: a reward term is considered, added, or banned; or the
> baseline method changes. This is the **binding do/don't list** — keep it
> short. The *why* (forks, passenger test) lives in
> `COUNTERFACTUAL_FRAMEWORK.md`.

## The only aligned signal

**Terminal verifiable reward.** Value is defined ONLY by terminal outcome
under a strict verifier. Nothing else is a reward, ever.

## Baselines

| Baseline | Status | Why |
|---|---|---|
| GRPO group baseline | **Essential** | The pool is deliberately heterogeneous in hardness; an unnormalized baseline conflates problem difficulty with intervention value. Normalizes away prompt difficulty in the broad phase. |
| Counterfactual no-op fork | The attribution method | Solves variance over 20k+ token traces; defines advantage `A = E[R\|s,a] − E[R\|s,no-op]`. |
| Amortized counterfactual critic | **Later** | A value function trained on fork outcomes — the legitimate dense signal. Downstream of fork machinery; does not exist before it. |

### Phase-3 fork competitors (no-op-equivalent baselines the controller must beat)

All training-free or cheap; a learned controller that only matches these
on forked outcomes has no value claim. External evidence: steering
papers (Manifold 2505.22411, RSP 2506.08390) use **one global, fixed-sign**
residual-stream direction and already cut tokens 41–71% with accuracy held or
improved on Qwen2.5-distills; decoding monitors (Loops 2601.05693,
e-CUSUM 2607.11317) catch loop onset or backtrack-and-re-explore without
training. None pass anything like the passenger test (no forked outcomes).
Run the controller against at least the fixed-direction and logit-penalty
baselines at the Phase-3 gate; if a fixed direction already matches on
forked outcomes, the controller line's value claim collapses.

| Competitor | Class | What it does |
|---|---|---|
| Lotfi logit penalty (per e-CUSUM §2) | training-free, decode-time | Penalizes `wait`/`but`/`alternatively` at high-entropy positions; cheapest baseline. Blunt — also hits success-leaning decisive replacers (`actually`/`however`/`just-to`). |
| Manifold fixed direction | training-free, fixed-sign | Difference-in-means overthinking direction, PCA-projected onto the low-dim activation manifold to null interference noise; applied at all layers/positions. |
| RSP constant-λ additive steering | training-free, fixed-sign | Add along the pre-allocated reasoning-length direction (cosine≈0.99 across difficulties); moves the end-of-reasoning logit; positive λ improves hard-math accuracy up to a ceiling. |
| CUSUM early-stop / early-intervention (Loops) | training-free, decode-time | CUSUM + persistence window on a hidden-state probe score; detects loop onset ~1.8k tokens early on Qwen3-8B. |
| e-CUSUM backtrack + re-explore | training-free, decode-time | Calibrated e-CUSUM on fused entropy-spike + verbatim-repetition alarm; backtracks + re-explores rather than terminates. |

The fixed-direction baselines are the load-bearing competitors: ACE's
own failure-mode taxonomy contains **opposite** pathologies (emission
paralysis vs premature commitment), so a global fixed-sign intervention
helps one and worsens the other — the precise argument for a
state-dependent controller, and the gap these baselines cannot close.

## Forbidden as rewards (diagnostics only, forever)

These are candidate *diagnostics* (`OBSERVABLES.md`), measured against
outcomes, never optimized:

- entropy (token or branch)
- tortuosity
- recurrence
- length
- RLAIF / judge scores
- any PRM-style step scorer

**Scope of the ban:** it governs what the controller is *trained
against* — reward and optimization targets. A judge that reads forked
windows to interpret what an intervention did (R4v2) is a measurement
instrument over yielded tokens, not a training signal, and is not
banned by this section (principal rulings 2026-08-28:
`lab_notes/2026-08-28-r4-attendance-stopped-design-change.md`,
redesign rationale, and the R4v2 build note in this week's lab notes).
What remains
forbidden: judge verdicts entering controller gradients, or replacing
terminal verified reward anywhere in training.

## The local-proxy trap

Any step-level scorer trained on what good reasoning *looks like* will
penalize productive struggle and reward premature convergence. **This is
the failure mode the fork architecture exists to avoid.** A locally ugly
expansion that eventually escapes a local maximum must beat a locally
elegant intervention that forces premature commitment.

## What the controller is NOT trained to do

Minimize entropy, recurrence, tortuosity, or trace length. Rewarding
shortness directly would reward premature convergence (length is an
*outcome* of search quality, never a target — `FAILURE_MODES.md`).

## Backdoor to watch: truncation

Terminal reward on a capped rollout leaks length in through the
backdoor: a trace that hits the backstop cap almost always scores 0, so
"finish before the cap" is reward-correlated signal shaped like the
banned length axis without being added as a term. Measured: 22% of q8 b2
rollouts and 6/24 first-bf16-smoke traces hit the 26k cap. Report
per-group cap-hit rate in every GRPO batch; all-truncated groups have
zero advantage anyway; treat the cap as a per-row covariate, never as a
silent scorer. (Evidence: `lab_notes/2026-08-25-grpo-h200-smoke-results.md`.)

## Quick reference: decision test

> "Is this thing I want to add a reward?" — If it is anything other than
> terminal verified outcome, **no**. It goes in `OBSERVABLES.md` as a
> measured diagnostic, and survives only if a kill test says it tracks
> outcome within-prompt.
