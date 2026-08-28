# Compute strategy — sequencing spend under a fixed budget

> Update when: the training-staging strategy, the incremental evaluation
> cadence, the budget envelope, or the scale-up plan changes. Companion to
> `TERMINAL_FORK_COMPUTE.md` (the problem this spends against) and
> `ANNOTATION_SIDESTEP.md` (the detection-first candidate source) — this doc
> states *how* compute spend is sequenced and gated so the fork phase runs
> within a fixed, hobbyist-sized budget, and how the approach scales to a
> larger model. 9B figures reuse measured scenario estimates from
> `TERMINAL_FORK_COMPUTE.md`; **every larger-model figure here is an estimate,
> not a measurement** — validate each rung on the new substrate before
> spending the next.

## 1. The strategy in one line

Do not commit to the full fork-phase compute upfront. Gate every spend on
evidence, cheapest signal first, and stop the moment a rung returns null.
The full-commitment figure is a worst case that assumes the controller is
already worth the full loop; the incremental ladder spends a small fraction
of it to find out whether that assumption holds. This is what makes the
experiment feasible on a fixed personal budget: the envelope is the sum of
the rungs actually passed, not the committed total.

## 2. What the fork phase costs and why

The expensive term is **advantage estimation** (terminal rollouts from fork
points), not candidate generation — the annotation sidestep already moved
candidate generation to scenario-A cost.

- **GRPO (cheap phase, no forks):** ~33 GPU-h. Trains an approximate
  controller from terminal reward. This alone is *cheaper* than one R4-shaped
  fork pass.
- **Fork-based refinement (expensive):** fork advantages are measured for the
  current controller's state distribution, so every training update requires
  re-forking (on-policy re-evaluation, `TERMINAL_FORK_COMPUTE.md` §6). With
  blind candidate placement the doc's scenario E is ~7,500–80,000 GPU-h for a
  12-iteration loop. With detector-nominated placement (scenario F shape,
  ~85 GPU-h per pass at m≈32) the same loop is **~1,000 GPU-h at m≈32,
  ~2,600 at m=80**.
- **The exponential** (forks of forks, interaction resolution) is structural
  and is not addressed by this doc — `TERMINAL_FORK_COMPUTE.md` §5.

## 3. The free baseline: a fixed direction steers without training

A fixed steering direction applied at a detector-nominated state alters
autoregressive yield with no controller training. R4's hook is exactly this:
during decode it adds `Δh = λ·d` to the residual at the focal layer on every
generated token (`annotate/fork_r4.py:make_hook`). It works in R4 because the
fork position is known in advance from the annotation.

For live steering (fork position not known in advance) the pieces assemble:

- **Detector** (R2 probe) reads `h_t` each decode step → the *when* (gating).
- **Fixed direction** (R3 mean-difference) → the *how* (`Δh = λ·d`).
- Join: `if detector(h_t) > τ: h_t += λ·d`.

This is a controller — a degenerate one, learned offline (detector from
annotations, direction as a mean difference), not from terminal reward during
generation. Two distinct kinds of conditionality:

| | *When* to fire | *How* to steer | Learns from terminal reward |
|---|---|---|---|
| Detector + fixed direction | conditional (detector) | fixed | no |
| Learned controller | conditional | conditional | yes |

The fixed-direction policy is the `TERMINAL_FORK_COMPUTE.md` §9.7 baseline
the learned controller must beat. R4 validates whether it steers causally at
all.

## 4. What training actually buys

The learned controller (`core/steering_controller.py`) makes both direction
and magnitude functions of the current state:

```
z     = SiLU(down(LayerNorm(h_t)))
d     = tanh(up(z))          # direction depends on h_t
scale = alpha * ||h_t|| / sqrt(4096)
g_t   = sigmoid(gate(z))     # magnitude depends on h_t
h'_t  = h_t + g_t * scale * d
```

This is the conditional *how*: intervene harder in one state, gentler in
another, possibly a different direction entirely. It is not a nice-to-have —
it is what the hypothesis needs. The explore→reheat→prune shape is temporal:
the right intervention changes over the course of a trace. A fixed direction
is temporally static; only a state-tracking controller can express "expand
now, prune later." The training is therefore the hypothesis under test, not an
optional refinement.

## 5. The incremental training ladder

Cheapest signal first; each rung gates the next spend. Stop at any rung.

| Rung | Spend | Learns | If null |
|---|---|---|---|
| 1. GRPO smoke | ~8 GPU-h, few iters | Does it learn more than a weak bias? Is the gate firing? (First attempt learned only a weak bias — substrate too easy, `STATUS.md` Q10.) | Diagnose, don't continue |
| 2. Fork evaluation | ~100 GPU-h, one R4-shaped pass on the trained controller | Passenger test: steered vs no-op on forked outcomes | Kill or reshape |
| 3. Baseline comparison | same fork data | Does it beat the detector-gated fixed direction? | Fixed direction may suffice |
| 4. Fork-based refinement | incremental, ~100/pass | Per-state credit → refine → re-evaluate; still improving? | Stop at plateau |

Evaluation signals, in the order to trust them:

1. **GRPO reward curve** — free, in-loop. Advisory only; can look good while
   the controller is a passenger.
2. **Intervention magnitude** — free. Is the gate firing meaningfully, or
   collapsed to ~0?
3. **Forked outcome advantage** — ~100 GPU-h. The binding check (the passenger
   test); the only signal that separates "causes improvement" from
   "recognizes upcoming success."

Do not gate the continuation decision on signal 1 alone. Gate the refinement
money on signal 3, which costs the same as a single R4 pass.

Caveat: incremental evaluation risks over-reading early noise (chasing an
uptick, or killing a real-but-slow improvement). The pre-registered kill
conditions and the passenger test as binding evidence keep the checks honest;
rest the decision on forked outcomes, not GRPO reward.

## 6. The sidestep compounds

Two mechanisms shrink the fork bill without touching the evidence standard:

- **Annotation sidestep** (`ANNOTATION_SIDESTEP.md`): moved candidate
  generation from exhaustive forking to supervised detection — scenario B's
  blind ~368×3 states became scenario F's detector-nominated states.
- **Amortized counterfactual critic** (`COUNTERFACTUAL_FRAMEWORK.md`): trained
  on fork outcomes, predicts advantage from state so subsequent iterations
  refine on critic estimates instead of fresh forks, with periodic fork
  recalibration. Legitimate because it is learned from fork outcomes, not a
  proxy. Reduces the 12-iteration fork loop (~1,000) toward a few R4-sized
  passes plus recalibrations (~400).

## 7. Scaling to a larger model (27B)

Estimates, not measurements — no 27B numbers have been measured.

**What transfers:** the method — annotation rubric, the
capture→probe→direction→fork-validate ladder, the incremental gating. The
expensive discovery (does the approach work at all) is spent once on 9B.

**What does not transfer:** the specific probes, layers, and steering
directions — trained on 9B's residual stream. A 27B model has different
norms, layer structure, and representations. Re-do R1–R3
(capture→probe→direction) on 27B.

**Compute reality (expectations):**

- Worse than linear per token: ~3× params → slower bandwidth-bound decode,
  and larger cards (27B bf16 ≈ 54 GB > 48 GB A6000; needs an 80 GB-class
  card).
- The owned-hardware advantage disappears: hookable activations need bf16 HF
  (`OPERATIONS.md` substrate policy), which does not fit the owned 24 GB
  cards, so even the cheap capture rungs need rental.
- Both make the incremental ladder *more* important at 27B, not less.

**Ladder at 27B (same order):** (1) redo detector + fixed direction — if it
steers causally, that is a useful result without training a controller; (2)
only if it works and the explore→prune nuance is wanted, train the controller
incrementally.

**Caveat — substrate redraws trajectories.** Even q8 vs bf16 of the same 9B
model changed which failure modes appeared and where (cross-substrate result,
`STATUS.md`). 9B→27B is a far larger substrate change; expect failure
species, onset signatures, and focal layers to differ. The pipeline re-runs
cleanly, but assumptions need re-validation, not a clean port.

The 9B run is the de-risking run: if it works, carry the *method* (not the
weights) to 27B; if it does not, that was learned cheaply.

## 8. Current state (drifts — verify)

- **R4** (detector-nominated forks, fixed direction) running on 2× A6000,
  ~1,944 rows. Validates rung-2 machinery and whether the fixed direction
  steers causally. See `annotate/runbook_r4_cancun.md`, `STATUS.md`.
- **Controller training** not yet started. The GRPO substrate-ease issue from
  the first attempt (Q10) is the open item for rung 1.
- **27B** not begun.
