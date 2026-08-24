# Lab note — 2026-08-24 — Machine kill-test: premature commitment survives, late-entropy champion dies

Scope: rigorous Phase-1 kill test on machine family, clean finishes only.
The doctrine's honest test is within-prompt: does an observable separate
correct from wrong samples of the SAME problem? Pooled numbers confound
with problem difficulty. This pass applies that test.

## The headline: our strongest pooled signal was a confound

**`ent_late` is dead.** Pooled rank-biserial +0.63 across all families.
Within machine: +0.037. Within-prompt across 3 mixed-outcome prompts:
p09 +0.25, p10 −0.60, p12 +1.00 — **sign-flipping.** The signal that
looked like our best finding was family composition: correct
certify/assign traces have high late entropy because those problems are
easy; wrong machine traces have low late entropy because machine is hard.
It never separated correct from wrong on the same problem. The pooled
view was lying. This is the test working as designed.

## What survived (sign-consistent across all 3 mixed prompts)

24 clean machine traces; 3 prompts with both outcomes (p09 4c/4w, p10 2c/6w,
p12 5c/3w). Verdict = sign of rank-biserial consistent across all 3.

| metric | p09 | p10 | p12 | direction |
|---|---|---|---|---|
| ent_early | −0.62 | −0.20 | −1.00 | wrong starts HIGHER entropy |
| ent_trend | +0.62 | +0.20 | +1.00 | correct rises, wrong falls |
| tortuosity | −0.75 | −0.40 | −0.33 | correct less tortuous |
| step_L6 | −0.25 | −0.20 | −0.33 | correct moves less early |
| step_L23 | +0.25 | +0.80 | +0.33 | correct moves more late-mid |
| step_L25 | +0.25 | +0.80 | +0.33 | same |
| frob_rec_14 | −0.38 | −1.00 | −0.33 | correct smaller recurrent state |
| frob_rec_18 | −0.31 | −0.60 | −0.33 | same |

Killed (sign-flipping within-prompt): ent_mid, ent_late, n_tokens,
recur_density, step_L15, step_L17, step_L31, frob_rec_2/9/12.

## The surviving story (provisional, 3 prompts)

Two patterns survived and they tell one story:

1. **Wrong machine traces start more confident and get more confident;
   correct traces start less certain and stay open or get more
   uncertain.** `ent_early` negative (wrong = higher starting entropy =
   premature commitment) + `ent_trend` positive (correct = entropy rises,
   wrong = entropy falls). The failure mode is the model locking in a
   trajectory before it has earned the right to, then gaining false
   confidence as it goes. Correct traces begin with genuine exploration
   and don't collapse. This is premature convergence with within-prompt
   evidence, not just a hypothesis anymore.

2. **The trajectory shape is layer-dependent.** Correct traces move
   LESS at L6 (early) but MORE at L23/L25 (late-mid). They consolidate
   early, then keep exploring late. Wrong traces do the opposite — churn
   early, lock in late. The recurrent state at L14/L18 is smaller for
   correct traces — less accumulated confusion in compressed memory.

## Connection to the counterfactual-escape observation

Reading the wrong machine traces revealed a specific failure mode: the
model escapes into counterfactuals about states the problem never
reaches ("what if n had been initialized to 1?", "if we were in ACTIVE,
could PAUSE fire?"). This is not "opens too many branches" in general
— on scheduling problems, correct traces opened MORE branches. It's
opening the WRONG branches: excursions into a different state machine
than the one being replayed.

This lines up with the kill-test survivors: wrong traces commit early
(low initial entropy = already on a locked path), then when stuck,
instead of backing up to re-examine the real problem, escape into
hypotheticals that can never be pruned (the trajectory never goes
there, so there's no outcome to learn from). It's not searching — it's
narrating, and the narration imports confusion back into the working
state.

**Honest scope:** we observed the counterfactual escape by reading
traces; we have NOT proven it CAUSES the failure. It might be a
symptom — the model already lost the thread, and hypotheticals are what
it does once lost. Telling those apart is the Phase-3 fork test's job.

## What this means for the controller (measurement, not reward)

"Is the model currently reasoning about a state the trajectory actually
reaches?" is a structural, verifiable property of the trace relative to
the problem. If the on-path → off-path transition is detectable in the
residual or recurrent state (the map suggests several candidate layers:
L6, L23, L25, rec_14, rec_18), that's a candidate fork point — the
moment the model stops re-examining the real problem and starts
escaping into hypotheticals. A controller that nudges it back to the
actual replay at that moment is exactly the "rescue the search it
already has" intervention class. It is a MEASUREMENT TARGET, never a
reward — per doctrine, anything that rewards how a trace looks instead
of where it lands is poison.

## Caveats on record

- **3 mixed prompts, 2-5 wrong each.** Sign-consistency across 3
  independent prompts is real but thin. Provisional keep, not confirmed.
- **p11 (5 clean, all wrong)** could not participate; with outcomes it
  might break consistency.
- **One family only.** ent_early and ent_trend must replicate on
  grid/hypothesis/adversary before being treated as general.
- **ent_late's death is machine-specific.** It might survive as a
  heuristic on families where it held pooled — but it is NOT a
  within-prompt signal and should not be trusted as one. If it can't
  separate same-problem correct from wrong, it can't guide a
  per-decision controller.

## State at note time

- Probe running (161/384, strict scoring, both GPUs).
- Survivors queued for cross-family replication when grid/hypothesis/
  adversary land.
- The counterfactual-escape detector (on-path vs off-path, in
  representation space) is the next instrument to build — it directly
  targets the failure mode the survivors describe.
