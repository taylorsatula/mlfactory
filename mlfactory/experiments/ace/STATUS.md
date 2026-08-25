# Status: the open-questions ledger

> Update when: **every batch**, and whenever a question resolves. This is
> the change-router: each row's `on resolve → update` column names the doc(s)
> to write back to when the question lands. Resolved rows get a dated
> `RESOLVED → see <doc>` pointer; the verdict *content* lives in the named
> doc, not here. A lab note's `Decisions` section is the manifest of what
> needs write-back — written at the moment evidence is fresh.

## Open questions

| # | Question | Current bet | Confidence | On resolve → update |
|---|---|---|---|---|
| Q2 | Terminal-loop early-stop in the collector (`stopped_reason=terminal_loop` + onset offset)? | yes for future batches (H200 harvest); do not retrofit b1 — method: CUSUM + persistence window on a hidden-state probe, not a token-entropy tripwire (3 sources agree entropy is the wrong trigger) | high | `OPERATIONS.md`, `core/collect_rollouts.py` |
| Q3 | Is emission paralysis common enough that a closure-nudge recovers meaningful reward mass? | yes — possibly highest-headroom single intervention class | moderate | `FAILURE_MODES.md`, `PHASES.md` |
| Q4 | Which Phase-1 observables survive the kill test cross-family? | survivors are ent_early, ent_trend, tortuosity, step_L6/L23/L25, frob_rec_14/18 (machine only) | moderate | `OBSERVABLES.md`, this row |
| Q5 | Does the loop-onset state become linearly separable with more loop traces? | re-opened: underpowered at n=4, not no-signal — regenerate hard-preset families to ≥50 loop traces, use CUSUM (not one-shot probe) + persistence window; input at ACE's stronger sites (L6/L17/L25, rec_2), not last-layer | moderate | `OBSERVABLES.md`, `LAYER_HYPOTHESES.md` |
| Q6 | Does the recurrent state carry drift/persistence info over time (not just final state)? | yes — v1 chunked-trajectory capture warranted | moderate | `LAYER_HYPOTHESES.md` |
| Q7 | Is counterfactual escape a *cause* of failure or a symptom? | symptom (model already lost the thread) — but unproven | moderate | `FAILURE_MODES.md`, `PHASES.md` (fork test) |
| Q8 | Does the on-path → off-path transition detect in residual/recurrent state? | yes, candidate layers L6/L23/L25/rec_14/rec_18 | moderate | `OBSERVABLES.md`, `LAYER_HYPOTHESES.md` |
| Q9 | Which layer carries causal leverage for steering (not just readability)? | unknown — readability map done, causal test is Phase 3 | n/a | `LAYER_HYPOTHESES.md`, `core/steering_controller.py` |
| Q10 | Does the controller learn a nontrivial state-dependent intervention from terminal reward alone? | first run learned a weak bias, not an explore/prune policy — substrate was too easy; bar sharpened: must beat a well-tuned constant λ on the pre-allocated reasoning-length axis (shown to exist, cosine≈0.99, R>0.8) on forked outcomes, not just a no-op | moderate | `HYPOTHESIS.md`, `PHASES.md` |

## Resolved questions

| # | Question | Verdict | Date | Evidence |
|---|---|---|---|---|
| R1 | certify at 2/24 soft-correct — dead-hard or per-prompt variance? | **DEAD-EASY at default knobs** (44/48 strict after extraction fix); needs hard preset | 2026-08-24 | `lab_notes/2026-08-24-verifier-fix-and-teacherforced-scan.md` → written back to `CALIBRATION.md` |
| R2 | Which layer carries the readable outcome signal? | **Not L15.** L6/L17/L25 (linear) + L23 (full); rec_2 (+0.759) strongest | 2026-08-24 | `lab_notes/2026-08-24-multi-layer-map-where-signal-lives.md` → written back to `LAYER_HYPOTHESES.md`, `OBSERVABLES.md` |
| R3 | Does the recurrent state carry information the residual does not? | **Yes.** rec_2/rec_9/rec_12 Frobenius-norm separation exceeds all residual layers | 2026-08-24 | same as R2 |
| R4 | Is the loop-onset state linearly separable from matched healthy states? | **INFEASIBLE at n=4** (degenerate probe, AUROC≈0.25) — data shortage, not method failure | 2026-08-24 | same as R2 → re-opened as Q5 |
| R5 | Which Phase-1 observables survive the kill test (machine)? | **ent_late KILLED within-prompt**; survivors listed in Q4 | 2026-08-24 | `lab_notes/2026-08-24-machine-kill-test-premature-commitment-survives.md` → written back to `OBSERVABLES.md` |
| R6 | Did the verifiers produce false negatives at scale? | **Yes** — extraction regexes assumed canonical serialization the prompts never specified; fixed at source | 2026-08-24 | `lab_notes/2026-08-24-verifier-fix-and-teacherforced-scan.md` |
| R7 | Do machine/adversary/grid/hypothesis land in band at default knobs? (b1 full calibration) | **Bet overturned.** Only **adversary** is well-calibrated at default (71.9%, 8/8 LIVE at depth 4). machine 93.8%, hypothesis 96.9%, assign 95.3%, certify 89.1%, grid 85.9% — all too easy → hard preset. 0 DEAD-HARD. Verifier false negatives repaired across 4 families; qwen38 judge audit cross-validated. 25-prompt LIVE pool | 2026-08-24 | `lab_notes/2026-08-24-b1-calibration-verifier-repair-judge-audit.md` → written back to `CALIBRATION.md` |
| R8 | Do the five too-easy families land in band at hard designs? (b2 iterative honing) | **Yes.** Four probe rounds moved each family from the HARD-preset prior to a locked design (grid's lever is clue composition, hypothesis's is VOIDED-record structure, certify's is trap non-announcement; numeric size knobs mostly add budget pressure, not reasoning). Final pool-expansion round: LIVE prompts adversary 8/8, machine 5/8, assign 4/8, certify 8/8, grid 5/8, hypothesis 3/8 → **46-prompt LIVE pool** (`data/acegen_live_b2.jsonl`), band spread 1/8–7/8, all q8_0+MTP — owes bf16 re-verification by regeneration before training | 2026-08-25 | `lab_notes/2026-08-25-b2-r1-hard-preset-landing.md`, `...-b2-r2-failure-species-taxonomy.md`, `...-b2-r3-r4-grid-clue-composition.md`, `...-b2-final-pool.md` → written back to `CALIBRATION.md` |
