# Session notes — 2026-08-24 — b1 calibration repair, substrate ruling, layer map

Scope: b1 probe calibration (48 candidates × 8), verifier false-negative
repair + judge audit, q8-vs-bf16 substrate ruling, teacher-forced
multi-layer map, machine kill test, branch-dynamics reading.

## What exists now that didn't before

- 25-prompt LIVE pool `data/acegen_live_b1.jsonl` (b1 verdicts; later
  superseded as the training pool by b2's 46 — see 08-25 note).
- Repaired verifiers across 4 families (`gen/`): extraction loosened,
  semantics untouched (`lab_notes/2026-08-24-verifier-fix-and-teacherforced-scan.md`).
- Teacher-forced capture/analysis split established (`analysis/`):
  GPU capture writes `.npz`, CPU analysis re-runs cheap (pattern C5).
- Lab notes: b1-calibration-verifier-repair-judge-audit; substrate-ruling;
  substrate-delta-smoke; multi-layer-map; branch-dynamics-elimination-species;
  external-literature-steering-loops-monitors.

## Findings (the durable knowledge)

1. **Verifier false negatives made four families look far harder than
   they are.** Extraction regexes assumed canonical serialization the
   prompts never specified; after repair machine 93.8%, hypothesis 96.9%,
   assign 95.3%, certify 89.1%, grid 85.9% — all DEAD-EASY at default,
   only adversary (71.9%, 8/8 LIVE) calibrated. Evidence: re-scored b1
   batch + qwen38 judge cross-audit (STATUS R6/R7).
2. **Within-group variance is the acceptance unit, not family means.**
   The frozen-30 lesson reaffirmed at b1 scale: family averages
   concealed near-ceiling prompts carried by one live sibling.
3. **Substrate ruling:** q8_0 GGUF+MTP accepted for text-level
   collection/calibration; anything consumed by bf16 training or
   hidden-state access must be measured on bf16 HF. The delta smoke
   showed substrate changes *how* the model fails even where pass rates
   land comparably (adversary 71.9% bf16 → 43.8% q8 later in b2).
   q8-banded prompts owe bf16 re-verification by regeneration.
4. **Outcome-readable signal is not at L15.** L6/L17/L25 (linear) +
   L23 (full) + rec_2 (+0.759) carry it (R2/R3). Loop-onset separation
   infeasible at n=4 — data shortage, not method failure (R4 → Q5).
5. **ent_late killed within-prompt** — the pooled-view lie, replicated
   (R5). The branch-dynamics reading sharpened the thrash candidate:
   the poison is expansion into states disjoint from the live
   trajectory (one family, provisional).
6. External literature (steering loops/monitors note): fixed-direction
   steering papers cut tokens 41–71% without forked outcomes; decoding
   monitors catch loops without training — the Phase-3 competitor set.

## Decisions with rationale

- **Score of record = strict gen check();** collector soft-correct is
  advisory. Rejected alternative: judge-as-scorer — structurally
  unusable where multiple answers are valid.
- **q8 accepted for collection speed while the effect is unproven;**
  purity cost deferred via the re-verification obligation, not avoided.
- **Hard preset for the five too-easy families** rather than new
  families — difficulty is tunable inside the existing topology space
  (this became b2's assignment).
- **L15 dropped as measurement site;** steering layer stays undecided
  until Phase 3 (readability ≠ causal leverage).

## Environment traps encountered

- None recorded beyond what the lab notes carry.

## State at note time

- b1 pool landed; b2 hone assignment written
  (`specs/b2_hone_assignment.md`); Phase-1 kill tests largely complete;
  two llama-servers (q8+MTP) on :3091/:3092 for collection; GPUs
  otherwise idle. Next: b2 iterative honing loop.
