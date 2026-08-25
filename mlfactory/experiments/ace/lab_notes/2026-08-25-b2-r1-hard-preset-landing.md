# Lab note — 2026-08-25 — b2 round 1: hard preset lands high, failures are almost all truncations

Scope: b2 round 1 (120 rollouts: 15 candidates × 8 samples, Qwen3.5-9B
q8_0+MTP via llama.cpp :3091/:3092, thinking on, cap 26000). Candidates
= first 3 staged per too-easy family (`data/b2/r1.jsonl`, pids 57–59
machine, 65–67 assign, 73–75 certify, 81–83 grid, 89–91 hypothesis;
hard preset, machine at n_rejected=3). Rollout artifacts:
`data/acegen_b2_r1_gpu{0,1}.jsonl` (merged analysis copy
`data/b2/r1_rollouts.jsonl`).

## Band table (strict, after one machine verifier fix)

| pid | family | strict | band |
|---|---|---|---|
| 57 | machine | 7/8 | LIVE |
| 58, 59 | machine | 8/8 | DEAD-EASY |
| 65, 66 | assign | 8/8 | DEAD-EASY |
| 67 | assign | 5/8 | LIVE* |
| 73 | certify | 4/8 | LIVE* |
| 74, 75 | certify | 8/8 | DEAD-EASY |
| 81 | grid | 3/8 | LIVE* |
| 82 | grid | 6/8 | LIVE |
| 83 | grid | 8/8 | DEAD-EASY |
| 89 | hypothesis | 7/8 | LIVE |
| 90, 91 | hypothesis | 8/8 | DEAD-EASY |

Family: machine 95.8%, assign 87.5%, certify 83.3%, grid 70.8%,
hypothesis 95.8%. LIVE prompts: 1/3, 1/3, 1/3, 2/3, 1/3. The hard
preset moved all five families harder (b1: 93.8/95.3/89.1/85.9/96.9)
but only grid is near the band goal; the preset undershoots.

## Verifier fix (machine, extraction only)

`machine._fields` rej-count regex `\brejected\b\s*[=:]\s*(\d+)` matched
"rejected: 1" *inside* "first rejected: 1", reading the first_rejected
value as the rejection count. p57 s0 was a fully correct completion
scored WRONG. Fixed with negative lookbehind `(?<!first[ _])` and
`reject(?:ed|ions?)` (also accepts the synonymous label "Rejections").
Positive/negative controls + full `--self-test` pass. p57: 6/8 → 7/8.

## Headline finding: zero wrong-but-completed answers

All 16 wrong samples across all 15 prompts are 26k truncations with no
Answer line. Reading the tails splits them into two failure species:

1. **Closure loops (q8 substrate artifact).** assign p67 s0/s4/s5,
   machine p57 s7, hypothesis p89 s7: the model has (nearly) solved the
   problem, then loops "I'll write the final answer / I'll write the
   reasoning" until the cap. Not difficulty — emission paralysis
   (STATUS Q3) / terminal-loop territory (Q2). These failures make a
   prompt *look* harder than its search pressure is.
2. **Genuine budget-exhausted search.** grid p81 s0–s3+s7, p82 s0/s7,
   certify p73 s2–s4+s7: real systematic case-splitting / coloring
   construction that didn't converge in 26k. This is the productive
   failure species — the problem sat on the frontier.

Per-family read:

- **machine**: correct traces are careful step-by-step guarded replay
  (mean 17.6k tokens, dozens of self-checks) — genuine sustained
  attention, no instant insight, but 23/24 correct: log_len 15 with
  n_rejected=3 is not hard enough. The one failure was a closure loop.
- **assign**: mean correct trace only 8.7k tokens; the delayed-
  constraint decoy never bit — the model validates against all rules
  before answering, so no committed wrong answers. All 3 failures were
  closure loops after solving. Actual search pressure is low.
- **certify**: p73 (10 edges) 4/8 via budget-exhausted search; p74/p75
  8/8. Per-instance variance is high. The prose spoiler ("greedy in
  audit order forces a 4th channel, yet a 3-channel assignment exists")
  announces the trap instead of letting the model discover it — the
  prompt gives away the problem's shape.
- **grid**: best-calibrated family at n_pos=6. p81 (13 clues) 3/8, p82
  (14) 6/8, p83 (15) 8/8 — more clues is *easier* (clues are grown to
  uniqueness then pruned; count is an output, not a difficulty knob).
  Failures are genuine search exhaustion. High instance variance:
  difficulty lives in the inference shape, not the knob.
- **hypothesis**: pure small arithmetic, 95.8%. p89 s7 drifted on
  *terminology* ("what does 'expected cash' mean — counted or total?")
  — a semantic-confusion failure the prompt design invites. The knob
  axes (n_sales, spread) add computation, not reasoning.

## Decisions (round-2 hones, write-back manifest)

1. **machine**: log_len 15 → 17 (more replay horizon; feasibility
   window must still hold); keep n_states=6, n_events=7, n_rejected=3.
   If still too easy, next lever is structural (entangled guards /
   second integer register), not more length.
2. **assign**: n_items 7 → 8 (same 4 bins → more collision constraints,
   deeper case split). Structural candidate if still easy: bias the
   constraint pool toward implications (missed contrapositive is a
   plausible committed-error trap).
3. **certify**: remove the prose spoiler note — the trap must be
   discovered, not announced (prompt information design is a difficulty
   knob). n_nodes 8 → 9. Ensure NONE instances appear (all 3 staged
   r1 instances were solvable; generate a mix).
4. **grid**: keep n_pos=6; structural: add a 4th category (deeper
   constraint lattice), keep grow-to-unique + prune. Expect high
   instance variance → generate ≥4 and keep what lands.
5. **hypothesis**: structural: add VOIDED distractor records (must be
   excluded from the reconciliation — attention/reading trap, real
   register-close realism) and keep arithmetic small; the numeric
   knobs stay. Terminology drift (p89 s7) is a feature to keep, not a
   bug to fix, as long as the format line pins the report fields.
6. **Loop species**: record, do not retrofit a detector now (Q2's bet
   stands: CUSUM on a hidden-state probe, future batch). q8-banded
   prompts already owe bf16 re-verification by regeneration, which
   re-measures loop-prone prompts under the training substrate.

Round 2 candidates get fresh pids from 97. `CALIBRATION.md` write-back
deferred to b2 finalization (knob→difficulty map is still moving);
`STATUS.md` gets no resolution row yet — b2 is mid-flight.
