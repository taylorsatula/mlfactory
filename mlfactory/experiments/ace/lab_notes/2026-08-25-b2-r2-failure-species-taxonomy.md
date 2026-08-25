# Lab note — 2026-08-25 — b2 round 2: machine/assign/certify land 3/3 LIVE; failure-species taxonomy

Scope: b2 round 2 (120 rollouts: 15 candidates × 8 samples, q8_0+MTP,
cap 26000). Candidates `data/b2/r2.jsonl` (pids 97–111): machine ll=17
with rejection window generalized to 1..max(3, ll//3) → 5 rejections;
assign n_items=8; certify n_nodes=9 with the prose spoiler note removed
(trap is discovered, not announced) + 1 NONE instance; grid n_pos=6
fresh seeds; hypothesis with new VOIDED-distractor structure (2 voided
sales). Rollouts: `data/acegen_b2_r2_gpu{0,1}.jsonl` (merged
`data/b2/r2_rollouts.jsonl`). Generator prose skins added after this
round — r2 measured pre-skin prose.

## Band table (strict)

| pid | family | strict | band |
|---|---|---|---|
| 97, 98, 99 | machine | 7/8 each | LIVE ×3 |
| 100, 101 | assign | 7/8 | LIVE |
| 102 | assign | 5/8 | LIVE* |
| 103, 104 | certify | 6/8 | LIVE (p104 = NONE instance) |
| 105 | certify | 7/8 | LIVE |
| 106, 108 | grid | 8/8 | DEAD-EASY |
| 107 | grid | 6/8 | LIVE |
| 109 | hypothesis | 6/8 | LIVE |
| 110, 111 | hypothesis | 8/8 | DEAD-EASY |

Family: machine 87.5% (3/3 LIVE), assign 79.2% (3/3), certify 79.2%
(3/3), grid 91.7% (1/3), hypothesis 91.7% (1/3).

## Headline: a failure-species taxonomy, built by reading every wrong trace

Counting alone would have read these bands correctly but hid *why*.
Every wrong sample was classified (script: read extracted answer line,
re-score it with the strict verifier, inspect tails for repetition):

1. **Emission paralysis on a CORRECT answer.** The model derives the
   right answer inside the think block, `
` closes with
   `visible_chars=0`, nothing emitted → scored wrong. machine p98 s5
   (exact reference in think), certify p104 s3/s7 (NONE, correct),
   p105 s5 (valid coloring), hypothesis p109 s2. STATUS Q3 confirmed
   live on this substrate at non-trivial rates (~4% of samples).
2. **Budget-exhausted search.** Truncated mid-replay / mid-case-split
   with partial or wrong answer line. machine (2), certify (2),
   hypothesis (1). Productive failure species.
3. **Closure loops.** rep-ratio > 1.9 on the tail ("I'll write the
   final answer" cycling): assign p100 s5, p101 s4, p102 s2. Solved or
   near-solved, then loops. Substrate artifact (q8; cf. machine p16 in
   the delta smoke).
4. **Committed wrong answers.** grid p107: 2 non-truncated wrong
   completions — the first genuine reasoning failures (not budget, not
   loops) seen in b2. This is the species calibration exists to
   produce; grid at n_pos=6 only produces it sporadically.

## Findings per family

- **machine** landed 7/8 ×3 at ll=17 / 5 rejections. Knobs are now at
  range max (n_states=6, n_events=7); remaining lever is length only,
  which increases budget exhaustion and cap-grinding — the wrong kind
  of hard. 7/8 is in-band (LIVE 1..7, sweet spot 2..7); accept.
- **assign** n_items=8 moved 87.5% → 79.2%, LIVE 3/3 but two prompts
  at 7/8. n_bins 4→5 next (higher branching factor), watching for
  closure-loop share.
- **certify** improved cleanly: removing the spoiler note plus
  n_nodes=9 gave 6/8, 6/8, 7/8 with the NONE instance at 6/8 (two of
  its failures were paralysis-on-correct-NONE). n_nodes at range max;
  design likely locked pending pool-expansion variance.
- **grid** still 1/3 LIVE. n_pos=6 instances split 3/8 | 6/8 | 8/8
  across r1/r2 — high instance variance around a too-easy mean. Pool
  values cap at 6, blocking n_pos=7 → pools extended to 7 values,
  HARD preset n_pos=7.
- **hypothesis** still 1/3 LIVE; sales-only voids (n_voids=2) moved
  one prompt to 6/8. Voids generalized to span sales AND payouts,
  n_voids=3.

## Decisions (round-3 hones; write-back manifest)

1. grid: POOLS extended to 7 values per category (Rowan / 1717 /
   silk); HARD n_pos=7. Answer format unchanged (positional parse).
2. hypothesis: voids now sampled across sales+payouts (n_voids ≤
   ns+npay−3); prose note generalized ("no cash moved for them");
   HARD n_voids=3.
3. assign: HARD n_bins 4→5.
4. machine/certify: no changes (knob-maxed, LIVE, budget-limited —
   length-only hardening rejected as cap-grinding risk).
5. Emission paralysis (species 1) recorded, not patched — Q2/Q3 work
   belongs to a future batch; the re-verification obligation
   re-measures these prompts under bf16 before training.

`CALIBRATION.md` write-back still deferred to b2 finalization (the
knob→difficulty map is converging but not locked). No STATUS rows
resolve yet.
