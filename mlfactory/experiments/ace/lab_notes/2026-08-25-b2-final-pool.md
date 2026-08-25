# Lab note — 2026-08-25 — b2 final: pool expansion at locked designs, 46-prompt LIVE pool

Scope: b2 pool-expansion batch (384 rollouts: 40 new candidates × 8
samples at locked hard designs + 8 adversary default × 8; q8_0+MTP).
Candidates `data/b2/expansion.jsonl` (pids 124–163 + staged adversary
49–56); rollouts `data/acegen_b2_pool_gpu{0,1}.jsonl` (merged
`data/b2/pool_rollouts.jsonl`). Accepted pool
`data/acegen_live_b2.jsonl` (46 prompts, sidecar caveats carry the
re-verification obligation).

## Final-round band table (8 candidates/family × 8 samples)

| family | strict | LIVE | bands (per prompt) |
|---|---|---|---|
| adversary | 43.8% | 8/8 | 4,7,3,3,3,2,3,3 |
| machine | 84.4% | 5/8 | 7,8,8,5,6,8,7,5 |
| assign | 90.6% | 4/8 | 7,8,8,8,7,7,5,8 |
| certify | 45.3% | 8/8 | 4,7,1,4,1,2,3,7 |
| grid | 40.6% | 5/8 | 3,0,4,8,5,3,0,3 |
| hypothesis | 93.8% | 3/8 | 8,8,7,8,6,8,8,7 |

Adversary replicates b1 (71.9% bf16 → 43.8% q8, 8/8 LIVE both) — note
the substrate-level shift is larger here than the delta smoke's mean
(1.17 eighths); adversary's depth-4 witness search is more
substrate-sensitive than mid-band families were. Recorded, consistent
with the caveat that q8 failure dynamics differ; bf16 re-verification
will settle it at training time.

## Pool composition (46 prompts)

Per-prompt bands: certify 1/8–7/8 (11), adversary 2/8–7/8 (8), machine
5/8–7/8 (8), grid 3/8–6/8 (7), assign 5/8–7/8 (6), hypothesis 6/8–7/8
(6). 13 rows fold in from earlier b2 rounds (same substrate, locked
designs): machine p97–99 + certify p103–105 (r2, pre-skin prose),
assign p118–119 + hypothesis p115–117 (r3), grid p121/p123 (r4).
Skin diversity: final-round prompts carry per-instance scenario skins;
the 13 folded rows split pre/post-skin.

## Trace-quality verdicts (read across all four rounds + expansion)

- **certify** is the quality anchor: bands span 1/8–7/8; failures are
  genuine coloring-search exhaustion and NONE-deliberation, not loops.
  Withholding the trap announcement worked — models discover the
  greedy failure themselves.
- **grid** at max_at=1 produces real relational-inference traces
  (chained immleft/samepos deduction) and both failure species:
  committed wrong answers and budget exhaustion. Two DEAD-HARD
  instances per eight is the price of the lattice; per-prompt banding
  filters them.
- **hypothesis** void trap produces clean committed errors (void
  miscounts) with short, structured traces — no cap-grinding; the
  family's residual 5/8 DEAD-EASY rate means its ceiling is high, but
  its LIVE prompts are genuinely discriminating.
- **machine** traces are sustained careful replay; difficulty is
  attention-over-horizon. 7/8-heavy because knobs are at max —
  accepted rather than lengthened (length → cap-grinding).
- **assign** remains the weakest on trace quality: failures are
  mostly closure loops + budget, few committed errors (the delayed
  decoy never bit). Its 6 LIVE prompts are acceptable pool members
  but the family offers the least search-and-prune texture.

## Decisions (write-back manifest — done)

1. `CALIBRATION.md`: pool status updated; knob/structure→difficulty
   map appended (this session, prior to the expansion result).
2. `STATUS.md`: R8 resolution row.
3. Re-verification obligation stated in the pool sidecar caveats and
   here: all 46 prompts q8-banded → bf16 regeneration before GRPO.
4. Open items for whoever runs training next: implement Q2/Q3
   loop/emission-paralysis handling before harvest-scale collection;
   consider an assign structure change (implication-biased
   constraints) if committed-error texture is wanted from that family.
