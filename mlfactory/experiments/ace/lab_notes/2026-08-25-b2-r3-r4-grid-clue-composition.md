# Lab note — 2026-08-25 — b2 rounds 3–4: hypothesis void trap lands, grid's difficulty knob is clue composition, all five families LIVE

Scope: b2 round 3 (72 rollouts: grid/hypothesis/assign × 3 candidates,
`data/b2/r3.jsonl` pids 112–120, artifacts `data/acegen_b2_r3_gpu{0,1}.jsonl`,
merged `data/b2/r3_rollouts.jsonl`) and round 4 (24 rollouts: grid × 3,
`data/b2/r4.jsonl` pids 121–123, artifacts `data/acegen_b2_r4_gpu{0,1}.jsonl`,
merged `data/b2/r4_rollouts.jsonl`). Substrate q8_0+MTP throughout.

## Round 3 bands

| pid | family | knobs change | strict | band |
|---|---|---|---|---|
| 112 | grid | n_pos=7 | 8/8 | DEAD-EASY |
| 113 | grid | n_pos=7 | 0/8 | DEAD-HARD |
| 114 | grid | n_pos=7 | 1/8 | LIVE |
| 115, 117 | hypothesis | voids across sales+payouts, n=3 | 7/8 | LIVE |
| 116 | hypothesis | same | 6/8 | LIVE |
| 118 | assign | n_bins=5 | 6/8 | LIVE |
| 119 | assign | n_bins=5 | 7/8 | LIVE |
| 120 | assign | n_bins=5 | 8/8 | DEAD-EASY |

**hypothesis fixed: 3/3 LIVE with 4 WRONG_COMMITTED failures** — the
void trap produces the species calibration exists for (committed wrong
answers: miscounting the voided records), not just budget exhaustion.
assign 2/3 LIVE at n_bins=5 — criterion met.

**grid at n_pos=7 overshot:** p113 0/8 with 10/16 wrongs =
TRUNC_NO_ANSWER. Reading p113 s0: the model derived the full correct
solution in-think, then burned the remaining budget re-verifying and
never emitted. n_pos=7's search depth exceeds the 26k budget even for
on-track samples — budget-bound, the wrong kind of hard. Reverted.

## Finding: grid difficulty is clue composition, not size

Counted clue types across all 9 grid instances (r1–r3) and joined with
bands:

| direct "at" clues | outcome |
|---|---|
| ≤1 | LIVE, 5 of 5 (3/8, 6/8, 6/8, 1/8@npos7, 0/8@npos7-budget) |
| ≥2 | DEAD-EASY, 4 of 4 |

Direct "at" clues are giveaway anchors — with ≥2 of them the constraint
lattice collapses into plug-and-check. Without them the solver must
chain relational clues (leftof/immleft/samepos/notat), which is genuine
inference at n_pos=6 without blowing the budget. Implemented as knob
`max_at` (cap on direct clues admitted during accretion; uniqueness
still enforced; `n_at` recorded in knobs). Verified feasible: 20/20
random instances at n_pos=6 with max_at=1, 11–16 clues.

## Round 4 bands (grid, n_pos=6 + max_at=1)

| pid | n_at | strict | band |
|---|---|---|---|
| 121 | 1 | 6/8 | LIVE |
| 122 | 0 | 0/8 | DEAD-HARD |
| 123 | 1 | 6/8 | LIVE |

Grid criterion met (2/3 LIVE). p122 0/8 is again budget/emission-bound
(s0 derived the correct solution in-think, hit the cap) — occasional
instances demand >26k even at n_pos=6; per-prompt banding filters
them, which is what the acceptance unit is for. Cumulative grid
evidence at at≤1: 3/8, 6/8, 6/8, 6/8, 1/8, 0/8 → LIVE rate 5/6.

## Decisions (locked designs — write-back manifest)

All five families now ≥2/3 LIVE in their last probe round. Locked
designs for pool expansion:

| family | locked design |
|---|---|
| machine | n_states=6, n_events=7, log_len=17, scaled rejection window (≈5 traps) |
| assign | n_items=8, n_bins=5, delayed=True |
| certify | n_nodes=9, k=3, trap=True, none_prob=0.3, no spoiler note |
| grid | n_pos=6, max_at=1 |
| hypothesis | n_sales=6, n_payouts=3, spread=12, n_voids=3 (sales+payouts) |
| adversary | default (b1-calibrated; staged p49–56 for expansion) |

Write-backs on finalization: `CALIBRATION.md` (knob/structure→difficulty
map incl. the grid clue-composition finding and the machine rejection
window), `STATUS.md` resolution row. Emission paralysis now observed
in every round (Q3): recorded, not patched this batch.
