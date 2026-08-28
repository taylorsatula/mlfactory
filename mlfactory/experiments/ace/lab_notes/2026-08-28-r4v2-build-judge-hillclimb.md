# 2026-08-28 — R4v2 build: windowed forks + judge hillclimb

> Session goal (principal, verbatim): "spend the next few hours
> hillclimbing the implementation of the new r4v2 evaluation. Once it
> works … set the local GPUs to power level 280 (to keep them cool)
> and smoke test the run … write a follow-up runbook for a GLM agent
> to administer it locally while I'm out of town." Predecessor notes:
> `2026-08-28-r4-attendance-stopped-design-change.md` (halt + recipe),
> `2026-08-28-r4-partial-trace-report.md` (what the 117 rows show).

## Design rulings received (principal, this session — verbatim where marked)

- Judge context: "2048 + tail that came before so the judge model knows
  what was happening in the lead up." → WINDOW=2048, TAIL=512.
- "We do not care about terminal correctness at this stage. It should
  not factor into your decision making or metrics. We're worried about
  the tokens slightly before the intervention, the intervention itself,
  and the 2048 tokens afterwards." → v2 harness stores no
  objective_check / correct / match_mode fields at all.
- "Yes, this is not a LLM-reward. This is an evaluation of yielded
  tokens to interpret the effect of the intervention. 2048 tokens is
  absolutely enough to know what change the intervention had." →
  REWARD_POLICY.md scope note written back: the judge ban governs
  training rewards, not measurement instruments.
- Keep m=24. substrate field: handle with best judgement (below).

## What was built

1. **`annotate/fork_r4v2.py`** — window harness. Inherits v1's proven
   machinery (plan, hook, FLASH_ATTENTION, OOM guards, resume); changes:
   `max_new = min(2048, CAP_ABS - fork_abs)`; rows store `prefix_tail`
   (last 512 pre-fork tokens) + `window` (generated text) instead of
   terminal scoring; seeds derive `base_hash(state_id) + seed_i` —
   keyed on seed_i, not pending-list position (v1 keyed on the batch
   index, which shifts on resume — latent v1 defect, never triggered
   because the production run had 0 restarts). v2 seeds verified equal
   to v1 fresh-run sub-1 seeds for all 24 seeds × 4 test states, so
   v1 pilot rows truncate losslessly into v2 windows.
2. **`annotate/judge_r4v2.py`** — blind three-branch judge (glm-5.2-vision
   via Lunaroute). Per triplet, arms go to labels A/B/C by a stable
   sha256 permutation; rubric v2 adds "presented in random order; judge
   each on its own content only." JSON verdict: per-label
   characterization + mode (progress/mixed/spinning), ranking with tie
   groups, differences.
3. **`annotate/analyze_r4v2.py`** — unblind + aggregate: wins, mean
   rank, healthy-vs-noop outcomes, mode distribution, gold-anchor
   agreement, residual-position-bias diagnostic.
4. **`scratch/convert_v1_pilot.py`** — one-shot migration of 39 v1 rows
   (3 states × seeds 0-7, noop + available toward_healthy) into v2
   schema for the judge hillclimb. Guard: the 2048 window slice itself
   must round-trip decode→retokenize exactly; full-length off-by-one
   mismatches are a stop-token boundary artifact (verified: all
   prefixes ≤7000 round-trip clean on the first affected row) and do
   not affect the window. 39 kept / 0 dropped.
5. **`scratch/pilot_gen.sh`** — fills the missing arms: toward_diverge
   seeds 0-7 × 3 states + toward_healthy gaps (GPU1, ~61s/row on the
   longest-fork state incl. 17.6k prefill — decode is ~4x faster than
   the A6000 boxes).

## Findings so far

**F1 — judge position bias is real and must be cancelled.** Rubric v1,
6-permutation probe on r4_cycle_00 seed 0: position-A arm won 6/6 —
the ranking followed the label order, not the arms. Rubric v2 (random-
order framing + content-only instruction): 4/6. Arm discrimination is
genuine underneath (v2 probe: toward_healthy top-or-second in 6/6,
never last; toward_diverge mostly last — the expected directional
structure; healthy-vs-noop agreed with the gold anchor in 5/6
permutations). Fix shipped: 3-pass ensemble per triplet — cyclic
rotations of the base permutation so every arm occupies every label
position exactly once; analysis averages ranks across passes
(additive position bias cancels exactly under that balance). Cost:
3 judge calls per triplet (~60-90s).

**F2 — rubric v1 already agrees with the hand-read gold where it
wasn't position-unlucky.** Pass-1 (rubric v1, 4 triplets, r4_cycle_00
seeds 0-3): gold anchors seeds 0,1 both OK; characterizations were
substantive (M1 reachability arguments, candidate re-enumeration,
format fixation flagged on seed 2's noop). healthy-vs-noop 4/4 toward
healthy on that state; noop spinning/mixed in all 4 — consistent with
the partial-trace report's read of r4_cycle_00 as a deep-thrashing
fork point.

**F3 — substrate:q8 resolved (best judgement, per principal).** The
field is the xsub corpus's per-trace label: which serving substrate
(q8_0 vs bf16) GENERATED the base trace the fork state sits in. It is
trace provenance, not run precision (forks all run bf16). Plan mix:
22 q8 / 5 bf16. Kept as-is; it is a covariate for analysis (does the
judge's read differ by substrate of origin), not an ambiguity to fix.

## Environment changes

- **llama-qwen38 stopped AND disabled** (principal: "Shut it down as
  this traffic takes precedence"). It was serving Qwen3.8-27B
  tensor-split across BOTH GPUs (~21GB each — the desktop overhead
  assumption of GPU0-only no longer held while it ran). Restore after
  the run: `sudo systemctl enable --now llama-qwen38`.
- Power limit 280 + smoke + runbook: PENDING (after hillclimb lands).

## Decisions

- Judge hillclimb rounds so far: v1 single-pass (gold 2/2 but 6/6
  position-A bias on probe) → v2 prompt fix (4/6 bias, arm signal
  visible) → 3-pass rotation ensemble (shipped; results next entry).
- Rejected levers: pairwise (2-arm) judging — 6 calls/triplet and the
  same bias structure; further prompt wording against position bias —
  diminishing returns against a structural prior, ensemble cancels it
  exactly.
- Full-run sharding draft (pre-smoke): GPU0 13 states/936 rows, GPU1
  14 states/1008 rows, prefill-units balanced 122k vs 125k (greedy on
  fork_abs). GPU0 holds the worst state (r4_cycle_00, fork 17580) —
  smoke must confirm it fits beside the desktop.

## Entry 2 — reuse assumption falsified; pilot regenerating locally

**F4 — cross-hardware generation is NOT bit-equivalent.** The
equivalence check (r4_cycle_02 noop seed 0 regenerated locally vs the
v1-derived truncated row) FAILED: prefix_tail identical, window
diverges at char ~744 (~token 350 of 2048). Same model, same seed
derivation (verified arithmetically), same sampling params, same
torch 2.11.0+cu128 — but v1 rows were generated on A6000 (Vast) and
v2 runs on local 3090s. Tiny hardware-numerics differences flip a
sampled token, then cascade. The v1 determinism proof (concurrent
same-seed runs bit-identical) held ON ONE HARDWARE+SOFTWARE STACK; it
was silently generalized across stacks. Corrected scope: determinism
is per-stack, and sampled continuations are chaotic amplifiers of any
numerics difference.

Consequences:
- Production run unaffected: all 1944 rows generate locally on one
  stack; arm pairing is within-process.
- Pilot reuse rejected: the judge hillclimb needs triplets whose arms
  share a generation stack, and the gold anchors must describe the
  SAME windows the judge reads. `data/fork_r4v2_pilot2.jsonl`
  regenerates all 3 pilot states x 8 seeds x 3 arms locally (GPU1,
  ~1h). Judge verdicts already produced (pilot_v2 file) were computed
  on hardware-mixed triplets — flagged; they inform rubric mechanics
  (parse rate, tie behavior, position-bias cancellation) but not the
  final calibration against gold.
- GPU0 smoke meanwhile PASSED under the 280W cap: worst state
  r4_cycle_00 (fork 17580) at 62s/row, 20.6GB peak beside the
  desktop, 53C. Power limit set to 280W on both GPUs, reboot-
  persistent via the edited nvidia-3090-power-limit.service (revert:
  363W, noted in the unit and the runbook).
- Runbook written: `annotate/runbook_r4v2_local.md` — 2 processes
  (one per GPU), 936/1008-row shards, ETA ~15h wall, llama-qwen38
  restore + power revert in the completion protocol.

## Entry 3 — clean-stack calibration + the anchor-stack lesson

Mixed-pilot calibration (v1-derived noop/healthy + local diverge):
5/6 window-based anchors, incl. the seed-5 format-fixation trap.
Then pilot2 (all 72 rows regenerated locally) recalibrated at **3/6** —
and the cause was instructive: the gold anchors had been blind-read
against the MIXED-pilot windows; locally regenerated windows diverge
from v1 ones after ~token 350 (F4), so three anchors no longer
described the windows being judged. **Lesson: calibration anchors must
be verified against the exact artifact stack they calibrate — the same
class of error as F4, one level up.** Re-blind-reads of the pilot2
windows: seed 5 anchor becomes a TIE (healthy's format fixation is
offset by new M2-chain structure), leaves 4/6 with two borderline
healthy-vs-noop misses.

**F5 — judge's remaining failure mode is context-diff blindness.**
On r4_cycle_00 seed 0 (pilot2), 2/3 passes labelled the noop branch
"progress" when it was re-deriving reachability arguments already
present in the prefix tail — "re-deriving content already derived"
requires diffing the window against the context, and the judge does
not do that reliably without being told. Rubric v3 adds one line:
content repeating the context is re-derivation, not progress — check
against the context explicitly. Under test on the 12 seed-{0,1,2,5}
triplets (judge_r4v2_pilot2_v3.jsonl).

Aggregate clean-stack picture (pilot2, rubric v2, 24 triplets):
toward_diverge comes out best (mean rank 0.74, 10/24 wins, fewest
spinning modes) — at these deep CYCLE forks, divergence-steering
injects novelty that reads as exploration; healthy ≈ noop on average
(11-11-2). This is the raw phenomenon the full run dissects; the
pilot's three states are all CYCLE and all deep-ish forks, so no
class-level conclusions yet.

**Production run launched 2026-08-28 08:30 UTC** on local GPUs (not
Vast): gpu0 936 rows / gpu1 1008 rows, power-capped 280W, llama-qwen38
stopped+disabled, ETA ~15h. Attendance: annotate/out/r4v2_attendance.log,
runbook annotate/runbook_r4v2_local.md.

## Entry 4 — hillclimb terminal state: rubric v2 wins, v3 rejected

Rubric v3 (explicit "content repeating the context is re-derivation"
instruction) scored 2/6 on the same 12 triplets where v2 scored 4/6 —
the added rule destabilized verdicts rather than fixing the context-
diff misses (s0 healthy→tie, s5 tie→healthy, s1 flipped). Consistent
with the GLM prompting lesson: appended edge-case clauses feed
wandering judgments. **v3 rejected; judge_r4v2.py TEMPLATE restored to
rubric v2** ("presented in random order; judge each on its own content
only"), the configuration that produced the 4/6 clean-stack
calibration.

Final calibration state (rubric v2, 3-pass rotation ensemble,
glm-5.2-vision):
  * mechanics: 100% parse (72/72 passes across 24 triplets), position
    bias cancels by construction (each arm occupies each label once
    per triplet); residual label-level gradient (A 0.64 / B 0.92 /
    C 1.44 at pass level) is exactly what the rotation averages out.
  * agreement: 4/6 pilot2-window anchors; the two misses (c00 s0,
    c02 s0) are healthy-vs-noop comparisons an independent blind read
    itself flags as borderline/close. The seed-5 format-fixation trap
    reads correctly on the mixed stack (noop > healthy) and as a tie
    on pilot2 windows where the fixation is offset by new structure —
    the judge tracks the window it is shown, which is the property
    that matters.
  * judge variance on genuinely close triplets is real: the same
    (state, seed) can flip direction across stacks and rubric runs.
    Analysis must therefore report margins and aggregates over the 648
    triplets, never single-triplet verdicts.

Rejected levers (do not retry): single-pass judging (position-A bias
6/6); pairwise judging (same bias structure, 2x calls); rubric v3
wording (regressed calibration).

Production rollouts: 1h in at launch+64min, 129 rows (61/936 gpu0,
68/1008 gpu1), ~120 rows/h aggregate, both GPUs 98% util at the 280W
cap, 59/45C — on the 15h ETA line.
