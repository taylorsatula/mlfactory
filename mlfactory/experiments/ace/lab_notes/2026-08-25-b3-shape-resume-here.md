# Lab note — 2026-08-25 — b3 shape established, PAUSED before probing: resume here

**Status: PAUSED by design.** User ruling 2026-08-25: run the first
serious GRPO attempt on the b2 pool (46 prompts,
`data/acegen_live_b2.jsonl`) FIRST; the two new families below are
built and guarded but unprobed, to be calibrated and pooled only if
the controller shows results (the gate protects against polishing a
substrate for an unproven effect — Q10).

## Why these two families (from b2 trace evidence)

b2's 504 rollouts showed the pool's texture gap: machine =
attention-over-bookkeeping, hypothesis = one-pass arithmetic, assign =
loop-prone CSP whose decoy never bit. Only grid/certify/adversary
carry real search-and-prune dynamics, and committed wrong answers (the
cleanest RL signal species) appeared only where a prompt planted a
plausible-but-wrong attractor. Design principle for b3: **build around
planted attractors and forced revision, not difficulty knobs.**

## What is built and verified (do NOT rebuild)

- `gen/construct.py` — bounded sequence construction (the catalog's
  never-built "bounded construction with a feasibility budget" row):
  arrange N items under before/at/notat/adj/notadj/parity constraints
  accreted to ≤2 surviving orders + a forced before-chain, optional
  transition-cost budget; NONE instances via planted before-cycles.
  `check()` validates ANY valid order (constraint_checker), tolerant
  name extraction (case-insensitive, numbering, last-N window),
  case-restored before constraint evaluation.
- `gen/revise.py` — two-stage evidence revision: 4 readings of a torn
  drop slip; batch 1 (records + fragment A compatible with ALL
  readings) + clerk's soft note favoring a RIVAL (attractor 1); batch
  2 fragment B completes only the true reading; fragment C sums with A
  to a rival reading but is stamped ACCT 7 (attractor 2, must be
  excluded). Answer `H# final=$X.XX`; drops constrained ≤ subtotal
  (realism + parser); sign-tolerant amount regex.
- Registered in `gen/generate.py` FAMILIES + all three presets (HARD:
  construct n_items=7/budget=True/none_prob=0.25; revise n_records=5/
  spread=12/decoy=True) and in `gen/calibrate.py` CHECK.
- Guards passed: `--self-test` 16/16 round-trip; construct swap-fail
  12/12 + NONE acceptance; revise 30-seed controls (round-trip,
  wrong-H rejection, wrong-amount rejection, decoy/trap ≠ true).
  Skins included (ceremony/lineup/relay; store/cafe/theater audit).

## Resume procedure (when the GRPO gate passes)

1. Probe round: `generate --family {construct,revise} --n-per 3
   --seed 8300 --start-id 164 --preset hard` (construct pids 164–166,
   revise 167–169); collect 8 samples each via
   `frontier.collect_rollouts_api` on the q8_0+MTP servers (:3091/:3092,
   cookbook in `specs/b2_hone_assignment.md`), `--quant Q8_0-MTP
   --backend llama.cpp`; name artifacts `data/acegen_b3_r1_gpu{0,1}.jsonl`.
2. Score with `gen.calibrate`; READ traces per the b2 method
   (`lab_notes/2026-08-25-b2-methodology.md` §3–5): failure-species
   classification before any hone; one dominant lever per family per
   round; construct's expected failure mode is budget exhaustion at
   n_items=7 (watch for correct-in-think truncations — the grid
   lesson); revise's risk is collapsing into one-pass arithmetic if
   models ignore the clerk note (if so, the attractor needs to be
   load-bearing, e.g. the note correct about the drop's parity/magnitude
   band, wrong about the exact reading).
3. Iterate (expect 2–3 rounds, b2 took four), then expand 8 candidates
   per family at locked designs; add LIVE prompts to a NEW consolidated
   pool file (`data/acegen_live_b3.jsonl` = b2 pool rows copied
   verbatim + b3 additions, fresh sidecar) — never mutate
   `acegen_live_b2.jsonl` (immutable evidence, sha256-sidecar'd).
4. Write-backs on completion: `CALIBRATION.md` family map, `STATUS.md`
   resolution row, lab notes per round.

## Decision (binding)

b3 families enter the training pool ONLY if the GRPO gate shows
controller signal. If the controller learns nothing and the diagnosis
is training-setup (not pool texture), b3 stays paused — a paused,
guarded generator costs nothing; a miscalibrated family costs a loop.
