# Lab note — 2026-08-24 — b1 calibration: systemic verifier false negatives, judge audit, corrected band table

Scope: completion of the b1 acegen probe (384 rollouts, 48 candidates × 8
samples, Qwen3.5-9B bf16 thinking-on) and its Phase-0 calibration. Discovery
and repair of systemic false negatives across four family verifiers; a
qwen38 LLM audit over all 384 completions; the corrected authoritative band
table. This supersedes the raw calibrate numbers and the interim read of Q1.

## Headline: the raw band table was materially wrong (too pessimistic)

The pre-fix verifiers failed on model serializations that the prompts never
pinned down (R6's blind spot, now confirmed at full scale across families).
Corrected strict counts (fixed `gen` check(), judge cross-validated):

| family | buggy strict | **corrected strict** | LIVE | DEAD-EASY | DEAD-HARD |
|---|---|---|---|---|---|
| adversary | 34.4% | **71.9%** | 8/8 | 0 | 0 |
| machine   | 37.5% | **93.8%** | 2/8 | 6 | 0 |
| assign    | 90.6% | **95.3%** | 3/8 | 5 | 0 |
| certify   | 89.1% | **89.1%** | 5/8 | 3 | 0 |
| grid      | 85.9% | **85.9%** | 5/8 | 3 | 0 |
| hypothesis| 95.3% | **96.9%** | 2/8 | 6 | 0 |

**Accepted LIVE pool: 25 prompts** (`data/acegen_live_b1.jsonl`): adversary
p17–24 (all 8), machine p11/p13, assign p1/p3/p8, certify p26/p28/p29/p31/p32,
grid p34/p35/p36/p38/p39, hypothesis p41/p43. **0 DEAD-HARD; 23 DEAD-EASY.**

## Findings (claim + evidence)

1. **machine "DEAD-HARD" p15 was a verifier artifact; machine is too easy.**
   The `_FIELD_RE` required `first[_ ]rejected\s*[=:]`, failing on the model's
   actual labels: `first rejected index: 1`, `firstrejectedindex=1`, `first=1`,
   `first index: 1`, count-before-label (`3 rejected`), bare positional
   (`ready, false, false, 0, 2, 11`). p15 went 0/8 → 8/8 (DEAD-EASY). machine
   corrected 37.5% → 93.8%. The buggy scoring had machine as the second
   "well-calibrated" family; it is actually near-ceiling at default.

2. **adversary false negatives from sequence/trace serialization.**
   `_WIT_RE` required contiguous `BAAB: 1,0,0,-1`; the model writes
   `B A A B`, `C,C,A,B`, `CBAB, CMDS: ...`, `(Credits: ...)`. A label word
   starting with a command letter (`CMDS`) can leak its `C` into the sequence
   if not stripped. Fixed extractor strips label words, tolerates separators,
   and **re-simulates the witness against the rule table** (accepts valid
   alternative witnesses of the shortest length; requires the stated trace to
   match the derived trace). adversary 34.4% → 71.9%. All 8 prompts LIVE —
   adversary is the **only well-calibrated family at default** (depth 4).

3. **assign / hypothesis label tolerance.** assign `S1: Ana` (colon) vs
   `S1=Ana` → fixed to `[:=]` case-insensitive. hypothesis `over/short=` vs
   `over_short=` → `over[_ /]?short`. Small but real (assign +3, hypothesis +1).

4. **grid / certify verifiers were already sound.** Every wrong grid/certify
   completion is a genuine truncation / emission loop / incomplete answer —
   no false negatives among finished answers. certify's structural check
   (valid 3-coloring, palette, all-edges, NONE handling) is correct.

5. **qwen38 judge audit (temp 0, reasoning disabled, YES/NO vs the
   solver-built reference).** Exact agreement with the corrected deterministic
   verifier on the single-answer families (machine 64/64, assign 64/64,
   hypothesis 64/64 rows). Its divergences were its own limits, not the
   deterministic checker's errors:
   - certify: 40× judge-NO/det-YES — the judge compares to a single reference
     and cannot verify a *different valid* 3-coloring. det is authoritative.
   - adversary: 3× judge-NO/det-YES = valid alternative witnesses (det
     re-simulates against rules); 1× judge-YES/det-NO = judge false positive
     on truncated reasoning (pid18 s0).
   - grid: 1× judge-YES/det-NO = judge false positive on truncated reasoning
     (pid39 s6).
   The judge **validated** the corrected deterministic verifiers and confirmed
   completeness on the variant-heavy single-answer families. It is the right
   tool for answer-equivalence on single-answer families; it is NOT a
   substitute for structural verification where multiple answers are valid
   (certify colorings, adversary witnesses).

6. **Validation evidence for the corrected verifiers.** Round-trip: every
   family reference passes its own check (12/12 gen self-test + per-family).
   Negative controls: machine 48/48 per-field mutations rejected; certify
   12/12 broken-edge + incomplete colorings rejected; cross-reference
   (correct completion vs other prompts' references) machine 0/420, assign
   0/427, hypothesis 0/434, adversary 0 cross-accepts. machine corrected
   extractor matches a full hand-read of all 64 machine answer lines (60/64).

7. **Truncation ≠ failure.** 24/63 truncated rows still emitted a *correct*
   answer before looping/hitting the cap. The emission-paralysis loop
   frequently sits **on top of** a completed answer rather than blocking it.
   Among det-final wrongs: adversary failures are mostly **search-budget
   exhaustion** (13 trunc-other, still reasoning at cap) not closure-blocking;
   machine/assign's few wrongs are emission-paralysis loops.

## Decisions with rationale (write-back manifest)

- **Corrected deterministic `gen` check() is the scoring authority** —
  reproducible, structural where multiple answers are valid, and now
  judge-cross-validated. The qwen38 judge served as an **audit oracle**, not
  the scorer: it confirmed the fixes and its divergences were its own
  structural limits. (User ruling 2026-08-24: using an LLM to clean messy 9B
  serialization is standard and does not pollute the experiment — the ground
  truth stays the solver-built reference, exact by construction. The judge
  verdicts are saved immutably at `scratch/judge_b1_verdicts.jsonl`.)
  → write back: `STATUS.md` (Q1), `CALIBRATION.md` (pool status).
- **adversary requires the full answer (sequence AND credit trace)** — the
  prompt asks for both, the reference always includes both, and the judge
  holds the same standard. 5 sequence-only accepts were dropped (adversary
  51→46). p20 flips 8/8→7/8 (stays LIVE). → `gen/adversary.py`.
- **Verifier fixes ported to source** (`gen/machine.py`, `gen/adversary.py`,
  `gen/assign.py`, `gen/hypothesis.py`), each validated with round-trip +
  negative controls. Fixes loosen extraction, never semantics (C4). → this
  note + the code.
- **Preset moves:** machine, assign, certify, grid, hypothesis all land too
  easy at default → regenerate at **hard** preset. adversary stays at default
  (only well-calibrated family). No family needs the easy preset (0
  DEAD-HARD). → `STATUS.md` (Q1), `CALIBRATION.md`.
- **Q1 resolved (bet overturned).** Bet was "grid and machine live; adversary
  at depth 4 hard." Corrected: adversary is the only well-calibrated family
  at default; machine/grid/hypothesis are too easy. → `STATUS.md`.

## Caveats on record

- The judge is not bit-reproducible; it is used only as a cross-check, at
  temp 0, with verdicts saved. The authoritative bands come from the
  deterministic verifier (reproducible).
- Mixed-cap rows (8 pre-ruling 32k + rest 26k) treated as a per-row
  covariate; only assign p6's DEAD-EASY verdict is cap-fragile (would be 7/8
  LIVE at a uniform 26k) — no family conclusion changes.
- The collector's stored `correct` field reflects the pre-fix verifier for
  machine/adversary/assign/hypothesis and is superseded; the rollouts
  themselves (completions) are immutable evidence, only the scoring changed.
- adversary judge count (44) is a single-reference undercount by design;
  certify judge count (17) is structurally meaningless. Neither is a verdict.

## State at note time

- Corrected LIVE pool (25 prompts) at `data/acegen_live_b1.jsonl`.
- Judge audit artifact `scratch/judge_b1_verdicts.jsonl`; harness
  `scratch/judge_b1.py`; reconciliation `scratch/final_reconcile_b1.py`;
  fixed-verifier prototype `scratch/fixed_verifiers.py` (ported to `gen/`).
- qwen38 (`llama-qwen38.service`) was started for the audit; reasoning
  disabled per-request (`enable_thinking:false`). Stopped after the audit.
- Immediate next: regenerate machine/assign/certify/grid/hypothesis at hard
  preset (b2), keep adversary at default; then re-run this exact pipeline
  (collect → strict re-score → judge audit → calibrate).
