# Lab note — 2026-08-25 — b2 methodology: how the iterative honing loop was actually run

Scope: consolidated methodology record for the full b2 effort (four probe
rounds + pool expansion, 504 rollouts, five families honed from the
HARD-preset prior to locked designs, 46-prompt LIVE pool). The
point-in-time notes — `2026-08-25-b2-r1-hard-preset-landing.md`,
`-b2-r2-failure-species-taxonomy.md`,
`-b2-r3-r4-grid-clue-composition.md`, `-b2-final-pool.md` — carry the
per-round evidence; this note records *how the loop was run and why*,
so the next calibration batch starts from method, not folklore.

## 1. Measurement design

**Substrate.** q8_0 GGUF + MTP via llama.cpp (ports 3091/3092, one
server per GPU, `--parallel 4`, ctx 131072, thinking on, temp 0.8,
top-p 0.95, cap 26000 tokens). Chosen by standing user ruling for
iteration speed, under the conditions in `OPERATIONS.md` → Substrate
policy: rows carry `backend`/`quant`, no pooling rows across
substrates, and every q8-banded prompt owes bf16 re-verification by
regeneration before training. Measured throughput 152–168 tok/s per
stream (MTP draft); a 120-sample round ran in ~1.5–2h across both GPUs.

**Round sizing.** Each probe round: 3 candidates × 8 samples per
family under test (~24 samples/family, 20–75 min). Rationale: n=8 is
the band unit (1..7 of 8 = LIVE) and matches b1; 3 candidates is the
minimum that distinguishes "family lands" (≥2/3) from one lucky
instance, at a cost that keeps the collect→read→hone cycle under an
hour. The assignment said expect 3–5 rounds; the loop took four plus
expansion.

**Seeding and identity.** New or changed prompts always got fresh pids
(r1: staged 57–91; r2: 97–111; r3: 112–120; r4: 121–123; expansion:
124–163 + staged adversary 49–56). Unchanged prompts would have been
re-sampled via `--sample-start 8` to keep seeds
(`seed_base + 17*pid + sample_i`) collision-free — never needed.
Every round is its own append-safe artifact pair
(`data/acegen_b2_r{k}_gpu{0,1}.jsonl`); merged analysis copies live in
`data/b2/` and are derived, never edited.

**Candidate generation.** Through `gen.generate` presets/module with
explicit seeds, self-checked at construction (round-trip invariant).
One deviation from blind preset-walking was deliberate: certify r2
seed-scanned until the 3-candidate set contained ≥1 NONE instance,
because a solvable-only probe would leave the NONE branch unmeasured;
the expansion batch (8 candidates) got its mix naturally (1 NONE).

## 2. Scoring discipline

- **Strict deterministic `check()` is the sole banding authority**
  (`gen/calibrate.py`); the collector's soft `correct` field was
  cross-checked every round and never diverged from strict in b2.
- **Verifier changes are extraction-only and control-tested.** One bug
  was found and fixed (machine: `\brejected\b\s*[=:]` matched inside
  "first rejected: 1", reading first_rejected as the rejection count).
  Fix protocol that worked: (1) read the actual completion before
  touching the verifier — the completion was fully correct, proving
  false-negative not judgment; (2) negative lookbehind, no semantics
  change; (3) full `--self-test` round-trip; (4) positive controls
  (correct variants must pass, incl. the exact serialization that
  failed) and negative controls (each field independently corrupted
  must fail). My first fix attempt shipped a broken regex
  (`rejections?` doesn't match "rejected") — the controls caught it
  before any rescoring. That is what controls are for.
- **Scored-after-verifier-change rows are re-scored, never
  re-collected** — the artifacts stay immutable; only the verdict
  moves (p57 6/8 → 7/8).

## 3. The reading protocol (where the actual work was)

Counts banded the prompts; reading classified *why*. Per round:

1. **Wrong traces, all of them.** For each wrong sample: tail ~1–3k
   chars first, full trace when the tail was ambiguous. Record whether
   an answer was ever derived, and where (think vs visible).
2. **1–2 correct traces per family** for quality judgment: genuine
   search-and-prune vs instant-insight vs cap-grinding.
3. **Failure-species classification**, mechanical where possible (the
   classifier re-scores the extracted answer line with the strict
   verifier and measures tail n-gram repetition):
   - `PARALYSIS_CORRECT` — right answer derived (in think or extracted
     line) but `visible_chars=0`; emission paralysis (Q3), scored
     wrong by terminal-reward definition and correctly so, but counted
     separately because it is not difficulty.
   - `TRUNC_NO_ANSWER` / `TRUNC_WRONG_ANSWER` — budget-exhausted
     search, partial or wrong answer at the cut.
   - closure-loop tails (rep ratio > 0.6; loops like "I'll write the
     final answer" score ~1.9) — substrate artifact on q8.
   - `WRONG_COMMITTED` — non-truncated, wrong answer: the species
     calibration exists to produce.
4. **Cross-instance joins.** The decisive grid finding came from
   joining clue-type counts (parsed from prose) with bands across all
   9 grid instances: at≤1 → LIVE 5/5, at≥2 → DEAD-EASY 4/4. When a
   family's variance looks random, count something in the prompt
   structure and join it with the band.

## 4. Hone protocol

- **One dominant lever per family per round**, chosen from the
  previous round's failure species — never a scatter of knob nudges.
  r2: machine length+traps, assign items, certify n_nodes+spoiler
  removal, hypothesis voids. r3: grid n_pos (failed), hypothesis
  void-coverage, assign bins. r4: grid clue composition only.
- **Structural beats numeric.** Numeric size knobs mostly added
  budget pressure (hypothesis spread/n_sales: 96.9%→95.8%; grid
  n_pos 6→7: budget-bound 0/8s with correct solutions derived
  in-think). Structure moved bands: withhold the trap announcement
  (certify), plant excludable distractors (hypothesis voids), remove
  giveaway anchors (grid max_at).
- **Prompt information design is a difficulty knob.** certify's prose
  had announced "greedy fails but a k-coloring exists" — a spoiler.
  Removing it (r2) was worth the node increase.
- **Feasibility probes before locked-in knob moves.** machine ll=17
  was infeasible under the fixed 1–3 rejection window (empirically:
  random logs reject 8–12 events); the window was generalized to
  1..max(3, ll//3) — a design-scaled constraint, documented in the
  module — before regenerating.
- **Rejected levers are recorded with reasons.** machine length-only
  hardening: rejected (cap-grinding, the failure mode already seen in
  100% of r1 wrongs). This prevents the next agent from re-trying it.
- **C4 guard on every generator change:** round-trip self-test,
  positive/negative controls where semantics could move, `py_compile`
  + prose inspection for the skin layer.

**Skin layer (realism/diversity).** Per-instance scenario skins
(3 per family; e.g. session controller / order pipeline / firmware;
deployment graph / broadcast network / exam venue; store register /
cafe till / box office) vary framing while the answer vocabulary the
verifiers parse stays pinned. Skins went in after r2, so r3/r4 and
expansion measured final-form prose; the 13 folded pre-skin rows are
annotated in the pool sidecar. The principle: *the prompt is part of
the difficulty and part of the product* — final bands must be measured
on final-form prompts, and the pool should read like problems real
users pose, with surface variety.

## 5. Decision rules used

- Family "lands" at ≥2/3 probe prompts LIVE (1–7/8); sweet spot 2–7,
  preferred 2–5 for drift robustness.
- A knob change is kept only if it converts DEAD-EASY to LIVE without
  manufacturing DEAD-HARD by budget exhaustion; budget-bound DEAD-HARD
  (correct solution derived in-think, cap hit) counts as overshoot,
  not difficulty — checked by reading, not by the 0/8 alone.
- Lock a design when its last round meets criterion AND its failures
  contain genuine species (committed errors or productive search
  exhaustion), not only substrate artifacts.
- Pool membership is per-prompt; expansion generates enough instances
  (8/family) to absorb known instance variance (grid especially:
  LIVE rate at max_at≤1 is 5/6, the 0/8s are filtered, not fixed).

## 6. What the method got wrong, honestly

- **r1's reading was undersampled on assign.** Closure loops were
  identified but assign's *lack of committed errors* (decoy never
  biting) only became a recorded finding in the final note; the r3
  assign hone (n_bins=5) was chosen on branching-factor intuition,
  and its measured effect was small and variance-heavy (r2 4-bin
  family strict 79.2% vs r3 5-bin 87.5% — bins up, family % up; knob
  effects at n=3 probes are inside noise). The honest read: assign's
  difficulty comes from item count and loop-proneness, not bin count,
  at this sample size.
- **grid n_pos=7 was predicted to work and didn't** — the budget
  ceiling is a property you only see by reading truncated traces. Cost
  one round; the fix (clue composition) was better than the original
  plan would have been.
- **Hypothesis remains skewed easy** (5/8 DEAD-EASY in expansion).
  The void structure created real discrimination where it bit, but
  most instances stay solvable in one arithmetic pass. A harder
  hypothesis design (voided payout *timing*, multi-slip readings) was
  not pursued — the pool already meets criterion; recorded as future
  work.
- The **adversary substrate shift** (71.9% bf16 b1 → 43.8% q8) is
  larger than the delta smoke's 1.17-eighths mean predicted — the
  smoke's six mid-band prompts undersampled depth-4 witness search.
  The delta-smoke method is sound but its error bars don't cover
  topology-specific sensitivity. Flagged for bf16 re-verification.

## 7. Evidence index

| Artifact | Content |
|---|---|
| `data/b2/r{1,2,3,4}.jsonl`, `data/b2/expansion.jsonl` | round candidates (pids 57–163) |
| `data/acegen_b2_r{1..4}_gpu{0,1}.jsonl`, `data/acegen_b2_pool_gpu{0,1}.jsonl` | 504 rollouts, immutable, sidecar'd |
| `data/acegen_live_b2.jsonl` (+sidecar w/ caveats) | 46-prompt LIVE pool; re-verification obligation stated |
| `lab_notes/2026-08-25-b2-r1…/-r2…/-r3-r4…/-final-pool.md` | point-in-time round notes |
| `CALIBRATION.md` § b2 pool status + knob/structure map | write-back |
| `STATUS.md` R8 | resolution row |
| `gen/{machine,assign,certify,grid,hypothesis}.py` | locked generators (windows, max_at, voids, no-spoiler, skins) |

## Decisions

All write-backs this note's content implies are already executed
(`CALIBRATION.md`, `STATUS.md`, pool sidecar). One new decision for
the record: **methodology sections 3–5 are the starting protocol for
the next calibration batch**; deviations need a lab note, not silence.
Open methodological debt: a loop/emission-paralysis detector (Q2/Q3)
would sharpen every future band measurement — today's bands count
paralysis deaths as difficulty, and the classifier above is the
manual workaround.
