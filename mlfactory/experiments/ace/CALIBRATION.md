# Calibration: the Phase-0 substrate

> Update when: the acceptance unit, band spec, or transformation catalog
> changes; or a family is killed/transformed. The calibration *loop* runs in
> `gen/calibrate.py`; the rulings it enforces live here.

## Substrate: reasoning-shaped, not computation-shaped

A 9B can do constraint satisfaction, guarded replay, bounded search, and
small-hypothesis elimination. It **cannot** enumerate combinatorial objects,
simulate machines at length, or do exponential optimization. Dead domain
shapes are 0% possible — computation, not reasoning — and no knob tuning
rescues them.

## The productive frontier (what "hard" means)

The substrate must place the model near its **productive reasoning
frontier**: hard enough to induce genuine search pressure, not so hard the
model collapses into unrecoverable failure. The goal is a broad mixture of
outcomes — traces that sometimes succeed directly, sometimes explore
several branches, sometimes make and repair mistakes, sometimes verify,
sometimes struggle or fail. The exact success percentage is not sacred; a
broad mixture is far more useful than a near-ceiling or near-zero set.

The first corpus overshot: questions so far beyond the model's capability
that traces were **reasoning stack overflow** — thousands of tokens of
speculative reconstruction, repeated branches, eventual truncation, never
a coherent reasoning arc. Those are interesting as capacity-boundary
failures but poor trajectory-preserving targets. The useful region is
closer to the frontier: enough capacity to form a real search process,
enough difficulty that the search is not trivial. This is why calibration
is per-prompt band membership, not "is it hard" (`CALIBRATION.md` §acceptance).

## The transformation catalog

Preserves search topology while removing computational impossibility:

| Dead shape | Transformation |
|---|---|
| Enumeration ("count all X") | Certification ("verify/complete this X") |
| Exponential optimization | Bounded construction with a feasibility budget |
| VM/code simulation | Decisive-discrepancy replay (short trace, one guarded trap) |
| Oversized bookkeeping | Small-hypothesis elimination (3–5 candidates) |

The generator (`gen/`) embodies this: solver-built exact answers by
construction, strict structural verifiers, scalar difficulty knobs per
family. See `gen/__init__.py` for the family list.

## The acceptance unit: within-group variance

All-wrong and all-right prompt groups both yield zero advantage under
group-relative RL — dead weight regardless of how interesting the prompts
look. **Per-prompt band membership, not domain averages.** Frozen-30
taught this directly: domain averages concealed 0/8 prompts carried by
successful siblings.

| Band | Per-prompt result | Verdict |
|---|---|---|
| DEAD-HARD | 0 of N correct | regenerate at easier knobs |
| LIVE | 1..N−1 correct | **ACCEPT** (gradient exists) |
| DEAD-EASY | N of N correct | regenerate at harder knobs |

**Preferred band: 2..5 of 8** (robust to upward controller drift).

## Proposal vs calibration (why the author doesn't decide difficulty)

The prompt-generation strategy separates proposal from calibration:

```
Madlibz envelope
        ↓
authored verifiable problem
        ↓
many untouched Qwen9B rollouts
        ↓
empirical difficulty/search behavior
        ↓
retain useful frontier problems
```

The authoring system does **not** decide whether a problem is good for
ACE; the model's actual sampled behavior determines that downstream. This
is deliberately different from asking an authoring model to make problems
that "cause the model to say Wait." The prompt should create structural
reasons for reconsideration, not script the solver's language. Examples:
initially plausible routes that later consume scarce capacity; hypotheses
that fit early evidence but fail under a later observation; local
algorithmic improvements that break a downstream invariant; bookkeeping
systems where early assumptions affect later state; representations that
become tractable only after transformation.

## The calibration loop

```
generate → probe → strict re-score → per-prompt band verdict → regenerate dead families at adjusted presets
```

- The collector's soft `correct` field is **advisory only**.
- `gen/calibrate.py` re-scores every completion against the strict
  per-family `check()` verifier and is the sole authority on band
  classification (DEAD-HARD / LIVE / DEAD-EASY).
- `gen/generate.py` ships default/hard/easy presets per family; the loop
  moves along those axes when prompts land DEAD-EASY or DEAD-HARD.

## Zero-init transfer argument (why base-model probing is valid calibration)

The controller is zero-initialized, so at training step 0 it *is* the base
model. Probing the base model's per-prompt success distribution therefore
calibrates the exact substrate the controller starts from. Controller drift
under terminal reward is one-directional (upward — toward more solvable),
so **calibrate to the hard side of the band**: a prompt at the LIVE
boundary for the base model drifts into the productive zone; a prompt
already easy drifts out to DEAD-EASY.

## Current pool status

**b1 (48 candidates, 8 samples each, corrected 2026-08-24):** accepted LIVE
pool = **25 prompts** (`data/acegen_live_b1.jsonl`). Band verdicts after
verifier false-negative repair + qwen38 judge audit (evidence:
`lab_notes/2026-08-24-b1-calibration-verifier-repair-judge-audit.md`):

| family | corrected strict | band verdict | move |
|---|---|---|---|
| adversary | 71.9% | **8/8 LIVE at default** (depth 4) — only well-calibrated family | stay at default |
| machine | 93.8% | 6/8 DEAD-EASY (too easy) | hard preset |
| assign | 95.3% | 5/8 DEAD-EASY | hard preset |
| certify | 89.1% | 3/8 DEAD-EASY, 5/8 LIVE (6–7/8) | hard preset (R1 confirmed) |
| grid | 85.9% | 3/8 DEAD-EASY, 5/8 LIVE (5–7/8) | hard preset |
| hypothesis | 96.9% | 6/8 DEAD-EASY | hard preset |

0 DEAD-HARD in the batch — no family needs the easy preset. b2 followed,
as below.

**b2 (2026-08-25, q8_0+MTP, four hone rounds + pool expansion, 504
rollouts):** accepted LIVE pool = **46 prompts**
(`data/acegen_live_b2.jsonl`). Every too-easy family was iterated from
the HARD preset to a locked design (knob/structure→difficulty map
below); verdicts measured in the final pool-expansion round (8
candidates × 8 per family, strict check()):

| family | locked design | final-round strict | LIVE prompts (band spread) |
|---|---|---|---|
| adversary | default (unchanged) | 43.8% | 8/8 (2/8–7/8) |
| machine | ll=17, 6 states/7 events, ≈5 traps | 84.4% | 5/8 + 3 from r2 (5/8–7/8) |
| assign | n_items=8, n_bins=5, delayed | 90.6% | 4/8 + 2 from r3 (5/8–7/8) |
| certify | n_nodes=9, no-spoiler trap | 45.3% | 8/8 + 3 from r2 (1/8–7/8) |
| grid | n_pos=6, max_at=1 | 40.6% | 5/8 + 2 from r4 (3/8–6/8) |
| hypothesis | n_voids=3 across sales+payouts | 93.8% | 3/8 + 3 from r3 (6/8–7/8) |

Pool membership is per-prompt; the pool carries rows from multiple
measurement rounds of the same substrate (q8_0+MTP) and locked designs
— never merged sample rows. **Re-verification obligation:** all 46
prompts are q8-banded and owe bf16 re-verification by regeneration
before controller training (OPERATIONS.md → Substrate policy).
Evidence: `lab_notes/2026-08-25-b2-r1-hard-preset-landing.md`,
`...-b2-r2-failure-species-taxonomy.md`,
`...-b2-r3-r4-grid-clue-composition.md`,
`...-b2-final-pool.md`. Live-pool membership is per-prompt (band unit),
never a family average — frozen-30's lesson, reaffirmed: the raw family
averages here (e.g. machine 37.5% buggy) concealed near-ceiling prompts.

**Verifier-authority note:** the deterministic `gen` check() is the scoring
authority. A 2026-08-24 pass repaired systemic extraction false negatives
(machine / adversary / assign / hypothesis) that made families look far
harder than they are; an LLM judge (qwen38, reasoning off, temp 0) audits
answer-equivalence against the solver-built reference as a cross-check — it
is not the scorer and is structurally unusable where multiple answers are
valid (certify colorings, adversary alternative witnesses), which is exactly
where the structural check() is required.

## b2 knob/structure → difficulty map (q8_0+MTP, 2026-08-25)

Built iteratively over four probe rounds (b2 r1–r4, ~340 rollouts). The
HARD preset shipped in `generate.py` was a prior; this map is what was
measured. Two general results dominate:

1. **Numeric size knobs add computation/budget pressure, not reasoning
   pressure.** Hypothesis's numeric axes (n_sales, spread) moved 96.9% →
   95.8% (nothing). Machine's only remaining axis after knob-max is log
   length, which only adds budget exhaustion. Difficulty that comes from
   *structure* — traps that must be discovered, giveaways that are
   withheld, distractors that must be excluded — is what moved bands.
2. **On q8, failures are overwhelmingly truncations** (emission paralysis,
   closure loops, budget-exhausted search). Committed wrong answers are
   rare and only appear where a prompt plants a genuine trap (hypothesis
   voids, grid relational lattices). Failure-species classification is in
   `lab_notes/2026-08-25-b2-r2-failure-species-taxonomy.md`.

Per family (default → locked hard design):

| family | axis moved | effect |
|---|---|---|
| machine | ll 13→17, rejection window scaled to 1..max(3,ll//3) (≈5 traps) | 93.8%→87.5%, all prompts 7/8 LIVE. Knobs at range max; length-only hardening rejected (cap-grinding). Fixed one extraction F/N (first_rejected misread as rejection count) |
| assign | n_items 6→8, n_bins 4→5, delayed | 95.3%→79–87%. Delayed-constraint decoy never bit (model validates all rules); failures are closure loops + budget, not committed errors. n_bins 4→5 raised branching but the effect is small and instance-variance-heavy |
| certify | n_nodes 7→9 + **removed the prose spoiler note** | 89.1%→79.2%. The note had announced the trap; removing it makes the solver discover greedy-failure. NONE instances land mid-band (6/8) |
| grid | **clue composition, not size**: knob `max_at` caps direct "X stands in slot k" clues at 1 | at≤1 → LIVE 5/6; at≥2 → DEAD-EASY 4/4. n_pos 6→7 overshoots (budget-bound: search depth > 26k even on-track). Direct at-clues are giveaway anchors; relational lattices are the real difficulty |
| hypothesis | **structure, not numbers**: VOIDED distractor records across sales+payouts (n_voids=3); spread/n_sales unchanged effect | 96.9%→83.3%, and the first reliable *committed-wrong* failures (miscounting voids). Terminology drift on "expected cash" kept — the format line pins the report fields |
