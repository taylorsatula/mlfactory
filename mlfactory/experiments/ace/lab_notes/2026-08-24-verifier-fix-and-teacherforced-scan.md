# Lab note — 2026-08-24 — Verifier extraction bugs, strict-at-source scoring, first teacher-forced scan

Scope: discovery and repair of systematic false negatives in b1 scoring;
retroactive re-score of collected rows; collector moved to strict scoring at
the source; first teacher-forced measurement pass (128 traces) and its
kill-test outcomes.

## Findings (claim + evidence)

1. **The verifiers produced false negatives at scale; two "failure" traces
   read in full were verifiably correct.** certify p25 s0 emitted a valid
   3-coloring, rejected because `_PAIR_RE`'s color group was lowercase-only
   and the model wrote `A=Green`. machine p9 s0 emitted exactly the gold
   tuple, rejected because `_FIELD_RE` required the literals `final=` and
   `first_rejected=` while the model wrote `State=` / `First Rejected=`.
   Root cause: the 48/48 round-trip validation tested *solver-output →
   check()*, never *model-output → check()*; prompts say "End with
   'Answer: ...'" but never specify the canonical serialization the regexes
   expect. Validation blind spot closed for these two families; grid
   (`;`-separated `Slot N:` fields), hypothesis (`expected=$X.XX
   over_short=$Y.XX`), adversary (`SEQ: t1,t2,...`) carry the same
   assumption class and are flagged for spot-check as their rows land.

2. **Re-scoring inverted a family-level verdict.** Strict-fixed scores:
   certify 13/48 (soft) → 22/48 (strict orig) → **44/48** (extraction fix).
   Per-prompt: p25 8/8, p26 7/8, p27 8/8, p28 6/8, p29 7/8, p30 8/8.
   certify is **DEAD-EASY at default knobs → needs the hard preset**, the
   opposite of the pre-fix plan. machine 0/15 → 6/15, with **p09 at 4/8 =
   LIVE in the prefer band**. assign −3 (soft substring false positives
   removed). b1 holds 4 confirmed LIVE prompts (p01, p08 assign; p09
   machine; p28 certify) where soft scoring showed 1–2. All band estimates
   from soft scores are pessimism-biased lower bounds; no preset decisions
   on pre-fix numbers.

3. **Emission paralysis replicates cross-batch and is truncation-exclusive.**
   9/14 truncated b1 traces (64%) contain verbatim terminal loops; 0/112
   finished traces do. 78% of loops carry closure intent in the loop unit
   ("I'll write the response" ×217–905). Mean post-onset mass: 40% of trace
   tokens; worst case p06_s3 at 71% (~18.5k dead tokens). This is a stable
   attractor-state failure mode, not a frozen-30 anomaly.

4. **Teacher-forced scan (128/128 traces, layer-15 hook + chunked full-vocab
   entropy): one strong keeper, one surprise, two muddies.**
   - `ent_late` (mean entropy, final third): rank-biserial +0.67 pooled,
     **+0.72 with crashouts excluded** — correct traces sustain late-stage
     optionality; wrong traces collapse entropy even without looping.
     Within-prompt strong in assign, weak/contradicted in machine
     (p10 −0.20): family-dependence unresolved.
   - `ent_early` +0.64 clean-only — unexpected; candidate premature-
     commitment signature; family-composition confound uncontrolled.
   - `recur_density` −0.33, `tortuosity` −0.32 (clean pooled): right
     direction, but within-prompt signs flip — neither killed nor supported
     at current n.
   - `n_ent` (length) −0.31 clean: known difficulty-composition confound;
     length tracks problem hardness, not per-sample search quality.

5. **Loop onset gives no prospective entropy warning.** Post-onset entropy
   collapses to 0.01–0.04 in all 9 loop traces (verbatim repetition is
   near-deterministic), but pre-onset entropy is bimodal: some traces decay
   into onset (p01_s1: 0.065 pre vs 0.43 baseline), others arrive from
   healthy entropy (p08_s1: 0.64 pre, above baseline). **Phase-3 fork
   placement cannot use an entropy tripwire**; candidate states must come
   from the controller's intervention magnitude or a trained state detector.

6. **Data-quality flags on record.** (a) b1 contains mixed-cap rows (32k
   rows from before the 26k ruling; no duplicate (proposal_id, sample_i)
   pairs) — treat cap as per-row covariate, chop analytically. (b) Collector
   `proposal_id` = provenance id = candidates line index + 1; ad-hoc joins
   by line index are off by one (calibrate.py and gen joins are correct).
   (c) Some looped traces carried soft `correct=True` via pre-loop answer
   substrings — strict re-score settled them.

## Decisions with rationale

- **Scoring moved to strict-at-source.** Collector `objective_check()`
  dispatches ace-gen candidates to `gen/` family verifiers
  (`match_mode: gen_strict_v2`); legacy madlibz candidates keep the soft
  fallback. The one-off `gen/rescore.py` was deleted after retroactive
  application (originals preserved in `correct_soft`, stamped
  `scorer: gen.check.v2`). A patch tool that persists invites drift between
  "collector truth" and "analysis truth"; the fix belongs in the verifier,
  not in a sidecar.
- **Fixes loosened extraction, not semantics.** certify colors match
  case-insensitively then lowercase; machine accepts `State=`/`final=`,
  `first rejected`/`first_rejected`, `:` separators. Structural validation
  (full pair-set, palette membership, per-edge constraint check; six-field
  exact tuple) unchanged. Negative controls (conflicted coloring, wrong
  register value) still reject; 48/48 reference round-trip still passes.
- **In-flight samples discarded on collector restart** (2 total). Resume is
  bit-stable per (proposal_id, sample_i); existing rows never regenerated.
- **Teacher forcing uses the frozen base model** — valid as controller
  step-0 substrate under the zero-init transfer argument. Re-tokenization
  from stored text may diverge from sampled tokenization at a few
  positions; acceptable for diagnostics, **not** for exact fork placement.

## Environment traps encountered

- GPU0 (desktop-resident, ~1.9 GB used) OOMed the scan shard on 26k-token
  traces at 3.8 GB attention workspace with 3.7 GB reserved-but-unallocated:
  fragmentation, not capacity. Fix: `PYTORCH_CUDA_ALLOC_CONF=
  expandable_segments:True`, plus `torch.cuda.empty_cache()` between rows;
  long traces routed to desktop-free GPU1. Scan rerun: 0 failures.
- `trace_diagnostics.py`/`teacherforced_scan.py` (now `core/trace_diagnostics.py` / `analysis/entropy_scan.py`) are CPU/GPU separable:
  text-level diagnostics need no GPU and can run during collection.

## State at note time

- Collectors running post-restart (already_done 49 + 79 = 128/384), strict
  scoring native, ETA ~16–19h. Remaining families: adversary (p16–23),
  grid (p32–39), hypothesis (p40–47) plus machine p11–15 tail.
- Artifacts: `trace_diagnostics.py` (now `core/trace_diagnostics.py`), `branch_ledger.py` (now `analysis/branch_ledger.py`),
  `teacherforced_scan.py` (now `analysis/entropy_scan.py`), `teacherforced_analyze.py` (now `analysis/analyze_scan.py`),
  `data/scan_b1/*.npz` (was `teacherforced_b1/`; 128 traces: entropy arrays, layer-15
  hidden states @ stride 4, onset token offsets).
- Immediate next: (1) v2 hidden-state segmentation + semantic
  return-after-elimination metric (see companion note); (2) linear probe on
  stored hidden states — is pre-onset state separable from matched healthy
  positions? (controller reads exactly this representation; separability =
  feasibility of a closure-nudge trigger); (3) merge + calibrate when the
  probe completes; (4) regenerate assign→hard, certify→hard presets.
