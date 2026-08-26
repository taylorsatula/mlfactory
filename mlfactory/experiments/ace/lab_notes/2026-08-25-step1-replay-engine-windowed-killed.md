# Lab note — 2026-08-25 — Step 1: replay engine wired; windowed replay KILLED

## Context

Step 0 (wiring) + Step 1 (loop health ladder) of the GRPO gated-execution
plan (`lab_notes/2026-08-25-handoff-grpo-gated-execution.md`), run on Vast
H200 #48673764. New code: `train/pool_adapter.py` (pool → items, CHECK
verify dispatch, deterministic stratified split), rewritten `train/grpo.py`
(pool mode, thinking ON, 26k backstop, per-sample append-safe rows +
persisted sequences, objective guards, replay engine), probes/tests in
`train/` (`probe_long_replay.py`, `test_nan_guard.py`,
`diag_window_drift.py`). Remote artifacts: `/workspace/s1_eq/`,
`/workspace/s1_diag.log`, `/workspace/s1_nan.log`, `/workspace/s1_long/`.

## Measurements

- **Split:** 46-prompt pool → train 30 / holdout 16, ~1/3 held out per
  family, deterministic (no RNG; stride selection), manifest written to
  the run dir. 46/46 pool items pass their own strict CHECK on
  `"Answer: " + reference`.
- **Equivalence (zero-init controller = exact no-op):**
  checkpointed full-trace replay vs no-grad single pass: **max_abs_diff
  0.0 at 3,000 AND at 26,000 completion tokens** — bit-exact at
  production length; grad flows (grad_ok true). Windowed (cached
  detached-prefix) replay: max 0.25 nats, mean 0.0075 at 26k — and see
  kill evidence below.
- **Full replay at production length (26,000 tokens, one backward step):**
  peak 125.2 GB reserved (allocator pre-polluted by the windowed eq pass;
  clean-state production number pending from the dry run), 15.9 s,
  grad_norm finite, base fingerprint holds.
- **NaN guard:** planted NaN in `ctrl.up.weight` trips exit 9 in full
  mode with positive control finite (`test_nan_guard.py`).
- **Generation:** batch-1 = 20.2–20.3 tok/s (consistent with smoke's 73
  tok/s ÷ 4); trace lengths at seed 123_456: run 1 = 21,427 (EOS), run 2
  = 26,000 (cap-hit) — cross-run sampling determinism is NOT assumed
  (see Findings 5).

## Findings

1. **Windowed replay via cache continuation is KILLED — twice over.**
   (a) Numerical: with zero-init controller (exact no-op), splitting a
   forward into cached-prefix + window corrupts the first ~50 tokens
   after every split boundary by up to **11.8 nats** (likelihood ratio
   off by >1e5; `diag_window_drift.py`, 3,000-token trace: A-split max
   11.82 argmax at pos 12; drift mean GROWS with split depth 0.14 →
   0.28 → 0.46). The cache mechanism itself is bit-exact (single pass
   with use_cache=True: diff 0.0) — the corruption is in the
   split/continuation, not the cache object. Content-dependent
   magnitude (0.19 on one trace, 11.8 on another). (b) Mechanical: the
   FLA gated-delta-rule chunk backward crashes on the continued cache
   state ("modified by an inplace operation … CopyBackwards … version
   2") — gradients through a cached window cannot even complete.
   Consequence: there is NO silent OOM fallback from full replay; a
   full-mode OOM is guard trip exit 8.
2. **Gradient-checkpointed full-trace replay is the replay mechanism.**
   Bit-exact vs single pass at 26k, one backward step finite, fits the
   140 GB card. It also satisfies the HYPOTHESIS requirement directly:
   gradients reach mid/late-trace positions (the windowed design was
   the workaround for a memory problem checkpointing removes).
3. **The first full-mode replay attempt OOMed at 26k — not from
   activations, from the logprob extraction graph.** The naive chunked
   float32 upcast in `completion_logprobs` retains ~2×63 MB per 64-token
   chunk (~52 GB at 26k). Fixed with `_TokenLogprobs`, a custom
   autograd function that saves only the bf16 logits reference +
   targets and recomputes softmax in backward (gradient verified vs the
   naive path on CPU, fp32 atol 1e-5 + bf16 finite). (A post-step
   full-vs-ref diff of 0.226 in the first probe was the updated
   controller's real effect, NOT drift — comparison ordering bug in the
   probe, corrected; zero-init checks bracket the step.)
4. **Crash-resume fidelity required persisted sequences.** Rows are
   written at generation time; a crash before replay would otherwise
   resume the group as "done" and silently drop its gradient. Group
   sequences are now saved next to the rows (`seqs/*.pt`) and replayed
   on resume.
5. **Anomaly (unresolved): cross-run sampling determinism broke at long
   length.** Same seed, same box, same settings: one run produced a
   21,427-token EOS trace, the rerun a 26,000-token cap-hit trace.
   Within-run determinism (the resume contract) is being tested by the
   dry run's iter-0 steered==base identity check; if cross-process
   regeneration is not bit-stable at 20k+ tokens, "in-flight samples
   redone bit-identically" degrades to "redone statistically
   identically" and the rows-file remains the evidence of record.
6. **Environment anomaly (unresolved):** the first long-probe process
   died silently ~5 min after launch (no traceback, no OOM in cgroup
   `memory.events`, tmux server gone); the identical relaunch with
   faulthandler completed. Watch for recurrence in the dry run.

## Decisions (write-back manifest)

- `PHASES.md` — Phase 2 replay paragraph: checkpointed full-trace replay
  replaces "segmented (windowed) replay … window size 8k"; windowed
  cache-continuation killed with reasons (this note).
- `STATUS.md` — resolved row R10 (windowed replay feasibility: NO) +
  Q10 row update (replay engine proven at production length).
- `OPERATIONS.md` — replay-mode ruling (full/checkpointed; window is
  investigation-only; no silent OOM fallback, exit 8).
- `ENVIRONMENT.md` — traps: hybrid-cache continuation corrupts boundary
  tokens + FLA chunk backward crashes on cache state; naive fp32
  vocab-chunk graphs OOM a 140 GB card at 26k.
- `HYPOTHESIS.md` — no change (mechanism, not claim).

## State at note time

Dry run (production settings, 3-prompt slice: assign-p132, certify-p140,
grid-p148; G=4, 2 iters, 26k cap, thinking on, fresh 80k seed space)
running on GPU0, started 22:12 UTC. Remaining Step-1 gates: per-sample
row completeness, iter-0 steered==base bit-identity, mid-run kill +
resume, fingerprint after 2 iters, production replay peak memory. Report
at the Step-1 checkpoint before Step 2 per the plan.
