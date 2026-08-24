# Lab note — 2026-08-24 — Multi-layer map: where the signal lives (measurement site ≠ steering site)

Scope: first full per-layer mapping pass over the b1 probe (144 clean
capturable traces ≤26k). Instruments: `teacherforced_map.py` (all 32
layer residuals + 24 final recurrent states + final-layer entropy) and
`teacherforced_map_analyze.py` (streaming, 1 GB RAM). This pass answers
*where each phenomenon is readable* and *whether the recurrent channel
carries outcome information the residual doesn't* — explicitly decoupled
from where the controller will eventually hook (the steering layer is a
separate later decision on its own criteria).

## Methodological ruling (on record)

Measurement site and steering site are two independent questions. This pass
maps where each phenomenon lives across all layers and both channels
(residual + recurrent), with no commitment to where the controller hooks.
Letting the steering choice (or a current hunch) pre-decide the measurement
site is the error this pass exists to correct. Readability ≠ causal
leverage; those can live at different layers, and the steering layer gets
chosen later on its own criteria.

## Findings (claim + evidence)

1. **Layer 15 (the current hook) is a weak measurement site.** Step-cosine-
   distance outcome separation at L15 = **+0.149** rank-biserial — near the
   bottom of the mid-layer range (L4–L29 all positive). The signal is 3–4×
   stronger elsewhere. We have been measuring from the wrong place.

2. **Outcome separation is broadly distributed across mid layers, strongest
   at linear-attention blocks, not full-attention blocks.** Top layers by
   rank-biserial (correct vs wrong, clean finishes, mean step-cosine-dist):
   | layer | type | rb |
   |---|---|---|
   | L25 | lin | +0.588 |
   | L17 | lin | +0.564 |
   | L6 | lin | +0.561 |
   | L24 | lin | +0.518 |
   | L23 | FULL | +0.443 |
   | L13 | lin | +0.490 |
   | L15 (current) | FULL | +0.149 |
   | L31 | FULL | −0.273 |
   Early L0 is negative (−0.747); the very final layer (L31) inverts. The
   architecture doc's prior — that linear-attention blocks carry
   state-tracking content the full-attention residuals read out less
   directly — is supported at the measurement level.

3. **The recurrent channel carries outcome information the residual does
   not.** Final recurrent-state Frobenius norm separates at several layers
   where the residual is weak:
   - rec_2: **+0.759** (stronger than any residual layer)
   - rec_12: +0.635, rec_9: +0.603, rec_10: +0.550
   - But sign flips by layer: rec_8 (−0.574), rec_20 (−0.686)
   The architecture doc's central claim — that the DeltaNet recurrent
   state carries information the residual readout does not expose — is
   **supported**. This channel is worth the v1 chunked-trajectory
   investment (~1 GPU-h) the earlier scoping identified.

4. **ent_late confirms at +0.629** (110 correct / 18 wrong, clean). This
   matches the single-layer scan (+0.72) within sampling noise and remains
   the strongest single scalar observable. It is a final-layer quantity so
   it does not by itself answer the per-layer question, but it is the most
   stable keeper across passes.

5. **Loop-onset separability is INFEASIBLE at current n.** Linear probe
   (pre-onset vs matched-healthy mid-trace states) returns AUROC ≈ 0.25
   (below chance) across all 32 layers — degenerate because only **4 loop
   traces** have usable onset offsets against 128 healthy controls. This
   is a data shortage, not a method failure: the crashout-exclusion
   instruction removed most loop traces from this analysis, and the doc's
   matched-control design needs more onset-bearing traces than the b1
   pool currently holds. Not a negative result — an underpowered one. The
   probe must be re-run when more loop traces are available (regenerated
   hard-preset families, or the H200 harvest).

## What the map does NOT establish (honest scope)

- **Causality.** All measurements are teacher-forced and observational.
  Separability is not steering-effectiveness. The Phase-3 passenger test
  (forked outcome distributions) still has to run before any of this
  licenses an intervention.
- **Where to steer.** Readability ≠ causal leverage. The steering layer is
  a separate decision; this pass deliberately does not make it.
- **Recurrent drift over time.** Only the *final* recurrent state is
  captured. "Did the eliminated branch persist in compressed memory at
  time *t*" (the doc's latent-persistence claim) needs the deferred v1
  chunked-trajectory pass; this pass validates only that the channel is
  worth that pass.
- **Generalization.** Three families only (assign/machine/certify). Any
  layer finding is provisional until grid/hypothesis/adversary land.

## Decisions with rationale

- **Measurement site should shift off L15** for any future diagnostic pass.
  L6, L17, L25 (linear) and L23 (full) are the evidence-based candidates;
  the recurrent channel (rec_2, rec_9, rec_12) is complementary. The
  steering layer remains undecided — it will be chosen on causal-leverage
  criteria from Phase-3 forks, not from these readability numbers.
- **Recurrent channel investment approved in principle.** rec_2's +0.759
  is the single strongest separability number in the entire map and it
  lives on the channel our residual-only instruments were blind to. The v1
  chunked-trajectory capture (recurrent state at ~80 chunk boundaries per
  trace, projected summaries not raw 2 MB/position matrices) is the next
  capture when we want drift/persistence, not before.
- **Onset probe deferred until more loop traces exist.** No conclusion
  about closure-nudge feasibility can be drawn from n=4; regenerating
  hard-preset families that produce more loop-onset cases is the gating
  data need.

## Environment traps encountered

- **GPU0 desktop overhead (~1.9 GB resident)** makes it effectively a
  ~22 GB card for 20k+ token forwards. A 26k-token forward peaks at
  ~21.7 GB; on GPU0 that + desktop = 23.6 GB = exactly the limit. The
  "leak" misdiagnosis was the desktop at the edge all along. Fix for
  GPU0-bound long forwards: route to GPU1, or free the desktop.
- **DynamicCache KV retention across rows.** `cache = DynamicCache(config=…)`
  created per row holds ~3 GB of KV states; without `del cache; gc.collect()`
  it accumulates until OOM. Fix applied: explicit delete + gc between rows.
- **`np.savez_compressed` was the GPU-idle bottleneck** (~10–15 s
  single-threaded zlib per 200 MB array vs ~6 s forward → ~30% GPU util).
  Switch to `np.savez` (uncompressed): disk is cheap (780 GB free), wall
  time per trace 25 s → 10 s.
- **Analysis memory: streaming, not caching.** Loading all 144 traces'
  residuals = ~26 GB > 25 GB available → zram thrash. Rewrote the analyzer
  as a single streaming pass (one file at a time, scalar features only,
  discard arrays): 26 GB → 1 GB RAM, 112 s wall.

## State at note time

- Probe collectors running (160/384, strict scoring native, both GPUs at
  98% util).
- Artifacts: `teacherforced_map.py`, `teacherforced_map_analyze.py`,
  `data/teacherforced_map_b1/*.npz` (144 traces: 32-layer residuals @
  stride 16, 24 final recurrent states, final-layer entropy, onset
  offsets). 32 GB on disk.
- Immediate next: (1) lab-note this finding (done); (2) when probe
  completes, merge + calibrate; (3) regenerate hard-preset families
  (assign→hard, certify→hard) — these should produce more loop-onset
  cases for the deferred onset probe; (4) v1 chunked recurrent-state
  trajectory capture if/when drift/persistence becomes the question.
