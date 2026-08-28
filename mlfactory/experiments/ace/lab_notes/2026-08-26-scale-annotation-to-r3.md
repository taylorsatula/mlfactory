# Annotation workstream — scale-up to R3 (capture → probe → directions)

> Session: 2026-08-26 (post-compaction, autonomous). Picking up from the
> converged v5 pilot (see 2026-08-26-annotation-pilot-prompt-hillclimb.md).
> Principal instruction: "do everything up until the forking task. Hold when
> complete." So: R0 annotation-at-scale + agreement → R1 capture → R2 probe →
> R3 directions; **hold before R4 forks** (the rental-spend rung).

## Objective (restated)

Climb the ladder from ANNOTATION_SIDESTEP.md on the 96-trace xsub corpus:
annotate every trace (R0), teacher-force-capture activations at annotated
positions (R1), test position-level separability per layer (R2), and extract
mean-difference steering directions (R3). Forks (R4) are the principal's
call — not spent autonomously.

## What was built this session (all under ace/annotate/)

- `build_plan.py` → `data/annotation_plan_xsub.jsonl`: 48 pairs covering
  all 96 traces (24 q8 + 24 bf16, 16 per domain). Contrast-priority pairing
  (correct↔cap, then correct↔wrong, cap↔wrong, within-class). Deterministic.
  Guard: every trace covered exactly once. 28/48 pairs are mixed-outcome.
- `run_batch.py`: applies locked v5 framing-C to every plan pair, resume-safe
  (a pair is done when C.json exists), A-fallback if C blows the thinking
  wall with no flags, consolidates to `data/annotations_xsub_pass{N}.jsonl`
  with resolved char offsets. pass2 = the 5-pair double-annotation subset.
- `r0_agreement.py`: K4 label-mush check — greedy best-IoU span matching
  between pass1/pass2 on the double subset (Jaccard + class-confusion +
  boundary drift).
- `capture_activations.py` (R1): teacher-forces each annotated trace through
  local HF bf16 Qwen3.5-9B in chunks (DynamicCache continuation,
  use_cache=True), hooks all 32 decoder layers and gathers residual rows at
  annotated positions (pre_onset/onset/mid/end) + depth-matched controls,
  snapshots DeltaNet recurrent state at REC_LAYERS [2,8,9,12,20]. One .npz
  per trace → `data/annot_captures/` + manifest.
- `probe_positions.py` (R2): per (class × kind × layer) pooled AUROC on the
  mean-diff direction + within-trace AUROC + sign consistency + escape-vs-
  reheat at onset (succ-trace vs fail-trace onsets). Adjudicates kill
  conditions K1–K3. CPU-only.
- `compute_directions.py` (R3): mean(controls) − mean(onsets) per
  (class × layer), unit-normalized, gated by probe AUROC →
  `data/steering_directions/`. These double as the constant-λ baseline of
  TERMINAL_FORK_COMPUTE constraint 7.

## Substrate / architecture facts verified live this session

- HF bf16 model at /home/admin/models/hf/Qwen3.5-9B (config nests the text
  model under config.text_config; 32 layers, hidden 4096, q/kv heads 16/4,
  vocab 248320; full-attn layers are 3,7,11,15,19,23,27,31; the other 24 are
  linear-attn GatedDeltaNet). AutoModelForCausalLM resolves to
  Qwen3_5ForCausalLM with model.model.layers accessible.
- Served-stream reconstruction is bit-exact: solver_prompt(candidate) +
  build_prompt_ids(enable_thinking=True) matches recorded n_prompt_tokens
  (verified on p53 ×6, p140 n_prompt=160).
- Chunked DynamicCache continuation needs use_cache=True on every call;
  omitting it corrupts get_seq_length and crashes single-token decode.
  With use_cache=True, prefill+decode and multi-token chunked continuation
  both work. One-shot vs chunked recurrent states differ by ~5e-2 (max
  |state| ~19) — FP re-association in the chunk kernel, not a semantic
  error; noted as a capture caveat, each channel is internally consistent.

## TeaLeaves (repo: taylorsatula/TeaLeaves, principal is the owner)

Clone now at /home/admin/TeaLeaves (moved off tmpfs /tmp, which is wiped on
reboot — the power outage erased the first clone). Generalizations made to
the repo this session (198 tests still pass):
- model_adapter.from_model unwraps nested text_config (vision wrappers).
- Hybrid architectures are first-class: full-attn and linear-attn layers are
  discovered separately; validation requires every layer to carry exactly one
  of the two (was: full-attn required on every layer → Qwen3.5 failed).
- run_analysis: --dtype {float16,bfloat16}, --residuals-only (skips the O(n²)
  attention matrix and lifts the 20k-char cap), --max-case-chars; metadata
  dtype now reflects the real load dtype.
These were verified against the real Qwen3.5-9B (ModelAdapter.from_model
returns 8 full-attn + 24 linear-attn layers).

## Decisions

- Annotation framing locked to C (compare & contrast) with A-fallback only
  if C hits the thinking wall with zero flags. Basis: pilot verdict.
- conf tier: probes default to conf=clear; conf=all is a switch. Probable
  tier was validated high-signal in the pilot audit.
- Capture saves recurrent state only at onset+control positions (R2 rec
  channel compares onset vs control); mid/end/pre_onset live in residuals.
  Keeps npz ~115 MB instead of ~190 MB for a 36-position trace.
- REC_LAYERS = [2,8,9,12,20] (the strong b1-map recurrent layers from
  LAYER_HYPOTHESES.md).
- Controls per annotation = 4, depth-matched (same token-index decile),
  outside all annotated spans, seeded per trace (reproducible).

## Bugs caught and fixed this session (record so they don't recur)

- `run_batch.consolidate` wrote `sample_i = int(f["trace"])` — the trace
  NUMBER (1|2), not the sample index. All rows collapsed to sample_i ∈
  {1,2}; downstream "only 24/96 traces have flags" was the symptom.
  Fix: `sample_i = int(labels[tnum][1:])` (trace_label was already
  correct). Lesson: any time a consolidated count looks implausibly
  small, check the join keys before believing the science.
- `consolidate` re-loaded the 96-row corpus inside the per-pair loop
  ("inefficient but it works" — principal: just fix it). Hoisted.
- `capture_activations` smoke-data generator first wrote trace numbers as
  sample_i too (same species of bug, caught in smoke verification).
- char_to_token / onset boundary spot-check initially looked like a
  capture bug; it was a check-script bug (compared the wrong onset).
  Re-check with the right target before patching working code.

## Preliminary pass1 stats (456 flags, pre-pass2; corrected sample_i)

- 456 flags: CYCLE 228, LOOP 208, MUSE 20. Resolved: 251 (55%):
  CYCLE 64%, LOOP 46%, MUSE 45%. 91/96 traces flagged; 79 with resolved
  material. Resolution rate consistent with the pilot's weak axis
  (quote ambiguity on repetitive traces); unresolved flags keep their
  quotes for review.
- Blunt cap-hit LOOP material present: 32 resolved spans > 10k chars,
  largest 56.9k (q8 p145 s2). Both substrates contribute all three
  classes.

## R0 agreement verdict (K4 adjudicated, not killed)

Double-annotation subset: 9 traces, 26 vs 24 spans.
- Exact-span Jaccard: 0.26 (below the 0.3 usability guide)
- BUT region agreement (any same-class overlap): 0.54 (CYCLE 10/15,
  LOOP 4/8, MUSE 0/3)
- Onset agreement: 0.23 within 500ch, 0.54 within 8k ch

Read: span Jaccard measures exact-span reproducibility at default temp —
known run-to-run variance (pilot lesson). What survives: within-pass
PRECISION is high (pilot audit 14/16 real episodes); cross-pass variance
is segmentation/recall noise, mostly granularity (one pass flags a 40k
wall as one span, the other as three). K4 verdict: the instrument is
noisy but usable; labels are noisy measurements as designed. Consequence
for R2: ~50% of onset labels carry multi-kchar jitter → onset AUROCs are
lower bounds; the onset-vs-mid/end profile (kill condition K2) reads the
jitter directly. Did NOT re-run annotation — v5 is already hill-climbed
and added strictness feeds GLM meander.

## R1 capture (done)

81 traces captured (all traces with resolved pass1 annotations), ~7s/trace
on GPU1, 17.7 GB peak. Per trace: residuals (32 layers × n_pos × 4096, fp16)
at pre_onset/onset/mid/end + 4 depth-matched controls each; DeltaNet
recurrent state at REC_LAYERS [2,8,9,12,20] at onset+control positions.
n_prompt reconstruction bit-exact (verified against recorded values).

## R2 probe results (fixed AUROC; conf=clear)

Onset separability — annotated onset vs depth-matched controls, LEAVE-ONE-
TRACE-OUT AUROC (direction fit on all other traces, the honest number):

| class | best layers        | LOO AUROC | n_pos |
|-------|--------------------|-----------|-------|
| cycle | L6, L16-L19        | 0.988-0.991 | 128 |
| loop  | L2, L17-L20        | 0.983-0.988 | 78  |
| muse  | L23-L29            | 0.979-0.982 | 5 (underpowered) |

- pre_onset (one token BEFORE the span starts) also ~0.97-0.98 — the
  divergence is already encoded in the state before the episode's tokens
  are written. This is the sharpest evidence for the HYPOTHESIS.md claim
  that divergence is a property of state dynamics, not token content.
- kind profile (best-LOO layer): cycle L18 pre_onset=0.979 onset=0.991
  end=0.986 (mid=0.69); loop L2 pre_onset=0.972 onset=0.988 end=0.974
  (mid=0.49). Onset/pre_onset/end separable; mid-span looks healthy-like.
- Escape vs reheat at onset (succ-trace vs fail-trace onsets, same class):
  cycle AUROC 0.781 (L31), loop AUROC 0.856 (L15). Same-class onsets in
  eventually-correct vs eventually-failed traces ARE separable → escape
  != reheat at onset → kill condition K3 does NOT fire.

Kill-condition verdicts:
- K1 onset-null: NOT fired (onset strongly separable).
- K2 post-hoc-only: NOT fired (onset/pre_onset separable, not just late).
- K3 escape==reheat: NOT fired (succ vs fail onsets separable).
- K4 label-mush: adjudicated above (noisy-but-usable, region agreement 0.54).

CAVEATS (read before over-trusting):
- All probes are teacher-forced and observational. Separability !=
  causal leverage. Phase-3 passenger test / R4 forks still required.
- Signal is broad (LOO AUROC ~0.98 across most layers), which is either
  genuinely distributed or residual position/texture confound not fully
  removed by depth matching. The escape-vs-reheat comparison (annotated
  vs annotated, not annotated vs control) is the cleaner test and is
  still strong (0.78-0.86).
- muse n=5; underpowered, treat as preliminary.

## R3 directions (done)

96 mean-difference directions -> data/steering_directions/directions_annot_clear.npz
(32 layers x {cycle, loop, muse}), unit-normalized, sign = divergence ->
healthy. These are the constant-lambda fixed-direction baselines of
TERMINAL_FORK_COMPUTE constraint 7. Gated by probe AUROC (all layers
passed; signal broad).

## Bugs fixed this session (see also earlier section)

- auroc() divided by len(pos) twice (Mann-Whitney win counts normalized by
  mean AND by n_pos*n_neg), deflating every AUROC by a factor of len(pos).
  Fixed in probe_positions.py AND analysis/analyze_map.py (the b1-map tool).
  Consequence: the recorded b1-map "loop-onset probe AUROC~0.25" was
  deflated; the underpowered verdict (n=4) may stand but that number is
  unreliable. Flagged for the user.

## HOLD — R4 forks not spent (principal's call)

Everything up to the fork rung is done. Detectors nominate; forks ratify.
R4 (concentrated forks on O(10) nominated states, the rental-spend rung)
is deliberately NOT started — awaiting principal.

## Open / next

- Wait for pass1 (48 pairs) then pass2 (5 pairs) to finish, consolidate.
- Run r0_agreement (K4), then capture_activations on the pass1 annotations,
  then probe_positions, then compute_directions.
- Hold before R4 forks; report to principal.

## Close-out (same session, later)

All "Open / next" steps above DONE. Decisions + write-back manifest:
- R0-R3 verdicts -> STATUS.md R12 (Q5 resolved; R4 annotated with the
  auroc-bug caveat), LAYER_HYPOTHESES.md (prior table + new evidence
  section), OBSERVABLES.md (position-level observables section),
  ANNOTATION_SIDESTEP.md §8, ace/AGENTS.md (annotate/ in directory +
  doc maps).
- Sidecars written for: annotation_plan_xsub, annotations_xsub_pass1/2,
  probe_results.json, annot_captures manifest, directions npz.
- auroc fix touches analysis/analyze_map.py — prior b1-era AUROC
  readings (incl. the loop-onset 0.25) were deflated by a factor of
  n_pos; rank-biserial readings unaffected.
- TeaLeaves generalizations at /home/admin/TeaLeaves UNCOMMITTED
  (model_adapter.py, run_analysis.py) — principal decides on commit/push.
- Nothing committed in mlfactory either — principal decides.
- Handoff: lab_notes/2026-08-26-handoff-r0-r3-complete-forks-pending.md.
  Holding before R4 forks.
