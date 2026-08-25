# Lab note — 2026-08-25 — GRPO H200 smoke results + regime measurements

## Context

Forecasting smoke (`train/smoke_h200.py`, tmux `smoke`, log
`/workspace/smoke1.log`, report `/workspace/smoke1/report.json`) on Vast
#46911241 (2× H200 140 GB, PCIe 5.0 interconnect, driver 590.48.01),
instance stack torch 2.13.0+cu130 / transformers 5.15.0, model
`Qwen/Qwen3.5-9B` bf16 at
`/workspace/models/hub/models--Qwen--Qwen3.5-9B/snapshots/c2022…`.
Setup and diagnostics: `lab_notes/2026-08-25-grpo-h200-setup.md`.

Protocol: 6 pool prompts (first per family: adversary-p49, machine-p124,
assign-p132, certify-p140, grid-p148, hypothesis-p158), group 4,
temp 0.8 / top_p 0.95 (matched to b2 collection), thinking on,
max_new 26000, seeds `70_000 + pid` — a fresh seed space; these rows are
a NEW bf16 artifact, not a regeneration of b2's q8 rows.

## Measurements

- **Load:** 8.95B params, bf16 alloc 16.68 GB, 8.4 s.
- **Generation (thinking on, batch 4):** 475,042 tokens / 6514 s =
  **73 tok/s aggregate**. Trace lengths min 6296 / **median 22349** /
  max 26000; **6/24 traces truncated at the cap** (adversary 2, machine
  0, assign 1, certify 1, grid 1, hypothesis 0). bf16 traces run longer
  than q8 (median 22.3k vs 18.4k over 720 b2 rollouts).
- **Per-group strict correctness (n=4):** adversary **4/4** (lens
  21301/26000/24281/26000), machine **4/4** (22349/25949/24244/21309),
  assign **3/4** (6296/11198/26000/10086), certify **1/4**
  (15058/26000/19900/24453), grid **2/4** (21893/23303/25540/26000),
  hypothesis **4/4** (18532/7413/8742/13195). Overall 18/24.
  Within-group outcome variance survives the substrate move on the
  mixed families — the forked-outcome precondition for Q10.
- **Replay memory curve** (gradients on, longest trace, completion
  25949 tok): cap 512→26.9 GB, 1024→32.6, 2048→44.0, 4096→66.6,
  **8192→111.6 GB (5.1 s), 16384→OOM** (peak 136.6 of 140). Linear in
  cap. `grad_finite=True` at every cap; base fingerprint unchanged.
- **Kernel A/B** (`probe_throughput.py`, GPU1, batch 4): sdpa
  151/155 tok/s short/long-context; flash_attention_2 (flash-attn
  2.8.3.post1, sm_90 source build) 147/150; +fla 0.5.2 +
  causal-conv1d 1.7.0 → 144/151. **Zero gain from any kernel.** Measured
  throughput is ~15% of the batch-4 bandwidth ceiling (~1066 tok/s), so
  generation is not bandwidth-bound; the kernel route is exhausted
  pending one check (is fla actually invoked, or silently falling back).

## Evidence-gap (own discipline)

Smoke kept per-sample correctness in memory but persisted only group
aggregates — `gen_rows.json` cannot join length↔outcome. Production
rollout writer must emit per-sample rows (correctness, length,
truncated, seed).

## Findings

1. Traces are long (median 22.3k) and the replay gradient window
   tops out at ~8k on this card: a single-prefix replay is structurally
   unable to reach mid/late-trace positions — the reheat and
   durable-pruning moments `HYPOTHESIS.md` says the controller must
   learn about. **Segmented replay is a hypothesis requirement, not
   just memory plumbing**; the memory curve sets the window size.
2. Truncated traces score ~0 under terminal reward, so completion
   length becomes reward-correlated through the backdoor — a shape of
   the banned length axis arriving without being added. Cap-hit rate
   must be reported per group in every GRPO batch.
3. Generation is the wall-clock bottleneck (~73–150 tok/s aggregate;
   ~15–25 min per group of 4 at 20k+ traces). The levers that survive
   the kernel results: rollout parallelism across the two GPUs (steered
   arm GPU0 / base arm GPU1, or prompt shards), and possibly a vLLM
   base arm (image-native, model-verified). TP=2 is a low-bet over
   PCIe at this model size; probe only if still needed.

## Decisions (write-back manifest)

- `REWARD_POLICY.md` — truncation-as-backdoor-length caveat (finding 2).
- `STATUS.md` — Q10 current-bet update (fork variance survives bf16;
  gradient plumbing proven); new Q11 (kernel/throughput); new R9
  (H200 infrastructure + regime) pointing to this note and the setup
  note.
- `ENVIRONMENT.md` — remote training stack section; FLA/conv1d row
  moved from "deliberately not installed" to "installed on remote,
  measured zero-gain"; flash-attn added with the same outcome.
- `OPERATIONS.md` — training-location ruling (GRPO on Vast H200; local
  3090s collection-only) + H200 concurrency condition.
- `CALIBRATION.md` — bf16 re-verification preview (this note's n=4
  per-family numbers, preliminary until the full first unsteered batch).
- `PHASES.md` — Phase 2 gets the replay measurements + segmented-window
  requirement; Phase 3 gets the attempt-prep status.
- `HYPOTHESIS.md` — **no change.** The smoke measured substrate, not
  observables; nothing sharpens or overturns the claim. The segmented
  replay requirement is *derived from* the hypothesis (decision points
  span the trace) and recorded here and in `PHASES.md`, not absorbed
  into the concept doc.
