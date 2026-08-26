# Operations: binding run rulings

> Update when: the run protocol changes. These are the standing rulings
> that govern every generation/probe/analysis run. Hardware staging and
> the falsification gate live in `PHASES.md`; environment setup in
> `ENVIRONMENT.md`.

## Generation protocol

| Ruling | Value | Rationale |
|---|---|---|
| Concurrency | **one sample at a time** (HF collectors on the 24 GB 3090s); llama-server API collection uses parallel slots (`--parallel N`, KV pooled); **same-prompt group batches are fine on the H200 training box** (140 GB; smoke ran batch-4 thinking-on groups at 26k caps) | 24 GB VRAM; an 8-wide batch of 32k-token reasoning traces OOMs a 24 GB card in HF; a GGUF quant leaves room for slot concurrency; the H200 changes the arithmetic (R9) |
| Thinking | **enabled** (`enable_thinking=True`) | native reasoning mode; the phenomenon of interest |
| Precision | **substrate identity between measurement and consumer** (bf16 HF for anything the bf16 training stack or hidden-state access consumes; q8_0 GGUF + MTP accepted for text-level collection/calibration, user ruling 2026-08-24) | q8 lands near-identically but fails differently — bands are substrate-flexible with re-verification by regeneration; failure-mode observables and training are not. See "Substrate policy" below |
| Per-sample seed | `seed_base + 17*proposal_id + sample_i` | bit-stable resume for unfinished samples (llama-server collection; **not** bit-stable across regenerations on the H200 training substrate — see §NEVER regenerate) |
| Backstop cap | **26000 tokens** | reduced from 32000 — terminal loops do not contribute to identical-conditions comparison; a trace needing >26k to be right is a recorded model flaw |
| Sampling | temperature 0.8, top-p 0.95 | default for reasoning rollouts |

## Training runs (GRPO)

Controller training runs on the **Vast H200 box** (2×140 GB, bf16 HF,
stack in `ENVIRONMENT.md` §remote) — the local 3090s are
collection-only for this phase (user ruling 2026-08-25). Box lifecycle
(stop/start) is controlled **from local** with the account key
(`ENVIRONMENT.md` §remote — Access and lifecycle); stop/start preserves
the container filesystem, so stopping during local code work is safe.
Standing
conditions there: template inference services stopped before training;
results rsynced home (no persistent volume); **replay = gradient-
checkpointed full-trace** (`STATUS.md` R10 — windowed replay killed
2026-08-25; full-mode OOM is guard trip exit 8, never silent fallback);
**rollout generation runs under a deterministic SDPA backend** (default
backend is call-to-call non-deterministic on this substrate — `STATUS.md`
R11; MATH backend verified bit-stable, FLASH untested); per-group cap-hit
rate reported
with every batch (`REWARD_POLICY.md` §backdoor). The first unsteered
rollout batch is the pool's bf16 re-verification (Substrate policy,
condition 3).

## Substrate policy (2026-08-24 ruling)

Rollout collection for calibration may run on **q8_0 GGUF + MTP**
(llama.cpp, `frontier/collect_rollouts_api.py`) instead of HF bf16. The
bet, made explicit: while the ACE effect itself is unproven, iteration
speed dominates; q8_0 is near-indistinguishable at the pass-rate level;
and if the program proves fruitful, every q8-banded prompt gets its band
**re-verified under bf16 at training time** (GRPO needs bf16 rollouts
anyway), so the purity cost is deferred, not avoided.

Hard conditions:

1. **Rows record their substrate.** Every API-collected row carries
   `backend` and `quant` fields. A batch's substrate is part of its
   identity.
2. **No pooling rows across substrates.** Calibration compares prompts
   within a substrate batch; the LIVE pool may combine prompts measured
   on different substrates only at the prompt level, never by merging
   their sample rows.
3. **Re-verification obligation.** Before a q8-banded prompt feeds
   controller training, its band is re-measured under bf16 HF — and
   "re-measured" means **regenerated**: fresh bf16 rollouts on the same
   prompt, fresh sample indices, same strict `check()`, recounted.
   Re-scoring an existing q8 completion is meaningless — the verifier is
   model-blind and reproduces the identical verdict; band membership is
   a property of the *policy's* sampling distribution, so it must be
   re-sampled from the bf16 policy. Timing: a dedicated pre-training
   pass, or piggyback on the first unsteered GRPO rollout batch (the
   honest measurement of the pool the controller is about to train on).
   If the prompt drops out of band, it leaves the pool — no special
   pleading.
4. **The binding requirement is substrate identity between a measurement
   and its consumer, not bf16 as a universal constant.** Test: *is this
   number consumed by bf16 training or hidden-state access?* If yes, it
   must be measured on bf16 HF — controller training (GRPO), teacher-
   forced capture/analysis, steering probes, residual-stream reads, and
   **failure-mode observables** (loop rates, truncation mix, emission-
   paralysis counts): the delta smoke showed the substrate changes *how*
   the model fails even where it lands comparably, so q8-collected
   failure statistics are q8's failure modes, not evidence about the
   bf16 policy. If the consumer is text-level only (calibration bands,
   verifier scoring), q8_0 GGUF is acceptable under conditions 1–3.
   llama.cpp cannot host the steering hook either way.
5. **The substrate delta is measured, not assumed.** When the substrate
   changes, a few b1 prompts are re-run on the new substrate and the
   pass-rate shift is recorded (`lab_notes/`). Baseline on record:
   `lab_notes/2026-08-24-substrate-delta-smoke-q8-mtp-vs-bf16.md`
   (mean 1.17 eighths across six mid-band prompts; worst 3/8 with a
   failure-mix change; landing-place equivalence confirmed at
   calibration granularity).

The HF-bf16 collector (`frontier/collect_rollouts.py`) stays the
reference implementation; the API collector imports its prompt
construction and `objective_check` so scoring is shared, not forked.

## NEVER regenerate or truncate existing rows

Existing rows are immutable. Chop analytically at analysis time if needed
(e.g. mixed-cap rows — see below). Resume skips done `(proposal_id,
sample_i)` pairs via `already_done()`; in-flight samples discarded on
restart are redone bit-identically (deterministic seeds).

**H200 training-substrate exception (2026-08-26, R11):** seeded HF
generation is deterministic within one `generate()` call but NOT
reproducible across calls/processes on the default SDPA backend (first
flip 422–1764 tokens — see
`lab_notes/2026-08-26-step1-sdpa-generation-nondeterminism-identity-gate.md`).
Resume evidence on this substrate = persisted rows + seqs; partial groups
are frozen from disk, never regenerated (`train/grpo.py` resume
semantics).

## Data-quality covariates on record

- **Mixed-cap file:** `data/acegen_probe_b1_gpu0.jsonl` contains 32k rows
  from before the 26k ruling + recent 26k rows. No duplicate
  `(proposal_id, sample_i)` pairs. Treat cap as a per-row covariate; chop
  analytically.
- **proposal_id offset:** Collector `proposal_id` = provenance id =
  candidates line index + 1. Ad-hoc joins by line index are off by one
  (`gen/calibrate.py` and `gen/*` joins are correct).

## Hardware staging

| Tier | Hardware | Role |
|---|---|---|
| Falsification engine (current) | local 2× RTX 3090 | runs through the Phase-3 fork-causality gate |
| Scale / harvest (later) | Vast.ai H200 — **Hopper, NOT Blackwell** | only after fork delta is proven |

Runbook: `/home/admin/facktry/BABYS_FIRST_VAST_ML_ENGINEER.md`.

## llama-server stays off

User ruling. Frees GPU0 desktop contention. The two llama.cpp systemd
services (`llama-laguna`, `llama-qwopus`) are mutually exclusive and must
never run both — OOM risk. For this experiment, neither runs.

## GPU0 desktop overhead trap

GPU0 is desktop-resident (~1.9 GB used). It is effectively a ~22 GB card
for 20k+ token forwards. A 26k-token forward peaks ~21.7 GB; on GPU0 that +
desktop = 23.6 GB = exactly the limit. Fix for GPU0-bound long forwards:
route to GPU1, or free the desktop. (Misdiagnosed once as a "leak" — it
was the desktop at the edge all along.)

## Environment trap fixes (applied, on record)

- **GPU0 fragmentation OOM:** `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
  + `torch.cuda.empty_cache()` between rows; long traces routed to
  desktop-free GPU1.
- **DynamicCache KV retention across rows:** `cache = DynamicCache(config=…)`
  holds ~3 GB; without `del cache; gc.collect()` it accumulates to OOM.
  Explicit delete + gc between rows.
- **`np.savez_compressed` GPU-idle bottleneck:** ~10–15 s single-threaded
  zlib per 200 MB array vs ~6 s forward → ~30% GPU util. Switch to
  `np.savez` (uncompressed): disk is cheap, wall time 25 s → 10 s per trace.
- **Analysis memory:** streaming, not caching. Loading all 144 traces'
  residuals = ~26 GB > 25 GB → zram thrash. The analyzers are single
  streaming passes (one file at a time, scalar features only, discard
  arrays): 26 GB → 1 GB RAM.

## Strict scoring at the source

The collector's `objective_check()` dispatches ace-gen candidates to `gen/`
family verifiers (`match_mode: gen_strict_v2`); legacy madlibz candidates
keep the soft substring/numeric fallback. The soft `correct` field is
advisory only — `gen/calibrate.py` re-scores with the strict per-family
`check()` and is the sole authority on band classification
(`CALIBRATION.md`).
