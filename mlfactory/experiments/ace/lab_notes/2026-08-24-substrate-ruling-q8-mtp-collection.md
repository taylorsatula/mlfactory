# Lab note — 2026-08-24 — substrate ruling overridden: q8_0 GGUF + MTP accepted for collection

Scope: a binding-context ruling change, made mid-b2-planning, plus the
tooling that implements it. No new measurements yet; the substrate-delta
numbers and b2 results land in follow-up notes.

## The decision (user ruling, 2026-08-24)

**Rollout collection/calibration may run on q8_0 GGUF + MTP (llama.cpp)
instead of HF bf16.** The prior ruling ("Precision: bf16 — q8 is a
different model") is overridden for *collection only*; it stands for the
training stack (see below).

Rationale, as argued and accepted:

- The ACE effect is unproven. While it is unproven, iteration speed
  dominates evidence purity — ~8x faster collection is a good bet.
- b2's prompts are new (pids 49–96, zero overlap with b1), so there is no
  pooling of rows across substrates; at worst the evidence chain gains one
  annotated batch.
- Calibration is a coarse, re-verifiable filter (band membership at n=8).
  Sampling noise at n=8 already moves prompts across 4/8↔5/8 between
  re-runs; a small quantization shift is not categorically worse than
  accepted noise.
- **The re-verification path exists and is cheap:** GRPO training needs
  bf16 HF rollouts regardless, so every q8-banded prompt gets its band
  re-measured on the true substrate at training time. If it drops out of
  band, it leaves the pool.

The agent's initial objection ("the evidence chain breaks") was overstated
and withdrawn: it conflated row-pooling (which doesn't happen across
substrates) with prompt-pooling (which is annotated and re-verified).

## What stays binding

- **bf16 HF remains mandatory** for controller training (GRPO), teacher-
  forced capture/analysis, steering probes, and anything reading hidden
  states — the steering hook lives in the transformers residual stream;
  llama.cpp cannot host it. This is physics, not ritual.
- Rows record substrate (`backend`, `quant` fields). No row-pooling across
  substrates. Re-verification obligation before q8-banded prompts feed
  training. (Full conditions: `OPERATIONS.md` → "Substrate policy".)

## Tooling implemented this session

- `frontier/collect_rollouts_api.py` — collector against an OpenAI-
  compatible llama-server endpoint. Same row schema, prompt construction,
  and `objective_check` as `collect_rollouts.py` (imported, not forked).
  Seeds recorded per request; caveat recorded in the module docstring:
  llama.cpp continuous batching is not bit-stable for a given seed.
  llama-server parses the CoT into `reasoning_content` (stripping the
  template's think tags); the collector re-wraps it so `objective_check`
  sees the HF-decode shape.
- Local serving: one llama-server per GPU (:3091/:3092), build 10336 at
  `/opt/llama.cpp/qwopus/current` (native `qwen35` arch + MTP graph
  support verified; same build already serves a 27B MTP model in prod).
  Model: `unsloth/Qwen3.5-9B-MTP-GGUF` Q8_0 → `/home/admin/models/
  Qwen3.5-9B-MTP-Q8_0.gguf`. NOTE: a non-MTP Q8_0 quant fails MTP-context
  creation at load — must be the MTP repo's file. `--spec-type draft-mtp
  --spec-draft-n-max 3` flags mirror the proven qwopus service.
- `--reasoning none` is NOT a valid flag value in this build (on|off|auto);
  and `--reasoning off` still splits CoT into `reasoning_content` — the
  split is template-driven, hence the collector's re-wrap.

## Pending (not evidence yet)

- Substrate-delta smoke: re-run 3 b1 prompts (one easy, adversary, grid)
  under q8_0+MTP, compare pass rates to b1 → quantifies "near-
  indistinguishable" on record. Result goes in the b2 note.
- b2 collection: 48 candidates (pids 49–96; adversary at default, machine/
  assign/certify/grid/hypothesis at hard preset) × 8 samples, candidates
  file `data/acegen_probe_b2.jsonl` (family-interleaved for shard balance).

## Decisions (write-back manifest)

- Ruling override + conditions → `OPERATIONS.md` (Precision row,
  Concurrency row, new "Substrate policy" section). Done this session.
- `CALIBRATION.md` gets the substrate of each batch in its pool-status
  table → to do with the b2 result note.
- This note is the reference for why q8 rows exist in the evidence chain.
