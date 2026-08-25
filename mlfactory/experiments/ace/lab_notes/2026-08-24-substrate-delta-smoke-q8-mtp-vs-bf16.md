# Lab note — 2026-08-24 — substrate-delta smoke: q8_0+MTP vs bf16 measured, variance acceptable for collection

Scope: the substrate-delta smoke promised in
`2026-08-24-substrate-ruling-q8-mtp-collection.md`. Six mid-band b1
prompts re-run under q8_0 GGUF + MTP (16 fresh samples each, sample_i
8–23) and compared to their b1 bf16 HF rates (8 samples each). This is
the measured answer to "how near-indistinguishable is q8_0 here" — on
record before the b2 launch.

## Setup

- Prompts (chosen mid-band, most shift-sensitive per family): assign p1
  (6/8), machine p16 (3/8), adversary p18 (3/8), certify p28 (6/8),
  grid p34 (5/8), hypothesis p41 (7/8).
- Substrate: `Qwen3.5-9B-MTP-Q8_0.gguf` (unsloth MTP repo — the non-MTP
  quant fails MTP-context creation at load), llama.cpp build 10336,
  `--spec-type draft-mtp --spec-draft-n-max 3`, 4 parallel slots, one
  server per GPU (:3091/:3092). Collector:
  `frontier/collect_rollouts_api.py`; same strict `check()` as b1.
- 96 samples total, all completed, zero collector errors.

## Results

| prompt | b1 bf16 (n=8) | q8+MTP (n=16) | shift (eighths) |
|---|---|---|---|
| assign p1 | 6/8 = 0.75 | 14/16 = 0.88 | +1.0 |
| machine p16 | 3/8 = 0.38 | 12/16 = 0.75 | **+3.0** |
| adversary p18 | 3/8 = 0.38 | 3/16 = 0.19 | −1.5 |
| certify p28 | 6/8 = 0.75 | 13/16 = 0.81 | +0.5 |
| grid p34 | 5/8 = 0.62 | 12/16 = 0.75 | +1.0 |
| hypothesis p41 | 7/8 = 0.88 | 14/16 = 0.88 | 0.0 |

Mean |shift| = **1.17 eighths**. With n=8 vs n=16, pure sampling noise
produces ~1.2–1.5 eighths of apparent difference at these pass rates —
the aggregate is inside noise. **No prompt flipped a band** (none
approached 0/8 or 8/8 on either substrate).

## Findings

1. **Landing-place equivalence holds at calibration granularity.** Four
   of six prompts sit inside the noise band; correct-rate ranking is
   preserved. For the ruling's bet ("near-indistinguishable at an eighth
   of the time"), the landing-place axis is confirmed.

2. **The failure *dynamics* are not equivalent — machine p16 is the
   evidence.** b1: 5/8 finished-wrong, 1/8 truncated. q8+MTP:
   **0/16 finished-wrong**, 4/16 truncated. b1's failures commit to a
   wrong answer; q8's failures keep searching to the cap. adversary p18
   shifts the same direction (3/8 → 13/16 truncated). The substrate
   changes *how the model fails* even where it lands comparably. This is
   the real content of "q8 is a different model" — it matters for
   failure-mode observables, not for band calibration.

3. **Consequence (carried into the ruling):** q8-collected failure-mode
   statistics (loop rates, truncation mix, emission-paralysis counts)
   are q8's failure modes and are not evidence about the bf16 policy.
   Band membership is re-verified by **regeneration** — fresh bf16
   rollouts, never re-scoring q8 completions (the verifier is
   model-blind; re-scoring reproduces the identical verdict).

4. **Throughput, measured:** q8_0+MTP single stream 152–161 tok/s vs
   Q5_K_M no-MTP 106 tok/s vs HF bf16 37 tok/s (4.1× baseline per
   stream; a single stream already ≈80% of 3090 bandwidth — MTP
   amortizes weight reads). Full b2 (384 samples) ETA **~5–6h** across
   both GPUs vs b1's 36h.

## Decisions with rationale (write-back manifest)

- Variance accepted for b2 collection on q8_0+MTP; this note is the
  measured substrate-shift baseline (mean 1.17/8, worst 3/8,
  failure-mix sensitivity flagged). → referenced by `OPERATIONS.md`
  "Substrate policy".
- Re-verification clarified as regeneration, not re-evaluation →
  `OPERATIONS.md` condition 3 wording. Done with this note.
- b2 launch proceeds on this evidence. Band verdicts and the b2 result
  table go in the next lab note.

## State at note time

- Delta artifacts: `/tmp/delta_gpu{0,1}.jsonl` (96 rows) + collector
  logs. (Promote to `data/` via sidecar if reused; this smoke is a
  decision input, not pool evidence.)
- Both Q8_0+MTP servers still up (:3091/:3092), b2 candidates staged
  (`data/acegen_probe_b2.jsonl`, 48 prompts, pids 49–96).
