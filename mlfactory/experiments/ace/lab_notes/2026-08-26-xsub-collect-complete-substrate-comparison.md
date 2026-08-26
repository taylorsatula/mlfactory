# 2026-08-26 — xsub dual-collect complete; first same-prompt q8-vs-bf16 comparison

Context: the annotation workstream's first collection (see
`ANNOTATION_SIDESTEP.md` §8, launch note
`2026-08-26-annotation-workstream-launch.md`). Six mid-band prompts
(adversary 53/56, certify 140/145, grid 150/152) × 8 samples × 2
substrates, paired seeds (seed_sample = 72000 + 17·pid + sample_i,
verified identical across substrates 48/48). q8 arm: local llama.cpp
Q8_0+MTP, fp16 KV, parallel-1. bf16 arm: rented Vast 48783410 (2× RTX
PRO 6000 Blackwell), llama.cpp BF16 GGUF + draft-mtp, two servers, one
GPU each.

Artifacts: `data/xsub_q8.jsonl` (48 rows, sha 9b8c14ff7771…),
`data/xsub_bf16_gpu0.jsonl` + `data/xsub_bf16_gpu1.jsonl` (24+24 rows,
sha 85f06aa61b77…/48d9c22a3ed6…, rsynced home and checksum-verified
against the remote). Sidecars written for all three. Analysis script:
`annotate/compare_xsub.py`.

## Findings

1. **Aggregate equivalence within noise.** q8 24/48 correct vs bf16
   22/48; cap-hits 13 each; median n_new 22266 vs 21970. The two
   substrates land in the same place at prompt-pool level.

2. **Per-prompt profiles diverge sharply — and in both directions.**
   adversary 53: q8 6/8 vs bf16 **2/8** (bf16 caps 6/8 — budget
   exhaustion, not wrong answers). grid 150: q8 5/8 (3 caps) vs bf16
   **8/8**. certify 140: 2/8 both, but different species (q8: 2 caps;
   bf16: 0 caps, committed wrongs). The 2026-08-24 species skew
   ("q8 truncation-heavy") did **not** replicate globally — it is
   prompt-specific in this set.

3. **Seed-paired outcome agreement is 26/48 = 0.54.** Same prompt, same
   seed index, different substrate → different outcome more than half
   the time (22 flips, both directions, all four species transitions
   observed). Aggregate rates hide a near-total per-trace reshuffle:
   substrate acts like a redraw of the sampling trajectory, not a
   perturbation of it. (Expected in principle — q8 vs bf16 logits
   diverge from token 1 — but the 0.54 quantifies it.)

4. **Both substrates supply blunt annotation material.** Species: q8
   13 budget / 11 committed-wrong; bf16 13 / 13. Cap-hit loop traces
   and committed-wrong traces exist for every prompt on at least one
   substrate. No prompt is dead (hardest, certify 145, still has 1/8
   on bf16).

5. **Consequence for the sidestep:** finding 3 makes kill condition 5
   (transfer null) live and measurable, not a formality — a probe
   trained on q8-trace positions must be tested against the bf16
   traces explicitly, because outcomes (and hence which spans exist
   where) reshuffle across substrates. Substrate stays a per-trace
   covariate in the annotation manifest.

## Timing & cost

- q8 arm: 48 rows in ~105 min local (~131 s/row at ~22k tokens).
- bf16 arm: 48 rows in ~85 min on 2 GPUs.
- Vast: instance up ~92 min at $3.5615/hr ≈ **$5.5**. Stopped
  (`vastai stop instance 48783410`, status `exited`, billing halted) —
  **not destroyed**; storage persists.

## Decisions

1. Collection phase of the annotation workstream is complete; the 96
   rows are the pass-1 annotation corpus candidate. → write back:
   `ANNOTATION_SIDESTEP.md` §8 (done this note).
2. Next step is the trace-centric annotation manifest restructure
   (all 96 traces first-class, same-prompt siblings as metadata,
   substrate as covariate), then R0 double-annotation pilot. The
   annotation model pick (lunaroute `-ballast` per provider preference)
   is a user decision — held for the user's return.
3. Vast instance 48783410 stays stopped-not-destroyed pending user
   call (destroy now vs keep for the next rented phase).
4. `annotate/compare_xsub.py` is the standing comparison tool for
   future xsub collections (schema-coercing, handles partial files).
