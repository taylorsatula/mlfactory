# Lab note — 2026-08-25 — cross-process sampling is NOT bit-stable; resume semantics changed

## Context

Follow-up to `2026-08-25-step1-replay-engine-windowed-killed.md` Finding 5
(anomaly: same seed, different traces across runs). Decided with
`train/probe_determinism.py` on the H200.

## Measurement

Two fresh processes, same prompt (assign-p132), same seed
(80_000 + 17*132), G=4, temp 0.8 / top_p 0.95, max_new 2000, bf16:
trace **lengths identical** ([2317]×4 both runs) but token ids diverge —
first flip at position **354 / 515 / 675 / 736** across the four
samples, and after a flip ~70% of remaining tokens differ (1554–1762 of
2317). One early sampling flip diverts the whole trajectory.

## Findings

1. On this stack (H200, torch 2.13/cu130, transformers 5.15, bf16 SDPA),
   seeded generation is deterministic WITHIN a process but NOT across
   processes. The collector-era "in-flight samples redone
   bit-identically" contract does not transfer to 20k-token
   thinking-on traces on this substrate.
2. Consequence: a crash-resumed group can never be regenerated to match
   its on-disk rows. Regeneration would mix two lineages inside one
   immutable group.

## Decisions (write-back manifest)

- `train/grpo.py` resume semantics changed (this session, uncommitted):
  completed groups reused verbatim; **partial groups frozen from disk**
  (rewards from rows, replay from the persisted sequences of the
  samples that made it); only zero-row groups generate fresh;
  iteration-level resume reads `train.jsonl` and skips done iterations
  (RNG shuffle still advanced so later batches are unchanged).
- `OPERATIONS.md` — owes a line: on the H200 substrate, seeded
  generation is not cross-process bit-stable at reasoning length;
  resume evidence = rows + seqs files, never regeneration.
- `HYPOTHESIS.md` — no change.

## State at note time

Dry run (production settings, 3-prompt slice) still generating on GPU0;
the planned mid-run kill will exercise the new resume path directly.
