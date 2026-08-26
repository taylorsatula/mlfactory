# Lab note — 2026-08-26 — annotation workstream launch + cross-substrate dual-collect

Scope: records the decisions launching the span-annotation workstream and
the cross-substrate collection now running. Results come in a later note;
this one is the design and its rationale, written while the collection arms
are mid-flight.

## Context (user rulings this session)

- Documentation axioms are treated as guidelines, not vetoes — the user
  rules case-by-case on evidence. Applies to the whole session (reward
  policy strictness included), not one clause.
- Annotation target is intermediate reasoning actions in ALL traces, not
  terminal-failure attribution: success traces carry recovered musings and
  productive cycles; cap-hit traces are loop-class material, not confounds
  to avoid.
- Blunt cases first; don't over-design day one.
- q8 iteration now, bf16 polish later — accepted as the working loop.
- fp16 KV for local q8 serving (user directive, supersedes the b2 q8-KV
  precedent for this collection).

## What was built

- `annotate/RUBRIC.md` — pass-1 rubric, three span classes: `muse` (idle
  musing / escape: content ungrounded in the prompt and never causally
  reused; canonical example: NFL-scoring trace musing about a Jared Goff
  injury represented nowhere in the prompt data), `cycle` (explore→reheat→
  prune exemplar), `loop` (rework). Quote-based spans (annotators quote
  start/end text; harness resolves to char offsets — LLMs can quote, they
  can't count characters). `basis` field mandatory (trace-internal
  evidence); annotator is outcome-blinded; noisy labels at scale are the
  design, not a defect.
- `annotate/build_pairs.py` → `data/annotate_pairs_p1.jsonl` (326 rows):
  71 same-prompt S/F pairs (44 hf-bf16, 27 q8-mtp) + 255 cap-hit loop
  targets, drawn from probe_b1 / frontier_p1 / b2 pools. Pairing was the
  initial frame; the annotation unit is the trace, pairs stay as sibling
  metadata. Manifest restructure to trace-centric happens post-collection.
- TeaLeaves (github.com/taylorsatula/TeaLeaves) inspected as the
  annotated-position capture vehicle: char-region→token resolution
  (cumulative decode mapping, BPE-robust), named query positions,
  per-layer residual hooks + logit lens. Verified against Qwen3.5-9B:
  three discovery failures as-is (nested `text_config` VL wrapper;
  `self_attn` present on only 8/32 layers — 24 expose `linear_attn`;
  GatedDeltaNet materializes no attention matrix). Residual hooks are
  architecture-agnostic and portable; attention-matrix capture only on the
  8 full-attention layers. 198/198 of its tests pass in the ace venv.

## The cross-substrate dual-collect (running now)

Design: same 6 mid-band b2-LIVE prompts (adversary 53/56, certify 140/145,
grid 150/152) under both substrates, paired seeds (seed-base 72000, same
(pid, sample_i) seed across substrates), n=8 each.

- **q8 arm — local 3090**, llama.cpp 10336 + `Qwen3.5-9B-MTP-Q8_0.gguf`,
  `--spec-type draft-mtp`, fp16 KV, ctx 32768 parallel 1. Output
  `data/xsub_q8.jsonl`.
- **bf16 arm — rented Vast box** (2× RTX PRO 6000 Blackwell, instance
  48783410, llama.cpp template): `unsloth/Qwen3.5-9B-MTP-GGUF`'s
  **BF16.gguf** via llama-server + draft-mtp, one server per GPU. Output
  `data/xsub_bf16_gpu{0,1}.jsonl`. Substrate recorded in rows as
  backend=llama.cpp, quant=BF16-MTP.

### Incident: badctx abort-1 (local q8)

First q8 launch used b2's `--parallel 4` with `--ctx-size 32768` — llama
server partitions ctx across slots (8192/slot), silently capping traces at
~7.9k tokens; collector flagged `truncated=True` at n_new≈7898. Two bad
rows archived as `xsub_q8.abort1_badctx.*` (not deleted). Relaunched
`--parallel 1` (full 32768 slot); first row then landed at 25,253 tokens,
untruncated. The fix was verified on the first post-fix row before
declaring the arm healthy.

## MTP on bf16 — the measured answer (was: "not a thing")

Initial claim this session: no MTP speedup available on the HF bf16 path.
User pushed back; re-verification confirmed the mechanism and found the
sidestep:

- transformers 5.14.1 HAS generic MTP speculative generation
  (`MtpModel` in `modeling_layers.py`, `MtpCandidateGenerator` keyed on
  config `num_mtp_layers`) — but the Qwen3.5 implementation ignores
  `mtp.*` checkpoint weights at load
  (`_keys_to_ignore_on_load_unexpected = [r"^mtp.*"]`) and its config
  exposes only the unused `mtp_num_hidden_layers` key. HF route: dead in
  this version.
- `unsloth/Qwen3.5-9B-MTP-GGUF` ships `Qwen3.5-9B-BF16.gguf` — genuine
  bf16 weights + MTP layers. llama.cpp self-speculation with acceptance
  testing preserves the target distribution (lossless). That is what the
  Vast arm runs.
- Cost, stated plainly: llama.cpp-BF16 numerics ≠ HF-bf16 numerics
  (kernel ordering/accumulation), closer than q8 but a distinct serving
  substrate. Rows carry the backend/quant tags; HF-bf16 remains the
  canonical training/fork substrate, and the 44 hf-bf16 pairs already in
  the annotation corpus are the transfer bridge. Precedent: the 2026-08-24
  substrate-delta-smoke accepted a larger delta (q8) for collection.

## Decisions with rationale (write-back manifest)

- Annotation workstream is live: trace-unit spans, three classes, quote
  format, outcome-blinded annotator, labels as noisy measurement. →
  Record in `STATUS.md` open questions; rubric owns the content
  (`annotate/RUBRIC.md`).
- Cross-substrate collection substrate policy gains a third entry
  (BF16-GGUF-MTP via llama.cpp) pending this collection's outcome stats.
  → `OPERATIONS.md` substrate policy, after arms land and bands are
  compared.
- TeaLeaves capture adaptations (nested config, linear_attn-aware
  discovery, bf16) are scoped but NOT built — build decision waits for
  probe-design results from this annotation pass. → nothing written yet;
  candidate infrastructure, not evidence.
- q8-vs-bf16 divergence: landing-place equivalence was already measured
  (2026-08-24 substrate-delta-smoke: mean shift 1.17/8, no band flips);
  failure-species mix is substrate-skewed (q8 truncation-heavy). The
  dual-collect adds the first same-prompt n=8×2 dataset for the
  annotation-era question. → results note supersedes this paragraph.

## State at note time

- Local q8 arm: running, ~6/48 rows, median ~136 s/sample, zero
  truncations post-fix.
- Vast arm: box up, BF16 GGUF downloading (~4/19 GB), collector venv
  verified (IMPORT_OK), servers not yet launched.
- Cost clock: Vast $3.766/hr since 14:57 UTC.
