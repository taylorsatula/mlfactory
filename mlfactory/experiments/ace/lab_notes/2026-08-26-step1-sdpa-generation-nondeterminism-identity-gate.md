# Lab note — 2026-08-26 — Step 1: dry-run identity gate FAILED — root cause is SDPA generation non-determinism (fix verified: MATH backend)

## Context

Step-1 gated execution (`lab_notes/2026-08-25-handoff-grpo-gated-execution.md`;
carried forward by `...-handoff-grpo-step1-inflight.md`). The production-
settings dry run (s1_dry, GPU0, launched 2026-08-25 22:12 UTC, pre-resume-
change code) completed naturally 2026-08-26 ~00:44 UTC: EXIT-CODE:0, 48
rows, 0 duplicate keys. Gate (a) — iter-0 steered==base identity under the
zero-init controller — was run while waiting for the crash-resume window
and **FAILED on all 12 pairs**, so the kill/resume test was subordinated
and the night went to diagnosis. Five probes (h1–h5) ran on GPU1 in
parallel, never touching the live dry run. Artifacts pulled to local
`/tmp/ace_s1_20260826/` (s1_dry dir, s1_dry.log, h2–h5 logs, verify_h*.sh).

## Measurements

- **Gate (a) failure:** 12/12 `(id, sample_i)` pairs mismatch on `n_new`
  and several on `reward` (iter-0 acc: base 0.75 vs steered 0.583).
- **The hook was a proven no-op:** every iter-0 steered row carries
  `rel_mean = rel_max = 0.0`; controller init verified — `up.weight`,
  `gate.weight`, `gate.bias` exactly zero (down is random; irrelevant).
  Identical logits therefore, and both arms re-seed identically before
  every group.
- **First-flip positions** (token index in full seqs, from persisted
  dry-run seqs): assign 586/690/854/855, certify 602/603/612/661,
  grid 446/598/1366/1512 — variable onset, total divergence after.
- **h2** (clean process, bf16 ctrl, 26k): flips 703/906/1126/1409 →
  reproduces outside the dry-run context.
- **h3** (fp32 ctrl, 1600 tokens): flips 701/891/938/703; **CUDA RNG
  consumption identical — exactly 6400 draws per run** (1600 steps ×
  batch 4) for both arms, both repeats. (Self-correction: the first
  consumption measurement was broken — it diffed the seed bytes; the
  philox offset lives at state bytes 8:16, not 0:8.)
- **h4:** base-vs-base flips (422/690/1409/1764) AND steered-vs-steered
  flips (422/434/690/854) — the controller/hook is irrelevant.
- **h5:** under `sdpa_kernel([SDPBackend.MATH])`: **0 flips, 8/8 samples,
  both conditions** (MATH alone; MATH + `use_deterministic_algorithms`).
- **MATH-backend throughput at 1600 tokens:** no measurable slowdown
  (~73+ tok/s aggregate batch-4; consistent with R9's "generation is ~15%
  of the bandwidth ceiling" — attention is a small fraction of decode cost
  at short length). 26k-cap throughput still unmeasured.
- **Other Step-1 gates (completed dry run):** 48 rows / 0 dups;
  `base_fingerprints_unchanged = true`; replay mode `full` for all 24
  replays, 0 OOM fallbacks; `peak_mem_gb` 47.78 (allocated) both iters;
  cap-hit rates 0.21 (iter 0) / 0.17 (iter 1); iter-1 `zero_var_groups=1`
  (assign-p132 all-correct → zero advantages; guard counted, no trip —
  `--stop-on-zero-var` not set); losses/grad_norm finite (grad_norm
  0.003 → 0.010); KL tiny (0 → 8.8e-4).

## Findings

1. **Root cause: default-SDPA generation is call-to-call non-deterministic
   on this substrate** (H200, torch 2.13/cu130, transformers 5.15, bf16).
   Two consecutive `generate()` calls with identical seed, weights, code,
   and *proven-identical* RNG streams (equal draw counts) diverge
   mid-trace (first flip 422–1764; everything after cascades). With RNG
   streams and inputs identical, the logits themselves must differ — the
   cache-growing decode attention path is not reproducible call-to-call.
2. **Fix verified:** forcing the MATH SDPA backend makes generation
   bit-stable across calls (0/8 flips). The FLASH backend (installed,
   sm_90) is untested — determinism + speed would be the ideal outcome;
   test before committing.
3. **This re-explains `probe_determinism` (2026-08-25):** the cross-
   process flips (354–736) were the same per-call phenomenon, not a
   process-seeding issue. The previous session's resume redesign (freeze
   partial groups from disk, never regenerate) is *more* justified, not
   less.
4. **Gate (a) as written is unreachable on the default backend.** Zero-
   init still means a mathematical no-op; what broke is the substrate's
   sampling reproducibility. Under the MATH backend the gate's premise
   holds (h5). Any future identity/determinism gate must pin the backend.
5. **The iterate-stage "matched-base arm" plan is dead as written**
   ("same seeds, same group construction, only the device differs, so
   determinism holds"). Two replacement routes: (i) run all rollouts
   under a verified-deterministic backend (pending 26k throughput), or
   (ii) matched comparisons via fork-from-prefix replay (bit-exact,
   proven at 26k) — which is also the passenger test's own mechanism
   (`COUNTERFACTUAL_FRAMEWORK.md`). Decide with tomorrow's numbers.
6. **Iter-1 sanity:** after one update the steered rows show
   `rel_max ≈ 0.005` (controller steering non-trivially), `gate_mean`
   still 0.5 / `gate_std` 0 (gate weights ~0 after one step — expected).

## Decisions (write-back manifest)

- `PHASES.md` — Phase 2 measurement paragraph: checkpointed full-trace
  replay replaces segmented/windowed replay (windowed KILLED — reasons in
  `2026-08-25-step1-replay-engine-windowed-killed.md`); add the
  determinism ruling: rollout generation requiring arm-to-arm
  reproducibility must run under a deterministic SDPA backend (MATH
  verified 2026-08-26; default backend non-deterministic). Phase 3
  attempt-prep: "segmented replay" → full-trace replay + determinism
  ruling.
- `STATUS.md` — new resolved rows: **R10** (windowed replay viability:
  NO, killed twice) and **R11** (SDPA generation non-determinism: root
  cause of the identity-gate failure and of probe_determinism; MATH
  backend bit-stable). Update Q10's current bet (replay engine proven at
  production length; determinism regime established).
- `OPERATIONS.md` — Training runs: replay ruling (full-trace checkpointed;
  no windowed fallback; OOM = exit 8) replacing "replay windows ≤8k";
  new generation-determinism ruling (default SDPA non-deterministic
  call-to-call on the H200; MATH verified, FLASH untested); "NEVER
  regenerate" section: annotate that "redone bit-identically" does NOT
  hold on the H200 training substrate — resume evidence = rows + seqs;
  per-sample-seed rationale: annotate likewise (still valid for
  llama-server collection).
- `ENVIRONMENT.md` — remote-stack traps: cache-continuation boundary
  corruption + FLA chunk-backward crash + fp32 vocab-chunk OOM (from
  `...-replay-engine-windowed-killed.md`) + SDPA generation
  non-determinism; fix stale instance id (#46911241 → 48673764).

## State at note time

Dry run done and pulled home; probe sessions closed. Box stops after
write-backs (user ruling: down for the night). Tomorrow's queue: (1) MATH
throughput at 26k cap; (2) FLASH-backend determinism; (3) backend fix in
`generate_batch` + rerun dry run on a fresh seed base (81_000 — the 80_000
space is consumed by s1_dry); (4) kill+resume test; (5) Step-1 checkpoint
report → Step 2. Handoff: `2026-08-26-handoff-step1-sdpa-fix.md`.
