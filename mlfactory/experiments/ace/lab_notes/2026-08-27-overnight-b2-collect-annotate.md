# LAB NOTE — 2026-08-27 — overnight b2 collect + annotate (unsupervised)

> Ran the b2 overnight runbook (`annotate/runbook_overnight_b2.md`)
> unsupervised while the principal slept. Append-only record of what
> ran and what landed — facts, not plans. The main session merges
> these artifacts into capture/probes.

## objective_and_constraints

```yaml
objective: |
  Generate q8 rollouts on the 46-prompt LIVE b2 pool (4 samples each =
  184), annotate through Lunaroute (glm-5.2-vision, framing C), leave
  a tidy morning record. The batch's purpose: the MUSE gap — extend
  span annotation into the six b2 families, three of which (machine,
  assign, hypothesis) are new to the annotation corpus.
bindings_from_runbook:
  - annotator glm-5.2-vision only (no -ballast/-flex/image); confirmed
    active in preflight
  - Lunaroute default temperature (never set), max_tokens 65536, <=6
    requests in flight
  - local server --parallel 1 (higher truncates by partitioning ctx),
    fp KV (--cache-type-k/v f16), ctx 32768, cap 26000
  - did NOT touch: Vast 48783410, llama-* systemd units, git, the
    capture/probe/direction scripts; deleted nothing
  - kill by explicit PID; verify effects after stop/start
preflight_lunaroute_models:
  - flux2-klein, glm-5.2-vision, glm-5.2-vision-background,
    glm-5.2-vision-flex, qwen-image-2512
```

## world_state_delta

```yaml
artifacts (all sidecar'd with sha256/size/created_at):
  data/annot_b2_q8.jsonl                 184 rollouts (46 prompts x 4)
  data/annotation_plan_b2.jsonl          92 pairs covering 184 traces
  data/annotations_b2_pass1.jsonl        879 flag rows, 505 resolved
  data/annotations_b2_pass2.jsonl        43 flag rows (6 double pairs)
  annotate/out/b2_pass1/ b2_pass2/        raw per-pair annotation outputs
  annotate/out/b2_{server,collect,annotate}.{log,pid}   run logs
collection:
  backend: llama.cpp Q8_0-MTP, build 10336 (system_fingerprint b10336-*)
  gpu1:3091, parallel-1, fp16 KV, ctx 32768, spec draft-mtp n-max 3
  seeds: seed_base 84000 (seed_sample = 84000 + 17*pid + si)
  throughput: ~156-159 tok/s measured (MTP); ~2.6 min/row avg
  wall: 00:36 -> 06:54 UTC (~6.3 h), collector exited clean
  outcome_mix: 121 correct, 47 cap (truncated), 16 wrong; acc 0.658
  coverage: adversary 32, machine 32, assign 24, certify 44, grid 28,
            hypothesis 24 (all 46 prompts x4)
plan:
  92 pairs, contrast-priority deterministic; 42/92 mixed-outcome
  6 double pairs, one per domain:
    adversary p49, machine p98, certify p103, hypothesis p115,
    grid p123, assign p138
annotation:
  model glm-5.2-vision, framing C, v5 prompt; workers 6 (pass1) / 5 (pass2)
  pass1: 879 flags — CYCLE 344, LOOP 504, MUSE 31
         conf clear 660 / probable 219
         resolved 505/879 = 57.5% (span_res OK 457, OK-NORM 48, UNRESOLVED 374)
         167/184 traces flagged
         2 A-fallbacks: p117, p142 (framing-C hit finish=length, no usable
         flags -> each trace annotated alone with framing A)
  pass2: 43 flags — CYCLE 21, LOOP 21, MUSE 1; 21 resolved (48.8%)
  retries: zero 429s, zero FAILED pairs across both passes
agreement (r0_agreement --tag b2, double subset n=6 pairs / 12 traces):
  pooled span Jaccard 0.417  (xsub was 0.26)
  matched=10, class-confused=1, pass1-only=7, pass2-only=7
  boundary drift: median=449 chars, p90=22478 chars
  per-trace jaccard: p138 s0=1.00, p123 s0=0.57, p115 s0/s3=0.25,
                    p123 s2=0.20, p49 s0=0.00
  K4 guide: 0.417 is in the borderline zone (>=0.5 usable, <0.3 fix
            rubric) — above xsub, still noisy-but-usable
muse_per_family (the point of this batch):
  assign      12  across 4/6 prompts (118,119,132,137)
  certify      8  across 3/11 prompts (141,142,147)
  adversary    4  across 3/8 prompts (50,51,53)
  hypothesis   4  across 4/6 prompts (116,117,158,160)
  machine      2  across 1/8 prompts (124)
  grid         1  across 1/7 prompts (152)
  total MUSE = 31  (xsub pass1 had 20)
machine_left_clean:
  server PID 285740 killed explicitly; port 3091 free; GPU1 2 MiB / 0%
  no commits, no pushes, nothing deleted, no stray processes
```

## Decisions

```yaml
# Write-back manifest. This note is a run record; the main session
# owns the merge into capture/probes and any concept-doc write-back.
docs_this_note_changes: []   # no concept doc changed by this run itself
for_main_session_to_review_before_writeback:
  - MUSE coverage expanded into all six b2 families (total 31, was 20
    in xsub); three new families (machine 2, assign 12, hypothesis 4)
    now carry MUSE. Whether this is enough to lift the muse onset AUROC
    (xsub n=5 was underpowered, LOO 0.979-0.982) is the main session's
    call after re-running probe_positions on the merged corpus.
  - R0 agreement shifted 0.26 -> 0.417 (pooled span Jaccard). Still
    below the 0.5 "usable" line; K4 stays noisy-but-usable. The shift
    is on a different corpus (b2 vs xsub) so it is not direct evidence
    the rubric improved — flag for ANNOTATION_SIDESTEP K4 readers.
  - Two A-fallbacks (p117, p142) are included in pass1; framing-A
    rows carry the same pair_id. If the main session re-consolidates,
    these are already in annotations_b2_pass1.jsonl.
nothing_skipped_or_aborted:
  - All phases completed; no abort conditions fired. Nothing skipped.
```

## next_for_main_session

- Merge `data/annot_b2_q8.jsonl` into the capture/probe pipeline
  (capture_activations -> probe_positions -> compute_directions) to
  re-score onset AUROCs on the larger, multi-family corpus —
  especially MUSE (was n=5 underpowered).
- The 4 sidecars are in `data/`; raw per-pair outputs in
  `annotate/out/b2_pass{1,2}/`; run logs in
  `annotate/out/b2_{server,collect,annotate}.log`.
