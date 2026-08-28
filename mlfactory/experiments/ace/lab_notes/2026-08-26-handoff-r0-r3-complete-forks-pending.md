# HANDOFF — 2026-08-26 — R0–R3 complete, holding before forks (night close-out)
# Concept: ANNOTATION_SIDESTEP.md — read it first.
# Session record (load-bearing): lab_notes/2026-08-26-scale-annotation-to-r3.md
# Verdicts were written back: STATUS.md R12 + Q5; LAYER_HYPOTHESES.md;
# OBSERVABLES.md (position-level section); ANNOTATION_SIDESTEP.md §8.
# Audience: next session. The detection side of the sidestep is BUILT and
# SCORED. The causal side (forks) is the principal's call — nothing spent.

## objective_and_constraints

```yaml
objective: |
  Annotation sidestep (ANNOTATION_SIDESTEP.md) rungs R0-R3 are complete:
  scale annotation -> capture -> probe -> directions. Results: onset
  positions of annotated episodes are prefix-causally separable from
  depth-matched controls (LOO AUROC 0.983-0.991; pre-onset ~0.97), and
  succ-vs-fail onsets of the same class separate at 0.78-0.86. Kill
  conditions K1/K2/K3 do not fire; K4 adjudicated noisy-but-usable.
goal_deltas_this_session:
  - built the full annotate/ pipeline (build_plan, run_batch,
    r0_agreement, capture_activations, probe_positions,
    compute_directions) — all module-qualified, resume-safe
  - TeaLeaves generalized upstream (/home/admin/TeaLeaves): nested
    text_config unwrap, hybrid linear-attn discovery, --dtype,
    --residuals-only, --max-case-chars; 198 tests pass; changes
    UNCOMMITTED in that repo (principal decides on commit/push)
bindings_from_principal:
  - NO -ballast Lunaroute models (server trouble; scope was "today")
  - Lunaroute default temperature; GLM room to think (max_tokens 65536)
  - subagents on qwen/qwen3.7-plus, parallelized+backgrounded, nearly
    all work stays in the main thread
  - remove dead code on sight; "inefficient but it works" -> fix it
  - hold before R4 forks — nothing spent without principal
  - Vast 48783410: principal said they'd destroy it when done with
    today's work — it is STILL stopped/reserved; confirm with them
```

## world_state_delta

```yaml
artifacts (all sidecar'd):
  data/annotation_plan_xsub.jsonl        48 pairs / 96 traces
  annotate/out/xsub_pass1/ + xsub_pass2/  raw annotation outputs
  data/annotations_xsub_pass1.jsonl      469 flags, 251 resolved
  data/annotations_xsub_pass2.jsonl      double subset (43 flags)
  data/annot_captures/*.npz              81 captures + manifest
  data/probe_results.json                pooled + LOO AUROCs per (class,kind,layer)
  data/steering_directions/directions_annot_clear.npz   96 directions
headline_numbers:
  cycle onset LOO 0.988-0.991 (n=128, L6/L16-19)
  loop  onset LOO 0.983-0.988 (n=78,  L2/L17-20)
  muse  onset LOO 0.979-0.982 (n=5,   L23-29, underpowered)
  pre_onset ~0.97-0.98; escape-vs-reheat 0.78 cycle / 0.86 loop
writebacks_paid:
  STATUS.md (R12 added, Q5 resolved, R4 annotated with auroc caveat)
  LAYER_HYPOTHESES.md (prior table updated + new evidence section)
  OBSERVABLES.md (position-level observables + loop-onset annotation)
  ANNOTATION_SIDESTEP.md §8; ace/AGENTS.md (directory + doc maps)
  host AGENTS.md (/tmp is tmpfs gotcha); session-handoff skill
```

## negative_knowledge

```yaml
- auroc() divided by len(pos) twice — deflated EVERY historical AUROC
  by a factor of n_pos (b1-map readings included). Fixed in
  probe_positions.py + analysis/analyze_map.py. Rule: sanity-check
  metrics on known-extreme inputs before trusting them.
- run_batch.consolidate wrote the trace NUMBER as sample_i (collapsed
  to {1,2}); tell = implausibly few traces. Fix verified against
  trace_label. Rule: when a consolidated count looks wrong, check the
  join keys before the science.
- Qwen3.5 chunked DynamicCache continuation needs use_cache=True on
  EVERY call; omitting it corrupts get_seq_length and crashes decode.
  One-shot vs chunked recurrent states differ ~5e-2 (FP re-association,
  not semantic). Rec states saved at onset+control positions only.
- /tmp is tmpfs — the TeaLeaves clone vanished on the power outage;
  durable clones live under /home/admin (host AGENTS.md gotcha).
- power outage mid-run: resume-safe design meant 3 completed pairs
  survived, one torn C.json detected by the parse-checking resume
  guard and re-run. Verify JSON parseability, not just file existence.
- bash cwd trap: the agent shell keeps cwd between commands; nohup
  log redirects need absolute paths (relative ones resolve against a
  stale cwd or die). Bit me twice this session.
- annotation span Jaccard 0.26 is segmentation variance, not label
  mush: region agreement 0.54, within-pass precision high (pilot
  audit 14/16 real). Do not re-run annotation chasing Jaccard.
```

## operational_state

```yaml
running: NOTHING. GPUs free (GPU0 desktop-resident ~1 GB, GPU1 0).
tombstones:
  - llama-qwen38 service still STOPPED (user direction, prior phase)
  - Vast 48783410 stopped/reserved ($0.09/hr) — user said they'd
    destroy it when today's work is done; CONFIRM before touching
  - annotate/out/pass1.log contains a harmless OOM traceback from a
    duplicate capture process that died double-loading the model
git: NOTHING committed anywhere — mlfactory tree (AGENTS.md, docs,
  ace/annotate/, data + sidecars, lab notes, living-doc updates) and
  /home/admin/TeaLeaves (2 modified files) are all uncommitted;
  principal decides.
```

## open_questions

```yaml
- OVERNIGHT BATCH (queued at principal's request before close-out):
  runbook at annotate/runbook_overnight_b2.md — collect 184 q8 rollouts
  on the 46-prompt LIVE b2 pool (new families: machine/assign/
  hypothesis — targeting the muse gap), then annotate via Lunaroute.
  On resume: check for data/annot_b2_q8.jsonl, annotations_b2_pass*.jsonl
  and the morning lab note 2026-08-27-overnight-b2-collect-annotate.md;
  if the batch ran, merge results into capture/probes next; if it
  aborted mid-way, the runbook's abort section says where to look.
  Supporting code was generalized for this: build_plan --corpus/--out,
  run_batch --tag/--plan/--corpus (out dirs renamed out/<tag>_<pass>/;
  xsub outputs moved to out/xsub_pass{1,2}/), r0_agreement --tag,
  call_with_retry now backs off 429s (60/120/240/480s x4).
- R4 fork placement: which nominated states/layers to fork, on what
  budget (Vast H200 vs local). Principal's call — this is the rung
  that spends money. Bet on the table: fork the top-LOO layers
  (cycle L18, loop L2/L17-20) on detector-nominated onsets.
- muse material is n=5 — the corpus is short on blunt muses; only 20
  flags total. Adversary/grid families produced almost none. If muse
  matters, author prompts that induce idle musing (the Goff species).
- onset AUROCs are broadly high across layers: distributed signal or
  residual position/texture confound. A permuted-position control
  (probe random positions vs controls) would bound it cheaply.
- rec channel at position level was captured but NOT yet probed (R2
  scored residuals only) — analyze rec_L{2,8,9,12,20} onset-vs-control
  when the question is needed.
```

## pointers

```yaml
concept: ace/ANNOTATION_SIDESTEP.md (kill conditions §6, ladder §7)
session_record: ace/lab_notes/2026-08-26-scale-annotation-to-r3.md
prior_handoffs: lab_notes/2026-08-26-handoff-annotation-pilot-converged-scaling-next.md
pilot_record: lab_notes/2026-08-26-annotation-pilot-prompt-hillclimb.md
rubric+prompt: ace/annotate/RUBRIC.md, ace/annotate/pilot/prompt_v5.md
harness_rules: mlfactory/AGENTS.md (subagents, hygiene, providers)
```

## checkpoint timeline (this session segment: R0-R3 climb)

1. compaction recovery: handoff + concept + pilot record re-read;
   live state verified (GPUs free, data present).
2. plan builder written; set.update bug caught by its own assert.
3. batch driver written; smoke-verified prompt construction (48
   prompts, worst-case ~38k tokens).
4. power outage: pass1 process killed; torn C.json found via
   parse-check resume guard; resume relaunched.
5. TeaLeaves re-cloned to /home/admin (tmpfs loss); generalizations
   implemented (nested config, hybrid layers, dtype, residuals-only);
   198 tests pass; verified against real Qwen3.5-9B.
6. Qwen3.5 cache mechanics established: use_cache=True mandatory for
   continuation; chunk-vs-oneshot divergence characterized as FP.
7. capture script written; smoke-tested end-to-end on pilot spans
   (boundary check first looked broken — check-script bug, not code);
   rec-state storage slimmed to onset+control positions.
8. pass1 completed (456->469 flags); sample_i consolidation bug found
   ("only 24/96 traces" tell), fixed, re-consolidated.
9. pass2 completed; R0 agreement adjudicated noisy-but-usable.
10. capture run: 81 traces (~7s each); one duplicate-process OOM
    (harmless).
11. R2 probe: AUROC ~0.007 everywhere — impossible by construction;
    found the double-division auroc bug (also in analyze_map.py);
    fixed both; added leave-one-trace-out scoring.
12. real R2: onset LOO 0.98-0.99, pre_onset ~0.97, escape-vs-reheat
    0.78/0.86 -> K1/K2/K3 do not fire.
13. R3: 96 directions saved (constant-lambda baselines).
14. write-backs: STATUS R12/Q5, LAYER_HYPOTHESES, OBSERVABLES,
    SIDESTEP §8, ace/AGENTS.md maps, data sidecars, this handoff.
    HOLD before R4.
