# HANDOFF — 2026-08-27 — R4 fork run: design locked, boxes up, harness mid-fix

> **SUPERSEDED as live state by `lab_notes/2026-08-27-r4-fork-launch.md`
> (run launched 2026-08-27 21:40–21:42 UTC; attendance log at
> `annotate/out/r4_attendance.log`). Kept for the build-phase history
> and negative knowledge.**
> Cold-reader pointers: `ANNOTATION_SIDESTEP.md` (R4 = rung 4, the
> fork rung), `TERMINAL_FORK_COMPUTE.md` §scenario F (R4 cost model),
> `docs/VAST_REMOTE.md` (box ops + the new selection directive),
> `lab_notes/2026-08-27-lookback-k5-rec-results.md` (why fork point =
> onset−6). Supersedes `2026-08-27-handoff-detection-complete-r4-pending.md`.
> Mission: launch the R4 fork run on two rented A6000s TODAY (principal
> leaves for Cancun tomorrow morning), with an attendance runbook so a
> Lunaroute GLM agent can tend it every 3h while they're away. "When I
> get home from Mexico it should be nearly done."

## objective_and_constraints

```yaml
objective: |
  R4 = concentrated counterfactual forks on detector-nominated onset
  states: does steering at/around the annotated divergence onset causally
  move terminal verified outcome? The only rung that spends rental money.
  Principal approved it this session by asking for the Cancun attendance
  setup. Everything below is LOCKED DESIGN — the attendance runbook must
  forbid changing any of it (principal's words: small adjustments to keep
  the run on track, no scope creep, no sweeping changes affecting the
  end-state goal).
locked_design:
  states: 27 (plan: /home/admin/mlfactory/artifacts/fork_plan_r4.jsonl,
    built via `fork_r4.py --plan`, datasaved). 10 CYCLE / 10 LOOP /
    7 MUSE (muse had only 7 eligible pids). Distinct pids per class,
    earliest conf=clear onset per trace, fork point = onset_abs - 6,
    PLAN_SEED=484000.
  arms: noop | toward_healthy (+lam*d) | toward_diverge (-lam*d).
    Saved directions point FROM divergence TOWARD healthy
    (directions_annot_clear_merged.npz sign_convention) — +lam is the
    therapeutic arm.
  layers: CYCLE L18, LOOP L2, MUSE L17 (probe-best focal layers)
  lam: 0.05 x median onset residual norm at focal layer =
    cycle 2.014 / loop 0.349 / muse 1.768 (half the 0.1||h|| bound)
  sampling: m=24 paired seeds per arm (identical seed sequence across
    arms), sub-batch 4, RUN_SEED_BASE=484100, temp 0.8, top_p 0.95
  model: bf16 HF /home/admin/models/hf/Qwen3.5-9B, sdpa with MATH
    backend FORCED (sdpa_kernel(SDPBackend.MATH)) — default backend is
    call-to-call nondeterministic (R11); backstop cap 26000 abs tokens
  scoring: frontier.collect_rollouts.objective_check vs the source
    trace's reference_answer; rows appended resume-safe to
    ace/data/fork_r4_results.jsonl keyed (state_id, arm, seed_i)
  totals: 27 states x 3 arms x 24 = 1944 continuations ~ 22M tokens
bindings:
  - Reward policy unchanged: terminal verified outcome only.
  - Do NOT touch: stopped Vast 48783410, any llama-* systemd service,
    git (no commits), the detection artifacts (immutable).
  - The two rented boxes ARE the spend: combined ~$0.97/hr; budget cap
    for the whole run incl. attendance interventions: $150.
```

## operational_state

```yaml
rented_boxes (verify with `vastai show instances`; FIRST PAIR WAS
  REPLACED — old ids 48916215/48916245 no longer exist, A vanished
  mid-provision and B's authorized_keys perms broke ssh):
  A: instance 48921611, 1x RTX A6000 48GB, Kansas US, $0.456/hr,
     label r4fork-a, alias `ssh r4fork-a` (root@209.137.198.14:28119)
  B: instance 48921612, 1x RTX A6000 48GB, Delaware US (dl 8.8Gbps),
     $0.511/hr, label r4fork-b, alias `ssh r4fork-b`
     (root@38.29.145.10:40340)
  image vastai/llama-cpp:b10182-cuda-12.9, disk 200, both RUNNING +
  SSH VERIFIED 2026-08-27 ~17:40 UTC. Aliases live in ~/.ssh/config
  (key ~/.ssh/id_vast). Instances were created WITH the SSH-permission
  repair --onstart-cmd (see negative_knowledge + runbook Appendix A).
local: smoke RUNNING on GPU1 (PID may change — find via
  `ps aux | grep fork_r4`): 4 states x 3 arms x m=2 into
  ace/data/fork_r4_smoke.jsonl, log annotate/out/fork_r4_smoke.log.
  CAVEAT: smoke states were picked for small PREFIX but have long
  horizons (r4_cycle_09: 14.4k) — first rows land ~18 min in, full
  smoke ~3h. GPU1 at ~20GB/99% is normal; do NOT restart it. Do not
  block launch on full smoke completion — the remote smoke ladder is
  the real gate; local smoke is corroboration (verify rows + hook
  liveness when available).
code: annotate/fork_r4.py FIXED — gen_batch is now the GRPO-proven
  full-forward batched generate (see negative_knowledge for the exact
  call); adaptive `batch_for(fork_abs, cap)` sizing (batch 1 >14k
  prefix, 2 >8k, else 4) + OOM-halving fallback. Syntax-checked;
  first rows pending in the smoke.
runbook: annotate/runbook_r4_cancun.md WRITTEN (binding rules,
  3h sampling protocol, permitted adjustments, abort/completion).
  FILL-AT-LAUNCH fields: shard --only lists (SHARDS below),
  target_rows (A 936 / B 1008), launch_time, baseline rows/hr.
shards (locked, snake-balanced by horizon):
  A (13 states, 936 rows): r4_muse_00,r4_muse_04,r4_cycle_08,
    r4_cycle_09,r4_cycle_02,r4_muse_06,r4_cycle_03,r4_loop_09,
    r4_loop_04,r4_cycle_05,r4_cycle_07,r4_muse_01,r4_loop_02
  B (14 states, 1008 rows): r4_loop_07,r4_loop_00,r4_cycle_01,
    r4_loop_08,r4_loop_05,r4_muse_05,r4_loop_03,r4_loop_06,
    r4_cycle_00,r4_muse_03,r4_cycle_04,r4_cycle_06,r4_loop_01,r4_muse_02
not_yet_built: launch lab note (runbook references it for the rsync
  command — write it AT launch), sidecars for results, requirements
  pin file for box rebuilds (requirements_r4_box.txt — runbook
  Appendix A references it; create from ace/.venv/bin/pip freeze at
  setup).
```

## negative_knowledge

```yaml
- THE ROADBLOCK (RESOLVED): fast path (chunked prefill into
  DynamicCache, then model.generate with past_key_values + short
  input) fails on this model in 3 different ways: empty input_ids ->
  IndexError in right-padding check; single-token input_ids ->
  _prepare_position_ids IndexError; inputs_embeds ->
  prepare_inputs_for_generation slicing IndexError (utils.py ~527).
  Cause: model loads as Qwen3_5ForConditionalGeneration (the VL
  wrapper class) with non-standard prepare_inputs (mrope-style
  get_rope_index). FIX APPLIED: gen_batch is now the GRPO-proven
  full-forward pattern (steering_controller.generate_batch):
  torch.manual_seed(seed); model.generate(input_ids=[prefix]*batch,
  attention_mask=ones, max_new_tokens, do_sample=True, temperature=0.8,
  top_p=0.95, eos_token_id=STOP_TOKEN_IDS [248044,248046],
  pad_token_id=248044); new tokens = out[row][fork_abs:] trimmed at
  first stop. The hook keys on seq length (seq_len>1 -> steer index
  fork-1-start during the monolithic prefill forward; seq_len==1 ->
  steer all decode steps) — st dict is set once {start:0,end:fork}.
- MATH backend memory: materializes q x kv attention. Config: 32 layers,
  full_attention every 4th (idx 3,7,...,31), 16 q heads, 4 kv heads,
  head_dim 256, hidden 4096. Long-prefix batched full-forward OOMs:
  batch_for() thresholds (>14k prefix -> 1, >8k -> 2, else 4, sized
  for 48GB cards) + OOM-catch halving. Full-forward recomputes the
  prefix per sub-batch — ~1.5-2x the token cost of cache reuse,
  accepted.
- Seed pairing caveat: batch-draw coupling is pair-matched only while
  each arm's pending-seed list is identical; after a mid-state resume
  the regenerated sub-batches may differ in composition across arms.
  Affects variance reduction only, never validity — do not "fix" it.
- objective_check returns keys 'correct'/'match_mode' (NOT 'ok'/'mode')
  and only runs the STRICT gen-family verifier when passed
  rec={'domain':..., 'knobs':...} — otherwise it silently degrades to
  the advisory soft-substring fallback (C4). Plan rows now carry
  surface_hash/domain/knobs; plan was rebuilt 2026-08-27 (same
  deterministic picks + new fields).
- Local 3090 (24GB) needs PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  even for batch-1 MATH prefill of ~11k prefixes (right at the edge);
  smoke only on small-fork states: r4_cycle_09 (fork 1357), r4_loop_09
  (1731), r4_muse_03 (2090), r4_loop_02 (3902). Boxes (48GB) have slack.
- transformers 5.14.1 / torch 2.11.0+cu128 / python3.12 (ace/.venv).
  fla imports fine; the model needs no external flash-attn (GRPO notes:
  zero gain anyway).
- VAST SSH GOTCHA (resolved, COST ~2h): fresh instances on the
  vastai/llama-cpp image came up with /root/.ssh/authorized_keys at
  wrong ownership/modes; sshd's server-side StrictModes refused EVERY
  key (log signature in `vastai logs <id>`: 'Authentication refused:
  bad ownership or modes for file /root/.ssh/authorized_keys'). This
  is a BOX-SIDE defect — no client flag fixes it (StrictHostKeyChecking
  is unrelated). Fix: create instances with the --onstart-cmd that
  appends the id_vast pubkey and runs a 30s background chmod/chown
  keeper loop (exact string in runbook Appendix A). Diagnose future
  ssh refusals with `vastai logs <id>` FIRST — it shows sshd's own
  auth log.
- Vast market moves fast: offers die between search and create (two did
  today). `create` may return success:false yet provision — verify with
  `vastai show instance <new_contract>` (happened with box A).
- gpu_ram query gotcha: GB in queries, MB in raw output, TOTAL across
  GPUs in both (per-GPU = gpu_ram/num_gpus). All in VAST_REMOTE.md now.
- datasave without manifest writes to mlfactory/artifacts/ (repo level),
  not ace/data — plan path is /home/admin/mlfactory/artifacts/fork_plan_r4.jsonl;
  fork_r4.PLAN_PATH resolves it. Include artifacts/ in the box rsync.
```

## next_steps

```yaml
1_check_smoke: rows in ace/data/fork_r4_smoke.jsonl (first ~18 min in);
  gates: plausible completions + verdicts; steered-arm text DIFFERS from
  noop (hook live); resume works (relaunch skips done). Remote smoke is
  the launch gate; this is corroboration.
2_box_setup (B first — already running; per VAST_REMOTE.md): inventory;
  `supervisorctl stop llama`; venv with EXACT pins from local
  `ace/.venv/bin/pip freeze` (also write requirements_r4_box.txt into
  the project for Appendix-A rebuilds); rsync project (exclude
  .venv/__pycache__; include artifacts/ and the ace/data inputs the
  harness reads: xsub_q8.jsonl, xsub_bf16_gpu{0,1}.jsonl,
  annot_b2_q8.jsonl, xsub_candidates.jsonl, acegen_live_b2.jsonl,
  annotations_xsub_pass1.jsonl, annotations_b2_pass1.jsonl,
  steering_directions/); model weights: try `hf download` via hf_xet
  if the Qwen3.5-9B repo is public, else rsync from local
  /home/admin/models/hf/Qwen3.5-9B (18GB — home uplink is the risk,
  check first; box B dl ~2000Mbps).
3_remote_smoke_ladder: on box B — load, then ONE fast state:
  --only r4_loop_02 --m 1 --sub-batch 1 (horizon 4233 -> minutes);
  expect 3 rows; measure tok/s + peak VRAM.
4_launch: supervisor configs (VAST_REMOTE.md template; autostart=false,
  autorestart=false, stopasgroup), PYTORCH_CUDA_ALLOC_CONF=
  expandable_segments:True in the launch script environment, full
  command per box:
    ace/.venv/bin/python -m mlfactory.experiments.ace.annotate.fork_r4
      --run --out <project>/mlfactory/experiments/ace/data/fork_r4_results.jsonl
      --only <SHARD from operational_state>
  then fill the runbook's FILL-AT-LAUNCH fields (incl. baseline rows/hr
  measured over the first ~2h — record the estimate in the runbook
  Topology as soon as it exists).
5_record: launch lab note lab_notes/2026-08-27-r4-fork-launch.md —
  the runbook's Appendix A points at it for the rsync command, so it
  must CONTAIN the exact rsync/venv/model commands used. Facts:
  design, boxes, kickoff counts, measured tok/s, shard targets.
6_finalize_before_principal_leaves: verify both runs flowing (log lines
  + rows on both boxes), attendance log started, report to principal.
```

## open_questions

```yaml
- Does the full-forward rewrite reproduce hook liveness locally? (verify
  in smoke before renting further time — boxes are already billing.)
- hf repo for Qwen3.5-9B weights: config _name_or_path is empty; if not
  downloadable on-box, rsync from local (slow home uplink is the risk).
- GLM attendance mechanics (how the principal runs the GLM agent against
  the runbook) are the principal's setup — the runbook must be
  self-contained either way.
still_pending_for_principal (unchanged): muse-K5 (STATUS Q12), Vast
  48783410 destroy/keep, TeaLeaves commits, mlfactory commits.
```

## checkpoint_timeline (this session, post-handoff)

1. Principal asked for R4 GPU-hour cost → agent grounded it in
   TERMINAL_FORK_COMPUTE + measured onset horizons from the 231-capture
   corpus → scenario F added to the doc (design center ~86 H200-GPU-h;
   1 GPU-h ≈ 2.6e5 tokens at 73 tok/s deep-context).
2. Principal: H200 ran at ~30% utilization and cost >2x a 2xRTX6000 box;
   right-size the server → agent measured HF decode is LATENCY-bound
   (37 tok/s/stream on 3090 ≈ H200), fork regime needs only ~30-35GB
   (140GB was training-replay-only) → swept Vast market via CLI
   (172 offers ≥40GB; 65 pass per-GPU+arch filters) → A6000 $0.43 vs
   2xH200 $8.45.
3. Principal: codify this — never pick from the Web UI → agent rewrote
   VAST_REMOTE.md "Choosing a Vast offer" as agent-driven search
   (4 steps + query-field gotchas + dated measured reference).
4. Principal (the green light): leaving for Cancun tomorrow, wants a
   GLM-attended runbook like runbook_overnight_b2.md — small corrections
   allowed, scope creep banned, "nearly done" on return → R4 approved.
5. Agent locked the design (27 states, 3 arms incl. both signs since
   saved directions are divergence→healthy, m=24, lam from measured
   norms), built the plan (datasaved), wrote fork_r4.py.
6. Debug cycle: directions key typo (dir_*_L18) → full-mode OOM on local
   (MATH materializes q×kv) → cache fast path built → empty-input_ids
   rejected → position-id IndexError → inputs_embeds IndexError → root
   cause: VL wrapper class prepare_inputs. Decision: GRPO-proven
   full-forward rewrite (in progress at compaction).
7. Agent provisioned box A (48916215, Kansas A6000) and box B (48916245,
   Virginia A6000); both loading at last check; ~$0.97/hr combined.
8. Post-handoff: gen_batch rewritten to the GRPO full-forward pattern
   with batch_for() sizing; runbook_r4_cancun.md written (binding rules /
   3h sampling / permitted adjustments / abort / completion / Appendix A
   rebuild); shards locked snake-balanced (A 13 states/936 rows, B 14
   states/1008 rows); box B running, A loading.
9. Smoke caught a verifier bug: objective_check keys are
   correct/match_mode, and the strict gen check needs rec={domain,knobs}
   (else silent soft fallback) — fixed in harness, plan rebuilt with
   domain/knobs/surface_hash (same picks), smoke relaunched on GPU1
   ~2026-08-27 early afternoon. Principal compacts the session here.
10. Post-compaction: smoke verified full-forward generation (noop rows
   correct=True on strict gen verifier) but crashed the hooked arm on a
   DTYPE bug (fp32 delta + bf16 hidden -> fp32 out -> next linear
   raises) — fixed (delta.to(h.dtype) in hook + directions cast to
   model.dtype), smoke resumed to corroborate steered arms. Then the
   SSH wall: old boxes unreachable (authorized_keys modes), box A
   vanished from the account mid-provision; destroyed B, recreated both
   with the onstart SSH repair (48921611 Kansas / 48921612 Delaware),
   SSH verified on both, ~/.ssh/config aliases r4fork-a/r4fork-b added,
   runbook updated (topology, alias-based sampling, Appendix A carries
   the mandatory --onstart-cmd).
```
