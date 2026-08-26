# HANDOFF — 2026-08-25 — GRPO Step 1 in flight (context-compaction carry-forward)
# Supersedes nothing: lab_notes/2026-08-25-handoff-grpo-gated-execution.md is
# STILL the parent plan (steps, bindings, stop criteria). Read it first.
# This file carries only what happened since: Step 0 done, Step 1 nearly done.
# Audience: the same agent after context compaction. Read this + the parent.

objective_and_constraints:
  goal: unchanged from parent handoff — first serious GRPO attempt on the b2
    46-prompt pool on the Vast H200, falsifying steps, pre-registered stops.
    Currently INSIDE step_1_loop_health; dry run in flight.
  goal_deltas_this_session:
    - "Replay mechanism decided: gradient-checkpointed FULL-trace replay.
       Windowed/cached-prefix replay is KILLED (see negative_knowledge).
       --replay-mode auto == full; window flag is investigation-only;
       full-mode OOM is guard trip exit 8, never silent fallback."
    - "Resume semantics changed: cross-process sampling is NOT bit-stable
       on this stack, so partial groups are FROZEN from disk (rewards from
       rows, replay from persisted seqs), never regenerated. Iteration-level
       resume reads train.jsonl and skips done iters (rng shuffle still
       advanced so later batches unchanged)."
  bindings: "ALL parent-handoff bindings still bind (terminal-only reward,
    thinking ON, bf16, rows immutable, local<->remote sync on ANY file
    change, NO git commits without explicit request, rsync artifacts home
    at every step boundary — remote workspace NON-persistent, b3 untouched,
    fresh seed spaces: train 80k, eval hold 90k, eval train 95k,
    revalidate 100k; smoke consumed 70k)."
  user_directives_this_session:
    - "Prefer --quiet outputs; preserve context window."
    - "After each iteration state out loud: success / what didn't work /
       next iteration intent (checkpoint discipline)."
    - "Never patch over janky code or accept 'good enough' outcomes."
    - "Harness quirk: duplicate 'user' messages get resent — disregard."
    - "Pass explicit timeouts on every ssh/remote command."

operational_state:
  remote_box: "Vast instance 48673764 RUNNING. ssh root@198.145.108.59
    port 30854, key ~/.ssh/id_vast with IdentitiesOnly=yes. Local control:
    ~/.local/bin/vastai start|stop instance 48673764 (key ~/.vast_api_key).
    vllm/model-ui/ray services STOPPED — keep stopped. Repo
    /workspace/mlfactory, model Qwen/Qwen3.5-9B via ACE_MODEL_PATH +
    HF_HOME=/workspace/models, venv /venv/main."
  live_job: "tmux s1dry on GPU0: DRY RUN (production settings), launched
    22:12 UTC with code state = _TokenLogprobs fix INCLUDED but
    partial-resume/iter-skip changes NOT included (synced ~23:20, process
    holds old import — fine as long as it doesn't crash). Cmd:
    /venv/main/bin/python -m mlfactory.experiments.ace.train.grpo
      --pool .../ace/data/acegen_live_b2.jsonl --only-pids 132,140,148
      --iters 2 --prompts-per-iter 3 --group-size 4 --max-new 26000
      --skip-eval --out /workspace/s1_dry   (log /workspace/s1_dry.log,
    EXIT-CODE appended at end; env: HF_HOME, PYTORCH_CUDA_ALLOC_CONF=
    expandable_segments:True, PYTHONFAULTHANDLER=1, PYTHONUNBUFFERED=1,
    ACE_MODEL_PATH=Qwen/Qwen3.5-9B, CUDA_VISIBLE_DEVICES=0).
    Progress at 23:21 UTC: 16/48 rows (iter 0, group 5 of 6 generating).
    ETA: iter0 done ~23:50, run done ~01:15 UTC. ~18-31 min per prompt
    pair (G=4 at 26k cap, ~73 tok/s aggregate batch-4 = ~18 min/group)."
  gpu1: "FREE (probes finished). tmux servers can die silently — always
    tmux ls + nvidia-smi before believing a job alive."
  local: "All code synced to remote as of 23:20 UTC. Local repo
    /home/admin/mlfactory uncommitted BY DESIGN. Use
    /home/admin/mlfactory/mlfactory/experiments/ace/.venv/bin/python
    (system python has no torch). 2x llama-server collection on local GPUs
    — leave alone."

next_actions_in_order:
  1: "Watch s1_dry. When rows >= 25 AND iter 0 line is in
      /workspace/s1_dry/train.jsonl (check via strings log | grep '^iter '):
      this is the crash-resume test window. SIGKILL the python mid iter-1
      generation (pkill -f on exact module path via a script file — pkill
      over ssh matches its own cmdline). Verify a partial iter-1 group
      exists in rollout_rows.jsonl (25-27 rows)."
  2: "Restart the same command (code on disk NOW has partial-resume +
      iter-skip). Verify: banner rows_resume=24+; 'already in train.jsonl
      — skipped' for iter 0; '[resume] ... partial group frozen from disk'
      for the iter-1 partial group; NO duplicate row keys; run completes
      EXIT-CODE:0."
  3: "Verify remaining Step-1 gates on /workspace/s1_dry: (a) iter-0
      steered==base identity — for every (id, sample_i), steered row
      reward/n_new == base row (zero-init controller = exact no-op; any
      mismatch = wiring bug); (b) 48 rows total, all keys present;
      (c) summary.json base_fingerprints_unchanged=true; (d) iter lines
      show cap=X/Y per-group cap-hit counts and peak_mem (production
      replay peak must stay < ~130 GB); (e) gate stats non-degenerate."
  4: "rsync artifacts home (remote workspace is non-persistent):
      rsync -az root@...:/workspace/s1_dry /tmp/s1_dry, plus s1_long,
      s1_eq, det/ logs. (Smoke artifacts live at /tmp/smoke1.)"
  5: "STATE THE STEP-1 CHECKPOINT REPORT out loud (all gates, numbers).
      If ALL pass -> proceed to Step 2 (within approved envelope). If any
      gate fails -> HOLD, diagnose, report to user."
  6: "Step 2 = revalidation: grpo.py --revalidate --revalidate-n 8 on the
      FULL 46-prompt pool, thinking on, unsteered. Parallelize: two
      processes, GPU0 + GPU1, split pids via --only-pids (rows/seqs keyed
      by pid, disjoint out dirs, merge revalidate_summary.json after).
      ~46 groups x ~18 min / 2 GPUs ≈ 7 h. Stop criterion: re-banded pool
      < ~15 LIVE -> pool-texture verdict, stop and rethink."

world_state_delta (new artifacts + purpose):
  - "train/pool_adapter.py — pool->items (solver_prompt), CHECK verify
     dispatch, deterministic stratified split (30/16), split manifest."
  - "train/grpo.py — REWRITTEN: pool mode, thinking ON, 26k, per-sample
     rows + persisted seqs (seqs/*.pt), guards (NaN->exit9, zero-var->
     exit6 with --stop-on-zero-var, full-OOM->exit8), replay engine
     (full = checkpointed layers via checkpointed_layers() ctx mgr;
     _TokenLogprobs custom autograd fn inside completion_logprobs),
     --revalidate mode, --check-equivalence, --skip-eval, iteration-level
     resume, partial-group-frozen resume."
  - "train/probe_long_replay.py (--from-trace reuses OUT/trace.pt),
     train/diag_window_drift.py, train/test_nan_guard.py,
     train/probe_determinism.py — evidence scripts, kept."
  - "tests/test_train.py GPU test updated to new train_iteration sig."
  - "Lab notes written: 2026-08-25-step1-replay-engine-windowed-killed.md
     and 2026-08-25-step1-determinism-resume-semantics.md (both carry
     Decisions write-back manifests — write-backs to PHASES/STATUS/
     OPERATIONS/ENVIRONMENT still OWED, do them at the Step-1 boundary)."

negative_knowledge:
  - "Windowed replay via cache continuation is DEAD, two independent
     reasons: (1) zero-init equivalence at 3k tokens: boundary corruption
     up to 11.8 NATS for ~50 tokens after every split point (likelihood
     ratios off >1e5), drift mean GROWS with split depth (0.14/0.28/0.46);
     the cache object itself is bit-exact (single pass use_cache=True diff
     0.0) — the split is the poison. (2) FLA gated-delta-rule chunk
     backward crashes on the continued cache state (inplace CopyBackwards
     version error) — gradients cannot even complete. Evidence:
     /workspace/s1_diag.log, s1_nan.log traceback, s1_eq/equivalence.json."
  - "Do NOT return to float32-vocab-chunk logprob extraction under grad:
     ~52 GB of retained autograd intermediates at 26k OOMs 140 GB.
     _TokenLogprobs (saves bf16 logits ref + targets, recomputes softmax
     in backward) is the fix; its backward sign is g*(onehot - softmax) —
     signed it wrong once, CPU test caught it."
  - "Cross-process sampling NOT bit-stable here: probe_determinism, same
     seed fresh processes, lengths identical [2317]x4 but first token flip
     at 354-736, ~70% tokens differ after. Resume must never regenerate
     over existing rows."
  - "Probe comparison-ordering trap: comparing replay paths AFTER an
     optimizer step measures the controller's effect, not drift. Always
     bracket equivalence checks with zero-init weights BEFORE any step."
  - "First long-probe attempt died silently (~5 min in, no traceback, no
     cgroup OOM, tmux server gone). Relaunch with faulthandler completed.
     Cause unresolved — watch for recurrence."
  - "Batch-1 generation is ~20 tok/s (73 tok/s is the BATCH-4 aggregate) —
     the 'astonishingly slow' probe was not a regression."
  - "HF gradient_checkpointing_enable only fires while model.training —
     base stays eval(), so grpo.checkpointed_layers wraps layer forwards
     directly; hook fires once per layer, recomputation never re-hooks."

verified_numbers (for reports):
  - "Full replay vs single-pass ref: max_abs_diff 0.0 at 3,000 AND 26,000
     tokens (zero-init). 26k backward: peak 125.2 GB reserved (allocator
     pre-polluted; clean-state number = dry-run iter peak_mem), 15.9 s,
     grad finite, fingerprint holds."
  - "NaN guard: trips exit 9 in full mode; positive control finite."
  - "Split: 46 -> train 30 / holdout 16; 46/46 pool self-verify pass."
  - "Probe trace: adversary-p49 seed 123_456 -> 21,427 tok EOS once,
     26,000 cap-hit on rerun (the determinism anomaly, since explained)."

open_questions:
  - "Production clean-state replay peak memory — bet: < 120 GB (dry-run
     iter line will say). Confidence moderate."
  - "Silent process death — unresolved; bet: one-off. Watch."
  - "bf16 pool texture (Step 2) — genuinely uncertain; stop if <15 LIVE."
  - "Eval wall clock at iterate stage: holdout 16 x 2 arms x G=4 x ~22k
     tokens ≈ 5-6 h per eval — may need subset evals; decide with Step-3
     timing data, not now."
  - "Q10/Q11 — unchanged from parent handoff."

tombstones:
  - "s1_eq/s1_diag/s1_nan/s1_long probe sessions are DONE — no live tmux
     for them. /workspace/s1_long/trace.pt (26k adversary-p49) is kept for
     --from-trace reruns."
  - "The first s1_dry attempt (21:38 launch, old completion_logprobs) was
     KILLED and its out dir deleted; current s1_dry = 22:12 launch."
  - "Original grpo.py shape (thinking-off, 640 tokens, arithmetic set,
     G=6) exists only as the --smoke/--no-pool fallback."

checkpoint_timeline:
  1: "agent read parent handoff + workin-hard skill; user confirmed
     autonomy, quiet outputs, checkpoint discipline."
  2: "agent verified state: pool 46 rows, local GPUs busy with llama
     servers (leave), git uncommitted by design."
  3: "agent wired Step 0: pool_adapter.py + grpo.py rewrite -> compile +
     import + CPU unit tests pass; split 30/16 deterministic; 46/46
     self-verify."
  4: "found+fixed: replay_logprobs import break (smoke/tests) and
     test_train signature drift."
  5: "user reminder: don't patch over jank -> agent applied throughout."
  6: "started box (vastai start 48673764), ssh up, GPUs clean, vllm/
     model-ui/ray stopped, pool md5 match, remote compile/import OK."
  7: "Step-1 gate 1: --check-equivalence 3k trace -> full bit-exact 0.0;
     window max 0.19 -> FAIL investigated, not accepted."
  8: "diag_window_drift -> cache harmless (D=0.0), split corrupts (11.8
     nats at boundary, growing with depth); window mode killed."
  9: "NaN guard test -> full-mode exit-9 trip verified; window-mode
     crashed FLA backward (second kill reason)."
  10: "first long-probe attempt died silently (no OOM) -> unresolved;
     relaunch with faulthandler ran."
  11: "user flagged 'astonishingly slow' -> explained batch-1 vs batch-4;
     no regression."
  12: "long probe v1: full bit-exact at 21.4k; post-step diff 0.226 was a
     comparison-ordering bug (controller had stepped) -> corrected."
  13: "long probe v2 at 26k: OOM in completion_logprobs fp32 chunk graph
     (~52 GB) -> wrote _TokenLogprobs custom autograd fn; gradient
     sign bug caught by CPU test and fixed; rerun (from-trace): full
     bit-exact 0.0 at 26k, backward peak 125.2 GB / 15.9 s, EXIT 0."
  14: "dry run launched 22:12 UTC (fixed code); probe_determinism ->
     DIVERGENT (flips 354-736) -> resume semantics redesigned: partial
     groups frozen from disk; iteration-level resume via train.jsonl;
     unit-tested."
  15: "two lab notes written with write-back manifests (write-backs owed).
     Synced code 23:20 UTC. At compaction: dry run 16/48 rows, group 5/6
     of iter 0 generating; next = mid-run kill + resume test."
