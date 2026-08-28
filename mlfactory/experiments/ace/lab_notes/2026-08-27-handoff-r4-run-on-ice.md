# HANDOFF — 2026-08-27 — R4 fork run launched, on ice until principal returns

> Cold-reader pointers: `ANNOTATION_SIDESTEP.md` (R4 = rung 4),
> `TERMINAL_FORK_COMPUTE.md` §scenario F (cost model),
> `annotate/runbook_r4_cancun.md` (GLM attendance protocol),
> `lab_notes/2026-08-27-r4-fork-launch.md` (launch facts + setup commands),
> `lab_notes/2026-08-27-r4-fork-launch-session.md` (this session's debugging log).
> Supersedes `2026-08-27-handoff-r4-build-inflight.md` (build phase).

## objective_and_constraints

```yaml
objective: |
  R4 fork run is RUNNING. 1944 rows (27 states × 3 arms × 24 seeds)
  across 4 processes on 2 Vast A6000 boxes. Expected completion:
  ~4.55 days (2026-08-30 ~10:00 UTC). Principal returns after
  2026-09-01. GLM agent attends every 3h per runbook. Session on ice.
locked_design:
  states: 27 (10 CYCLE / 10 LOOP / 7 MUSE), plan at
    artifacts/fork_plan_r4.jsonl
  arms: noop | toward_healthy (+lam*d) | toward_diverge (-lam*d)
  layers: CYCLE L18, LOOP L2, MUSE L17
  lam: 0.05 × median onset residual norm (cycle 2.014, loop 0.349, muse 1.768)
  sampling: m=24 paired seeds per arm, sub-batch 1, temp 0.8, top_p 0.95
  model: bf16 HF Qwen3.5-9B, sdpa with FLASH_ATTENTION backend
    (determinism proven 2026-08-27: concurrent same-seed 4096-token
    runs → identical SHA256; MATH also deterministic but OOMs co-residency)
  backend: FLASH_ATTENTION (not MATH — see negative_knowledge)
  cap: 26000 tokens
  scoring: frontier.collect_rollouts.objective_check vs reference_answer
bindings:
  - Reward policy: terminal verified outcome only
  - Do NOT touch: Vast 48783410 (stopped, reserved), llama-* services,
    local GPUs, git, detection artifacts, other experiments
  - Budget cap: $150 total (projected ~$106)
  - Wall cap: 5.5 days (projected ~4.55 days — within cap)
```

## world_state_delta

```yaml
backend_switch: |
  MATH → FLASH_ATTENTION. MATH materializes q×kv attention (34GB prefill
  spike), making two-process co-residency impossible on 48GB cards for
  fork >9.5k (11 of 27 states). FLASH has no spike (~17GB steady),
  empirically proven deterministic (SHA256 match across concurrent
  processes, 4096 tokens). Numerics differ from MATH but run is
  self-consistent (paired arms share backend).
seed_fix: |
  batch_seed changed from hash(state_id) to sha256(state_id). Python
  hash() is per-process randomized (PYTHONHASHSEED), silently broke
  cross-process and cross-resume seed pairing. Caught by bit-mismatch
  in determinism check.
oom_guards: |
  Model-load retries (3×, 30/60s backoff), generation retries (2×,
  25/60s backoff, loud raise after), periodic empty_cache() hook (every
  128 decode steps), post-generate empty_cache(). Zero OOM crashes in
  production.
checkpointing: |
  Cron job every 2h syncs results to local (Vast workspace not
  persistent). Script: data/checkpoint_sync.sh. GLM agent notified but
  doesn't manage it.
runbook_updates: |
  Added "⚠️ CONTEXT PRESERVATION" section: spawn subagents for
  debugging, use own model (not qwen 3.7-plus), concise outputs,
  one-line log entries. Removed manual rsync (automated).
```

## negative_knowledge

```yaml
- VAST SSH PERMISSIONS: authorized_keys at wrong ownership/modes →
  sshd StrictModes refuses every key. Diagnose with `vastai logs <id>`
  (shows sshd auth log: "Authentication refused: bad ownership or
  modes"). Fix: mandatory --onstart-cmd that chmod/chown every 30s
  (runbook Appendix A has exact string). First box pair (48916215/48916245)
  lost to this; replaced with 48921611/48921612.
- MATH MEMORY SPIKE: prefill materializes q×kv → 34GB transient
  (16.7GB weights + 17.2GB spike). Two processes can't co-reside on
  48GB when fork >9.5k. FLASH_ATTENTION has no spike (~17GB steady).
  Switched backend; determinism verified.
- BATCH ANTI-SCALING: decode is launch-overhead + bandwidth bound on
  this hybrid model. Measured: b1 19.6 → b2 15.7 → b4 10.3 tok/s
  aggregate. Two time-sliced processes = 1.85× one process. Run with
  sub-batch 1, 2 processes per box.
- HASH() INSTABILITY: batch_seed = hash(state_id) is per-process
  randomized. Cross-process seed pairing silently broken. Fixed to
  sha256(state_id). Bit-verified after fix.
- EMPTY_CACHE TIMING: MATH prefill spike persisted through entire
  decode phase (allocator caches freed blocks). empty_cache() only at
  sub-batch end → co-process couldn't load. Fixed: empty_cache()
  immediately after generate() returns.
- EFFICIENT_ATTENTION: "No available kernel" on this stack (xformers
  not installed). Only FLASH_ATTENTION and MATH are viable.
- COST ANALYSIS: A6000 ($5.48/1M tokens) beats A100 ($6.20/1M tokens)
  for latency-bound decode. H100 3× more expensive. Stayed with A6000.
```

## operational_state

```yaml
running_processes:
  box_a:
    instance: 48921611
    label: r4fork-a
    ssh: r4fork-a (root@209.137.198.14:28119)
    programs:
      - r4fork1: 7 states → 504 rows (r4_muse_00,r4_cycle_09,r4_cycle_02,
          r4_loop_09,r4_loop_04,r4_muse_01,r4_loop_02)
      - r4fork2: 6 states → 432 rows (r4_muse_04,r4_cycle_08,r4_muse_06,
          r4_cycle_03,r4_cycle_05,r4_cycle_07)
    status: RUNNING, GPU 97-100%, ~36GB resident
  box_b:
    instance: 48921612
    label: r4fork-b
    ssh: r4fork-b (root@38.29.145.10:40340)
    programs:
      - r4fork1: 7 states → 504 rows (r4_loop_07,r4_loop_08,r4_loop_05,
          r4_loop_06,r4_cycle_00,r4_cycle_06,r4_loop_01)
      - r4fork2: 7 states → 504 rows (r4_loop_00,r4_cycle_01,r4_muse_05,
          r4_loop_03,r4_muse_03,r4_cycle_04,r4_muse_02)
    status: RUNNING, GPU 97-100%, ~36GB resident
launch_time: 2026-08-27 21:40-21:42 UTC
expected_completion: ~2026-08-30 10:00 UTC (~4.55 days)
projected_cost: ~$106 (well under $150 cap)
checkpointing:
  frequency: every 2 hours via cron
  script: /home/admin/mlfactory/mlfactory/experiments/ace/data/checkpoint_sync.sh
  log: /home/admin/mlfactory/mlfactory/experiments/ace/data/checkpoint_sync.log
  local_backup: /home/admin/mlfactory/mlfactory/experiments/ace/data/r4fork-{a,b}/
attendance:
  agent: GLM (per runbook_r4_cancun.md)
  frequency: every 3 hours
  log: /home/admin/mlfactory/mlfactory/experiments/ace/annotate/out/r4_attendance.log
verification_evidence:
  - Local smoke (3090): 24/24 rows, 23/24 correct, 16/16 steered pairs differ
  - Remote smoke (box B): 3/3 rows correct
  - Determinism: FLASH same-seed concurrent 4096-token runs → identical SHA256
  - Co-residency: long-fork + twin, zero OOM retries, both generating at 98% util
tombstones:
  - Old boxes 48916215/48916245: destroyed (SSH perms issue)
  - Old box 48783410: stopped, reserved (do NOT touch)
```

## open_questions

```yaml
- None critical. Run is stable, checkpointed, on track.
- Analysis begins after run completes (counterfactual framework).
```

## pointers

```yaml
runbook: annotate/runbook_r4_cancun.md (GLM attendance protocol)
launch_note: lab_notes/2026-08-27-r4-fork-launch.md (setup commands, design deviations)
session_note: lab_notes/2026-08-27-r4-fork-launch-session.md (this session's debugging)
attendance_log: annotate/out/r4_attendance.log (GLM appends every 3h)
checkpoint_script: data/checkpoint_sync.sh (cron every 2h)
checkpoint_log: data/checkpoint_sync.log (cron appends)
local_checkpoints: data/r4fork-{a,b}/fork_r4_results_*.jsonl
remote_results: /root/mlfactory/mlfactory/experiments/ace/data/fork_r4_results_{1,2}.jsonl
plan: artifacts/fork_plan_r4.jsonl
harness: annotate/fork_r4.py (FLASH_ATTENTION, sha256 seeds, OOM guards)
```

---

## checkpoint_timeline

1. Principal asked for R4 cost estimate → agent grounded it in TERMINAL_FORK_COMPUTE + measured horizons → scenario F added.
2. Principal: H200 over-provisioned, right-size → agent measured HF decode is latency-bound → swept Vast market → A6000 $0.43 vs 2×H200 $8.45.
3. Principal: codify this → agent rewrote VAST_REMOTE.md "Choosing a Vast offer" as agent-driven search.
4. Principal: leaving for Cancun, wants GLM-attended runbook → R4 approved.
5. Agent locked design (27 states, 3 arms, m=24, lam from measured norms), built plan (datasaved), wrote fork_r4.py.
6. Debug cycle: directions key typo → full-mode OOM → cache fast path built → empty-input_ids rejected → position-id IndexError → inputs_embeds IndexError → root cause: VL wrapper prepare_inputs. Decision: GRPO-proven full-forward rewrite.
7. Agent provisioned box A (48916215) and box B (48916245); both loading.
8. Post-handoff: gen_batch rewritten to GRPO full-forward with batch_for() sizing; runbook_r4_cancun.md written; shards locked snake-balanced; box B running, A loading.
9. Smoke caught verifier bug: objective_check keys are correct/match_mode, strict gen check needs rec={domain,knobs} (else silent soft fallback) — fixed in harness, plan rebuilt with domain/knobs/surface_hash, smoke relaunched.
10. Post-compaction: smoke verified full-forward generation (noop rows correct=True) but crashed hooked arm on dtype bug (fp32 delta + bf16 hidden → fp32 out → next linear raises) — fixed (delta.to(h.dtype) in hook + directions cast to model.dtype), smoke resumed.
11. SSH wall: old boxes unreachable (authorized_keys modes), box A vanished mid-provision; destroyed B, recreated both with onstart SSH repair (48921611 Kansas / 48921612 Delaware), SSH verified on both, ~/.ssh/config aliases added, runbook updated.
12. Throughput diagnosis (subagent): 9-19 tok/s per stream is launch-overhead + bandwidth bound; batching anti-scales; two time-sliced processes = 1.85× one. Decision: 2 processes per box, sub-batch 1.
13. Two real bugs caught by verification: hash(state_id) batch seeds per-process randomized (silently broke seed pairing → sha256), batch-1 OOM path silently dropped seeds (now loud raise).
14. Memory wall: MATH prefill spikes to ~34GB, two-process co-residency impossible on long-fork states. FLASH_ATTENTION proven bit-deterministic (concurrent same-seed 4096-token runs → identical SHA256) and spike-free → adopted.
15. Launched all 4 programs (21:40-21:42 UTC), verified rows flowing, GPUs at 97-100%, zero errors.
16. Principal asked for completion estimate → agent computed from actual data (34 rows): median continuation 9,947 tokens, wall time 4.55 days, cost $106. A6000 is most cost-effective.
17. Principal asked about checkpointing → agent found workspace not persistent, created checkpoint_sync.sh + cron every 2h, updated runbook to remove manual rsync.
18. Principal asked for context preservation rules → agent added "⚠️ CONTEXT PRESERVATION" section to runbook: spawn subagents, use own model, concise outputs, one-line log entries.
19. Session going on ice. Run stable, checkpointed, GLM attending. Expected completion ~2026-08-30 10:00 UTC, principal returns after 2026-09-01.
