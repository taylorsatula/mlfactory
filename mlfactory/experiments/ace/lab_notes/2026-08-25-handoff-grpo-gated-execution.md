# HANDOFF — 2026-08-25 — GRPO execution (fresh session)
# Supersedes lab_notes/2026-08-25-handoff-grpo-wiring-next.md. Read this one.
# Audience: the next agent, running autonomously (workin-hard skill).
# Deliberately lossy on completed phases; dense on what happens next.

objective_and_constraints:
  goal: >
    Run the FIRST SERIOUS GRPO attempt on the b2 46-prompt pool on the
    Vast H200. The plan is a sequence of small falsifying steps with
    pre-registered stop criteria, so a bug or a null result is caught
    early instead of late. You are running this autonomously: continue
    within the approved envelope, hold only on a stop criterion or a
    surprise, report at each step boundary.
  success_bar (Q10, STATUS.md): >
    The controller must beat a WELL-TUNED CONSTANT-LAMBDA on the
    pre-allocated reasoning-length axis on FORKED outcomes — not merely a
    no-op. A learned controller that only matches fixed-direction /
    constant-lambda baselines has no value claim.
  bindings:
    - "Terminal verified reward ONLY (REWARD_POLICY). No entropy/length/
       recurrence/judge/PRM terms. Truncation is a backdoor length term —
       report per-group cap-hit rate every batch."
    - "Thinking ON for every rollout (overrules grpo.py's old
       thinking-off/640-token/arithmetic-set shape)."
    - "bf16 HF substrate. The first UNSTEERED bf16 batch = the pool's
       re-verification-by-regeneration; prompts that drop out of band
       leave the pool. No special pleading."
    - "Replay windows ≤8k (measured ceiling). Segmented replay is
       HYPOTHESIS-mandated, not optional plumbing."
    - "Existing rows/artifacts immutable; per-sample rollout rows are
       append-safe, resume-keyed JSONL (fixes the smoke's evidence gap)."
    - "Keep local and remote in sync for ANY file change. No git commits
       without explicit user request."
    - "Remote workspace is NON-persistent (workspace_is_volume=false):
       rsync artifacts home at every step boundary."
    - "b3 (construct/revise) stays paused at guarded shape until the
       controller gate resolves. Do not touch it."

operational_state:
  remote:
    instance: "Vast #46911241, 2x H200 140GB (PCIe5, NO NVLink)"
    ssh: "root@198.145.108.59 port 30854, key ~/.ssh/id_vast (-i, IdentitiesOnly)"
    repo: "/workspace/mlfactory (git-archive + rsync, pip install -e .)"
    model: "/workspace/models/hub/models--Qwen--Qwen3.5-9B (bf16). HF_HOME=/workspace/models. Load via ACE_MODEL_PATH=Qwen/Qwen3.5-9B"
    venv: "/venv/main — torch 2.13.0+cu130, transformers 5.15.0, flash-attn 2.8.3(sm90), fla 0.5.2, causal-conv1d 1.7.0"
    state: "Box STOPPED at handoff (user request 2026-08-25, deliberate
      clean-slate state). Start it from local when GPU work resumes:
      `~/.local/bin/vastai start instance 48673764` (key via
      ~/.vast_api_key), then wait for ssh
      root@198.145.108.59:30854 to accept again. Filesystem intact;
      tmux sessions and GPU state do NOT survive the stop — re-verify
      everything (nvidia-smi, ls /workspace/smoke1, tmux ls) after boot,
      and re-stop the vllm/model-ui/ray template services if they
      autostarted."
    vllm_services: "vllm/model-ui/ray are STOPPED (autostart off). They hold GPUs if started — keep stopped for training."
  local:
    repo: "/home/admin/mlfactory (uncommitted by design; b2 pool + docs + notes staged)"
    servers: "2x llama-server q8+MTP on :3091/:3092 for collection — leave alone"
  pool: "data/acegen_live_b2.jsonl — 46 prompts, 6 families, band spread 1/8..7/8, ALL q8_0+MTP-banded, owes bf16 re-verification"
  key_numbers:
    gen_throughput: "~73 tok/s aggregate batch-4 (SDPA); traces median 22.3k, 6/24 cap-hit in smoke"
    replay_ceiling: "8k window = 111.6GB peak; 16k OOMs; memory linear in window"
    load: "bf16 8.95B params = 16.7GB"

  restart_constraint: >
    RESOLVED. An account-level Vast API key is stored in mlfactory
    secrets (VAST_API_KEY) and at ~/.vast_api_key (chmod 600); the local
    vastai CLI (~/.local/bin/vastai) authenticates with it. The agent CAN
    stop/start the box from local: `vastai stop instance 48673764` and
    `vastai start instance 48673764` (add `--api-key $(python3 -m
    mlfactory secrets get VAST_API_KEY | tail -1)` if ~/.vast_api_key is
    absent). stop/start preserves the whole container filesystem. Note the
    in-container CONTAINER_API_KEY is only SELF-scoped (401s externally)
    — always control the box from local with the account key.

the_plan (technical sequence — each step falsifies before the next):
  step_0_wiring:
    note: "Local code work. The box is already STOPPED and stays stopped
      during wiring. Order matters: wire locally FIRST, then start the box
      (`vastai start instance 48673764`), wait for ssh, THEN rsync the
      code over (rsync needs ssh — it cannot run against a stopped
      instance). stop/start preserves the filesystem (see
      restart_constraint)."
    actions:
      - "Wire train/pool_adapter.py: load pool, build items via
         frontier.collect_rollouts.solver_prompt(), score-dispatch through
         gen.calibrate.CHECK per family, deterministic stratified
         train/holdout split (~31/15, hold out 1/3 per family) written as
         a split manifest into the run dir."
      - "Wire train/grpo.py changes: --pool flag + CHECK-based verify
         (keep legacy arithmetic path as fallback); thinking ON default;
         max_new 26000; SEGMENTED REPLAY (gradient-checkpointed full-trace
         replay first, detached-prefix ≤8k windows as fallback — the dry
         run picks); per-sample rollout rows (prompt id, seed, length,
         truncated, reward) append-safe JSONL; objective guards
         (NaN/Inf stop, per-group truncation+EOS rate, reward variance,
         KL, grad norm, memory peaks); ACE_MODEL_PATH env override."
      - "Verify: py_compile, import-smoke, determinism of the split, then
         rsync to remote."
  step_1_loop_health:
    actions: "Ladder (load, thinking-on gen on 3 pool prompts,
      checkpointed replay at real window, one backward step); 2-iter dry
      run at production settings on a 3-prompt slice with equivalence
      checks (windowed vs single-pass logprob agreement, per-sample rows
      complete, a planted-NaN trips the guard, fingerprint holds)."
    report_point: "Report before proceeding to step_2."
    stop: "any guard trip, logprob mismatch, memory creep"
  step_2_revalidation:
    actions: "All 46 prompts x n=8, UNSTEERED, thinking on, both GPUs.
      Re-bands the pool on the policy it will train against."
    stop: "re-banded pool collapses below ~15 LIVE prompts (pool-texture
      verdict, not controller verdict -> stop and rethink)"
  step_3_first_gradients:
    actions: "2-4 GRPO iterations + first eval on the frozen pool."
    stop: "zero reward variance, gate collapse/saturation, wall-clock far
      outside forecast"
  iterate:
    actions: "Iterations to ~10-12 at G=4, eval every 4 iters; run the
      constant-lambda baseline arm CONCURRENTLY (generation-only).
      GPU division of labor (both H200s): STEERED rollouts (controller
      hook, must be HF transformers) on GPU0, MATCHED-BASE rollouts on
      GPU1 — same seeds, same group construction, only the device
      differs, so determinism holds; distribute the replay-with-gradient
      passes across both GPUs the same way. Alternatively shard prompts
      across the GPUs if arm-parallelism proves awkward. Do NOT TP=2
      (PCIe all-reduce tax, and we are not bandwidth-bound)."
    stop: "at two consecutive evals, steered-vs-base separation inside CI
      AND gate dynamics degenerate (gate_std->0 or saturated) -> genuine
      null, terminate per PHASES kill rule"
  verdict:
    actions: "Full iteration budget, baseline tuning, holdout verdict,
      Q10 write-back (STATUS), b3 gate decision."
    gate: "Proceed to verdict only if iterate showed state-dependent gate
      structure and a separation trend."
  default_config: "G=4 (not 6), 4 prompts/iter, lr 1e-3, beta_kl 0.02,
    lambda_mag 0.1, temp 0.8, top_p 0.95. Seeds: the SMOKE consumed the
    70_000+pid space — every new batch takes a FRESH base (e.g.
    80_000 + 17*pid + sample_i per OPERATIONS.md), never b2's q8 space
    and never the smoke's."

what_is_already_done (lossy — do not re-do):
  - "b2 calibration done: 46-prompt LIVE pool exists (q8-banded)."
  - "Forecasting smoke COMPLETE: generation/replay/gradient numbers above;
     per-family bf16 preview 4/4,4/4,3/4,1/4,2/4,4/4; report at
     /workspace/smoke1/ (remote) and /tmp/smoke1/ (local)."
  - "Kernel optimization exhausted: flash-attn AND fla+conv1d gave ZERO
     gain (~150 tok/s). Do NOT chase kernels; bottleneck is not bandwidth.
     (Q11 open: confirm fla is actually invoked, one instrumentation run.)"
  - "Docs written back: REWARD_POLICY (truncation caveat), STATUS
     (Q10/Q11/R9), ENVIRONMENT (remote stack), OPERATIONS (training
     location), CALIBRATION (bf16 preview), PHASES (Phase2/3)."
  - "Session notes + this handoff lineage exist."

negative_knowledge:
  - "flash-attn/FLA don't speed this workload; generation is ~15% of
     bandwidth ceiling (overhead/latency-bound)."
  - "flash-attn env var is FLASH_ATTN_CUDA_ARCHS (not FLASH_ATTENTION_...);
     wheels are torch-version-locked (none for torch 2.13 -> source build)."
  - "pkill -f over ssh matches its own command line; use script files or
     pkill by exact name."
  - "supervisorctl stop can orphan `vllm serve`; kill by pid."
  - "HF cache layouts differ by writer; verify config.json present before
     trusting a snapshot dir."
  - "TP=2 over PCIe at 9B is a low-bet; prefer 2-GPU rollout parallelism
     (steered GPU0 / base GPU1, or prompt shards)."

open_questions:
  - "Q10 unchanged — this attempt tests it (genuinely uncertain)."
  - "Q11: is fla silently falling back? (one instrumentation run decides;
     bet: fallback or not the bottleneck)."
  - "Scope: the gate structure is approved in principle; do not escalate
     to the verdict stage without a GO at the iterate stage."

first_actions_for_next_session:
  - "Read this file, load workin-hard skill."
  - "Do Step 0 wiring locally first (box is STOPPED and stays stopped
     during wiring — see operational_state)."
  - "THEN start the box (`vastai start instance 48673764`), wait for ssh
     to accept, rsync the wired code, gather state (nvidia-smi, tmux ls,
     df — tmux/GPU state did not survive the stop), re-stop the
     vllm/model-ui/ray services if any came back, and run Step 1."
  - "Report at the Step 1 checkpoint before Step 2."

pointers:
  pool: "mlfactory/experiments/ace/data/acegen_live_b2.jsonl"
  smoke_note: "lab_notes/2026-08-25-grpo-h200-smoke-results.md"
  setup_note: "lab_notes/2026-08-25-grpo-h200-setup.md"
  b3_resume: "lab_notes/2026-08-25-b3-shape-resume-here.md"
  status_ledger: "STATUS.md (Q10/Q11/R9)"
  reward_policy: "REWARD_POLICY.md (truncation backdoor caveat)"
