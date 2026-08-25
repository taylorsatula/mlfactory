# HANDOFF — 2026-08-25 (evening) — GRPO wiring next
# SUPERSEDED (same day) by
# lab_notes/2026-08-25-handoff-grpo-gated-execution.md — read that one.
# Supersedes 2026-08-25-handoff-grpo-next.md. Read this one.

objective_and_constraints:
  current_goal: >
    Wire and run the first serious GRPO attempt on the Vast H200:
    thinking on, b2 46-prompt pool, segmented replay, 2-GPU rollout
    split. User guidance pending on scope/budget before the full
    launch; a 2-iter dry run at production settings comes first.
  bindings:
    - "Terminal verified reward only; truncation backdoor reported per
       group (REWARD_POLICY.md caveat written today)."
    - "bf16 HF substrate; first unsteered batch = pool re-verification;
       prompts dropping out of band leave the pool."
    - "Replay windows ≤8k (measured ceiling, R9); segmented replay is
       hypothesis-mandated, not optional plumbing."
    - "Per-sample rollout rows this time (smoke's evidence gap)."
    - "Remote workspace is non-persistent: rsync results home."
    - "b3 paused at guarded shape until the controller gate resolves."
    - "No commits without explicit user request; local/remote stay in
       sync for any file change."
    - "workin-hard skill is the working protocol."

world_state_delta:
  - "Smoke COMPLETE (report /workspace/smoke1/, local /tmp/smoke1/):
     load 16.7 GB; generation 73 tok/s agg batch-4; traces median 22.3k,
     6/24 cap-hit; replay cap curve 26.9→111.6 GB (512→8192), 16k OOM;
     grad path finite; fingerprint holds; per-family correctness
     4/4,4/4,3/4,1/4,2/4,4/4."
  - "Kernel A/B: flash-attn 2.8.3 (sm90 build) and fla 0.5.2 +
     causal-conv1d 1.7.0 installed on remote; BOTH zero-gain (~150 tok/s).
     Q11 open (is fla actually invoked?)."
  - "Docs written back today: REWARD_POLICY (truncation caveat), STATUS
     (Q10 updated, Q11 new, R9 new), ENVIRONMENT (remote stack section),
     OPERATIONS (training-location ruling), CALIBRATION (bf16 preview),
     PHASES (Phase-2 measurements, Phase-3 prep)."
  - "Session notes written: session_notes/2026-08-24-*.md and
     session_notes/2026-08-25-*.md."
  - "Next free pid still 164 (b3 untouched)."

negative_knowledge:
  - "flash-attn helps nothing here (bottleneck ≠ full attention)."
  - "FLASH_ATTN_CUDA_ARCHS is the env var (not FLASH_ATTENTION_…);
     wheels are torch-version-locked."
  - "pkill -f over ssh matches its own command line; use script files."
  - "supervisorctl stop can orphan vllm serve; kill by pid."
  - "Smoke persisted group aggregates only — length↔outcome join lost;
     production writer emits per-sample rows."
  - "HF generation ≈15% of bandwidth ceiling → don't chase TP before
     kernel-invocation check and rollout parallelism."

operational_state:
  remote: >
    Vast #46911241, ssh root@198.145.108.59:30854 (key ~/.ssh/id_vast).
    tmux: smoke (finished), flashbuild (done), conv1dbuild (done).
    GPUs idle (vllm/model-ui/ray stopped, autostart off). Model at
    /workspace/models/hub/models--Qwen--Qwen3.5-9B. Stale duplicate
    download dir /workspace/models/models--Qwen--Qwen3.5-9B (~18 GB)
    flagged for deletion once cleanup authorized.
  local: >
    2× llama-servers :3091/:3092 still up (q8+MTP, collection).
    Worktree uncommitted by design (today's docs/notes + smoke script
    among it; user commits on their schedule).
  next_commands:
    - "Await user guidance on run scope/budget."
    - "Wire grpo.py: --pool adapter (solver_prompt + CHECK dispatch,
       stratified train/holdout split manifest), --thinking default on,
       --replay-cap + segmented windows, per-sample rows, NaN/truncation
       guards, ACE_MODEL_PATH env override, attn_implementation arg."
    - "2-iter dry run → report → full attempt under tmux with named log;
       rsync artifacts home at milestones."

open_questions:
  - "Q10 unchanged — this attempt tests it; genuinely uncertain."
  - "Q11: is fla silently falling back? One instrumentation run decides;
     bet: fallback or not the bottleneck, confidence moderate."
  - "Run scope/budget: user's call (group size, prompts/iter, iters,
     whether re-verification runs as a standalone full-n pass first)."

pointers:
  smoke_results_note: lab_notes/2026-08-25-grpo-h200-smoke-results.md
  setup_note: lab_notes/2026-08-25-grpo-h200-setup.md
  b3_resume: lab_notes/2026-08-25-b3-shape-resume-here.md
  pool: data/acegen_live_b2.jsonl
  local_report_copy: /tmp/smoke1/

checkpoint_timeline:
  - "Re-read post-compaction handoff; held state."
  - "User moved GRPO to Vast (3090s insufficient) → transfer plan presented."
  - "User committed the tree (e3f039d, 4dd181b, 0e83090) → git-archive snapshot valid."
  - "Instance chosen: H200 #46911241 (VRAM first, then DLPerf/$) → user spun it up."
  - "Connected via id_vast; read vast agent guide; stopped vllm template services + killed orphan → GPUs free (12 GB residual → 0)."
  - "Model download completed via snapshot_download (config/tokenizer were missing)."
  - "User + agent rulings: thinking ON (grpo.py's thinking-off shape overruled); HYPOTHESIS re-read confirmed it; replay-window requirement surfaced."
  - "workin-hard skill triggered; TRAINING_STACK/steering_controller/pool interfaces read."
  - "Smoke written (train/smoke_h200.py), synced, launched in tmux."
  - "User flagged slowness → diagnosed not-hung via GPU1 probes: ~150 tok/s agg, SDPA, flash absent."
  - "User: try flash; use the median → b2 median 18.4k found; flash-attn source-built (env-var name corrected mid-flight) → zero gain."
  - "Bottleneck relocated to linear-attention blocks → fla installed, causal-conv1d built; smoke reached 5/6 then 6/6 groups."
  - "User asked about tensor splitting → recommendation: 2-GPU rollout parallelism, not TP=2 (PCIe tax, not bandwidth-bound)."
  - "Smoke completed: replay ceiling 8k, fingerprint holds, mixed rewards on bf16."
  - "Report parsed; HYPOTHESIS impact assessed (no claim change; truncation caveat + segmented-replay derivation written to proper owners)."
  - "Housekeeping checkpoint: docs written back, session notes written, this handoff."
