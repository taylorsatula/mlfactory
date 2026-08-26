# HANDOFF — 2026-08-26 — Step 1: SDPA determinism fix pending (context-compaction / overnight carry-forward)
# Supersedes lab_notes/2026-08-25-handoff-grpo-step1-inflight.md. The parent
# plan (steps, bindings, stop criteria) is STILL
# lab_notes/2026-08-25-handoff-grpo-gated-execution.md — read it first.
# Tonight's evidence record: lab_notes/2026-08-26-step1-sdpa-generation-nondeterminism-identity-gate.md
# Audience: the same agent, tomorrow morning. Read this + parent + evidence note.

objective_and_constraints:
  goal: unchanged — first serious GRPO attempt on the b2 46-prompt pool,
    Vast H200, falsifying steps, pre-registered stops. Currently INSIDE
    step_1_loop_health: one gate FAILED (zero-init identity), root cause
    found and fix verified in probe; code fix + dry-run rerun tomorrow.
  goal_deltas_this_session:
    - "ROOT CAUSE of the identity-gate failure: default-SDPA generation is
       call-to-call NON-DETERMINISTIC on this substrate (identical seed,
       weights, code, and proven-equal RNG draw counts still diverge at
       422-1764 tokens). Not the controller (same-arm pairs flip), not RNG
       consumption. Fix verified: sdpa_kernel([SDPBackend.MATH]) -> 0/8
       flips. FLASH backend untested. Also re-explains probe_determinism."
    - "The iterate-stage matched-base-arm design ('same seeds ... so
       determinism holds') is dead as written; replacement routes: (i)
       deterministic backend for all rollouts, or (ii) fork-from-prefix
       replay matching (bit-exact, proven). Decide after tomorrow's
       26k-throughput + FLASH-determinism measurements."
    - "Write-backs PAID: PHASES (full replay + determinism ruling), STATUS
       (R10 windowed killed, R11 SDPA nondeterminism, Q10 updated),
       OPERATIONS (replay + determinism rulings, H200 resume exception),
       ENVIRONMENT (traps + stale instance id fixed)."
  bindings: "ALL parent-handoff bindings still bind (terminal-only reward,
    thinking ON, bf16, rows immutable, local<->remote sync on ANY file
    change, NO git commits without explicit request, rsync artifacts home
    at every step boundary, b3 untouched). Seed spaces: 70k smoke, 80k
    train (CONSUMED by s1_dry iters 0-1), 90k eval-hold, 95k eval-train,
    100k revalidate. The dry-run RERUN takes a fresh 81_000 base."
  user_directives_carried_forward:
    - "Prefer --quiet outputs; preserve context window."
    - "Checkpoint discipline: after each iteration state success / what
       didn't work / next intent, out loud."
    - "Never patch over janky code or accept 'good enough' outcomes."
    - "Blocking sleeps are fine; pass explicit timeouts on every command
       that can hang (ssh/remote/builds/probes)."
    - "Read docs into context during waits (user: it prevents misphrasing)."
    - "End of day: notes + write-backs + data pulled local + BOX STOPPED."
    - "Success-bar phrasing: the controller must beat a well-tuned
       constant-lambda baseline (which steers along the pre-allocated
       length-DERIVED direction) on FORKED TERMINAL-VERIFIED outcomes —
       length is the competitor's intervention axis, never a metric
       (REWARD_POLICY.md)."

operational_state:
  remote_box: "Vast instance 48673764 STOPPED for the night (user ruling).
    Start: ~/.local/bin/vastai start instance 48673764 (key
    ~/.vast_api_key), wait for ssh root@198.145.108.59 port 30854 (key
    ~/.ssh/id_vast, IdentitiesOnly=yes). Filesystem intact; tmux/GPU state
    does not survive; re-stop vllm/model-ui/ray if autostarted. Repo
    /workspace/mlfactory synced as of 2026-08-25 23:20 (NO code changes
    since — the determinism fix is NOT in the code yet); model via
    ACE_MODEL_PATH=Qwen/Qwen3.5-9B + HF_HOME=/workspace/models; venv
    /venv/main. Remote evidence: /workspace/s1_dry (done), h2-h5probe.log,
    verify_h*.sh, kill_s1dry.sh + relaunch_s1dry.sh (staged, untested)."
  local: "Artifacts pulled: /tmp/ace_s1_20260826/ (s1_dry dir + logs +
    probe scripts). Local repo /home/admin/mlfactory uncommitted BY
    DESIGN; local ace venv for CPU checks:
    /home/admin/mlfactory/mlfactory/experiments/ace/.venv/bin/python.
    2x llama-server collection on local GPUs — leave alone."
  verified_numbers: "Gate results from the completed s1_dry: 48 rows /
    0 dups; base_fingerprints_unchanged=true; replay full x24, 0 OOM
    fallbacks; peak_mem_gb 47.78 (allocated); cap-hit rates 0.21/0.17;
    zero_var_groups iter1=1 (assign-p132 all-correct); EXIT 0. Identity
    gate: FAILED 12/12 pairs (root cause R11). Probes: RNG draws exactly
    6400/run; MATH backend 0/8 flips; flip positions 422-1764 default
    backend."

next_actions_in_order:
  1: "Start box (vastai start 48673764); wait for ssh; gather state
      (nvidia-smi, tmux ls — expect none, df); re-stop template services."
  2: "Two quick probes on GPU0 (both: assign-p132, seed 82244, batch-4,
      base-vs-base flip test): (a) FLASH backend
      sdpa_kernel([SDPBackend.FLASH_ATTENTION]) — determinism AND tok/s;
      (b) MATH backend throughput at max_new 26000 (attention cost grows
      with length — the 1600-token probe is not sufficient). Pick the
      deterministic backend with the best 26k throughput."
  3: "Apply the fix in core/steering_controller.py generate_batch: force
      the chosen deterministic SDPA backend around model.generate (context
      manager, scoped to rollout generation only). Verification: local
      py_compile + import-smoke; remote two-call flip test (must be 0
      flips); teacher-forced paths unaffected but rerun
      --check-equivalence once as a control."
  4: "Dry-run RERUN (fresh out dir, seed base 81_000): same production
      settings (--pool acegen_live_b2 --only-pids 132,140,148 --iters 2
      --prompts-per-iter 3 --group-size 4 --max-new 26000 --skip-eval).
      Gate (a) MUST pass now (steered==base bit-identical, zero-init).
      Watch: iter 0 line in train.jsonl after ~24 rows."
  5: "Crash-resume test (the window missed tonight): when rows >= 25 and
      iter-0 line present — SIGKILL via /workspace/kill_s1dry.sh, relaunch
      via /workspace/relaunch_s1dry.sh (check its out dir/log still point
      at the rerun's paths — they were written for s1_dry). Verify:
      rows_resume, iter-0 skipped, partial group frozen from disk, no dup
      keys, EXIT 0. THEN run /tmp/step1_gates.py (local copy; adjust cap/
      peak key names if the schema differs) + the full gate list."
  6: "rsync artifacts home; STATE THE STEP-1 CHECKPOINT REPORT (all gates,
      numbers). ALL pass -> Step 2 (revalidation, 46 x n=8, two GPUs,
      ~7h; stop if <15 LIVE). Any fail -> HOLD, report to user."

open_questions:
  - "MATH-vs-FLASH at 26k: which deterministic backend, at what
      throughput cost? (bet: FLASH deterministic and fast — unverified)."
  - "If NO backend is both deterministic and fast enough: fallback is
      fork-from-prefix matching via replay for all arm comparisons;
      generation determinism then only needed within a single call."
  - "bf16 pool texture (Step 2) — genuinely uncertain; stop if <15 LIVE."
  - "Q10/Q11 — unchanged from parent handoff."
  - "Silent process death (first long probe, 2026-08-25) — no recurrence
      tonight; keep watching."

tombstones:
  - "All probe sessions closed. s1_dry is DONE (EXIT 0) — do not rerun it
      (evidence complete at /tmp/ace_s1_20260826/ and /workspace/s1_dry)."
  - "kill_s1dry.sh/relaunch_s1dry.sh are staged for step 5 but were never
      exercised; verify paths before use."
  - "/tmp/writebacks_draft.md and /tmp/step1_gates.py — working drafts;
      write-backs are PAID into the docs, the draft file is superseded."

checkpoint_timeline:
  1: "agent oriented from step1-inflight handoff + parent plan; verified
      box/tmux/row state live."
  2: "user granted overnight autonomy + reminders (no patching jank;
      timeouts on everything; read docs during waits; box down at night)."
  3: "dry run iter 0 landed; gate (a) ran early: 12/12 pair MISMATCH."
  4: "diagnosis chain h1-h5 on GPU1 in parallel with the live dry run:
      no-op hook confirmed (rel_max 0) -> clean-process reproduction (h2)
      -> equal RNG consumption 6400 draws, after fixing a broken
      measurement (h3) -> same-arm pairs flip too (h4) -> MATH SDPA
      backend = 0 flips (h5). Root cause: call-to-call SDPA
      non-determinism."
  5: "user check-ins: explained kill window; corrected success-bar
      phrasing (length = baseline's axis, never a metric); big-picture
      recap."
  6: "dry run completed naturally (48 rows, EXIT 0); other gates read:
      fingerprint true, peak 47.78 GB, caps reported, zero-var guard
      counted. Artifacts rsynced to /tmp/ace_s1_20260826/."
  7: "evidence note + four doc write-backs written and verified; this
      handoff written; box being stopped for the night."
