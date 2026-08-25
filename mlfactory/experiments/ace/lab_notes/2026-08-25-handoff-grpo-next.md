# HANDOFF — 2026-08-25 — post-compaction successor state
# Read this first. Durable detail lives at the pointers; this file carries
# only what the artifacts do not.

objective_and_constraints:
  current_goal: >
    Run the FIRST SERIOUS GRPO attempt on the b2 pool
    (data/acegen_live_b2.jsonl, 46 prompts). User ruling 2026-08-25:
    this gates everything else — if the controller shows signal, resume
    b3 (calibrate + pool the two new families); if not, diagnose
    training-setup vs pool-texture before touching b3.
  goal_deltas_this_session:
    - b2 calibration loop: COMPLETED (was the session's assigned task).
    - Two new families (construct, revise) designed and built after a
      user-requested brainstorm; user delegated their construction; then
      user RE-ORDERED: GRPO first, b3 paused at guarded shape.
  bindings:
    - "Substrate policy (OPERATIONS.md): GRPO consumes bf16 HF. The pool
       is q8-banded → the first unsteered GRPO rollout batch IS the
       re-verification-by-regeneration pass; prompts that drop out of
       band leave the pool, no special pleading."
    - "Existing rows/artifacts immutable. Never mutate
       data/acegen_live_b2.jsonl (sha256-sidecar'd)."
    - "No llama-* systemd services; the two nohup llama-servers
       (:3091/:3092) are the only allowed llama processes and they
       occupy both GPUs — GRPO needs those VRAM budgets, so plan the
       server shutdown WITH the user or per their next instruction
       before killing anything (gather state first: ps/nvidia-smi/ss)."
    - "workin-hard skill (~/.pi/agent/skills/workin-hard/SKILL.md) is the
       user-endorsed working protocol for this kind of long autonomous
       run — load it."
    - "User working preferences: autonomous; quiet outputs to preserve
       context; explicit checkpoint reports per iteration (worked /
       didn't / next); realism and diversity over metric-passing
       ('assessment pass, ship it' is a named failure mode); honest
       numbers with sources, loud self-corrections."
    - "Do not commit without explicit user request. Worktree is
       deliberately uncommitted."

world_state_delta:
  - "b2 DONE: 46-prompt LIVE pool (band spread 1/8–7/8), all six
     families calibrated, evidence 504 rollouts in
     data/acegen_b2_*_gpu*.jsonl + data/b2/. Docs written back:
     CALIBRATION.md (pool status + knob/structure→difficulty map),
     STATUS.md R8, four round lab notes + methodology note."
  - "b3 shape exists and is GUARDED but UNPROBED: gen/construct.py
     (bounded sequence construction + transition-cost budget, NONE via
     before-cycles) and gen/revise.py (two-stage evidence revision,
     planted decoy clerk-note + foreign-account fragment). Registered
     in generate.py presets and calibrate.py CHECK. Guards passed:
     self-test 16/16, swap-fail 12/12, 30-seed controls. Full resume
     procedure: lab_notes/2026-08-25-b3-shape-resume-here.md."
  - "PID allocation: b1=1–48, b2=49–163. Next free pid = 164 (b3
     probes), never reuse."
  - "Skill installed globally: ~/.pi/agent/skills/workin-hard/SKILL.md."

negative_knowledge:
  - "Numeric size knobs buy budget pressure, not reasoning pressure
     (grid n_pos 6→7 produced budget-bound 0/8s with correct solutions
     derived in-think). Difficulty comes from structure: withhold
     spoilers, remove giveaway anchors, plant excludable distractors."
  - "On q8, wrongs are overwhelmingly truncations; classify failure
     species (emission paralysis / closure loop / budget exhaustion /
     committed error) before reacting — the classifier pattern is in
     lab_notes/2026-08-25-b2-methodology.md §3."
  - "machine is knob-maxed; length-only hardening rejected as
     cap-grinding. assign's delayed decoy never bites (model validates
     all rules first); its failures are closure loops."
  - "adversary shifted 71.9% (bf16, b1) → 43.8% (q8, b2) — larger than
     the delta smoke predicted; depth-4 witness search is
     substrate-sensitive. Watch this at GRPO re-verification."
  - "Tooling traps: pkill self-kill (use bracket trick); stray
     /tmp/*.py shadows stdlib modules when cwd scripts import them;
     verify file state after every edit call (a spurious-edit bug
     corrupted two files once; py_compile caught it)."

operational_state:
  llama_servers: >
    nohup llama-server ×2 on :3091 (GPU0, desktop-resident card,
    ~15.8 GB used) and :3092 (GPU1), model
    /home/admin/models/Qwen3.5-9B-MTP-Q8_0.gguf, q8_0 + MTP, parallel
    4. Health was OK at last check but RE-CHECK before relying:
    curl -s http://127.0.0.1:309{1,2}/health; nvidia-smi.
  collectors: none running. GPUs idle apart from the servers.
  vcs: >
    Uncommitted: b2+b3 generator changes, docs, lab notes, data
    artifacts under mlfactory/experiments/ace/ plus pre-existing
    unrelated worktree changes (deletions etc.) that are NOT mine.
    User has not authorized a commit; stage only this session's paths
    if asked.
  next_commands:
    - "Read lab_notes/2026-08-25-b3-shape-resume-here.md (b3 is parked)."
    - "For GRPO: read PHASES.md, docs/TRAINING_STACK.md (GPU memory,
       smoke ladder, OOM, objective safety), core/steering_controller.py,
       train/grpo.py (run as .venv/bin/python -m
       mlfactory.experiments.ace.train.grpo from the ace dir), and
       STATUS.md Q10 for the sharpened success bar: the controller must
       beat a well-tuned constant lambda on the pre-allocated
       reasoning-length axis on forked outcomes, not just a no-op."
    - "GRPO runs on bf16 HF → needs GPU VRAM; the q8 servers likely
       must stop first. Confirm the plan with the user before killing
       them (they said they'd check back before GRPO work proceeds:
       'I'll check back in with you in a bit')."

open_questions:
  - "Q10: can the controller learn a nontrivial state-dependent
     intervention from terminal reward alone? Current bet: unknown —
     first run learned a weak bias on a too-easy substrate; the b2 pool
     fixes the substrate half. This GRPO attempt is the test."
  - "Whether pool texture is sufficient or b3 families are needed —
     exactly what the gate decides; genuinely uncertain."
  - "bf16 memory shape for GRPO rollouts on 2×24 GB with 26k-token
     traces and 46 prompts — TRAINING_STACK.md governs; check before
     promising ETAs."

pointers:
  b2_assignment_spec: specs/b2_hone_assignment.md
  b2_methodology: lab_notes/2026-08-25-b2-methodology.md
  b3_resume: lab_notes/2026-08-25-b3-shape-resume-here.md
  pool: data/acegen_live_b2.jsonl (+ .meta.json caveats)
  calibration_map: CALIBRATION.md
  status_router: STATUS.md
  substrate_policy: OPERATIONS.md
  training_rules: /home/admin/mlfactory/docs/TRAINING_STACK.md
