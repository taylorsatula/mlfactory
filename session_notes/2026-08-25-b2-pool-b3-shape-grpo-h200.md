# Session notes — 2026-08-25 — b2 pool landed, b3 shape guarded, GRPO moves to H200

Scope: b2 iterative honing to a 46-prompt LIVE pool; two new families
(construct, revise) built and guarded, paused at the GRPO gate; GRPO
training moved to a Vast H200; forecasting smoke completed.

## What exists now that didn't before

- **46-prompt LIVE pool** `data/acegen_live_b2.jsonl` (sidecar'd,
  sha256'd; band spread 1/8–7/8 across six families, q8_0+MTP).
  Evidence rollouts: `data/acegen_b2_{r1..r4,pool}_gpu{0,1}.jsonl`
  (504 rows), candidates `data/b2/`. Commits e3f039d / 4dd181b / 0e83090.
- `gen/construct.py`, `gen/revise.py` (b3, registered, C4-guarded:
  self-test 16/16, swap-fail 12/12, 30-seed controls; UNPROBED — resume
  at `lab_notes/2026-08-25-b3-shape-resume-here.md`, pids from 164).
- Lab notes: b2 r1 / r2 failure-species-taxonomy / r3-r4 grid /
  final-pool / methodology; b3 resume; grpo-h200 setup + smoke-results;
  two handoffs.
- `train/smoke_h200.py` (remote `/workspace/mlfactory`); remote smoke
  report `/workspace/smoke1/` (Vast #46911241, nothing there persists).
- Global skill `~/.pi/agent/skills/workin-hard/SKILL.md` (autonomous
  long-horizon protocol); `session-handoff` skill gained the checkpoint
  timeline as a second artifact.

## Findings (the durable knowledge)

1. **Structure beats numeric knobs.** grid's lever is clue composition
   (`max_at` caps giveaway at-clues: at≤1 LIVE 5/6, at≥2 DEAD-EASY);
   hypothesis's is VOIDED-record structure (first committed-wrong
   failures); certify's is trap non-announcement. Numeric size axes
   (spread, n_sales, n_items) mostly bought budget pressure, not
   reasoning (CALIBRATION.md knob→difficulty map).
2. **On q8, wrongs are overwhelmingly truncations.** Failure-species
   classifier before reacting: emission paralysis / closure loop /
   budget exhaustion / committed error — counts look identical, fixes
   differ (b2-r2 taxonomy).
3. **bf16 regime on H200 (smoke, R9):** thinking traces median **22.3k**
   (longer than q8's 18.4k); generation 73 tok/s aggregate batch-4;
   replay with gradients fits **≤8k window** (112 GB peak; 16k OOMs);
   gradient path finite, frozen base holds. 6/24 traces cap-hit →
   truncation is a backdoor length term under terminal reward
   (REWARD_POLICY caveat written).
4. **Kernel route exhausted (for now).** flash-attn 2.8.3 sm_90 and
   FLA+causal-conv1d both measured zero gain vs SDPA (~150 tok/s
   batch-4 ≈ 15% of the bandwidth ceiling) — generation is not
   bandwidth-bound; remaining levers are 2-GPU rollout parallelism and
   possibly vLLM for the unsteered arm; TP=2 low-bet over PCIe (Q11).
5. **Segmented replay is a hypothesis requirement**, not memory
   plumbing: the learnable decision points (reheat, durable pruning)
   sit mid/late trace; an 8k prefix window can't reach them
   (PHASES.md Phase 2 write-back).
6. **Thinking-on ruling:** grpo.py as committed ran thinking-off /
   640-token / arithmetic-set — the wrong regime for the explore→prune
   hypothesis; overruled for this attempt (HYPOTHESIS re-read).

## Decisions with rationale

- **GRPO first, b3 paused** (user ruling): the controller gate decides
  whether the pool needs the new families, not the other way around.
- **Training on Vast H200** (user ruling): 3090s lack the VRAM for bf16
  + optimizer + 26k traces. Local boxes stay collection-only.
- **Slice before full pool; first unsteered batch doubles as the bf16
  re-verification** (OPERATIONS substrate policy condition 3).
- **Machine/certify locked at knob-max;** length-only hardening
  rejected as cap-grinding. Emission-paralysis work (Q2/Q3) deferred —
  bf16 re-verification re-measures it.
- **adversary 71.9%→43.8% q8 shift** noted as larger than the delta
  smoke predicted; depth-4 witness search is substrate-sensitive —
  watch at re-verification.

## Environment traps encountered

- **pkill -f self-kill over ssh**: the remote bash command line contains
  the pattern text; bracket tricks don't help when the pattern string
  itself appears later in the same command. Write a script file or pkill
  by exact name.
- **flash-attn env var is `FLASH_ATTN_CUDA_ARCHS`** (not
  FLASH_ATTENTION_CUDA_ARCHS); wheels are torch-version-locked (none
  for torch 2.13) → source build; restrict archs or pay 4× build time.
- **vLLM-template instances**: `supervisorctl stop vllm` can orphan the
  actual `vllm serve` process — check nvidia-smi and kill by pid.
- **HF cache layouts differ by writer**: vllm `--download-dir` puts
  `models--X` at the dir root; `snapshot_download` with HF_HOME uses
  `$HF_HOME/hub/`. A snapshot dir can hold weights but no config.json —
  verify before trusting.
- **Edit-tool discipline**: verify file state after every edit call;
  py_compile after multi-edit batches (a malformed match once failed
  silently-looking but real).
- Remote smoke: `/tmp/*.py` name hygiene (module-shadowing), tmux for
  every long job, named logs, GPU state checked before and after.

## State at note time

- Smoke complete; report parsed; docs written back (REWARD_POLICY,
  STATUS R9/Q11, ENVIRONMENT, OPERATIONS, CALIBRATION preview, PHASES).
- Remote: model + repo + kernels in place; vllm services stopped; old
  duplicate download dir pending cleanup; flashbuild/conv1dbuild done.
- Next: user guidance on scope/budget, then wire production grpo.py
  (thinking on, pool adapter, segmented replay ≤8k windows, per-sample
  rollout rows, 2-GPU split), 2-iter dry run, then the attempt.
