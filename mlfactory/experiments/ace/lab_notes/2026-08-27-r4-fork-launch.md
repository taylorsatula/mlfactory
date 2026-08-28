# 2026-08-27 — R4 fork run LAUNCHED (2× A6000, 4 processes, FLASH_ATTENTION)

> Run note. Predecessor: `2026-08-27-handoff-r4-build-inflight.md`
> (superseded as the live state doc by this note + the attendance
> log). Attendance runbook: `annotate/runbook_r4_cancun.md`.
> Attendance log: `annotate/out/r4_attendance.log`.

## Facts

- **Launched 2026-08-27 21:40–21:42 UTC** (programs staggered ~40s).
- **Boxes** (both image `vastai/llama-cpp:b10182-cuda-12.9`, disk 200,
  created WITH the SSH-permission-repair `--onstart-cmd` — see runbook
  Appendix A for the mandatory string):
  - A: instance **48921611**, Kansas US, RTX A6000 48GB, $0.456/hr,
    `ssh r4fork-a` (root@209.137.198.14:28119)
  - B: instance **48921612**, Delaware US (dl 8.8Gbps), RTX A6000 48GB,
    $0.511/hr, `ssh r4fork-b` (root@38.29.145.10:40340)
  - (First pair 48916215/48916245 died to a Vast-side
    authorized_keys-permissions defect; replaced, not reused.)
- **Programs** (supervisor, `autostart=false autorestart=false
  stopasgroup=true`, logs `/tmp/r4fork<N>.log|.err`, scripts
  `/root/r4fork_run<N>.sh`):
  - A/r4fork1: 7 states → 504 rows: r4_muse_00,r4_cycle_09,r4_cycle_02,
    r4_loop_09,r4_loop_04,r4_muse_01,r4_loop_02
  - A/r4fork2: 6 states → 432 rows: r4_muse_04,r4_cycle_08,r4_muse_06,
    r4_cycle_03,r4_cycle_05,r4_cycle_07
  - B/r4fork1: 7 states → 504 rows: r4_loop_07,r4_loop_08,r4_loop_05,
    r4_loop_06,r4_cycle_00,r4_cycle_06,r4_loop_01
  - B/r4fork2: 7 states → 504 rows: r4_loop_00,r4_cycle_01,r4_muse_05,
    r4_loop_03,r4_muse_03,r4_cycle_04,r4_muse_02
  - Each program: `--run --sub-batch 1 --only <states>`, output
    `/root/mlfactory/mlfactory/experiments/ace/data/fork_r4_results_<N>.jsonl`,
    env `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- **First row:** B/r4fork1 21:53Z — r4_cycle_00 noop, 8420 tok in 656s
  (13 tok/s) at fork 17580 (the longest fork in the plan) — the
  previously impossible long-fork + co-resident case, working.
- **Total:** 1944 rows (27 states × 3 arms × 24 seeds); expected
  wall-time ~1.5–2 days at the measured rates; spend at kickoff
  $0.967/hr combined; budget cap $150.

## Design deviations from the pre-launch freeze (recorded, justified)

1. **Backend MATH → FLASH_ATTENTION.** MATH is deterministic but
   materializes q×kv: prefill spikes to ~34GB (2× steady), which makes
   two-process co-residency OOM on every fork ≳9.5k state (11 of 27
   states have fork >8k). FLASH was proven deterministic for this exact
   generation pattern: two concurrent processes, same seed, 4096 new
   tokens → identical SHA256 (2026-08-27, box B). Same-seed 512-token
   runs also bit-identical; different seeds differ. EFFICIENT_ATTENTION
   has no available kernel on this stack. Numerics differ from MATH
   captures-era runs (irrelevant: the run is self-consistent; paired
   arms share backend, seeds, and everything else).
2. **Two batch-1 processes per box** instead of one batch-4 process:
   decode on this hybrid model is launch-overhead + bandwidth bound;
   batching anti-scales (b1 19.6 → b2 15.7 → b4 10.3 tok/s aggregate,
   measured), two time-sliced processes measured 1.85× one.
3. **batch_seed = sha256(state_id)-derived**, replacing
   `hash(state_id)`: Python str-hash is per-process randomized
   (PYTHONHASHSEED) — the old formula silently broke cross-process and
   cross-resume seed pairing (caught by the determinism check).
4. Harness OOM guards: model-load retries (3×, 30/60s backoff),
   batch-1 generation retries (2×, 25/60s backoff, loud raise after),
   periodic decode-phase `empty_cache` hook (every 128 steps),
   post-generate `empty_cache`. All memory-maintenance only — no
   numerics effect.
5. Per-program result files (`fork_r4_results_1/2.jsonl`) to avoid
   concurrent-append interleaving; merged + deduped at completion
   (runbook §Completion).

## Box setup commands as executed (reference for Appendix-A rebuilds)

```bash
# LOCAL — project transfer (per box; ~209MB):
rsync -az --exclude '.git/' --exclude '.venv' --exclude '__pycache__' \
  --exclude 'runs/' --exclude '.mlfactory/' --exclude 'annot_captures*' \
  --exclude 'ace-legacyapproach/data/' \
  /home/admin/mlfactory/ r4fork-<x>:/root/mlfactory/

# BOX — venv + pinned deps (requirements_r4_box.txt is in the project
# root; torch installs from the cu128 extra-index-url line inside it):
python3 -m venv /root/venv-r4
/root/venv-r4/bin/python -m pip install -r /root/mlfactory/requirements_r4_box.txt
/root/venv-r4/bin/python -m pip install pydantic==2.13.4 \
  flash-linear-attention==0.5.2   # pydantic via core/manifest; fla for the DeltaNet fast path

# causal-conv1d has NO wheel for this torch/CUDA combo and source-build
# fails on a toolkit-version check — copy the binary package from the
# jump host's ace venv:
SP=/home/admin/mlfactory/mlfactory/experiments/ace/.venv/lib/python3.12/site-packages
rsync -az $SP/causal_conv1d $SP/causal_conv1d-1.7.0.dist-info \
  $SP/causal_conv1d_cuda.cpython-312-x86_64-linux-gnu.so \
  r4fork-<x>:/root/venv-r4/lib/python3.12/site-packages/

# BOX — model (public repo, hf_xet; 19GB in ~20s on box B):
HF_HOME=/root/.hf_home /root/venv-r4/bin/hf download Qwen/Qwen3.5-9B
mkdir -p /home/admin/models/hf   # MODEL_PATH is hardcoded to this path
ln -sfn /root/.hf_home/hub/models--Qwen--Qwen3.5-9B/snapshots/<snap> \
  /home/admin/models/hf/Qwen3.5-9B

# BOX — supervisor (the image ships the binary but does NOT start it;
# PID1 is bash):
supervisord -c /etc/supervisor/supervisord.conf
supervisorctl stop llama          # template's idle server (was not running
                                  # on these boxes; check nvidia-smi anyway)
# then drop /root/r4fork_run{1,2}.sh + /etc/supervisor/conf.d/r4fork{1,2}.conf
# (contents recorded in the runbook Topology / Facts above), then:
supervisorctl reread && supervisorctl update
```

## Verification evidence behind the launch decision

- Local smoke (3090, GPU1): 24/24 rows, 23/24 strict-verifier correct,
  16/16 steered-vs-noop text pairs differ (hook live). `data/fork_r4_smoke.jsonl`.
- Remote smoke (box B, pre-launch): r4_loop_02 m=1 → 3 rows correct.
- Determinism (box B): FLASH same-seed concurrent 4096-token runs →
  identical SHA256; same-seed 512-token runs identical; different seeds
  differ.
- Co-residency gate (box A): long-fork process + twin, zero OOM
  retries under FLASH, both generating (35.8GB resident total, 98% util).
- Seed stability: `hash()` bug found via bit-mismatch between two
  processes' same-state rows; fixed to sha256; re-verified bit-identical.

## Decisions

- Attendance handed to the GLM agent per `runbook_r4_cancun.md`
  (3-hourly sampling; small mechanical corrections only; design frozen;
  $150 cap). Analysis of results is reserved for the main session after
  the principal's return.
- Status docs to update on completion: STATUS.md (R4 resolved row →
  COUNTERFACTUAL_FRAMEWORK/OBSERVABLES write-backs), TERMINAL_FORK_COMPUTE.md
  (measured tok/s vs scenario-F estimate).
