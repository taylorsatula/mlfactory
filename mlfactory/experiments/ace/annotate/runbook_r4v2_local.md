# Attendance runbook — R4v2 window-fork run, LOCAL GPUs (unsupervised)

> You are tending the ACE R4v2 fork run on the local machine while the
> principal is away. It generates 1944 fork-continuation rows
> (27 states × 3 arms × 24 seeds, 2048-token windows) on the two local
> RTX 3090s. The design is frozen and the harness is resume-safe. Your
> job: sample the run every 3 hours, keep the attendance log, make
> small mechanical corrections when something drifts — and nothing
> else. Principal's ruling, verbatim: **"make small adjustments to get
> the run back on track but avoid scope creep or sweeping changes that
> affect the final endstate goal."** When in doubt, stop and write what
> happened — never improvise anything that changes the design or
> touches something this doc doesn't name.

## ⚠️ CONTEXT PRESERVATION — READ THIS FIRST

**You are running for days. Context window is your scarcest resource.**

1. **Spawn subagents for investigation and repair.** When something
   needs debugging (log inspection, process diagnosis, restart
   verification), spawn a subagent to do it. Do NOT investigate in
   your main context. The subagent returns a terse verdict; you act
   on it.
2. **Subagent model: use YOUR model, not qwen 3.7-plus.** The
   `AGENTS.md` default is `qwen/qwen3.7-plus` — override it. Pass
   your own model via the Agent tool's `model` parameter.
3. **Prefer concise tool outputs.** Use `tail -N`, `head -N`,
   `wc -l`, `grep -c`, `--format=csv,noheader` — anything that
   returns a short summary instead of a firehose. Do NOT `cat` large
   files. Do NOT dump full logs. Do NOT read files you don't need.
4. **Attendance log entries: one line per sample.** No prose. No
   narration. The format in §Sampling protocol is the entire entry.

## Topology

```yaml
process_gpu0:
  name: r4v2-gpu0
  gpu: 0 (RTX 3090, 280W cap; desktop compositor resident, ~0.3 GB)
  pid_file: /home/admin/mlfactory/mlfactory/experiments/ace/logs/r4v2_gpu0.pid
  out:  /home/admin/mlfactory/mlfactory/experiments/ace/data/fork_r4v2_run_gpu0.jsonl
  log:  /home/admin/mlfactory/mlfactory/experiments/ace/logs/r4v2_gpu0.log
  states: 13 (936 rows)
  only: r4_cycle_00,r4_muse_06,r4_loop_04,r4_muse_02,r4_cycle_04,
        r4_loop_08,r4_loop_07,r4_loop_06,r4_cycle_05,r4_loop_02,
        r4_cycle_06,r4_loop_03,r4_muse_03
process_gpu1:
  name: r4v2-gpu1
  gpu: 1 (RTX 3090, 280W cap)
  pid_file: /home/admin/mlfactory/mlfactory/experiments/ace/logs/r4v2_gpu1.pid
  out:  /home/admin/mlfactory/mlfactory/experiments/ace/data/fork_r4v2_run_gpu1.jsonl
  log:  /home/admin/mlfactory/mlfactory/experiments/ace/logs/r4v2_gpu1.log
  states: 14 (1008 rows)
  only: r4_muse_01,r4_loop_05,r4_loop_01,r4_cycle_08,r4_cycle_03,
        r4_loop_00,r4_muse_05,r4_cycle_01,r4_muse_00,r4_muse_04,
        r4_cycle_07,r4_cycle_02,r4_loop_09,r4_cycle_09
harness: annotate/fork_r4v2.py (FROZEN after the 2026-08-28 smoke —
  never edit it, never pass arguments other than the restart command)
plan: /home/admin/mlfactory/artifacts/fork_plan_r4.jsonl (frozen)
model: bf16 HF Qwen3.5-9B at /home/admin/models/hf/Qwen3.5-9B,
  FLASH_ATTENTION sdpa (bit-deterministic on one machine)
rate_baseline: 55-63 s/row measured at 280W (worst state r4_cycle_00,
  fork 17580: 62s; short forks ~55s). ~52s/row average.
eta: ~13.5h gpu0 + independent ~14.5h gpu1 -> done in ~15h from launch
run_end_state: 936 + 1008 = 1944 rows, keys (state_id, arm, seed_i)
  unique, arms noop/toward_healthy/toward_diverge
```

## Binding rules (violating these fails the run)

- **The design is frozen.** Never change the plan, arms, seeds, lam
  values, focal layers, WINDOW=2048, TAIL=512, temperature 0.8 /
  top_p 0.95, the model, the backend, or the harness. Never pass
  `fork_r4v2.py` arguments other than the exact restart command in
  §Permitted adjustments.
- **Do NOT touch:** git (no commits/pushes), the detection artifacts
  (`data/annot_captures*`, `data/probe_results*`,
  `data/steering_directions/`), the pilot/judge files
  (`data/fork_r4v2_pilot*.jsonl`, `data/judge_r4v2_*.jsonl`,
  `data/fork_r4v2_smoke.jsonl`, `data/fork_r4v2_equivalence.jsonl`),
  the v1 partial results (`data/r4fork-*`), anything in other
  experiments, any `llama-*` service except the single restore step
  in §Completion. Delete nothing, ever.
- **No analysis.** Judging, aggregation, and lab notes beyond the
  attendance log and abort/complete notes are the main session's work.
- The desktop uses GPU0 continuously (~0.3 GB); that is expected.
  Never start other GPU jobs on either card.
- Use absolute paths in every redirect and nohup. Kill by the PID in
  the pid file only — the two processes share a command line, so any
  pattern kill hits both.

## Sampling protocol

Every 3 hours, including overnight. All checks run from this machine:

```bash
cd /home/admin/mlfactory/mlfactory/experiments/ace
# 1. processes alive
for p in r4v2_gpu0 r4v2_gpu1; do
  pid=$(cat logs/$p.pid 2>/dev/null);
  ps -o pid,etime,pcpu,comm -p "$pid" --no-headers 2>/dev/null \
    || echo "$p: DEAD";
done
# 2. GPU busy + power cap still 280
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,power.limit,temperature.gpu --format=csv,noheader
# 3. progress + log freshness
wc -l data/fork_r4v2_run_gpu*.jsonl
for p in r4v2_gpu0 r4v2_gpu1; do tail -2 logs/$p.log; done
```

Healthy means: both PIDs alive; both GPUs at high utilization (a
between-rows dip to 0 for a few seconds is normal); power limit 280 W;
row counts growing (~70 rows/process per 3h interval, ±50%); a log
line newer than 30 minutes in each log. One missed window is not an
escalation — recheck at the next sample first.

Append one attendance-log entry per sample
(`annotate/out/r4v2_attendance.log`, local, append-only):

```
<UTC time> | gpu0: <status> rows=<n> | gpu1: <status> rows=<n> | temps=<C0,C1> | <anomalies/actions or ok>
```

## Permitted adjustments (exhaustive list — anything else is forbidden)

1. **Process dead (PID gone):** relaunch with its exact command
   (below) — the run resumes; done rows are skipped. Log it. If it
   dies again before 10 new rows, read the last 30 lines of its log,
   record them in the attendance log, and wait for the next sample
   before trying once more. Two consecutive failed restarts → §Abort.
2. **Process alive but no new log line for 90+ minutes:** kill the PID
   from its pid file (`kill <pid>`, verify gone with `ps -p <pid>`),
   then relaunch. Log it. (The in-flight row regenerates identically
   seeded; rows are never duplicated.)
3. **Power limit not 280 W** (e.g. after a reboot where the service
   was edited back): `echo '4231' | sudo -S nvidia-smi -pl 280` —
   that's all; the service file itself is never edited by you. Log it.
4. **Disk pressure (df -h / below 20GB free):** delete only files in
   `/tmp`. Nothing else, anywhere.

Relaunch commands (exact):

```bash
cd /home/admin/mlfactory/mlfactory/experiments/ace
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CUDA_VISIBLE_DEVICES=0 nohup .venv/bin/python -m \
  mlfactory.experiments.ace.annotate.fork_r4v2 --run \
  --out /home/admin/mlfactory/mlfactory/experiments/ace/data/fork_r4v2_run_gpu0.jsonl \
  --only r4_cycle_00,r4_muse_06,r4_loop_04,r4_muse_02,r4_cycle_04,r4_loop_08,r4_loop_07,r4_loop_06,r4_cycle_05,r4_loop_02,r4_cycle_06,r4_loop_03,r4_muse_03 \
  </dev/null >> /home/admin/mlfactory/mlfactory/experiments/ace/logs/r4v2_gpu0.log 2>&1 &
echo $! > /home/admin/mlfactory/mlfactory/experiments/ace/logs/r4v2_gpu0.pid

CUDA_VISIBLE_DEVICES=1 nohup .venv/bin/python -m \
  mlfactory.experiments.ace.annotate.fork_r4v2 --run \
  --out /home/admin/mlfactory/mlfactory/experiments/ace/data/fork_r4v2_run_gpu1.jsonl \
  --only r4_muse_01,r4_loop_05,r4_loop_01,r4_cycle_08,r4_cycle_03,r4_loop_00,r4_muse_05,r4_cycle_01,r4_muse_00,r4_muse_04,r4_cycle_07,r4_cycle_02,r4_loop_09,r4_cycle_09 \
  </dev/null >> /home/admin/mlfactory/mlfactory/experiments/ace/logs/r4v2_gpu1.log 2>&1 &
echo $! > /home/admin/mlfactory/mlfactory/experiments/ace/logs/r4v2_gpu1.pid
```

Every adjustment gets an attendance-log entry: what, why, before/after
state. If an adjustment doesn't restore health within one sample
interval, stop acting and §Abort.

## Forbidden (explicit)

- Changing design constants, CLI args beyond the exact restart
  command, shard assignments, code, or data.
- Editing this runbook; the attendance log is the only file you add
  lines to (plus abort/complete notes named below).
- Editing the power-limit service file or any systemd unit.
- Starting or enabling any `llama-*` service before §Completion.
- Analysis of results, steering experiments, judge runs, restarting
  detection work, using the GPUs for anything else.
- Asking the model being served anything (monitoring reads
  process/log/device state only).

## Abort conditions (stop the run, preserve, wait for the principal)

- Two consecutive failed restarts of one process (§Permitted 1), or
  the same process needing a third restart in 24h.
- A log shows ≥3 OOMs (`grep -c OutOfMemory logs/r4v2_gpu*.log`), or
  any traceback that repeats across two samples.
- Disk <20GB after §Permitted 4, or any filesystem error.
- Wall time exceeds 3 days from launch (2× the ETA with margin).
- Anything requiring judgment this doc doesn't cover.

On abort: kill both PIDs (from pid files, verify gone); write
`lab_notes/<date>-r4v2-attendance-abort.md` with the attendance log
contents, what happened, and current row counts. Leave the output
files exactly as they are. Wait.

## Completion protocol (when BOTH logs print "run complete")

1. Verify locally:
   ```bash
   cd /home/admin/mlfactory/mlfactory/experiments/ace/data
   wc -l fork_r4v2_run_gpu*.jsonl          # 936 + 1008
   python3 -c "
   import json
   keys=set(); rows=0
   for f in ('fork_r4v2_run_gpu0.jsonl','fork_r4v2_run_gpu1.jsonl'):
       for l in open(f):
           r=json.loads(l); rows+=1
           k=(r['state_id'],r['arm'],r['seed_i'])
           assert k not in keys, ('duplicate', k)
           keys.add(k)
           assert r['arm'] in ('noop','toward_healthy','toward_diverge'), r
   print('rows',rows,'unique keys',len(keys))"   # want 1944 / 1944
   sha256sum fork_r4v2_run_gpu*.jsonl >> fork_r4v2_results_SHA256SUMS
   ```
2. Restore the machine to its pre-run state:
   ```bash
   echo '4231' | sudo -S systemctl enable --now llama-qwen38.service
   systemctl is-active llama-qwen38        # want: active
   echo '4231' | sudo -S sed -i 's/-i 0 -pl 280 && \/usr\/bin\/nvidia-smi -i 1 -pl 280/-i 0 -pl 363 \&\& \/usr\/bin\/nvidia-smi -i 1 -pl 363/' /etc/systemd/system/nvidia-3090-power-limit.service
   echo '4231' | sudo -S systemctl daemon-reload
   echo '4231' | sudo -S nvidia-smi -pl 363
   nvidia-smi --query-gpu=power.limit --format=csv,noheader  # 363, 363
   ```
   (The service's Description line may still say 280 W / R4v2 — leave
   it; it is a record of why the limit changed.)
3. Write `lab_notes/<date>-r4v2-forks-complete.md`: final row counts,
   wall time, any adjustments made, and STOP — no analysis. Analysis
   and judging are the main session's work.

## What success looks like

1944 window rows on disk and checksummed, the machine restored
(llama-qwen38 active, power limit 363 W), a complete attendance log,
and nothing touched that this doc didn't name.
