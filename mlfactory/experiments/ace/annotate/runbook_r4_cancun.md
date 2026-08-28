# Attendance runbook — R4 fork run, Cancun stretch (unsupervised, multi-day)

> You are tending the ACE R4 fork run while the principal is away. The
> run was launched by the main session (2026-08-27) on two rented Vast
> boxes; the design is frozen and the harness is resume-safe. Your job:
> sample the run every 3 hours (first three checks at ~45-minute
> intervals to confirm the kickoff is healthy), keep the attendance
> log, make small mechanical corrections when something drifts — and
> nothing else. Principal's ruling, verbatim: **"make small adjustments
> to get the run back on track but avoid scope creep or sweeping
> changes that affect the final endstate goal."** When in doubt, stop
> and write what happened — never improvise anything that spends money,
> changes the design, or touches something this doc doesn't name.

## ⚠️ CONTEXT PRESERVATION — READ THIS FIRST

**You are running for days. Context window is your scarcest resource.**

1. **Spawn subagents for investigation and repair.** When something
   needs debugging (log inspection, process diagnosis, box rebuild,
   etc.), spawn a subagent to do it. Do NOT investigate in your main
   context. The subagent returns a terse verdict; you act on it.

2. **Subagent model: use YOUR model, not qwen 3.7-plus.** The
   `AGENTS.md` default is `qwen/qwen3.7-plus` — override it. Pass
   your own model via the Agent tool's `model` parameter.

3. **Prefer concise tool outputs.** When you DO run commands directly
   (sampling checks, status queries), use `tail -N`, `head -N`,
   `wc -l`, `grep -c`, `--format=csv,noheader` — anything that
   returns a short summary instead of a firehose. Do NOT `cat` large
   files. Do NOT dump full logs. Do NOT read files you don't need.

4. **Attendance log entries: one line per sample.** No prose. No
   narration. The format in §Sampling protocol is the entire entry.

5. **Do NOT break the agent loop.** You are running autonomously with
   no one at the computer. If you break out of your loop (ask a
   question, request user input, wait for approval), you will stall
   because there is no one to respond. Continue the loop unless you
   hit a true edge case where continuation is impossible (e.g., both
   boxes destroyed, budget exhausted, abort condition triggered, bugs 
   that are unrepairable after an earnest try). In
   those cases, write a final log entry, transfer data from the remote
   box to local and power down (not destory) the instances, and stop.

## Topology

```yaml
box_a: {instance: 48921611, label: r4fork-a, gpu: 1x RTX A6000 48GB,
        Kansas US, dph: 0.456,
        ssh alias: r4fork-a (root@209.137.198.14:28119),
        programs: r4fork1 (states r4_muse_00,r4_cycle_09,r4_cycle_02,
          r4_loop_09,r4_loop_04,r4_muse_01,r4_loop_02 — 504 rows),
          r4fork2 (states r4_muse_04,r4_cycle_08,r4_muse_06,r4_cycle_03,
          r4_cycle_05,r4_cycle_07 — 432 rows)}
box_b: {instance: 48921612, label: r4fork-b, gpu: 1x RTX A6000 48GB,
        Delaware US (dl 8.8Gbps), dph: 0.511,
        ssh alias: r4fork-b (root@38.29.145.10:40340),
        programs: r4fork1 (states r4_loop_07,r4_loop_08,r4_loop_05,
          r4_loop_06,r4_cycle_00,r4_cycle_06,r4_loop_01 — 504 rows),
          r4fork2 (states r4_loop_00,r4_cycle_01,r4_muse_05,r4_loop_03,
          r4_muse_03,r4_cycle_04,r4_muse_02 — 504 rows)}
launch_time: 2026-08-27 21:40 UTC (r4fork-a/r4fork1 first; the other
  three programs staggered ~40s apart, all up by 21:42 UTC)
baseline_rows_per_hour_per_box: PROVISIONAL 20-30 (sum of the box's
  two programs; first measured row: r4_cycle_00 noop 8420 tok in
  656s at 13 tok/s — continuations vary 500-15000+ tokens, so the
  rate is continuation-length-limited, not token-rate-limited).
  Recalibrate from observed deltas at the first 3h sample; until
  recalibrated, healthy = process alive + GPU busy + a new log line
  within 30 min>
parallelism_model: two concurrent batch-1 processes per box. This is
  deliberate: decode on this model is launch-overhead + bandwidth
  bound, batched generation is SLOWER per token than batch-1, and two
  time-sliced processes measured ~1.85x one process. Never raise
  --sub-batch above 1, never run a third GPU process on a box.
remote_layout: project /root/mlfactory; results per program
  /root/mlfactory/mlfactory/experiments/ace/data/fork_r4_results_<1|2>.jsonl;
  supervisor programs `r4fork1`/`r4fork2`; logs /tmp/r4fork<1|2>.log
  and .err
local_jump_host: this machine (192.168.1.9). SSH aliases are configured
  in ~/.ssh/config — use `ssh r4fork-a` / `ssh r4fork-b` and
  `rsync ... r4fork-a:<remote path> <local path>`. (Underlying key:
  ~/.ssh/id_vast. If an alias stops resolving after a box replacement,
  update ~/.ssh/config from `vastai ssh-url <new instance id>`.)
budget_cap_usd_total: 200   # both boxes + any replacement combined
run_end_state: 1944 result rows total (27 states x 3 arms x 24 seeds)
  across the four program files (504/432/504/504); rows are
  resume-keyed (state_id, arm, seed_i)
```

## Binding rules (violating these fails the run)

- **The design is frozen.** Never change any of: the plan
  (`artifacts/fork_plan_r4.jsonl`), arms, lam values, m=24, seeds,
  focal layers (CYCLE L18 / LOOP L2 / MUSE L17), temperature 0.8 /
  top_p 0.95, the 26000-token backstop, the FLASH_ATTENTION sdpa
  backend (determinism proven 2026-08-27: concurrent same-seed runs
  bit-identical at 4096 tokens; the MATH backend is also
  deterministic but its prefill spike OOMs two-process co-residency
  on fork >~9.5k states — do not switch back), the
  model, or the verification path. Never pass `fork_r4.py` arguments
  other than the exact restart command in §Permitted adjustments.
  Never edit `fork_r4.py` or any experiment code.
- **Money:** no new rentals except the single replacement path in
  §Permitted adjustments; combined spend ceiling $200; every attendance
  action that could cost money gets a log entry with the reason.
- **Do NOT touch:** Vast instance 48783410 (stopped, reserved), any
  `llama-*` systemd service on the local machine, the local GPUs (the
  desktop uses GPU0; do not start local jobs), git (no
  commits/pushes/reverts), the detection artifacts
  (`data/annot_captures*`, `data/probe_results*`,
  `data/steering_directions/`), anything in other experiments. Delete
  nothing, ever.
- Use absolute paths in every redirect and nohup (the shell's cwd is
  not reliable between commands). Kill by explicit PID only; verify
  effects after stop/start.

## Sampling protocol

First three checks at ~45-minute intervals (kickoff confirmation),
then every 3 hours, including overnight. Each sample, from THIS
machine:

```bash
# 1. boxes alive + spend
vastai show instances --raw | python3 -c "
import json,sys
for r in json.load(sys.stdin):
    print(r.get('label'), r.get('id'), r.get('actual_status'),
          '\$', r.get('dph_total'))"

# 2. per box (repeat for r4fork-a and r4fork-b; each box runs TWO
#    programs):
ssh r4fork-a '
  supervisorctl status r4fork1 r4fork2;
  tail -3 /tmp/r4fork1.log; tail -3 /tmp/r4fork2.log;
  wc -l /root/mlfactory/mlfactory/experiments/ace/data/fork_r4_results_*.jsonl;
  nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader'
```

Healthy means, for EACH box: instance status `running`; both programs
`RUNNING` (or `EXITED` with the final "run complete" line only at the
very end — §Completion); a log line newer than 30 minutes in each
program's log; row counts growing at roughly the baseline rate
(±50%); GPU memory ~40+ GB used (two resident models) and
utilization high. One missed log line window is not an escalation —
recheck at the next sample first.

Append one attendance-log entry per sample
(`annotate/out/r4_attendance.log`, local, append-only):

```
<UTC time> | A: <status> rows=<n> rate=<rows/hr> | B: <status> rows=<n> rate=<rows/hr> | spend=$<est> | <anomalies/actions or ok>
```

(Note: results are automatically checkpointed to the local machine every 2 hours via cron. You don't need to rsync manually.)

Spend estimate: instance uptime (hours) × dph, summed over both boxes
plus any replacement. If you can't get uptime, bound it: hours since
`launch_time` × dph.

## Permitted adjustments (exhaustive list — anything else is forbidden)

1. **Worker crashed or exited without "run complete":** on the box:
   `supervisorctl start r4fork<N>` (the run resumes — done rows are
   skipped). Log it. If it exits again immediately, read
   `/tmp/r4fork<N>.err` and `.log` tails, log the last 30 lines in the
   attendance log, and wait for the next sample before trying once
   more. Two consecutive failed restarts → §Abort.
2. **Worker alive but no new log line for 90+ minutes:** find the PID
   (`supervisorctl pid r4fork<N>`), `kill <that PID>`, confirm gone,
   then `supervisorctl start r4fork<N>`. Log it. (The in-flight
   continuation is lost and regenerates identically-seeded; rows are
   never duplicated.)
3. **Box dead/offline ≥2 hours** (status `offline`/`unknown`, or SSH
   refused repeatedly): destroy it (`vastai destroy instance <id> -y`
   — results are already in the attendance log as row counts; rows on
   the dead box are lost and its shard reruns from zero), then rebuild
   ONE replacement via Appendix A and restart that box's exact shard
   command. Log it. At most one replacement per box over the trip.
4. **Disk pressure on a box (df < 20GB free):** delete only files in
   `/tmp` on that box. Nothing else, anywhere.

Every adjustment gets an attendance-log entry: what, why, before/after
state. If an adjustment doesn't restore health within one sample
interval, stop acting and §Abort.

## Forbidden (explicit)

- Changing design constants, CLI args beyond the exact restart
  command, shard assignments (`--only` lists), code, or data.
- Editing this runbook except: Topology fill-in fields (before the
  first sample) and appending nothing else. The attendance log is the
  only file you add lines to.
- New rentals beyond the single replacement path; bidding; changing
  box images; resizing disks; stopping the run "to be safe" (a running
  healthy run is left running).
- Analysis of results, writing lab notes beyond abort notes, steering
  experiments, restarting detection work, touching the local GPUs.
- Asking the model being served anything (never probe the run by
  generating samples — monitoring reads process/log/device state only).

## Abort conditions (stop the run, preserve, wait for the principal)

- Spend estimate reaches **$200**.
- Wall time exceeds **6.5 days** from `launch_time`.
- Two consecutive failed restarts on one box (§Permitted 1), or the
  same box needing a second rebuild.
- A box logs ≥3 OOMs at batch 1 (search `/tmp/r4fork1.log` and
  `/tmp/r4fork2.log` for
  `OutOfMemory`), or any traceback that repeats across two samples.
- Disk <20GB on a box after §Permitted 4, or any filesystem error.
- Anything requiring judgment this doc doesn't cover.

On abort: `supervisorctl stop r4fork1 r4fork2` on each box; rsync all result
files home (Completion step 1); then `vastai stop instance <id>` on
each box (STOP, not destroy — disk keeps state cheaply); write
`lab_notes/<date>-r4-attendance-abort.md` with the attendance log
contents, what happened, and current row counts. Wait.

## Completion protocol (when ALL FOUR programs print "run complete")

1. Final checkpoint — results are automatically synced every 2 hours via cron. Run one final sync to capture the last rows:
   ```bash
   /home/admin/mlfactory/mlfactory/experiments/ace/data/checkpoint_sync.sh
   ```
2. Verify locally (merge + dedup check across the four files):
   ```bash
   cd /home/admin/mlfactory/mlfactory/experiments/ace/data
   wc -l r4fork-a/fork_r4_results_*.jsonl r4fork-b/fork_r4_results_*.jsonl
   python3 -c "
   import json, glob
   keys=set(); rows=0
   for f in glob.glob('r4fork-*/fork_r4_results_*.jsonl'):
       for l in open(f):
           r=json.loads(l); rows+=1
           k=(r['state_id'],r['arm'],r['seed_i'])
           assert k not in keys, ('duplicate', k)
           keys.add(k)
           assert r['arm'] in ('noop','toward_healthy','toward_diverge'), r
   print('rows',rows,'unique keys',len(keys))"   # want 1944 / 1944
   sha256sum r4fork-*/fork_r4_results_*.jsonl >> fork_r4_results_SHA256SUMS
   ```
3. Verify the same sha256 for the remote copies (record both in the
   attendance log).
4. `vastai stop instance 48921611` and `vastai stop instance 48921612`
   (STOP — the principal decides on destroy). Confirm status `stopped`.
5. Write `lab_notes/<date>-r4-forks-complete.md`: final row counts,
   total spend (from the attendance log), wall time, any adjustments
   made, and STOP — no analysis. Analysis is the main session's work.

## Appendix A — rebuilding a replacement box (only via §Permitted 3)

1. Search (from this machine):
   ```bash
   vastai search offers 'gpu_ram>=44 gpu_ram<=50 num_gpus==1 rentable=true verified=true reliability>0.985' \
     -o 'dph_total' --storage 200 --limit 15 --raw
   ```
   Pick the cheapest with `compute_cap` in (800, 860, 890, 900),
   `inet_down` ≥ 300 Mbps, dph ≤ $0.70, `duration` > 7 days.
2. Create with the SSH-permission repair baked in. **This `--onstart-cmd`
   is MANDATORY** — the 2026-08-27 boxes came up with
   `/root/.ssh/authorized_keys` at wrong ownership/modes, and sshd's
   server-side StrictModes refused every key (log signature:
   `Authentication refused: bad ownership or modes for file
   /root/.ssh/authorized_keys`). The onstart fixes permissions and
   keeps them fixed:
   ```bash
   ONSTART='mkdir -p /root/.ssh; grep -qF vast-dft-20260726 /root/.ssh/authorized_keys 2>/dev/null || echo "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAID+EYvEp2UOqpXCYUfdsZt7fZc+6cuAPQuU31/Owlp95 vast-dft-20260726" >> /root/.ssh/authorized_keys; nohup bash -c "while true; do chown -R root:root /root/.ssh 2>/dev/null; chmod 700 /root/.ssh 2>/dev/null; [ -f /root/.ssh/authorized_keys ] && chmod 600 /root/.ssh/authorized_keys 2>/dev/null; sleep 30; done" >/dev/null 2>&1 &'
   vastai create instance <offer_id> --image vastai/llama-cpp:b10182-cuda-12.9 \
     --disk 200 --ssh --direct --label r4fork-<a|b>-r1 \
     --onstart-cmd "$ONSTART" --raw
   ```
   — if the response has `new_contract`, poll `vastai show instance
   <new_contract>` until `running` (a response with `success: false`
   can still provision; trust `show instance`, timeout after 30 min).
   Then add/update the `~/.ssh/config` Host alias with the new
   host/port from `vastai ssh-url <id>` and verify `ssh r4fork-<x>`
   works before doing anything else.
3. On the new box: start supervisord (`supervisord -c
   /etc/supervisor/supervisord.conf` — it does not auto-start in this
   image), then `supervisorctl stop llama` (the template's idle
   server); verify GPUs near 0 MiB with nvidia-smi.
4. Rsync the project exactly as at launch (the launch command is in
   `lab_notes/2026-08-27-r4-fork-launch.md`), INCLUDING
   `/home/admin/mlfactory/artifacts/fork_plan_r4.jsonl` and the model
   weights directory named there.
5. Recreate the venv from the pinned requirements file shipped in the
   project (`requirements_r4_box.txt`), then ALSO copy the
   causal-conv1d binary package from the jump host (no wheel exists
   for this torch/CUDA combo; source build fails on a version check):
   ```bash
   SP=/home/admin/mlfactory/mlfactory/experiments/ace/.venv/lib/python3.12/site-packages
   rsync -az $SP/causal_conv1d $SP/causal_conv1d-1.7.0.dist-info \
     $SP/causal_conv1d_cuda.cpython-312-x86_64-linux-gnu.so \
     r4fork-<x>:/root/venv-r4/lib/python3.12/site-packages/
   ```
   smoke ONE state with m=1
   (`--only <first state of the shard> --m 1 --sub-batch 1` — expect 3
   rows), then recreate BOTH supervisor programs and launch scripts
   (exact templates in the launch note), and restart that box's exact
   original shard commands. Update this runbook's Topology with the
   new instance id and dph (allowed edit).

## What success looks like

~1944 fork-result rows home and checksummed, both boxes stopped, a
complete attendance log, and nothing touched that this doc didn't
name — the experiment's fork evidence intact for the principal's
return.

Good luck! You're going to do great. Thank you for your help.
