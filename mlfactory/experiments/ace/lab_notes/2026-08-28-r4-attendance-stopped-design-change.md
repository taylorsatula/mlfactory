# 2026-08-28 — R4 fork run STOPPED for design change (principal); boxes destroyed

> ## ⚠️ WHY THE RUN WAS HALTED — principal's design rationale (verbatim)
>
> The run was generating each trace to its **terminal state — many
> thousands of tokens after applying a steady linear change to all
> downstream tokens. That is NOT what we should be testing.**
>
> The correct design: take every intervention point, run it out for
> **2048 tokens**, then have an **LLM judge review the traces of the
> three branches (noop, healthy, diverge) and assess whether the
> intervention did something meaningful.**
>
> Conflating a correct terminal answer with the intervention working
> is an incorrect approach — and that doesn't even touch on the fact
> that we're spending an astronomical amount of yielded tokens just to
> look at the span right after the intervention point. It makes no
> sense.
>
> **The design must change before this run continues.** The recipe and
> partial data below are preserved for the redesign.

> Run note. Principal directed halt: stop the run, pull data home, write
> a full A6000 setup snapshot for recreation, then destroy the boxes.
> **The experiment design must change before this run continues** —
> principal's decision, not this agent's. Analysis of the 117 partial
> rows is reserved for the main session. Predecessor live-state doc:
> `2026-08-27-r4-fork-launch.md`. Attendance runbook:
> `annotate/runbook_r4_cancun.md`. Attendance log:
> `annotate/out/r4_attendance.log`.

## What happened

The R4 fork run launched 2026-08-27 21:40 UTC was stopped by the
principal at 2026-08-28 05:35 UTC (~7h55m wall) to change the
experiment design. The attendance agent (this session) executed the
halt: stopped all four supervisor programs, synced results home,
verified local copies byte-identical to remote, captured the full box
setup below, then destroyed both instances. The run reached 117/1944
rows (~6%) — internally consistent and preserved, but partial.

## Final run state

| box | instance | program | states | target rows | rows at stop | last progress line |
|---|---|---|---|---|---|---|
| A | 48921611 | r4fork1 | 7 | 504 | 26 | r4_cycle_02 toward_healthy seed 1, done 26/504 |
| A | 48921611 | r4fork2 | 6 | 432 | 29 | r4_cycle_03 toward_healthy seed 4, done 29/432 |
| B | 48921612 | r4fork1 | 7 | 504 | 42 | r4_cycle_00 toward_healthy seed 17, done 42/504 |
| B | 48921612 | r4fork2 | 7 | 504 | 20 | r4_cycle_01 noop seed 19, done 20/504 |
| | | | 27 | 1944 | **117** | noop=92, toward_healthy=25, toward_diverge=0 |

- **Wall time:** 7h55m (21:40 UTC 2026-08-27 → 05:35 UTC 2026-08-28).
- **Spend:** ~$7.65 (7.917h × $0.9667/hr combined; A $0.456/h, B $0.511/h).
  Well under the $200 runbook cap / $150 launch-note cap.
- **Rate at stop:** ~7–8 rows/h/box (continuation-length-limited: noop
  and toward_healthy arms run 8–20k tok at 12–13 tok/s, ~10–25 min/seed).
  Projected full-run wall would have been ~5.3–5.6 days, ~$130 — the
  run was healthy and on-track when halted; the halt is a design
  decision, not a health event.
- **0 OOMs**, 0 restarts across the whole run; all four programs
  `RUNNING` with uptime matching launch at every check until stop.

## Data preservation (verified safe before destroy)

Local files (jump host, this machine):

```
/home/admin/mlfactory/mlfactory/experiments/ace/data/r4fork-a/fork_r4_results_1.jsonl   26 rows
/home/admin/mlfactory/mlfactory/experiments/ace/data/r4fork-a/fork_r4_results_2.jsonl   29 rows
/home/admin/mlfactory/mlfactory/experiments/ace/data/r4fork-b/fork_r4_results_1.jsonl   42 rows
/home/admin/mlfactory/mlfactory/experiments/ace/data/r4fork-b/fork_r4_results_2.jsonl   20 rows
```

Integrity check (run before destroy): **117 rows, 117 unique keys** —
no duplicate `(state_id, arm, seed_i)`; all `arm` in
`{noop, toward_healthy, toward_diverge}`. The run had not yet reached
`toward_diverge` on most states (expected — arms run in order noop →
toward_healthy → toward_diverge).

sha256 (local == remote, confirmed byte-identical before destroy):

```
471102fa3005e4a83bf4210c5175b141050d872d6e1276372b7f5bc01341cc6a  r4fork-a/fork_r4_results_1.jsonl
7f1521f5b3ae0fdee4c5deaa07d76dbda81189130f60f99d59091c0dc883e90c  r4fork-a/fork_r4_results_2.jsonl
50cf38e395fb87bddeec5f9815cfd47e855f7bd7d43bad01832e24acfd17cb4d  r4fork-b/fork_r4_results_1.jsonl
792e6c33d30bebfeb6507715b1dcad2557a11b7fb9629a0490bbef3bcfb46224  r4fork-b/fork_r4_results_2.jsonl
```

A `fork_r4_results_SHA256SUMS` file was NOT written — these are
partial results of a halted run, not the completion artifact; the
sha256s above are the record. The 2-hourly cron checkpoint
(`data/checkpoint_sync.sh`) had already been syncing throughout; the
final manual sync run at halt captured the last rows.

## A6000 box setup snapshot — recreation recipe

> Self-contained recipe to stand up a fresh A6000 box identical to the
> two that ran. Both boxes were identical except shard assignment and
> driver version (cosmetic). Captured live at halt (2026-08-28 05:35
> UTC) and cross-checked against `2026-08-27-r4-fork-launch.md`.

### Instance / image

- **GPU:** 1× RTX A6000 48GB. **Disk:** 200GB. **Image:**
  `vastai/llama-cpp:b10182-cuda-12.9`. **OS:** Ubuntu 24.04.4 LTS.
- **CUDA:** 12.8 (via torch cu128 wheel). **Driver:** 570.x (A:
  570.133.20, B: 570.195.03 — both work; driver is image-provided).
- **Python:** 3.12.3 (system, symlinked into venv).
- **dph at launch:** A $0.456 (Kansas), B $0.511 (Delaware, 8.8Gbps
  down). Recreate via the search in runbook Appendix A.
- **PID 1 is bash** — supervisord does NOT auto-start in this image.

### Mandatory: SSH-permission-repair `--onstart-cmd`

The 2026-08-27 boxes came up with `/root/.ssh/authorized_keys` at wrong
ownership/modes; sshd StrictModes refused every key (log signature:
`Authentication refused: bad ownership or modes`). The onstart fixes
permissions and keeps them fixed. **This is mandatory on every create:**

```bash
ONSTART='mkdir -p /root/.ssh; grep -qF vast-dft-20260726 /root/.ssh/authorized_keys 2>/dev/null || echo "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAID+EYvEp2UOqpXCYUfdsZt7fZc+6cuAPQuU31/Owlp95 vast-dft-20260726" >> /root/.ssh/authorized_keys; nohup bash -c "while true; do chown -R root:root /root/.ssh 2>/dev/null; chmod 700 /root/.ssh 2>/dev/null; [ -f /root/.ssh/authorized_keys ] && chmod 600 /root/.ssh/authorized_keys 2>/dev/null; sleep 30; done" >/dev/null 2>&1 &'
vastai create instance <offer_id> --image vastai/llama-cpp:b10182-cuda-12.9 \
  --disk 200 --ssh --direct --label r4fork-<a|b> \
  --onstart-cmd "$ONSTART" --raw
```

Then add/update `~/.ssh/config` Host alias from `vastai ssh-url <id>`.

### Project transfer (jump host → box, ~209MB)

```bash
rsync -az --exclude '.git/' --exclude '.venv' --exclude '__pycache__' \
  --exclude 'runs/' --exclude '.mlfactory/' --exclude 'annot_captures*' \
  --exclude 'ace-legacyapproach/data/' \
  /home/admin/mlfactory/ r4fork-<x>:/root/mlfactory/
```

The box copy is NOT a git repo (rsync excludes `.git/`); code is a flat
snapshot. Frozen plan lands at `/root/mlfactory/artifacts/fork_plan_r4.jsonl`
(1.34MB, 27 states × 3 arms × 24 seeds).

### venv + pinned deps

```bash
python3 -m venv /root/venv-r4
/root/venv-r4/bin/python -m pip install -r /root/mlfactory/requirements_r4_box.txt
/root/venv-r4/bin/python -m pip install pydantic==2.13.4 flash-linear-attention==0.5.2
```

`/root/mlfactory/requirements_r4_box.txt` (pinned):

```
--extra-index-url https://download.pytorch.org/whl/cu128
torch==2.11.0+cu128
transformers==5.14.1
numpy==2.5.1
safetensors==0.8.0
tokenizers==0.22.2
huggingface_hub==1.28.0
hf-xet==1.6.0
accelerate==1.14.0
einops==0.8.2
pydantic==2.13.4
```

`pydantic` and `flash-linear-attention==0.5.2` are installed
separately (pydantic via core/manifest; fla for the DeltaNet fast path).

### causal-conv1d binary (NO wheel — copy from jump host)

`causal-conv1d` has no wheel for this torch 2.11 / cu128 combo and
source-build fails a toolkit-version check. Copy the binary package
from the jump host's ace venv:

```bash
SP=/home/admin/mlfactory/mlfactory/experiments/ace/.venv/lib/python3.12/site-packages
rsync -az $SP/causal_conv1d $SP/causal_conv1d-1.7.0.dist-info \
  $SP/causal_conv1d_cuda.cpython-312-x86_64-linux-gnu.so \
  r4fork-<x>:/root/venv-r4/lib/python3.12/site-packages/
```

### Model (Qwen3.5-9B, public, hf_xet — ~19GB)

`MODEL_PATH` is hardcoded in `fork_r4.py` to
`/home/admin/models/hf/Qwen3.5-9B`. Download to the hf cache and
symlink that path to the snapshot:

```bash
HF_HOME=/root/.hf_home /root/venv-r4/bin/hf download Qwen/Qwen3.5-9B
mkdir -p /home/admin/models/hf
ln -sfn /root/.hf_home/hub/models--Qwen--Qwen3.5-9B/snapshots/<snap> \
  /home/admin/models/hf/Qwen3.5-9B
```

At halt, snapshot was `c202236235762e1c871ad0ccb60c8ee5ba337b9a`
(4 safetensors shards, bf16). `<snap>` is whatever `hf download`
checks out — pin by reading `ls /root/.hf_home/hub/models--Qwen--Qwen3.5-9B/snapshots/`.

### Supervisor (does not auto-start; stop the idle llama server)

```bash
supervisord -c /etc/supervisor/supervisord.conf
supervisorctl stop llama          # template's idle server; verify GPU ~0 MiB
```

The r4 programs use `autostart=false autorestart=false stopasgroup=true`
so they run once and stay down on exit (resume is via the output file).

### Run scripts (exact, per box/program)

`/root/r4fork_run1.sh` and `/root/r4fork_run2.sh` — identical structure,
differ only in `--only` shard list and `--out` file. Each:

```bash
#!/bin/bash
cd /root/mlfactory
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec /root/venv-r4/bin/python -m mlfactory.experiments.ace.annotate.fork_r4 \
  --run --out /root/mlfactory/mlfactory/experiments/ace/data/fork_r4_results_<N>.jsonl \
  --sub-batch 1 --only <comma-separate state list>
```

Shard assignments (27 states total, split 4 ways; each state = 3 arms ×
24 seeds = 72 rows):

| box | program | out file | --only states | rows |
|---|---|---|---|---|
| A | r4fork1 | fork_r4_results_1.jsonl | r4_muse_00,r4_cycle_09,r4_cycle_02,r4_loop_09,r4_loop_04,r4_muse_01,r4_loop_02 | 504 |
| A | r4fork2 | fork_r4_results_2.jsonl | r4_muse_04,r4_cycle_08,r4_muse_06,r4_cycle_03,r4_cycle_05,r4_cycle_07 | 432 |
| B | r4fork1 | fork_r4_results_1.jsonl | r4_loop_07,r4_loop_08,r4_loop_05,r4_loop_06,r4_cycle_00,r4_cycle_06,r4_loop_01 | 504 |
| B | r4fork2 | fork_r4_results_2.jsonl | r4_loop_00,r4_cycle_01,r4_muse_05,r4_loop_03,r4_muse_03,r4_cycle_04,r4_muse_02 | 504 |

### Supervisor confs (drop into /etc/supervisor/conf.d/)

`r4fork1.conf` and `r4fork2.conf` (identical except program name):

```
[program:r4fork1]
command=/root/r4fork_run1.sh
directory=/root/mlfactory
autostart=false
autorestart=false
stopasgroup=true
stdout_logfile=/tmp/r4fork1.log
stderr_logfile=/tmp/r4fork1.err
```

Then `supervisorctl reread && supervisorctl update`, and
`supervisorctl start r4fork1 r4fork2`.

### Frozen design constants (DO NOT change without principal — but the
design IS being changed now; record what they were)

m=24 seeds, 3 arms (noop, toward_healthy, toward_diverge), temperature
0.8 / top_p 0.95, 26000-token backstop, `FLASH_ATTENTION` sdpa backend
(determinism proven 2026-08-27; MATH backend prefill-spike OOMs
two-process co-residency on forks >~9.5k states), `--sub-batch 1`
(batch-1 decode anti-scales on this hybrid model; two time-sliced
processes measured 1.85× one). `batch_seed = sha256(state_id)`-derived.
Focal layers CYCLE L18 / LOOP L2 / MUSE L17.

## Attendance log summary

```
2026-08-27 21:42Z | LAUNCH | A: r4fork1+r4fork2 RUNNING rows=0/936 | B: r4fork1+r4fork2 RUNNING rows=0/1008 | spend=~$0 (started) | all four programs up 21:40-21:42Z, first row B/r4fork1 21:53Z (r4_cycle_00 noop 8420tok 656s)
2026-08-27 22:27Z | A: RUNNING rows=4/936 rate=~5/h | B: RUNNING rows=5/1008 rate=~6/h | spend=~$0.76 | kickoff check ok; rate length-limited by first long seeds, recalibrate at 3h
2026-08-27 22:30Z | CHECKPOINT | Initial checkpoint created: A=4 rows, B=6 rows (10 total)
2026-08-28 01:29Z | A: RUNNING rows=25/936 rate=7.0/h | B: RUNNING rows=29/1008 rate=7.9/h | spend=~$3.69 | ok; rate recalibrated ~7-8 rows/h per box
2026-08-28 01:52Z | EXTRA | ...principal flagged B GPU 0% on glance — verified transient between-seeds dip... no action
2026-08-28 04:29Z | A: RUNNING rows=46/936 rate=7.0/h | B: RUNNING rows=53/1008 rate=7.9/h | spend=~$6.60 | ok
2026-08-28 04:49Z | EXTRA | A: RUNNING rows=50/936 | B: RUNNING rows=56/1008 | spend=~$6.91 | ok; one-off check
2026-08-28 05:35Z | HALT | principal stopped run for design change; both boxes supervisorctl stop; 117/1944 rows preserved; destroying boxes
```

## Decisions

- **Run halted and boxes destroyed per principal** (2026-08-28 ~05:30
  UTC). Design must change before continuing — principal's call.
- **117 partial rows preserved** at
  `data/r4fork-{a,b}/fork_r4_results_*.jsonl`, byte-verified against
  remote before destroy. Usability of partial data under the new
  design is for the main session to decide.
- **No analysis written** — reserved for the main session, per
  runbook. This note records state + the recreation recipe only.
- **STATUS.md:** no write-back — the R4 question is not resolved, it is
  paused for redesign. Leave the open row as-is.
