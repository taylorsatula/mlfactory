# Overnight runbook — b2 corpus collect + annotate (unsupervised)

> You are carrying out one batch of the ACE annotation workstream while
> the principal sleeps. Everything below is code that already exists and
> ran successfully on 2026-08-26 (the xsub batch). Your job: generate q8
> rollouts on the 46-prompt LIVE b2 pool, then annotate them through
> Lunaroute, then leave a tidy morning record. **Do exactly this, in
> order, checking each gate before moving on. When in doubt, stop and
> write what happened — never improvise a fix that spends money or
> touches something this doc doesn't name.**

## Binding rules (violating these fails the run)

- Models: annotator is **glm-5.2-vision** only (no `-ballast`, no
  `-flex`, no image models). Query `GET https://gw.lunaroute.com/v1/models`
  once in preflight and record what's active in the morning note.
- Lunaroute: default temperature (never pass one), `max_tokens 65536`,
  at most **6 requests in flight**. Wall-hits and duds are costless —
  never treat `finish_reason=length` as an error to engineer around.
- Local server: `--parallel 1` exactly (higher values silently truncate
  traces by partitioning the context). fp KV (`--cache-type-k f16
  --cache-type-v f16`).
- Do NOT touch: Vast (instance 48783410), any `llama-*` systemd
  service, git (no commits/pushes), the capture/probe/direction scripts
  (main-session work). Delete nothing.
- Use absolute paths in every redirect and nohup (the shell's cwd is
  not reliable between commands).
- Read `mlfactory/AGENTS.md` "Shell discipline" before your first
  command: verify before you speculate, kill by explicit PID, check
  effects after stop/start.

## Phase 0 — preflight (5 min)

```bash
df -h /                          # need >= 30G free, else STOP + note
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
ss -tlnp | grep -E "3091|3090"   # both must be FREE, else STOP + note
```

Check Lunaroute is alive and record the active model list:

```bash
cd /home/admin/mlfactory
mlfactory/experiments/ace/.venv/bin/python - <<'EOF'
import json, urllib.request
from pathlib import Path
from mlfactory.core.secrets import SecretsStore
key = SecretsStore(Path(".mlfactory/secrets.yaml")).get("LUNAROUTE_API_KEY")
req = urllib.request.Request("https://gw.lunaroute.com/v1/models",
                             headers={"Authorization": f"Bearer {key}"})
print([m["id"] for m in json.load(urllib.request.urlopen(req, timeout=60))["data"]])
EOF
```

Gate: `glm-5.2-vision` in the list, GPU1 mostly free, port 3091 free,
disk fine. If any fails: write a note, stop.

## Phase 1 — start the collection server

```bash
CUDA_VISIBLE_DEVICES=1 LD_LIBRARY_PATH=/opt/llama.cpp/qwopus/current/bin \
  nohup /opt/llama.cpp/qwopus/current/bin/llama-server \
  --model /home/admin/models/Qwen3.5-9B-MTP-Q8_0.gguf \
  --alias Qwen3.5-9B --jinja --reasoning on --reasoning-preserve \
  --host 127.0.0.1 --port 3091 --n-gpu-layers 999 \
  --ctx-size 32768 --parallel 1 \
  --cache-type-k f16 --cache-type-v f16 \
  --spec-type draft-mtp --spec-draft-n-max 3 \
  </dev/null > /home/admin/mlfactory/mlfactory/experiments/ace/annotate/out/b2_server.log 2>&1 &
```

(LD_LIBRARY_PATH is mandatory — the build links
`libllama-server-impl.so` from its own bin dir; without it the binary
fails instantly. This is build 10336, the one the xsub q8 arm ran on.)

Wait for health (up to 5 min): `curl -s http://127.0.0.1:3091/health`
until it returns ok. If it never does: read the tail of the server log,
record it, STOP (do not try another model or flags).

Smoke one generation:

```bash
curl -s http://127.0.0.1:3091/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3.5-9B","messages":[{"role":"user","content":"Say ok."}],"max_tokens":64}' | head -c 400
```

Gate: a real completion comes back. Then note the server PID
(`ps aux | grep llama-server | grep -v grep`) — you kill THAT PID later.

## Phase 2 — collect rollouts (~8-9 h, unattended)

184 rollouts: 46 LIVE-b2 prompts x 4 samples, strict gen verifiers,
resume-safe by (pid, sample_i).

```bash
cd /home/admin/mlfactory
nohup bash -c 'cd /home/admin/mlfactory && mlfactory/experiments/ace/.venv/bin/python -m mlfactory.experiments.ace.frontier.collect_rollouts_api \
  --candidates mlfactory/experiments/ace/data/acegen_live_b2.jsonl \
  --out mlfactory/experiments/ace/data/annot_b2_q8.jsonl \
  --port 3091 --n-samples 4 --seed-base 84000 --quant Q8_0-MTP' \
  </dev/null > /home/admin/mlfactory/mlfactory/experiments/ace/annotate/out/b2_collect.log 2>&1 &
```

Confirm the banner in the log shows `already_done` and rows start
landing: `wc -l .../data/annot_b2_q8.jsonl` should grow by ~1 every
2-4 minutes. Check coarsely (every 30-60 min; do not poll tightly):
process alive, server healthy, row count growing. Individual collector
errors self-retry/resume — only escalate if row count is unchanged for
>60 min AND the server is down (then: restart the server from Phase 1
and relaunch this exact command — it resumes).

Gate to proceed: 184 rows (46 prompts x 4). If still short after the
night, proceed anyway with what exists IF every prompt has >= 2 rows;
otherwise keep collecting and delay Phase 4.

## Phase 3 — stop the server

Kill the PID from Phase 1 (explicit PID, no pkill patterns). Verify:
process gone, `ss -tlnp | grep 3091` empty, GPU1 memory ~0.

## Phase 4 — annotate through Lunaroute (~1.5-2 h)

Build the plan (deterministic pairing; 6 double-annotation pairs, one
per family):

```bash
cd /home/admin/mlfactory
mlfactory/experiments/ace/.venv/bin/python -m mlfactory.experiments.ace.annotate.build_plan \
  --corpus mlfactory/experiments/ace/data/annot_b2_q8.jsonl \
  --out mlfactory/experiments/ace/data/annotation_plan_b2.jsonl \
  --double-per-domain 6
```

Gate: banner says pairs cover exactly the collected row count, one
double pair per domain present. If the coverage assert fails, some
prompt has <2 rows — back to Phase 2.

Run both annotation passes (pass2 = independent re-annotation of the
double subset; chained sequentially, resume-safe):

```bash
nohup bash -c 'cd /home/admin/mlfactory && \
  mlfactory/experiments/ace/.venv/bin/python -m mlfactory.experiments.ace.annotate.run_batch \
    --pass pass1 --tag b2 --workers 6 \
    --plan mlfactory/experiments/ace/data/annotation_plan_b2.jsonl \
    --corpus mlfactory/experiments/ace/data/annot_b2_q8.jsonl && \
  mlfactory/experiments/ace/.venv/bin/python -m mlfactory.experiments.ace.annotate.run_batch \
    --pass pass2 --tag b2 --workers 5 \
    --plan mlfactory/experiments/ace/data/annotation_plan_b2.jsonl \
    --corpus mlfactory/experiments/ace/data/annot_b2_q8.jsonl' \
  </dev/null > /home/admin/mlfactory/mlfactory/experiments/ace/annotate/out/b2_annotate.log 2>&1 &
```

Monitor coarsely: `[i/N]` lines with `finish=stop`, resolved counts
> 0 on most pairs. 429s self-backoff (60/120/240/480 s) — leave them.
Escalate (stop, keep outputs, note) only if 30+ consecutive requests
fail.

## Phase 5 — morning record (do all of it)

1. Label-agreement check on the double subset:
   ```bash
   cd /home/admin/mlfactory
   mlfactory/experiments/ace/.venv/bin/python -m mlfactory.experiments.ace.annotate.r0_agreement --tag b2
   ```
2. Write sidecars for the new data files
   (`data/annot_b2_q8.jsonl`, `data/annotation_plan_b2.jsonl`,
   `data/annotations_b2_pass1.jsonl`, `data/annotations_b2_pass2.jsonl`)
   matching the schema of the existing `*.meta.json` files in
   `data/` (name/title/description/format/tags/path/sha256/size/created).
3. Quick stats: flags by class and conf, resolved fraction, traces
   flagged, per-domain flag counts (especially MUSE counts per family —
   the point of this batch is the muse gap; today's corpus had n=5).
4. Write the lab note
   `lab_notes/2026-08-27-overnight-b2-collect-annotate.md`: what ran,
   counts, agreement numbers, failures/retries, Lunaroute model list
   from preflight, anything skipped or aborted. Append-only record —
   facts, not plans.
5. Leave the machine clean: no stray processes, GPUs idle, nothing
   deleted, nothing committed.

## Abort conditions (stop, record, wait for the principal)

- Server won't start / crashes twice.
- Disk < 30G.
- Collector stalled >60 min with the server down AND the restart fails.
- 30+ consecutive Lunaroute failures.
- Anything not covered above that would require judgment about money,
  other machines, services, or deleting data.

## What success looks like

~184 new q8 rollouts on the six b2 families (machine/assign/hypothesis
are new to the annotation corpus), ~90 annotated pairs with resolved
span offsets in `data/annotations_b2_pass{1,2}.jsonl`, an agreement
reading on the double subset, and a lab note — ready for the main
session to merge into capture/probes.
