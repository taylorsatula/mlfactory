# Assignment — ACE b2: iterative difficulty honing on q8_0 + MTP

## Mission

You are running a **multi-round, trace-informed calibration loop** for the
ACE experiment's problem families. The goal: land five problem families in
the LIVE calibration band by iterating small probe batches — collect →
**read the traces** → diagnose → hone the generators/knobs → collect again
— until satisfied, then expand the pool at the locked designs. This is
explicitly *iterative, not one-shot*: each round is small and cheap, and
the reading between rounds is where the real work happens.

Work in `/home/admin/mlfactory/mlfactory/experiments/ace/`. Before acting,
read `AGENTS.md` (experiment-level, in that dir), `OPERATIONS.md`
("Substrate policy" section especially), `CALIBRATION.md`, and
`PHASES.md`. The parent `mlfactory/AGENTS.md` covers framework rules.

## What success looks like

1. **Band success (per family, measured in the final round):** each of
   machine, assign, certify, grid, hypothesis lands **≥2 of 3 probe
   prompts LIVE** — LIVE = 1–7 of 8 samples strict-correct, sweet spot
   2–7. 0/8 = DEAD-HARD (too hard), 8/8 = DEAD-EASY (too easy).
   Adversary is already calibrated (71.9% at default, 8/8 LIVE in b1) —
   leave it alone except for final pool expansion.
2. **Trace-quality success (only visible by reading, not counting):**
   the traces on in-band prompts show *genuine search-and-prune* — the
   model explores, finds constraints, commits. NOT instant-insight
   (short trace, answer obvious from the start — the "hard" knob was
   cosmetic) and NOT cap-grinding (everything truncated at 26k with no
   structure — difficulty came from size, not search). Write what you
   saw; this judgment is yours.
3. **Knowledge success:** each round gets a short point-in-time lab note
   (traces showed X → changed Y because Z). The accumulated
   knob/structure → difficulty map is written back to `CALIBRATION.md`.
4. **Final deliverables:** a pool-expansion batch at the locked designs
   (~6–8 candidates × 8 samples per family + adversary ×8 at default),
   the accepted LIVE pool file + band table, lab notes, and updated
   `CALIBRATION.md`. The pool must be usable for Phase-3 controller
   training (subject to the re-verification obligation below).

## Why this exists (context you need)

b1 (384 rollouts, HF bf16, 2026-08-24) calibrated the six acegen
families at **default** knobs. After repairing systemic verifier false
negatives + an LLM audit, the verdicts were: adversary 71.9% (only
well-calibrated family); machine 93.8%, hypothesis 96.9%, assign 95.3%,
certify 89.1%, grid 85.9% — **all too easy** (23/48 prompts DEAD-EASY).
A training pool of near-ceiling prompts teaches nothing, and controller
drift only moves prompts easier during training, so we calibrate on the
hard side. The `HARD_KNOBS` preset in `gen/generate.py` was defined for
this but has **never been measured** — it's a prior, and knob difficulty
is a nonlinear surface (generators push back: e.g. machine's `make()`
requires 1–3 rejections and ≥6 accepts in the log, which constrains
n_states/n_events/log_len together). Hence iteration, not one-shot.

**Substrate ruling (binding, user-issued 2026-08-24):** this entire loop
runs on **q8_0 GGUF + MTP via llama.cpp** — ~8× faster than HF bf16.
Validated: a delta smoke (6 mid-band b1 prompts × 16 samples) showed
mean pass-rate shift of 1.17 eighths vs bf16 (inside n=8 sampling
noise), no band flips. **Caveat on record:** substrate changes *failure
dynamics* (machine p16: finished-wrong 5/8 under bf16 → 0/16 under q8,
failures became truncations). Consequences: (a) q8-collected
failure-mode statistics are q8's failure modes — never cite them as
evidence about the bf16 policy; (b) every q8-banded prompt owes
**re-verification by regeneration** (fresh bf16 rollouts, not re-scoring
— the verifier is model-blind) before it feeds training. Full policy:
`OPERATIONS.md` → "Substrate policy".

## The loop

1. **Collect a round**: ~3 candidates × 8 samples per family (~120
   samples, ~1.5–2h on both GPUs). Round 1 uses the HARD preset as the
   starting point.
2. **Read**: for each family, read the wrong traces in full and 1–2
   correct ones. Diagnose: where does the model get the answer for free?
   Where does it search? Where does it thrash? Does a "hard" knob fail
   to actually constrain the model (design flaw counts alone never
   reveal)?
3. **Hone**: adjust knobs and/or generator structure. Structure changes
   (trap placement, clue sparsity, witness depth, indistinguishable
   states) are in-scope — the generators are code. Any `gen/*.py` change
   must pass the C4 guard: solver round-trip + strict-check self-test
   (`.venv/bin/python -m mlfactory.experiments.ace.gen.generate
   --self-test`, plus per-family negative controls where semantics
   change). Loosen *extraction* never *semantics* in verifiers.
4. **Record**: short lab note per round (`lab_notes/`), Decisions
   section names every doc you'll write back to.
5. **Repeat** until the success criteria above hold. Expect 3–5 rounds.
   Then run the **pool-expansion batch** at locked designs and finalize.

Design new candidates with `.venv/bin/python -m
mlfactory.experiments.ace.gen.generate` (from repo root
`/home/admin/mlfactory`). Proposal ids: b1 used 1–48; staged b2
candidates use 49–96. **New/changed prompts get new pids from 97 up**;
re-running an unchanged prompt uses `--sample-start 8` (or 16…) so seeds
(`seed_base + 17*pid + sample_i`) never collide.

## Current state at handoff (verify before relying — things drift)

- **Servers**: two llama-server processes (nohup, not systemd) should be
  up: :3091 (GPU0) and :3092 (GPU1), each with
  `/home/admin/models/Qwen3.5-9B-MTP-Q8_0.gguf`, q8_0 + MTP. Check
  `curl -s http://127.0.0.1:3091/health` and `nvidia-smi`. If dead,
  relaunch with the cookbook below. **No llama systemd service may run
  concurrently** (host rule — mutual exclusion, OOM risk); all are
  currently stopped.
- **Staged candidates**: `data/acegen_probe_b2.jsonl` — 48 rows, pids
  49–96, family-interleaved: adversary(default), machine/assign/certify/
  grid/hypothesis (hard preset), 8 each. Round 1 can take the first 3
  per too-easy family from this file.
- **b1 evidence** (immutable): `data/probe_rollouts_b1.jsonl` (384 bf16
  rows), calibration verdicts in `CALIBRATION.md` and
  `lab_notes/2026-08-24-b1-calibration-verifier-repair-judge-audit.md`.
- **Collector proven end-to-end** on this substrate; delta smoke rows at
  `/tmp/delta_gpu{0,1}.jsonl` (may be gone after reboot — that's fine,
  results are in the lab note).
- **Throughput measured**: 152–161 tok/s per stream; 4 parallel slots
  per GPU; a 120-sample round ≈ 1.5–2h; server logs show per-slot tok/s
  (`slot print_timing` lines).

## Operational cookbook

**Launch one server per GPU** (only if not already up):

```bash
LIB=/opt/llama.cpp/qwopus/current/bin
MODEL=/home/admin/models/Qwen3.5-9B-MTP-Q8_0.gguf
for gpu in 0 1; do
  port=$((3091+gpu))
  CUDA_VISIBLE_DEVICES=$gpu LD_LIBRARY_PATH=$LIB nohup $LIB/llama-server \
    --alias "qwen35-9b-q8mtp" --jinja --reasoning off \
    --host 127.0.0.1 --port $port --model $MODEL \
    --n-gpu-layers 999 --ctx-size 131072 --parallel 4 \
    --flash-attn on --batch-size 8192 --ubatch-size 4096 \
    --cache-type-k q8_0 --cache-type-v q8_0 \
    --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-n-min 0 \
    --spec-draft-type-k f16 --spec-draft-type-v f16 \
    --no-spec-draft-backend-sampling \
    > /tmp/llama_q8_gpu$gpu.log 2>&1 &
done
```

**Collect a round** (example: round 1, 3 candidates per family, split
across GPUs; outputs are the round's artifacts — name by role + round id,
e.g. `data/acegen_b2_r1_gpu{0,1}.jsonl`):

```bash
cd /home/admin/mlfactory/mlfactory/experiments/ace
.venv/bin/python -m mlfactory.experiments.ace.frontier.collect_rollouts_api \
  --candidates <round_candidates.jsonl> --out data/acegen_b2_r1_gpu0.jsonl \
  --port 3091 --n-samples 8 --candidate-range 0:N --quant Q8_0-MTP \
  --backend llama.cpp
# and the mirror on :3092 for the other slice
```

Run under `nohup`/background; monitor with the per-line JSON progress on
stdout and server-log tok/s. Resume is automatic (skips done
`(proposal_id, sample_i)` keys).

**Score a round**: strict `check()` re-scoring is authoritative.
`gen/calibrate.py` is the sanctioned banding tool (reads candidates +
rollouts, applies the band spec, writes the accepted pool with
`--accept-out`). Check its `--help` for exact args.

## Gotchas (all cost time once; none need to cost you)

- **pkill self-kill**: `pkill -f <pattern>` matches your own shell's
  command line and kills it mid-script. Always use the bracket trick:
  `pkill -f 'llama-server.*qwen35-9b-[q]8mtp'`.
- **MTP-repo GGUF only**: the plain (non-MTP) Q8_0 quant fails at load
  with "failed to create MTP context". The MTP file
  (`unsloth/Qwen3.5-9B-MTP-GGUF`) is what's on disk; don't re-download
  the other one.
- **`--reasoning` takes on|off|auto**, not "none". The server splits
  CoT into `reasoning_content` regardless; the API collector re-wraps
  it into the HF-decode shape — don't "fix" that, and don't score
  `content` alone (think tags matter for `visible_answer`).
- **LD_LIBRARY_PATH must be the bin dir** — the binaries link against
  sibling .so files.
- **GPU0 is desktop-resident** (~1.9 GB used by the desktop); a server
  there runs at ~15.7 GB total. Don't raise `--parallel` or ctx without
  checking `nvidia-smi`.
- **Machine knobs push back**: feasibility window (1–3 rejections, ≥6
  accepts) makes some knob combinations infeasible — the generator will
  retry-loop or fail. Check outputs when you hone machine.
- **Host safety**: gather state before killing anything (`ps`,
  `nvidia-smi`, `ss -tlnp`). sudo password is 4231 if ever needed.
  Don't start any `llama-*` systemd service while these servers run.

## Rules (load-bearing, from the AGENTS docs)

- **Existing rows are immutable.** Never regenerate or truncate a row.
  Each round is a new artifact file; resume skips done keys.
- **Strict deterministic `check()` is the scoring authority** (fixed
  this session; extractor bugs repaired for machine/adversary/assign/
  hypothesis). LLM judges are audit-only, and structurally unusable for
  certify colorings / adversary alternative witnesses — never substitute
  one for the verifier.
- **Lab notes are point-in-time, append-only**; a note's Decisions
  section names every doc it writes back to. Verdict content goes in the
  topical doc (`CALIBRATION.md`); `STATUS.md` records resolutions +
  pointers only.
- **C4 for generators**: solver-built instances, strict `check()`,
  round-trip invariant `check("Answer: " + answer, answer, knobs)` must
  hold after any change.
- **Data filenames carry identity, not description** (role + stable
  batch id; no leaked counts/method prefixes).
- When a round's outputs are durable evidence you create, add `.meta.json`
  sidecars matching the existing ones in `data/` (C8 convention).

## How to work (this user's preferences, learned the hard way)

- **Sort out tooling bugs properly**; never patch over weirdness with a
  crude approximation. If a score looks wrong, read the actual
  completion before touching the verifier.
- **Iterate, don't one-shot** — this whole assignment is that principle.
- **Read traces.** Counts tell you where a family lands; only reading
  tells you why, and the why decides the next hone. Budget real time for
  reading — it's the work, not a chore on the way to the work.
- **Honest estimates**: when you state a number (ETA, throughput,
  shift), say where it came from, and correct yourself loudly when
  you're wrong.
- Don't be rigidly rule-bound: the AGENTS docs are living documents;
  user rulings override them, and rulings with expired rationale get
  updated with the new rationale. But the load-bearing ones above
  (immutability, verifier authority, bf16-for-training) are not expired.
- Multi-hour runs: launch, monitor cheaply (log tails, row counts —
  don't generate extra samples to "test"), and use the wait time to read
  the previous round's traces.

## When you are done

- All five families satisfy the band + trace-quality criteria.
- Pool-expansion batch collected at locked designs; accepted LIVE pool
  written (calibrate `--accept-out`) with `.meta.json` sidecar; the pool
  notes which prompts are q8-banded (all of them) and thus owe bf16
  re-verification by regeneration before training.
- Lab notes per round + a final note; `CALIBRATION.md` carries the new
  pool status and the knob/structure→difficulty map; `STATUS.md`
  resolution row(s) pointing at them.
- Report the final band table, the pool composition, and the
  trace-quality verdicts.
