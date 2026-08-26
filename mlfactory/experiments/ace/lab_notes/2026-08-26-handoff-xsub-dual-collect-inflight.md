# HANDOFF — 2026-08-26 — xsub dual-collect inflight (context-compaction carry-forward)
# STATUS (2026-08-26 16:50 UTC): SUPERSEDED — both arms finished (96/96
# rows archived + checksummed), Vast 48783410 stopped (not destroyed),
# local GPUs freed, comparison written:
# lab_notes/2026-08-26-xsub-collect-complete-substrate-comparison.md.
# Pending user calls: annotation model pick (R0); destroy-vs-keep Vast.
# The concept this workstream runs on: ANNOTATION_SIDESTEP.md — read it first.
# Decisions record: lab_notes/2026-08-26-annotation-workstream-launch.md.
# Live monitor: `mlfactory dashboard --config
#   mlfactory/experiments/ace/annotate/dashboard.json`.
# Audience: the same agent, post-compaction. Read concept doc + this file,
# then resume monitoring; this file is transient — the run state it
# describes supersedes it.

## objective_and_constraints

```yaml
objective: |
  Run the annotated-pattern workstream (ANNOTATION_SIDESTEP.md): detect
  reasoning episodes (muse/cycle/loop spans) in collected traces via
  LLM annotation, then probe prefix-causal activations at those
  positions — a cheap candidate-generation organ in front of the fork
  machinery. Current phase: cross-substrate dual-collect RUNNING
  (feeds annotation pass R0 + first same-prompt q8-vs-bf16 comparison).
goal_deltas_this_session:
  - annotation unit is the TRACE, not the failure member of a pair
    (success traces and cap-hit traces are first-class material)
  - blunt cases first (idle musing / loops), nuanced escape-vs-reheat
    deferred to a later pass
  - iterate on q8 rollouts, polish on bf16 — accepted working loop
  - documentation axioms are GUIDELINES; the user rules case-by-case on
    evidence (applies session-wide, not to one clause)
bindings_from_principal:
  - fp16 KV for local q8 llama serving (supersedes b2 q8-KV precedent)
  - use hf_xet for ALL Hugging Face transfers (curl from HF is fallback)
  - prefer MTP-enabled model variants when downloading (lossless, ~4x)
  - continue monitoring the running collection after compaction
```

## world_state_delta

```yaml
new_artifacts_this_session:
  - ace/ANNOTATION_SIDESTEP.md — bedrock concept doc (idea verbatim,
    relation to fork architecture, kill conditions, ladder R0–R4)
  - ace/annotate/RUBRIC.md — pass-1 rubric: classes muse/cycle/loop,
    quote-format spans, mandatory basis field, outcome-blinded annotator
  - ace/annotate/build_pairs.py + data/annotate_pairs_p1.jsonl —
    71 S/F pairs (44 hf-bf16, 27 q8-mtp) + 255 loop targets
  - ace/annotate/dashboard.json — validated live monitor (all 16 probes
    tested headlessly via _run_probe)
  - ace/lab_notes/2026-08-26-annotation-workstream-launch.md —
    point-in-time decisions note
  - mlfactory/AGENTS.md — new sections "Shell discipline" +
    "Model downloads and transfers" (per user directive)
  - mlfactory/docs/VAST_REMOTE.md — image-drift warning, "Fast path:
    llama.cpp serving template" recipe, hf_xet note
  - mlfactory/remote/vast.py — DEFAULT_IMAGE fixed to
    vastai/llama-cpp:b10182-cuda-12.9 (old tag 404s)
rented_infrastructure:
  vast_instance: 48783410   # $3.766/hr since ~14:57 UTC
  image: vastai/llama-cpp:b10182-cuda-12.9
  ssh: ssh -i ~/.ssh/id_vast -p 36867 root@154.59.156.23
  layout: |
    /workspace/models/Qwen3.5-9B-BF16.gguf (unsloth MTP repo, 18407321728 B)
    two llama-servers: GPU0 :3091, GPU1 :3092 (logs /tmp/llama_bf16_*.log)
    /workspace/venv-xsub (torch-cpu+transformers+numpy+hf_xet; collector
      import chain only — no GPU torch needed)
    /workspace/mlfactory (rsynced tree, PYTHONPATH usage, no pip -e)
    outputs: /workspace/mlfactory/mlfactory/experiments/ace/data/
      xsub_bf16_gpu{0,1}.jsonl (+ .log siblings)
```

## negative_knowledge

```yaml
- pkill/pgrep self-match: `pkill -f "llama-server.*3091"` killed its own
  wrapper shell mid-fix → kill by explicit PID; bracket trick
  ('collect_rollouts_[a]pi') for pgrep patterns. Now in AGENTS.md.
- llama-server --parallel N partitions --ctx-size: parallel 4 + ctx
  32768 = 8192-token slots → silent truncation at ~7.9k tokens. Use
  --parallel 1; verify n_ctx_slot in startup log. Now in VAST_REMOTE.md.
- HF transformers 5.14.1 CANNOT use Qwen3.5 MTP: modeling ignores mtp.*
  weights (_keys_to_ignore_on_load_unexpected), config exposes only the
  unused mtp_num_hidden_layers, generator requires num_mtp_layers. MTP
  speedup rides with llama.cpp GGUF only (unsloth *-MTP-GGUF repos ship
  BF16 too).
- vastai CLI is verb-noun (`vastai create instance`, `vastai show
  instances`); `update instance --image` 404s on a loading instance —
  destroy+recreate instead; house DEFAULT_IMAGE tag was stale (404).
- HF CDN throttles single-stream curl to 15–22 MB/s; hf_xet measured
  ~450 MB/s (18 GB in ~40 s) on the same link.
- dashboard shell probes: escaped quotes inside ssh remote command broke
  under the executor (unexpected EOF) — fixed with unquoted echo; ALWAYS
  validate probes via core.dashboard._run_probe before shipping.
- local /opt llama builds need LD_LIBRARY_PATH=<build>/bin (read the
  systemd unit: systemctl cat llama-qwen38 showed it).
```

## operational_state (at 15:49 UTC — RECHECK ON RESUME)

```yaml
collection_progress: q8 23/48, vast g0 11/24 + g1 11/24 (56/96 total)
rates: q8 ~136 s/sample; vast ~170 s/sample → both arms ETA ~16:40-16:50 UTC
monitor: |
  cd /home/admin/mlfactory && mlfactory dashboard --config \
    mlfactory/experiments/ace/annotate/dashboard.json
  # or one-liner:
  wc -l mlfactory/experiments/ace/data/xsub_q8.jsonl && ssh -i ~/.ssh/id_vast \
    -p 36867 root@154.59.156.23 'D=/workspace/mlfactory/mlfactory/experiments/ace/data; \
    wc -l $D/xsub_bf16_gpu0.jsonl $D/xsub_bf16_gpu1.jsonl'
local_processes:
  - q8 llama-server: pid 2584994, port 3091, GPU0, log /tmp/xsub_llama_q8.log
  - q8 collector: pid 2586250 (wrapper 2586247), log
    ace/data/xsub_q8.log, out ace/data/xsub_q8.jsonl (seed-base 72000,
    quant Q8_0-MTP, fp16 KV)
tombstones:
  - llama-qwen38 service STOPPED at user direction (was Qwen3.8-27B on
    :3090, tensor-split both GPUs). Do NOT auto-restart.
  - xsub_q8.abort1_badctx.* = the two bad rows from the ctx bug.
    Archived evidence of the incident; NOT collection data.
on_completion_do:
  1. rsync vast rows home:
     rsync -az -e "ssh -i ~/.ssh/id_vast -p 36867" \
       root@154.59.156.23:'/workspace/mlfactory/mlfactory/experiments/ace/data/xsub_bf16_gpu*' \
       mlfactory/experiments/ace/data/
  2. sha256 both ends; parse-check every jsonl; confirm 24+24 rows
  3. vastai stop instance 48783410 (halts GPU charges) — destroy only
     after local archive verified
  4. kill local q8 llama-server (pid above); GPUs then free
  5. write .meta.json sidecars for the three new xsub files (house style)
  6. substrate band comparison on the 6 prompts (first same-prompt
     q8-vs-bf16 n=8x2 dataset) → lab note
  7. restructure annotation manifest trace-centric; then R0 (annotation
     model choice: lunaroute -ballast variants per AGENTS provider
     preference; key in `mlfactory secrets` LUNAROUTE_API_KEY) — confirm
     model pick with user before spending
```

## open_questions

```yaml
- do blunt annotated spans yield prefix-causal separable state patterns?
  bet: yes for muse/loop (blunt), confidence: genuinely uncertain — this
  is the R2 gamble; kill conditions pre-registered in ANNOTATION_SIDESTEP.md §6
- do the 6 prompts stay mixed-outcome on BOTH substrates? data landing ~16:50
- will any prompt show the substrate failure-species skew measured on
  2026-08-24 (q8 truncation-heavy vs bf16 committed-wrong)? expect yes
- annotation model pick (lunaroute ballast; which one) — user decision
```

## pointers

```yaml
concept: ace/ANNOTATION_SIDESTEP.md (verbatim idea text lives there)
rubric: ace/annotate/RUBRIC.md
decisions_note: ace/lab_notes/2026-08-26-annotation-workstream-launch.md
dashboard: ace/annotate/dashboard.json
vast_recipe: mlfactory/docs/VAST_REMOTE.md §fast-path
shell_rules: mlfactory/AGENTS.md §shell-discipline
substrate_precedent: ace/lab_notes/2026-08-24-substrate-delta-smoke-q8-mtp-vs-bf16.md
```

## checkpoint timeline (2026-08-26 session)

1. principal asked to explore a sidestep to TERMINAL_FORK_COMPUTE → agent
   read 6 docs (README, HYPOTHESIS, COUNTERFACTUAL_FRAMEWORK,
   LAYER_HYPOTHESES, OBSERVABLES, TERMINAL_FORK_COMPUTE).
2. principal shared github.com/taylorsatula/TeaLeaves → agent cloned,
   installed in ace venv (198/198 tests pass), inspected hook machinery.
3. agent verified TeaLeaves vs Qwen3.5-9B substrate → 3 discovery
   failures (nested text_config; 24/32 layers linear_attn-only; DeltaNet
   has no attention matrix); residual hooks portable.
4. principal pitched the annotation idea (pair traces → LLM annotator →
   span labels → activation patterns → gentle nudges) → agent mapped it
   onto the constraint set: candidate generation gets cheap, fork gate
   stays binding ("detector nominates, forks ratify").
5. agent measured pairing reality: 71 clean pairs, not "several hundred"
   → principal: fine, new pairs generate quickly.
6. principal ruled axioms are guidelines; blunt cases first (Goff idle-
   musing example); annotations noisy-by-design → rubric/manifest design
   absorbed all three.
7. principal asked q8-vs-bf16 functional-blocker question → agent found
   the 2026-08-24 substrate-delta-smoke note: landing places equivalent
   (1.17/8 mean shift), failure dynamics NOT equivalent (q8 truncation-
   heavy) → iterate-on-q8 accepted with bf16 control subset built in.
8. principal offered Vast box + split plan (bf16 there, q8 local after
   stopping qwen38) → agent stopped llama-qwen38, built
   xsub_candidates.jsonl (6 prompts), launched q8 server.
9. badctx incident: --parallel 4 partitioned ctx → traces capped at 7.9k
   → agent archived 2 bad rows, relaunched parallel-1, verified first
   row 25,253 tokens.
10. principal: use fp kv → agent relaunched server without cache-type
    flags (fp16 KV default).
11. principal: new instance 48783410 on llama.cpp template → agent read
    box guide, stopped template llama service.
12. principal challenged MTP-on-bf16 conclusion → agent re-verified
    (transformers ignores mtp.* weights) but found
    unsloth/Qwen3.5-9B-BF16.gguf → redesigned vast arm as llama.cpp
    BF16+MTP: no torch stack needed, ~125 tok/s measured on Blackwell.
13. curl download throttled (15 MB/s) → principal: use xet → hf_xet
    fetched 18 GB in ~40 s; hf_xet installed locally too.
14. agent launched both vast arms (samples 0-3 / 4-7, seed-base 72000),
    smoke-tested both servers, wrote lab note.
15. principal directed 4 write-backs → AGENTS.md (shell discipline,
    downloads), VAST_REMOTE.md (fast path + pitfalls), vast.py
    DEFAULT_IMAGE fixed.
16. principal asked for a monitor dashboard → agent built + headlessly
    validated annotate/dashboard.json (16 probes; fixed one remote-quote
    bug found by the executor test).
17. principal: write the concept doc + handoff → ANNOTATION_SIDESTEP.md
    written; this handoff.
