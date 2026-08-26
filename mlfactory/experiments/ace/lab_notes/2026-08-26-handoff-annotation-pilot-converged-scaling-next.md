# HANDOFF — 2026-08-26 — annotation pilot converged, scaling next (context-compaction carry-forward)
# The concept this workstream runs on: ANNOTATION_SIDESTEP.md — read it first.
# Pilot history + verdict (the load-bearing record):
#   lab_notes/2026-08-26-annotation-pilot-prompt-hillclimb.md
# Collection/comparison phase record (complete):
#   lab_notes/2026-08-26-xsub-collect-complete-substrate-comparison.md
# Prior handoff (collection phase) carries a SUPERSEDED banner; its
# on-completion steps 1-6 are DONE — only step 7 (trace-centric manifest
# + R0) remains, and it is now reshaped by the pilot below.
# Audience: the same agent, post-compaction. Read concept doc + pilot
# lab note + this file, then scale the annotation pass.

## objective_and_constraints

```yaml
objective: |
  Annotation-pattern workstream (ANNOTATION_SIDESTEP.md): LLM span
  annotation of reasoning episodes (muse/cycle/loop) in collected
  traces, then position-level activation probes. The prompt-hill-climb
  pilot has CONVERGED (v5 validated on 2 problem families, probable
  tier audited 14/16 real). Next phase: scale annotation from the 2
  pilot traces to the broader corpus.
goal_deltas_this_session:
  - annotation prompting is a hill-climbed artifact, versioned files
    prompt_v1..v5.md containing ONLY what gets sent to the model;
    revision provenance lives in the lab note, never in the prompt
  - framing variants A (single trace), B (unlabeled pair), C (pair +
    compare & contrast) are all implemented; C is richest and
    best-resolved (7-12 flags, best quote resolution), B competitive,
    A the fallback
  - conf=probable greyzone flags are KEPT, not withheld — audited
    high-signal (13/16 correct-class real episodes); droppable later
  - thinking-wall hits / dud runs are costless (Lunaroute not
    pay-per-token); retry once, keep whatever lands
bindings_from_principal:
  - NO -ballast models today (Lunaroute server trouble) — scope today;
    always GET /v1/models first, the active set changes
  - Lunaroute default temperature unless diverging with a reason
  - GLM is a big thinker — large max_tokens (runner uses 65536)
  - subagents on qwen/qwen3.7-plus, parallelized + backgrounded;
    nearly all work stays in the main thread; subagents absorb bulk
    content (full-trace reviews, audits), return terse verdicts
  - read annotation results via ⟦-anchor greps, never whole traces
  - remove dead code on sight (nothing is "harmless")
  - Vast 48783410 stays stopped-not-destroyed ($0.09/hr reserved);
    the user destroys it when today's work is done
```

## world_state_delta

```yaml
pilot_harness (annotate/pilot/):
  run_pilot.py: runner — ⟦-anchor parsing (splits on class head
    wherever it occurs; survives wrapped + concatenated flags),
    ordered-pair quote resolution (end must follow start; first valid
    pairing wins; whitespace-normalized fallback marked OK-NORM),
    case-encoded outputs out/<ver>_p<pid>s<succ>s<fail>, max_tokens
    65536, no temperature override, ThreadPool <=6, secrets loaded
    via REPO-anchored path (SecretsStore path is cwd-relative)
  prompt_v5.md: the winning prompt (classes muse/cycle/loop with
    CYCLE-eats-discarded-branches boundary; quotes verbatim-as-
    appearing-markdown-included 4-12 words distinctive; flag-every-
    might-qualify opener with clear/probable split)
  out/{v1,v2,v3,v4,v5_p140s7s1,v5_p53s0s6}/: raw responses + digests
  out/probables_check.json: the 16 probable flags handed to auditors
pilot_cases:
  p140 certify (3-coloring): s7 success + s1 cap-hit
  p53 adversary (command sequences M0-M3/credit c): s0 success + s6
    cap-hit. Pair order deterministic (SHUFFLE_SEED 42)
audit_result: 14/16 probable flags real episodes; 13/16 correct-class.
  Failures: one LOOP that was actually a CYCLE; one false MUSE
  ("prompt never mentions a vendor" — gen/adversary.py contains the
  vendor line; annotator misremembered the prompt). Method note
  recorded: muse judgments rest on prompt text the annotator can
  misremember — keep surface_question adjacent to every trace (the
  harness does).
writebacks_paid:
  mlfactory/AGENTS.md: "Subagents (delegation discipline)", "Code and
    prompt hygiene", Provider preferences expanded (Lunaroute billing
    + GLM prompting lessons)
  ~/.pi/agent/skills/session-handoff/SKILL.md: "Placement and naming"
    + verbatim-loss sharpening
  ace/lab_notes/2026-08-26-annotation-pilot-prompt-hillclimb.md:
    full v1-v5 evidence chain, incident, audit, verdict (all uncommitted)
```

## negative_knowledge

```yaml
- shared outdir keyed only on prompt version → p53 run overwrote p140
  v5 raw outputs. Fixed: case-encoded outdir. Rule: any output path
  two runs can share must encode every input that varies.
- GLM pair-framing occasionally runs thinking to the 65k wall (v3-B
  p140, v5-C p53; one gateway 504 on retry). Costless per user ruling
  — do not engineer around it; retry once, accept duds.
- ⟦-flags wrap across lines AND concatenate without newlines — parse
  by class-head split, never line-start anchoring.
- quote failures are mostly the model reconstructing from memory and
  dropping **markdown** markers; generic verification quotes are
  genuinely ambiguous in repetitive traces → ordered-pair resolution.
- passing a labels dict where texts are expected fails SILENTLY (all
  resolutions miss) — check the dict contents, not just the count.
- SecretsStore(".mlfactory/secrets.yaml") is cwd-relative: running
  from a subdir loads an empty store → Bearer None → 401. Anchor to
  repo root.
- never put "what changed since last time" in a prompt file — the
  model has no memory; it confuses the model and risks leaks.
```

## operational_state

```yaml
running: NOTHING. GPUs free (GPU0 desktop-resident 1745 MiB, GPU1 ~0).
tombstones:
  - llama-qwen38 service STOPPED (user direction, previous phase) — do
    NOT auto-restart.
  - Vast 48783410: exited/stopped, reserved at $0.09/hr — user
    destroys; do not destroy or restart without user.
  - xsub_q8.abort1_badctx.*: archived incident evidence, NOT data.
scale_launch_recipe: |
  cd /home/admin/mlfactory
  mlfactory/experiments/ace/.venv/bin/python \
    -m mlfactory.experiments.ace.annotate.pilot.run_pilot \
    --prompt-version v5 --model glm-5.2-vision \
    --pid <pid> --success <s> --fail <s> [--only A1,A2,B,C]
  # parallelize across cases; <=6 Lunaroute requests in flight.
  # candidate pairs: data/annotate_pairs_p1.jsonl (71 pairs: 44
  # hf-bf16 + 27 q8-mtp; 255 cap-hit loop targets) and the 96 xsub
  # traces (data/xsub_*.jsonl, sidecar'd).
  # audit probable tiers afterwards with qwen/qwen3.7-plus subagents
  # (recipe: probables manifest + rubric + trace offsets; see this
  # session's three-agent audit).
git: NOTHING committed this session — AGENTS.md, skill, lab notes,
  pilot harness+outputs all untracked/modified; user decides commits.
```

## open_questions

```yaml
- framing lock: C primary + A fallback is the evidence lean; user
  saw the recap but hasn't explicitly locked it. Bet: lock C+A.
- scaling subset first: q8 xsub traces (the iteration substrate) vs
  the annotate_pairs_p1 pair set. Bet: xsub first — same-prompt
  sibling pairs exist within it and it feeds trace-centric manifest.
- trace-centric annotation manifest restructure (previous handoff's
  step 7) — fold in the pilot's span schema when built.
- R0 double-annotation at scale — design when scaling starts.
```

## pointers

```yaml
concept: ace/ANNOTATION_SIDESTEP.md (verbatim idea text lives there)
pilot_record: ace/lab_notes/2026-08-26-annotation-pilot-prompt-hillclimb.md
collection_record: ace/lab_notes/2026-08-26-xsub-collect-complete-substrate-comparison.md
winning_prompt: ace/annotate/pilot/prompt_v5.md
harness: ace/annotate/pilot/run_pilot.py
rubric: ace/annotate/RUBRIC.md
harness_rules: mlfactory/AGENTS.md (subagents, hygiene, providers)
handoff_convention: ~/.pi/agent/skills/session-handoff/SKILL.md
```

## checkpoint timeline (this session segment: pilot hill-climb)

1. principal: no ballast today (server trouble); hillclimb the probe-
   point prompt, read results via anchor greps; don't destroy Vast
   ($0.09/hr reserved) → agent recovered RUBRIC.md post-compaction,
   found gw.lunaroute.com/v1 + active models (glm-5.2-vision only).
2. principal: try single-trace, pair, and compare-&-contrast framings;
   parallelize Lunaroute <=6 → agent built pilot pair case (p140 s7
   success / s1 cap-hit), prompt_v1.md + run_pilot.py.
3. principal: default temp; leave GLM room to think → runner set
   max_tokens 65536, no temperature override.
4. v1 ran → semantics strong; agent's own harness bugs surfaced
   (cwd-relative secrets → 401; labels-vs-texts dict → silent
   resolution failure; KeyError digest bug) → fixed, v1 re-parsed
   baseline: pair framings surfaced more episodes than single.
5. principal: clean up dead code ("harmless" urllib.error) → removed;
   lesson later persisted to AGENTS.md.
6. v2 (verbatim + distinctiveness rules) → tighter spans; agent's
   splitter bug hid B's flags → fixed to class-head splitting.
7. principal (run of notes): GLM wants direct instructions, rewrite
   don't append; full-trace reviews via subagent; revision notes in a
   lab note; NEVER put "what changed" in prompts → prompt files now
   contain only sent content; PILOT_LOG became the lab note.
8. v3 → B blew the 65k thinking budget; v4 (class boundary +
   markdown-in-quotes) → B recovered, class confusion resolved.
9. principal: relax the confidence language → v5 (flag everything that
   might qualify; clear/probable split; borderline bullet removed).
10. v5 → best round: flag counts 2-4x up, C 7/11 resolved, greyzone
    material captured → generalization run p53 (adversary family):
    semantics transfer; fabricated-"vendor" muse caught.
11. incident: p53 run's shared outdir overwrote p140 v5 → case-encoded
    outdir fix, outputs preserved, both re-run; p53-C later blew the
    budget again + one 504 retry — accepted as costless duds.
12. principal (lunch check): thinking wall fine (not pay-per-token);
    audit probables now; propose compaction point when reached →
    agent launched 3 parallel backgrounded qwen/qwen3.7-plus auditors
    on 16 probable flags.
13. audit landed: 14/16 real, 13/16 correct-class; vendor-muse
    disproven against gen/adversary.py → probable tier validated,
    verdict recorded in the lab note.
14. principal: persist session directives → AGENTS.md (subagents,
    hygiene, Lunaroute/GLM) + session-handoff skill (placement/naming,
    verbatim sharpening) → this handoff; principal compacts next.
