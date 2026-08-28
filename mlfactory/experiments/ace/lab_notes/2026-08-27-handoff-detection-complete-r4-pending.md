# HANDOFF — 2026-08-27 — detection side complete incl. extensions; R4 pending principal

> **SUPERSEDED same day by
> `2026-08-27-handoff-r4-build-inflight.md`** — R4 was approved and
> build started; read that handoff instead.
> Cold-reader pointers: `HYPOTHESIS.md` (the claim), `ANNOTATION_SIDESTEP.md`
> §1–§7 (the detection design and its kill conditions),
> `REWARD_POLICY.md` (what may never be rewarded),
> `COUNTERFACTUAL_FRAMEWORK.md` (why forks remain the evidence standard).
> Supersedes `2026-08-27-handoff-lookback-k5-rec.md` (that handoff's
> "inflight recapture" is DONE — do not relaunch it).
> Session record: `2026-08-27-overnight-b2-collect-annotate.md` (overnight
> batch), `2026-08-27-b2-merge-capture-probe.md` (merge),
> `2026-08-27-lookback-k5-rec-results.md` (lead time, K5, rec channel).

## objective_and_constraints

```yaml
objective: |
  ACE (autoregressive context engineering): test whether a small prefix-
  causal controller can steer a frozen reasoning model's search dynamics,
  using only terminal verified outcome as reward. The annotation sidestep
  converted "where to intervene" from exhaustive forking into annotated
  position-level detection. DETECTION SIDE IS NOW COMPLETE, including all
  three extensions (lookback lead-time, K5 substrate transfer, recurrent
  channel). The experiment stands at the R4 decision: concentrated forks
  on detector-nominated states — rental-spend rung, HELD for principal.
goal_deltas_this_session:
  - "Principal stance (load-bearing): 'Detection is half the battle. Now
    that we know it exists we can just try again if our first approach to
    doing something about it isn't a success.' — the intervention side is
    now treated as iteratable; the detection stack is the durable asset
    that survives failed steering attempts. Structure of failures was
    agreed: shallow (mechanism fails, detector intact, re-nominate) vs
    deep (passenger null — readable but not leverage; reshapes hypothesis,
    still not back to zero)."
  - Principal asked for results explained in real-world terms; plain-
    language framings used below carry that intent forward.
bindings:
  - Reward: terminal verified outcome ONLY (REWARD_POLICY.md); annotation
    hindsight is legal for MEASUREMENT, never as a reward term.
  - Detector nominates, forks ratify — no steering claims without fork
    comparison (passenger test).
  - R4 forks: NOT SPENT without principal's explicit go; placement and
    budget are theirs. Lead-time curve now constrains placement: nominate
    the 4–8-token pre-onset window, not the onset token.
  - No -ballast Lunaroute models (ruling 2026-08-26, "today only" —
    re-confirm with principal before using; reason was server-side trouble).
  - Artifacts immutable: never regenerate/truncate existing rows or
    capture dirs; extensions go to NEW dirs (pattern followed:
    annot_captures_{xsub,b2}_lb alongside annot_captures{,_b2}).
  - llama-server: --parallel 1 for long-trace collection (higher
    partitions ctx and silently truncates); fp KV; kill by explicit PID.
```

## world_state_delta

```yaml
result_set (all written back to docs; numbers live in the named notes):
  corpus: 280 traces (xsub 96 q8+bf16 + b2 184 q8), 231 captures,
    1349 pass1 flags / 756 resolved; 6 problem families; acc 0.658 on b2
  detection (LOO AUROC, onset vs depth-matched controls):
    cycle L18 0.992 (n=291) | loop L2 0.978 (n=285) |
    muse L16–19 0.947–0.952 clear / 0.967–0.968 all (n=13/29)
  pre_onset 0.95–0.99 (divergence encoded before the episode's tokens)
  escape-vs-reheat: loop 0.86 (stable), cycle 0.72 (deflated from 0.78),
    muse 0.957 conf=all (23v6)  -> K1/K2/K3 do not fire
  K4 label agreement: Jaccard 0.26 (xsub) / 0.417 (b2) — noisy-but-usable
  lead time (lookback): 0.97 at 2 tokens pre-onset, 0.86 at 4, 0.74 at 8,
    chance from 32 — sharp flip over last ~8–16 tokens; earliest reliable
    intervention point ~4–8 tokens upstream
  K5 transfer: q8->bf16 cycle L18 0.995, loop L2 0.985 -> PASSES;
    muse UNTESTED (bf16 muse onsets < 10; STATUS Q12)
  rec channel: real but auxiliary — within-trace perfect everywhere; LOO
    best rec_L2 cycle 0.892 / loop 0.859 / muse 0.710 (n=9); residual
    stream stays primary steering substrate
  code generalized (principal-sanctioned, "we'll be using this script
    more"): build_plan --corpus/--out; run_batch --tag/--plan/--corpus
    (429 backoff); capture_activations --corpus/--candidates/--tag +
    LOOKBACK_KS=(2,4,8,16,32,64) + decile-tagged controls;
    probe_positions --cap-dirs + lb kinds (decile-matched negs) +
    lead_time section; compute_directions --cap-dirs/--tag;
    NEW transfer_test.py; NEW probe_recurrent.py
artifacts_all_sidecared: yes (data/*.meta.json incl. overnight agent's)
```

## negative_knowledge

```yaml
- xsub pids are a SUBSET of b2 pids (53/56/140/145/150/152 recur with new
  seeds): (pid, sample_i) keys collide across corpora -> per-tag capture
  dirs. Never merge corpora into one capture dir.
- /home/admin/llama.cpp/build/bin/llama-server DOES NOT EXIST (host
  AGENTS.md is stale on this). Working binary:
  /opt/llama.cpp/qwopus/current/bin/llama-server with
  LD_LIBRARY_PATH=/opt/llama.cpp/qwopus/current/bin (build 10336).
- auroc() once divided by len(pos) twice — fixed in probe_positions.py
  and analysis/analyze_map.py; ALL pre-2026-08-26 AUROCs were deflated.
- probe lb kinds MUST use decile-matched controls (anchor depths not
  comparable); core kinds keep all-controls semantics — regression on old
  captures is byte-identical, preserve that.
- muse is concentrated in q8/b2 material; xsub bf16 traces carry <10 muse
  onsets — don't promise bf16-muse results without new collection.
- GLM annotation: framing C hits the thinking wall on rare pairs
  (p117/p142) — A-fallback is built in; don't "fix" by tightening v5.
```

## operational_state

```yaml
processes: none running; GPUs idle (GPU0 desktop-only ~1GB, GPU1 free)
tombstones:
  - lookback recapture COMPLETED (81 xsub_lb + 150 b2_lb, zero failures)
    — wrapper gone; do not relaunch
  - annotate/out/pass1|pass2 renamed to out/xsub_pass1|xsub_pass2
  - data/annot_captures_lbsmoke deleted (was scratch)
  - Vast 48783410: STOPPED/reserved $0.09/hr — untouched, decision pending
vcs: mlfactory uncommitted since ~2026-08-25; /home/admin/TeaLeaves has 2
  uncommitted generalizations (model_adapter.py, run_analysis.py).
  Principal decides both — do not commit autonomously.
next_actions: nothing pending on the agent side. Await principal on:
  1. R4 forks (placement: 4–8-token pre-onset window; budget; substrate)
  2. optional bf16 muse collect to close K5/Q12 (cheap, local+Vast question)
  3. Vast 48783410 destroy/keep
  4. TeaLeaves + mlfactory commits
```

## open_questions

```yaml
- R4 forks (top question): detection nominates with full evidence
  (231 captures + lead-time + K5 pass). Fork placement should target
  detector-nominated states 4–8 tokens before annotated onsets; baseline
  to beat = directions_annot_clear_merged.npz (constant-lambda).
  Confidence detection is ready: high. Confidence steering will work:
  genuinely uncertain — that is what forks answer.
- muse-K5 (STATUS Q12): needs bf16 collection on muse-rich prompts
  (assign/certify/hypothesis families). Confidence it will pass if
  tested: moderate (cycle/loop transferred at focal layers).
- muse best-layer set unsettled (xsub L23–29 vs merged L16–19); muse
  n still smallest class (13 clear / 29 all onsets, 19 traces).
- where causal leverage actually lives (residual L16–19/L2/L18 vs rec
  L2) — forks will discriminate; current bet: residual primary.
```

## pointers

```yaml
concepts: HYPOTHESIS.md, ANNOTATION_SIDESTEP.md, COUNTERFACTUAL_FRAMEWORK.md,
  REWARD_POLICY.md, TERMINAL_FORK_COMPUTE.md, PHASES.md
living: STATUS.md (R12 resolved+extended; Q12 open), OBSERVABLES.md,
  LAYER_HYPOTHESES.md, FAILURE_MODES.md
records: lab_notes/2026-08-27-*.md (four notes this date),
  lab_notes/2026-08-26-scale-annotation-to-r3.md (xsub R0–R3 full record)
data: data/{xsub_*,annot_b2_q8,annotations_*_pass*,annotation_plan_*}.jsonl,
  data/annot_captures{,_b2,_xsub_lb,_b2_lb}/, data/probe_results*.json,
  data/transfer_test.json, data/probe_recurrent.json,
  data/steering_directions/directions_annot_clear{,_merged}.npz
pipeline: annotate/{build_plan,run_batch,r0_agreement,capture_activations,
  probe_positions,compute_directions,transfer_test,probe_recurrent}.py —
  run as modules from repo root with ace/.venv/bin/python
runbook (reusable overnight pattern): annotate/runbook_overnight_b2.md
```

## checkpoint_timeline (this session, post-compaction)

1. Principal: overnight b2 batch ran; check results, orient → agent read
   morning note, verified ALL claims against data (flags 879/505 resolved,
   MUSE 31 across 6 families incl. machine/assign/hypothesis, machine clean).
2. Agent generalized capture/probe/directions for multi-corpus use; CAUGHT
   xsub⊂b2 pid collision before it silently mixed captures → per-tag dirs;
   regression-checked all xsub defaults (byte/bit-identical).
3. Agent captured 150/150 b2 traces (zero failures) → merged probe on 231
   captures: cycle/loop held at 3× n; muse upgraded underpowered→supported;
   cycle escape-vs-reheat deflated 0.78→0.72 (only inflation found).
4. Agent wrote back OBSERVABLES/STATUS/LAYER_HYPOTHESES/ANNOTATION_SIDESTEP
   + lab note + sidecars → principal asked for plain-language explanation
   of the numbers → agent re-read concept docs, explained AUROC/LOO/
   pre_onset/escape-vs-reheat in real terms with the what-it-does-NOT-prove
   caveat (passenger test).
5. Principal: "I can't believe it worked" → agent affirmed with the five
   pre-registered ways it could have died, kept the steering caveat.
6. Principal: detection is half the battle; intervention side now
   iteratable → agent agreed, structured shallow-vs-deep failure modes,
   flagged three free detection works (K5, rec channel, more muse).
7. Principal: would multi-token lookback be worth it? → agent proposed
   three-question design (lead time, t−1 audit, drift shape), essentially
   free capture cost → principal: yes.
8. Principal: prepare the other free works too; will compact after handoff;
   at breakfast.
9. Agent built lookback capture+probe (smoke-verified, regression identical),
   launched 231-trace recapture, ran K5 transfer test → PASSES (cycle 0.995
   L18, loop 0.985 L2 q8→bf16; muse untestable), ran rec probe → real but
   auxiliary (rec_L2 best); wrote interim handoff.
10. Recapture finished clean → lead-time curves adjudicated: sharp flip,
    ~8–16-token horizon, nominate 4–8 pre-onset for forks; all write-backs
    + sidecars + lab note done; machine left idle.
