# Annotated-pattern sidestep — the detection-first approach

> Update when: the approach is validated, killed, or reshaped (a probe
> verdict, a rubric revision, a fork spent on a nominated state). Concept
> tier. States the sidestep to `TERMINAL_FORK_COMPUTE.md` adopted
> 2026-08-26, in the originators' words where the wording is
> load-bearing, and its exact relationship to the fork architecture it
> sidesteps. Operational state (what's built, what's running) lives at
> the end and drifts — verify before acting on it.

## 1. The idea, in the author's words

Proposed by the user 2026-08-26, as a sidestep to the compute problem in
`TERMINAL_FORK_COMPUTE.md`:

> 1. We already have several hundred verified success/failure rollouts
>    (from the frontier collection and calibration passes — b1 pool etc.).
> 2. Pair them up (same problem, one success, one failure), and hand them
>    to an LLM-as-annotator.
> 3. The annotator marks spans in place (token positions within the trace)
>    for: (a) unproductive off-topic hypothetical reasoning (i.e., the
>    "counterfactual escape" failure mode — divergence into states that
>    split off from the live trajectory), and (b) places that exemplify
>    the explore→reheat→prune cycle.
> 4. Hypothesis: if there are enough annotated examples, reproducible
>    patterns will emerge at the annotated token positions in the
>    activations — something like the model's "epistemic representation of
>    a productive reasoning path" contrasted with the "divergence path."
>    Autoregressive yield doesn't know about tokens coming in the future
>    however it does have the residual stream of past tokens and
>    conditions the yield on them every time it outputs a tokens. We don't
>    need to intercept every single unproductive reasoning trace that
>    hasn't happened yet. We only need to detect them and gently nudge the
>    model to finish the mental branch in a productive way. As the
>    residual stream grows the model will become predisposed to pattern
>    matching onto productive traces because they're already
>    well-represented in the models OEM post-training.

## 2. The problem it sidesteps (and what it does NOT change)

`TERMINAL_FORK_COMPUTE.md` in one paragraph: per-state credit and the
evidence standard both require counterfactual forks — same prefix, steered
vs no-op, both run to terminal verification (~160 continuations per fork
state for a powered comparison). Candidate intervention states can only be
*discovered* by forking; chained interventions grow forks of forks,
O((2m)^d · L), re-explored per controller update. Measured: even one
underpowered marginal-fork snapshot is ~630 GPU-hours of rented hardware.
Hobbyist budget. The exponential is structural.

The sidestep attacks exactly one term of that cost: **candidate
generation.** It converts "find where to intervene" from exhaustive
forking into supervised position-level detection on already-collected
traces:

- Annotation is hindsight, and hindsight is legal for measurement — the
  ban on judges is a ban on *reward terms*, not on labels.
- Teacher-forced capture at annotated positions costs one forward per
  trace (the b1-map cost regime, ~10 s/trace on owned hardware), not
  terminal executions.
- Probes trained on those positions are candidate detectors; detectors
  nominate states; forks are then spent on O(10) nominated states per
  controller version instead of 368×3 explored blind.

**What does not change:** the passenger test remains the evidence
standard. Stated during design and kept: **"the detector nominates, forks
ratify"** — not because a doc says so but because fork comparison is the
only design that separates "controller causes" from "controller
recognizes." A detector's output is advisory until fork-validated. The
binding reward rules (`REWARD_POLICY.md`) are otherwise untouched. On the
axioms generally, the user ruled this session:

> For now we're going to kinda treat axioms in the documentation as
> suggestions and guidelines. I'd hate to ignore an approach because
> documentation written weeks ago was too strict. I am a grown adult and
> can make choices for myself based on the evidence presented. This does
> not just apply to the LLM-annotator fwiw.

## 3. The scientific core: an information asymmetry, on purpose

The annotator reads the *full* trace (it must — whether an excursion paid
off is only knowable from its downstream). The detector built from the
annotations sees only the **prefix-causal residual state h_t**, exactly
like the controller it will eventually gate. The asymmetry is not a
defect; it is the test:

- If h_t at a divergence onset already encodes the divergence before the
  divergent tokens are written, divergence is a property of **state
  dynamics** — which is precisely what `HYPOTHESIS.md` claims thrashing
  is — and a prefix-causal detector can exist.
- If h_t does not encode it (divergence is only recognizable from how the
  tokens play out), the detector is dead for steering. The annotations
  still feed hindsight curation in the harvest pipeline
  (`COUNTERFACTUAL_FRAMEWORK.md`), so the work is not wasted either way.

This is also why the design beats every observable tried so far:
`OBSERVABLES.md`'s killed metrics died as **trace-level statistics vs
trace-level labels** with prompt difficulty leaking through the pooling.
This approach moves both sides to the position level — span labels, h_t
vectors, within-trace matched controls (annotated span vs healthy span of
the same trace) — which cancels prompt difficulty and trace-specific
noise more completely than any prior design. The loop-onset probe was the
only prior position-level attempt and died of data starvation (4 loop
traces); annotation fixes n without new rollouts.

## 4. Refinements adopted this session (user, verbatim where marked)

**(a) The annotation unit is the trace, not the failure member of a
pair.** User:

> Also, please remember that we aren't exclusively looking for examples
> where the model got the terminal answer wrong. Trying to pin what
> causes a correct/failed terminal answer in a rollout is a bigger ask
> than anyone can answer. We're interested in those intermediate
> reasoning actions.

Consequences: success traces are annotation targets (a recovered musing
is the escape-vs-reheat contrast the hypothesis lives on); cap-hit traces
are loop-class material, not confounds to exclude; pairs remain as
sibling metadata for within-prompt contrasts, not as filters.

**(b) Blunt cases first.** User, on the risk that escape-onset and
reheat-onset might be indistinguishable:

> perhaps it is indistinguishable in nuanced cases but there are clear
> and present examples of the model simply getting lost in traces.
> Yesterday I saw a model go off the rails working on a NFL Scoring algo
> because it said that Jared Goff should be scoring higher and hes had a
> productive history but maybe he got hurt. The model has nothing to go
> off of that Jared Goff got hurt. Nothing in the data represented that.
> It was idle musing. Perhaps after the idea is validated on some low
> hanging fruit we can hone it to capture nuanced examples but for now we
> shouldn't try to commit to a perfect design on day one. The key is to
> work towards the right direction.

The Goff episode is the canonical `muse` example in `annotate/RUBRIC.md`:
fabrication of context-absent facts, idle and ungrounded. Pass 1
annotates blunt, high-precision cases only; the nuanced
escape-vs-reheat separability question is deferred, not dropped — it
remains the pre-registered kill for the *steering* use (§6).

**(c) Annotations are noisy measurements, not ground truth.** User, on
the judge-smuggling risk:

> we aren't treating the annotations as god, my thought is that we
> annotate a whole mess of traces and even if the model is right most of
> the time patterns will hopefully emerge.

Design response: the rubric requires a `basis` field (trace-internal
evidence) per span so weak labels are filterable; the annotator is
outcome-blinded; a double-annotated subset measures label noise as
instrumentation. The smuggling path to stay closed: probe→fire→nudge
trained end-to-end without fork validation would launder a judge score
into per-state credit. The firewall is §2's: detector nominates, forks
ratify.

**(d) Iterate on q8, polish on bf16.** User asked whether anything
functionally blocks iterating on Q8_0+MTP rollouts (orders of magnitude
faster to produce) and switching to bf16 only for polished results.
Mechanically verified this session — the pipeline has three stages with
different substrate dependencies:

1. **Annotation** — substrate-blind (annotators read text).
2. **Capture** — teacher-forced through the HF bf16 model regardless of
   where the tokens came from; activations measured are always the bf16
   model's. llama.cpp q8 cannot produce hookable activations at all, so
   q8 traces cannot smuggle q8 numerics into the measurement stack.
3. **Detection at inference** — the one real transfer step (probe trained
   on teacher-forced q8 tokens must fire during live bf16 generation).
   Measurable, not assumed: the corpus keeps hf-bf16 traces as an
   internal control.

Evidence already on record: the 2026-08-24 substrate-delta smoke
(`lab_notes/`) measured q8-vs-bf16 on the same prompts: landing-place
equivalence within noise (mean shift 1.17/8, no band flips), but
**failure dynamics are not equivalent** — bf16 failures commit wrong
answers; q8 failures keep searching to the cap. So q8 material is
loop/truncation-rich and bf16 material carries more committed-wrong and
mid-trace divergence; both substrates earn their place in the corpus for
different failure species.

## 5. Relation to the earlier architecture

- **Two-scale design (`COUNTERFACTUAL_FRAMEWORK.md`):** the framework
  already reserved a "candidate sources" slot (surviving observables +
  controller intervention magnitude). Annotated-pattern detectors are a
  strictly richer candidate source entering that slot. Nothing else in
  the two-scale loop moves.
- **Amortized counterfactual critic:** still strictly downstream of fork
  machinery. Detectors are NOT the critic — they produce candidate
  states, not advantage estimates.
- **Layer hypotheses (`LAYER_HYPOTHESES.md`):** probes are trained
  per-layer on captured residuals (all 32 layers + recurrent channel);
  whichever layers carry separability feed the still-open steering-site
  decision. Readability ≠ causal leverage still holds — separability at a
  layer is a measurement result until forks say otherwise.
- **Harvest pipeline:** even a killed-for-steering detector result feeds
  hindsight curation (select among causally generated trajectories),
  which is explicitly legal.
- **HYPOTHESIS.md `Refined by`:** class-(a) annotation operationalizes
  the provisional sharpening "expansion into states disjoint from the
  live trajectory" (one family, machine). Cross-family annotation either
  promotes it into the core claim with evidence or kills it — the pass
  pays the concept docs back regardless of probe outcomes.

## 6. Kill conditions (pre-registered)

1. **Onset null:** no layer's probe separates annotated divergence-onset
   positions from matched healthy positions within-prompt (AUROC ≈ 0.5)
   → detection dead; annotations survive for curation only.
2. **Post-hoc only:** separability exists mid/late-span but not at or
   before onset → cannot detect early enough to nudge; curation use
   only.
3. **Escape ≡ reheat at onset:** rejouined excursions (success traces)
   inseparable from disjoint excursions (failure traces) at onset → a
   nudge-on-detect policy would suppress productive exploration, which is
   the poison `REWARD_POLICY.md` exists to prevent. **Kills the steering
   use; the diagnostic survives.** This is the nuanced case deferred by
   refinement (b) — pass 1 targets blunt `muse`/`loop` where this is not
   expected to bind.
4. **Label mush:** double-annotator span agreement poor → the rubric is
   the suspect before the models are; fix rubric, not data.
5. **Transfer null:** probe trained on q8 traces fails on the bf16
   control set → patterns are token-distribution-specific, not semantic;
   retrain on bf16 or kill.

## 7. The ladder (each rung falsifiable at the previous rung's cost)

- **R0 — annotation.** Protocol + double-annotator agreement on ~10–12
  traces. Cost: API bill. RUBRIC: `annotate/RUBRIC.md` (quote-based
  spans — annotators quote span boundaries, harness resolves to offsets;
  LLMs can quote, they cannot count characters).
- **R1 — capture.** Teacher-forced activations at annotated positions,
  all 32 layers + recurrent channel, within-trace matched controls.
  Scenario-A cost on owned hardware. Capture vehicle evaluated:
  **TeaLeaves** (github.com/taylorsatula/TeaLeaves) — its
  char-region→token resolution and named query positions are exactly the
  annotation→position bridge; verified adaptations for Qwen3.5-9B:
  nested `text_config` (VL wrapper), `self_attn` exists on only 8/32
  layers (24 are `linear_attn` GatedDeltaNet with no attention matrix),
  bf16 not fp16, lift 20k-char cap. Residual hooks are
  architecture-agnostic and portable; attention-matrix capture is a bonus
  on the 8 full-attn layers only (where the model *looks* when it
  escapes).
- **R2 — probe.** Position-level within-trace/within-prompt separation,
  per layer and channel. Kill conditions §6 apply.
- **R3 — direction.** Mean-difference steering direction from
  productive-minus-divergence states. This is also the constant-λ
  fixed-direction baseline that `TERMINAL_FORK_COMPUTE.md` constraint 7
  requires the controller to beat — produced from the same data.
- **R4 — concentrated forks.** Forks on O(10) detector-nominated states.
  The only rung that spends rental money; ~2 orders of magnitude smaller
  than scenario B by construction.

## 8. Operational state at writing (drifts — verify)

- **Built:** `annotate/RUBRIC.md` (pass-1 rubric, three classes:
  `muse`/`cycle`/`loop`, quote-format spans, mandatory `basis`,
  outcome-blinded annotator); `annotate/build_pairs.py` →
  `data/annotate_pairs_p1.jsonl` (71 S/F pairs: 44 hf-bf16 + 27 q8-mtp;
  255 cap-hit loop targets); `annotate/dashboard.json` (live monitor for
  the collection below).
- **Done (2026-08-26):** cross-substrate dual-collect — 6 mid-band b2
  prompts (adversary 53/56, certify 140/145, grid 150/152) × 8 samples
  × 2 substrates, paired seeds (base 72000). q8 local (Q8_0+MTP, fp16
  KV), bf16 on rented Vast Blackwell (BF16 GGUF + draft-mtp). 96 rows
  archived: `data/xsub_q8.jsonl`, `data/xsub_bf16_gpu{0,1}.jsonl`.
  Results + decisions:
  `lab_notes/2026-08-26-xsub-collect-complete-substrate-comparison.md`
  (headline: aggregates equivalent 24/48 vs 22/48; per-prompt profiles
  diverge sharply; seed-paired outcome agreement 0.54 — substrate
  redraws trajectories, kill condition 5 is live).
- **Not built (decision waits for R2):** TeaLeaves capture adaptations;
  manifest restructure from pair-centric to trace-centric.
- **Next (waits for user decision):** trace-centric annotation manifest
  over the 96 rows, then R0 (annotation model pick: lunaroute
  `-ballast` per provider preference). Also pending: destroy-or-keep
  for Vast instance 48783410 (stopped, not destroyed).

## 9. Why this may be bedrock

The fork architecture answers "did this intervention help?" — it cannot
scale to "where should we look?" at hobbyist budget. This approach is the
first design in the experiment that answers the second question at
scenario-A cost while keeping the first question's answer standard intact.
If R2 shows prefix-causal state separability at annotated spans, the
experiment gains a reusable, cheap *where-to-look* organ in front of
every expensive mechanism it has — fork placement, critic training data,
and eventually controller gating all consume the same detector. If R2
kills it, the kill costs an API bill and a day of capture, and the
annotations still feed curation. Either way the bet is priced correctly,
which is what the fork compute problem never was.
