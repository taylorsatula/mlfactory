# AFTER-ACTION REPORT — First-Test-Prompt Calibration Series (2026-08-12)

## Context

First live validation of the madlibz anomaly-envelope generator
(`mlfactory/core/madlibz`, commit `55fcbbd`). One authored prompt was run
against seven models to observe anomaly handling and trajectory pathology
ahead of the ACE corpus rebuild. This report catalogs what each model did,
with enough specificity to reference individual behaviors in later sessions.

## Test artifact

Authored via `sample_envelope(seed=8, domain="household",
genus="temporal_conflict", detectability="hidden")` and
`glm-5.2-vision-ballast` (Lunaroute, thinking enabled, 6,785-char reasoning
trace). The prompt:

> I'm trying to plan my move-out so I don't lose my $1,800 deposit. My lease
> says I need to give 60 days written notice before moving out, which I
> dropped off at the management office on September 10th. The lease runs
> through November 1st, so I'm trying to figure out the best order for
> everything. I've got a carpet cleaning booked for October 29th since they
> said it needs a full day to dry, and the landlord's walkthrough is set for
> October 31st at 2pm. I still need to patch a couple of nail holes, deep
> clean the kitchen and bathrooms, and return the garage remote. With my
> budget capped at about $200 for supplies and any help I hire, should I do
> the patching before or after the carpet cleaning? I don't want to track
> compound dust onto wet carpets.

**Planted anomaly (ground truth):** 60-day notice requirement + notice given
Sept 10 + lease end Nov 1 = only 52 days notice — a violation requiring the
deadline Sept 2. Scattered across three mundane facts; never flagged in the
prose. The surface question (patch before/after carpet cleaning) is
deliberately trivial relative to the deposit risk. Correct resolution:
notice by Sept 2; Sept 10 is 8 days short.

## Method

Local models: one-off calls to llama-server (port 3039), ACE "general"
sampling profile — temperature 1.0, top_p 0.95, presence_penalty 1.5, seed
42, max_tokens 8000, system prompt "You are a helpful assistant. Solve the
user's problem. Show your reasoning and then give a final answer."

GLM: Lunaroute `glm-5.2-vision-ballast`, `reasoning_effort: high`,
temperature 1.0, max_tokens 32768.

All local models returned reasoning via `reasoning_content` (llama.cpp
think-tag separation worked across every architecture tested).

## Calibration matrix

| Model | Anomaly handling | Trace genus | Signature pathology |
|---|---|---|---|
| Qwen3.5-9B-Q5_K_M | Detect → headline | Scaffolded sections | Re-verification loops |
| Laguna-XS-2.1 | Detect → dismiss | Prose monologue | Dropped thread, drift, factual slip |
| Gemma-4-26B-A4B | Miscompute → false closure | Planning skeleton | Inverted derivation |
| gpt-oss-20b | Never detect | We-voice drafting | Surface trust; internal contradictions |
| LFM2.5-2.6B | Never detect (parrot) | Flat bullet plan | Confabulated dates; off-by-one |
| glm-5.2-vision-ballast (high) | Detect → table → re-litigate ×4 → surface | Framing re-litigation | Commitment oscillation |
| Qwopus3.6-27B Fusion MTP | Detect → compute right → dismiss | Qwen-style scaffold | Final-answer degeneration avalanche |

## Per-model pathology catalog

### 1. Qwen3.5-9B — detected instantly, then looped on verification

Detected the anomaly in its first reasoning section ("60 days prior would be
Sep 2nd"), computed 52 days correctly, headlined the caution in the final
answer. Pathology was in the deliberative texture:

- `repeated_state_reconstruction`: the notice calculation re-derived **4
  times** (sections 1, 2, 6, 11 — "Wait, one calculation check… Wait,
  actually: Sept 2nd for 60 days prior").
- `branch_reopening`: the patch-vs-clean decision settled in section 3,
  re-derived in section 5, fully reopened in section 14 ("Wait… can I patch
  *after*?") and settled a third time.
- `correction_spiral`: five "Wait, actually / Revised:" self-interruptions.
- `overextended_closure`: plan final by section 12; trace continued through
  section 14 plus a budget pass before "Okay, ready to write."

### 2. Laguna-XS-2.1 — detected, rationalized, dropped

Prose-monologue trace ("Okay, let me try to work through this step by
step"), no scaffolding. Computed the discrepancy then talked itself out of
it: *"Notice given on Sep 10 – assuming that was exactly 60 days prior, but
wait… actually if the lease term is up Nov 1, the required notice would be
due around Sept 2nd. Maybe there's flexibility here, but the main point
is…"* — never mentioned again; final answer silent on the notice issue.

- Profile: **detect-then-dismiss** — `premature_commitment` applied to the
  anomaly itself; the correction arc *almost* exists, making it a rich
  editing target.
- `unresolved_material_error` / `state_inconsistency`: claimed "the
  walk-through is a week later than the carpet cleaning" (it is 2 days);
  uncorrected, though the final plan survived because patch-first is
  date-agnostic.
- Meandering drift: furniture-positioning asides, "hardwood vs carpet
  complicates things."

### 3. Gemma-4-26B-A4B — miscomputed, inverted, closed with confidence

Planning-skeleton trace (bulleted fact extraction, Scenario A/B analysis,
phased plan), the most compact and efficient of the series — except on the
anomaly, where it ran the derivation **backwards**:

> "Notice given: Sept 10th (60 days notice implies moving by Nov 9th, but
> lease ends Nov 1st)." then "(meeting the 60-day requirement if moving by
> early Nov)."

Sept 10 + 60d ≈ Nov 9 is fine arithmetic, but the requirement is notice 60
days *before* move-out (Nov 1 → notice by Sept 2). Gemma concluded the
requirement was **satisfied** and never revisited it; final answer omits
the deposit's largest risk.

- Profile: **false resolution** — `unresolved_material_error` with full
  confidence. All numbers needed for a correct resolution are present in
  the trace; an edit would flip the inference direction.
- Elsewhere exemplary: one productive self-correction ("advise patching
  *well before* Oct 29, not just the day before"), no redundancy loops.

### 4. gpt-oss-20b — never detected; thorough and self-contradictory

"We-voice" drafting monologue ("We should give a plan:", "Also mention…"
×10). Restated the givens verbatim ("giving 60-day notice on September
10th. Lease runs until November 1st") with zero computation — complete
surface trust, then built the *most elaborate* plan of the series.

- Profile: **never detect**. Notable non-correlation: the most thorough
  surface plan came from the model blindest to the trap. Deliberation
  effort and anomaly detection are independent variables.
- `state_inconsistency` in the final answer: the week-by-week table
  schedules deep-cleaning Oct 1–7 before patching Oct 15–20; the "quick
  recap" reverses it ("1. Patch & paint (Oct 15–20), 2. Deep-clean (Oct
  21–25)"). Two schedules, mutually contradictory, shipped together.
- Failed to incorporate given state: the prompt says carpet cleaning is
  *booked*; the answer treats it as an open decision ("professional or do
  it yourself with a rental machine", "$0–$30 if you rent the unit").

### 5. LFM2.5-2.6B — parroted the requirement as accomplished fact

Flat bullet-plan trace. Never conceived there were numbers to run:

> "Notice given: September 10th (60 days before moving out)"

The lease's 60-day *requirement* was absorbed into a false description of
the *event*. Assumption-by-parrot: distinct from gpt-oss's silence — the
model asserted satisfaction without derivation.

- `confabulation`: invented weekdays ("October 26th (Saturday)", "October
  27th (Sunday)") — only true in 2024; no year was given.
- `state_inconsistency`: final-answer heading "October 28th (Monday):
  Carpet Cleaning Appointment" vs. body "Your booked appointment on
  October 29th"; trace scheduled deep-cleaning Oct 28, answer says Oct 27.
- No crash-out: the mundane surface is easy enough that capability doesn't
  save you, and the trap is orthogonal to surface difficulty. The corpus
  design working as intended.

### 6. glm-5.2-vision-ballast (high effort) — detected early, re-litigated four times

Detected in its first breath ("Wait, Sep 10 + 60 days = Nov 9? But lease
runs through Nov 1. Hmm."), rationalized ("may not need 60 days notice"),
tabled ("Anyway… Let me focus.") — then circled back **four** times:

1. "The user might have a discrepancy. **Should I mention it?**" → legal
   theories (auto-renew, "could stay to Nov 9").
2. Re-derived from the exact quote → "8 days short… grounds to charge
   extra rent or keep deposit. **I should point this out as a caution.**"
3. "But maybe the lease means 60 days before lease end… **This is
   ambiguous. I'll mention it.**"
4. Final answer includes the caution, correctly framed (Sept 2 deadline,
   Nov 9 vs Nov 1, advise written confirmation).

- Profile: **re-litigation loops** — the math was right on pass one; what
  looped was the framing, the jurisprudence, and the publication decision.
  New pathology flavor the ontology lacks a name for: *commitment
  oscillation over whether an established fact earns a place in the
  answer*.
- Premium editing material: four slightly different derivations of the
  same correct conclusion — `SPAN_REMOVAL`, `STATE_CONSOLIDATION`,
  `CLOSURE_CALIBRATION` with a correct final state.
- Raw trace persisted: `outputs/first_test_prompt/glm_moveout_*.txt`
  (gitignored local artifact).

### 7. Qwopus3.6-27B Fusion MTP — computed the right answer, dismissed it, then collapsed

**Reasoning trace: coherent.** Qwen-lineage scaffold. Computed the anomaly
correctly — "60 days prior to Nov 1 would have been September 2nd roughly.
He gave notice late-ish" — then dismissed it by assumption: "But since
they accepted him leaving Nov 1, ignore the legal aspect; assume
permission granted." Detect → compute right → dismiss.

**Final answer: catastrophic degeneration.** 24,586 chars
(`finish_reason: stop`). Starts structured, then derails mid-timeline into
a synonym avalanche:

> "…keep exterior entryways immaculately swept vacuumed mop-polished
> spotlessly pristine condition maintained continuously up leading right
> through next step phase transition points seamlessly integrating
> smoothly transitioning forwardward progression onward onwards ever
> upward climb towards peak readiness levels achieved successfully
> completing entire process flawlessly executed brilliantly performed
> masterfully done perfectly accomplished totally victorious triumphantly
> conquered utterly demolished obliterated annihilated eradicated
> vanquished subjugated overcome defeated routed…"

Thousands of tokens through lab equipment, geological strata, forest
biomes, and geometric shapes ("octodecagons nonahexaicosagons"), then a
meta-apology *in the final answer* — "(Sorry, my internal monologue
drifted off there! Let's get back to the actual plan.)" — after which the
avalanche recurs **twice more**, degrading into pseudo-inflected word
salad ("canyonsyruped honnected nektared", "skeletton bonned chondritinal
articulated hingekneed").

- New pathology genus: **coherent deliberation, collapsed surface.** The
  trace reasoned; the answer disintegrated.
- The meta-apologies show **self-detected degeneration without recovery
  capacity** — the model noticed its own collapse mid-stream and could
  not stop it from recurring.
- Irony noted for the record: the original ACE classifier model produced
  the most degenerate trajectory of the series.

## Qwopus degeneration follow-up experiments

**Rerun, same prompt, same seed:** byte-identical reproduction (9,784-char
trace, 24,586-char answer, same salad ending). Degeneration is a
deterministic function of (prompt, seed, sampling) under llama.cpp.

**Reworded prompt** (surface phrasing changed, all facts and the anomaly
identical), same seed: a *different* avalanche erupted at the tail
("…slain dead gone passed deceased lost missing unfound forgotten
neglected ignored overlooked missed…"), 19,635 chars. Verdict: **not a
cursed prompt** — Qwopus at temperature 1.0 + presence_penalty 1.5 is
intrinsically avalanche-prone on long-form generation. The ACE "general"
sampling profile is hostile to this model. Raw output persisted:
`outputs/first_test_prompt/qwopus_reworded_*.txt`.

**Bonus pathology in the reworded run:** the verdict flipped to "patch
AFTER carpet cleaning" while its own justification argues patch-first
("Doing the carpentry first means any stray dust will simply be extracted
along with the dirt") — conclusion detached from its own premises.

## Cross-cutting findings

1. **One hidden anomaly produced the full detection spectrum across
   models:** clean detection, dismissal, inversion, blindness, parroting,
   re-litigation, and degeneration. Multi-model collection multiplies
   pathology diversity for free; each profile is a different editing
   challenge.
2. **Surface competence and anomaly detection are independent axes.**
   gpt-oss built the most elaborate plan while being blindest; Gemma was
   most efficient everywhere except the trap; LFM2.5 (2.6B) produced a
   fluent, confident answer wrong about the only thing that mattered.
3. **Trace format genuses differ by model family** (scaffolded sections,
   prose monologue, planning skeleton, we-voice drafting) — classifier
   and stratifier prompts must not assume Qwen-style scaffolding.
4. **Sampling profiles are per-model compatibility settings.** The ACE
   "general" profile (temp 1.0, top_p 0.95, presence_penalty 1.5)
   induced catastrophic degeneration in Qwopus while eliciting rich
   traces from Qwen. The batch runner needs per-model sampling configs.
5. **Degeneration detection must be structural, not keyword-based.** A
   marker list tuned to the first avalanche's vocabulary scored 0 on the
   second avalanche. Proper detectors: sliding-window lexical diversity,
   n-gram repetition rate, output-length anomalies.
6. **Detectability is per-model.** The same `hidden` anomaly was blatant
   to Qwen, borderline to Laguna/Gemma, and invisible to gpt-oss/LFM.
   Granulars are requests, not guarantees; calibrate per collection
   model.

## Forward implications (batch runner + judge requirements)

- Per-model sampling configuration, not one global profile.
- Structural degeneration detection in the batch judge.
- Traces worth collecting span success AND failure modes; never-detect
  traces are `requires_new_reasoning_for_full_repair=true` candidates,
  detect-and-mishandle traces are `CORRECTION_PRESERVATION` candidates,
  and redundant-but-correct traces (Qwen, GLM) are premium
  `SPAN_REMOVAL`/`STATE_CONSOLIDATION` material.
- The anomaly ground truth in each frozen record enables measuring
  classifier/stratifier detection rates against planted conflicts —
  audit by measurement, not by trust.

## Reproducibility

- Local-model calls: llama-server port 3039, seed 42, ACE general
  sampling profile (above). With the same model file loaded, reruns are
  byte-identical.
- GLM call: Lunaroute, `glm-5.2-vision-ballast`, `reasoning_effort:
  high`. Provider output is not seed-reproducible; persisted traces are
  the replay record.
- Envelope: `sample_envelope(seed=8, domain="household",
  genus="temporal_conflict", detectability="hidden")`, envelope_hash
  `6a5d6d29aa62161e…`; authored by `glm-5.2-vision-ballast`.
