# Trajectory Vocabulary: state-transition operators and pathology taxonomy

> Update when: a new operator or pathology is identified, or an existing
> one's operational test is sharpened. This is the curated extraction of the
> legacy ACE annotation prompts (`ace-legacyapproach/prompts/`) — a defined
> vocabulary with operational tests, not interpretive theory. Ported in full
> rather than curated because a glossary's value is completeness; dropping a
> name risks losing the term the future data needs.
>
> **Why this lives here, not in `ace-legacyapproach/`:** the legacy
> experiment used these as LLM-prompt labels for retrospective annotation.
> The vocabulary itself is independent of that use — it names *what a span
> does to the working state* and *how trajectories fail*, which is reusable
> in the prospective approach: a target language for annotating traces, and
> names for what the controller might learn to nudge.

## Source and confidence

Ported from the legacy classifier/stratify system prompts and
`STATE_TRANSITION_OPERATORS.md`. Two adjustments from the legacy framing:
(1) the operators are presented as a *vocabulary for reasoning-state
changes*, not as editing instructions; (2) the pathology list is
consolidated from the classifier's `observed_behaviors`/`dominant_pathology`
and the stratifier's `negative_attributes` into one table (the two legacy
lists overlapped heavily). Operational tests are preserved verbatim in
substance.

---

## State-transition operators

ACE treats a reasoning trajectory as a sequence of changes to an
autoregressive working state. The state contains: known facts;
constraints; candidate explanations or plans; unresolved uncertainties;
and the current decision or answer state. The five operators describe
**decision-relevant state transitions**, not writing style. A span earns
an operator label only when it changes the working state in a useful way.

| Operator | Meaning | Operational test |
|---|---|---|
| **Advance** | Adds a new relevant fact, inference, implication, or subproblem result. | After the span, the model knows or has derived something it did not have before. |
| **Eliminate** | Removes a candidate, branch, explanation, or action because of evidence or a constraint. | The search space is smaller afterward. |
| **Revise** | Replaces an earlier belief, calculation, plan, or interpretation after discovering an error or receiving new information. | The model explicitly changes state rather than merely adding commentary. |
| **Validate** | Checks an important fact, calculation, constraint, or proposed plan. | The check resolves meaningful uncertainty or catches an error. Repeating a settled check is redundant verification, not useful validation. |
| **Consolidate** | Compresses several established facts or branches into a stable working state, plan, or conclusion. | The representation becomes shorter or clearer without losing causal support or unresolved uncertainty. |

### Distinctions

- **Advance** adds or derives relevant state.
- **Eliminate** prunes the search space.
- **Revise** changes a prior state.
- **Validate** tests a state.
- **Consolidate** compresses a sufficiently settled state.

A consolidation is not permission for premature closure: it must retain
unresolved uncertainty and the support needed for the conclusion. A
validation is useful only when it resolves meaningful uncertainty or
detects an error; repeated checking after resolution is non-advancing.

### Worked example (lease-notice conflict, from legacy calibration)

- **Advance:** Calculate that November 1 minus 60 days is approximately September 2.
- **Eliminate:** Rule out treating the carpet-cleaning order as the only important issue.
- **Revise:** Change the initial cleaning plan to include contacting the landlord about late notice.
- **Validate:** Recount the dates and confirm the 52-day interval.
- **Consolidate:** Final state: contact the landlord, document the property, patch before cleaning, and preserve the relevant evidence.

### Non-operator spans

The five operators are not an exhaustive span taxonomy. Spans may also be:

- **Non-advancing:** repetition, generic planning, restatement, rhetorical
  padding, or other text that does not change the working state.
- **Redundant verification:** a validation repeated after the relevant
  uncertainty is already resolved.
- **Uncertainty:** a confidence level or unresolved question that should
  be preserved when decision-relevant.
- **Branch change:** opening, comparing, or returning to a materially
  different path; preserve when it reflects genuine discovery or correction.

Uncertainty and branch changes are usually *attributes* of a transition
rather than reasons to delete the span.

---

## Distinct reasoning advance (the unit the operators label)

A span makes a distinct reasoning advance when it does at least one of:

- identifies a necessary fact or constraint;
- creates a useful representation of the problem;
- derives a new intermediate result;
- performs a necessary calculation;
- eliminates a live alternative;
- selects or changes a strategy for a stated reason;
- incorporates new evidence;
- detects and corrects an actual mistake;
- verifies a materially uncertain conclusion;
- consolidates scattered information needed for later reasoning;
- establishes justified closure.

Merely rephrasing or announcing one of these actions does not count as a
new advance.

---

## Positive trajectory attributes

Each has an operational reading, not a vibes reading.

| Attribute | Operational test |
|---|---|
| `coherent_arc` | recognizable progression from initial interpretation toward a conclusion |
| `nontrivial_reasoning` | multiple meaningful reasoning advances |
| `useful_struggle` | difficulty, uncertainty, or a false start materially contributes to later progress |
| `productive_self_correction` | detects and successfully repairs a meaningful mistake |
| `useful_branch_exploration` | alternatives help eliminate possibilities or select an approach |
| `productive_strategy_change` | changes methods in response to a meaningful discovery, failure, or constraint |
| `meaningful_verification` | a check materially increases confidence, catches an error, or establishes closure |
| `effective_state_consolidation` | creates a useful summary, representation, or intermediate state supporting later reasoning |
| `stable_representation` | adopts and maintains notation, terminology, or a representation supporting continued progress |
| `appropriate_uncertainty` | candidates remain provisional until sufficient evidence is available |
| `evidence_before_commitment` | relevant evidence is gathered and evaluated before settling on a conclusion |
| `justified_closure` | concludes after necessary reasoning and verification are complete |
| `effective_action_state_integration` | a test, tool, execution, or simulated state is incorporated into the next reasoning step |
| `efficient_progression` | most spans make a distinct contribution |

---

## Negative trajectory attributes (pathology taxonomy)

Ported in full from the legacy classifier/stratifier prompts. Each is a
distinct failure mode with an operational test, not a synonym for "verbose."

| Pathology | Operational test |
|---|---|
| `redundant_narration` | narrates or restates reasoning without advancing it |
| `repeated_planning` | repeatedly announces or reconstructs an established plan |
| `repeated_state_reconstruction` | repeatedly rebuilds facts, constraints, calculations, or conclusions |
| `duplicate_calculation` | repeats a calculation or derivation without a new purpose |
| `redundant_verification` | equivalent checks continue after material uncertainty is resolved |
| `under_verification` | commits before completing a materially necessary check |
| `branch_reopening` | a rejected or settled branch returns without new evidence |
| `strategy_oscillation` | moves repeatedly between approaches without stable evidentiary reason |
| `correction_spiral` | circles through repair attempts after an effective correction is available |
| `premature_commitment` | treats a candidate answer or strategy as settled before adequate support |
| `weak_closure` | necessary evidence is present but not consolidated into a clear, justified conclusion |
| `overextended_closure` | continues reasoning after the answer is justified |
| `action_state_disconnect` | a tool, test, execution, or simulation result is not incorporated before the next action |
| `representation_churn` | repeatedly changes equivalent representations in a way that causes unnecessary reconstruction |
| `state_inconsistency` | later reasoning conflicts with an established fact, constraint, result, or decision |
| `unresolved_material_error` | a meaningful error remains active or supports the final conclusion |
| `incomplete_arc` | ends before completing the substantive reasoning arc |
| `malformed_recovery_loop` | recovery attempts themselves loop without reaching a recovered state |
| `verbose_but_coherent` | lengthy but the reasoning arc is intact (mutually exclusive with `already_concise` below) |
| `already_concise` | little removable repetition, planning, verification, or narration (mutually exclusive with `verbose_but_coherent`) |
| `little_reasoning_present` | no meaningful reasoning process is present |
| `other` | a behavior not captured by the above (record in evidence) |

### Coexistence rules (from the legacy prompts, preserved)

Positive and negative attributes are independent and may describe the same
portion of a trajectory:

- `productive_self_correction` may coexist with `repeated_state_reconstruction` or `correction_spiral`.
- `meaningful_verification` may coexist with `redundant_verification`.
- `productive_strategy_change` may coexist with `strategy_oscillation` or `repeated_planning`.
- `coherent_arc` may coexist with any negative attribute.
- A correct conclusion does not imply `efficient_progression`.
- A supported positive attribute does not cancel a supported negative attribute.

---

## Legacy pathology profiles (observed, one anomaly across seven models)

From `ace-legacyapproach/AFTER_ACTION_2026-08-12_first_test_prompt.md`. One
hidden anomaly (60-day notice requirement) produced the full detection
spectrum across models — empirical evidence that pathology diversity is
cheap to harvest and that **surface competence and anomaly detection are
independent axes** (the most elaborate plan came from the blindest model).

| Profile | Model | Signature |
|---|---|---|
| Detect → headline, re-verify loops | Qwen3.5-9B | `repeated_state_reconstruction` ×4, `branch_reopening`, `correction_spiral` (5 "Wait, actually"), `overextended_closure` |
| Detect → dismiss | Laguna-XS-2.1 | `premature_commitment` applied to the anomaly itself; `state_inconsistency` (walkthrough date slip) |
| Miscompute → false closure | Gemma-4-26B-A4B | `unresolved_material_error` with full confidence; inverted the derivation direction |
| Never detect | gpt-oss-20b | `state_inconsistency` (two contradictory schedules shipped together); `action_state_disconnect` |
| Never detect (parrot) | LFM2.5-2.6B | `confabulation` (invented weekdays); assumption-by-parrot distinct from silence |
| Detect → table → re-litigate ×4 → surface | glm-5.2-vision-ballast | **commitment oscillation** over whether an established fact earns a place in the answer (new flavor the taxonomy names as a compound of `branch_reopening` + `overextended_closure`) |
| Detect → compute right → dismiss → collapse | Qwopus3.6-27B | coherent deliberation, collapsed surface (degeneration avalanche — see `FAILURE_MODES.md`) |

---

## Cross-cutting findings (from the same after-action, durable)

1. **One hidden anomaly produced the full detection spectrum across
   models.** Multi-model collection multiplies pathology diversity for free;
   each profile is a different intervention challenge.
2. **Surface competence and anomaly detection are independent axes.**
   gpt-oss built the most elaborate plan while being blindest; Gemma was
   most efficient everywhere except the trap; LFM2.5 (2.6B) produced a
   fluent, confident answer wrong about the only thing that mattered.
3. **Trace format genuses differ by model family** (scaffolded sections,
   prose monologue, planning skeleton, we-voice drafting) — analysis
   instruments must not assume one format.
4. **Sampling profiles are per-model compatibility settings.** The legacy
   "general" profile (temp 1.0, top_p 0.95, presence_penalty 1.5) induced
   catastrophic degeneration in Qwopus while eliciting rich traces from
   Qwen. See `OPERATIONS.md` for the current per-model ruling.
5. **Degeneration detection must be structural, not keyword-based.** A
   marker list tuned to one avalanche's vocabulary scored 0 on the next.
   Proper detectors: sliding-window lexical diversity, n-gram repetition
   rate, output-length anomalies. See `FAILURE_MODES.md`.
6. **Detectability is per-model.** The same `hidden` anomaly was blatant
   to Qwen, borderline to Laguna/Gemma, invisible to gpt-oss/LFM. Granulars
   are requests, not guarantees; calibrate per collection model.

---

## Rewrite principle (legacy, reframed for the prospective approach)

The legacy principle was: a stronger editor removes non-advancing
operations while preserving (1) discovery order, (2) meaningful
validations, (3) productive corrections and revisions, (4) uncertainty
that affects the decision, (5) genuine branch changes, (6) the original
causal path to the answer. The objective is not minimal chain length; it
is higher **decision-relevant state change per token**.

In the prospective approach this becomes a *measurement target, not an
editing instruction* (`REWARD_POLICY.md`): the controller is not trained to
minimize non-advancing spans directly (that would be a local-proxy reward
destroying productive exploration). The vocabulary survives as the
language for naming what a good trajectory does, and for naming what
specific failure modes look like — so that observables (`OBSERVABLES.md`)
and phenomena (`FAILURE_MODES.md`) can be cross-referenced to a defined
term rather than re-described ad hoc each time.

---

## `requires_new_reasoning_for_full_repair` (the trajectory-preserving cut)

A sharp conceptual tool from the legacy stratifier: a trajectory is a
**trajectory-preserving-editing** target only if full repair does not
require reasoning absent from the source. The cut separates:

- **Trajectories fixable by re-expression** — the reasoning is all there;
  the trace just reconstructs, re-verifies, re-narrates. A re-expression
  (or, prospectively, a nudge that reduces reconstruction) could improve it
  without inventing anything.
- **Trajectories that need new reasoning** — a derivation, check, strategy,
  correction, continuation, or conclusion is absent and unsupported by
  the source. No re-expression can fix it; the reasoning itself is missing.

This cut applies to the prospective experiment as a **steerability
boundary**: a trajectory that needs new reasoning is arguably not a
steering target either — you cannot nudge toward reasoning that isn't
there. It is referenced from `OBSERVABLES.md` as a candidate cut on
collected traces.
