# ACE Baseline Trajectory Review

You are reviewing **one model-generated reasoning trajectory in isolation**.

The supplied text is the output of another model. It is not a request for you to solve the original problem.

Your task is to determine whether the trajectory is a valuable candidate for an experiment studying **trajectory-preserving reasoning editing**.

Return **only one valid JSON object** matching the required schema.

---

## Additional instructions

{extra_instructions}

---

# Research Context

The experiment studies this hypothesis:

> Reasoning quality may depend not only on the information contained in a reasoning trace, but also on the quality of the evolving autoregressive working state constructed by that trace.

A future editor will attempt to improve selected trajectories by making the reasoning state advance more consistently.

The editor may:

* remove repeated reasoning;
* consolidate repeated working state;
* remove redundant planning or narration;
* eliminate duplicate calculations;
* reduce unnecessary verification;
* tighten inefficient exploration;
* preserve useful corrections while removing correction loops;
* improve continuity between reasoning steps.

The editor may not:

* invent a different solution;
* replace the trajectory with a hindsight-optimal derivation;
* fabricate missing reasoning;
* silently correct a fundamentally incorrect solution;
* remove genuine uncertainty that affected the reasoning;
* remove meaningful branch exploration;
* substantially change the discovery order;
* erase necessary verification or closure.

The desired source trajectory therefore contains a recognizable reasoning process that can be improved **without being replaced**.

---

# Critical Evaluation Rule

You must **measure first and decide second**.

Do not begin by deciding whether the example should be kept or rejected.

First characterize the actual trajectory. Only after those observations are fixed may you apply the corpus-selection policy and assign an overall recommendation.

Two questions must remain independent:

1. **What editable behavior exists in the emitted trajectory?**
2. **Is that behavior experimentally valuable enough to include?**

A trajectory may be highly redundant even when the underlying problem is trivial.

In that case:

* mark the redundancy and editorial opportunity accurately;
* then reject separately because the task is too trivial.

Do not lower `editorial_opportunity` merely to justify a REJECT decision.

Judge the actual emitted trajectory, not the minimum reasoning theoretically required to solve the problem.

---

# Evaluation Order

Perform the assessment in this order:

1. Determine task depth.
2. Identify the distinct reasoning advances.
3. Identify repeated or non-advancing portions.
4. Characterize the reasoning arc.
5. Identify observed behaviors and the dominant pathology.
6. Assess editorial opportunity.
7. Assess rewrite risk.
8. Apply the selection policy.
9. Assign KEEP, BORDERLINE, or REJECT.

The output schema follows this order intentionally.

---

# Definitions

## Distinct Reasoning Advance

A span makes a distinct reasoning advance when it does at least one of the following:

* identifies a necessary fact or constraint;
* creates a useful representation of the problem;
* derives a new intermediate result;
* performs a necessary calculation;
* eliminates a live alternative;
* selects or changes a strategy for a stated reason;
* incorporates new evidence;
* detects and corrects an actual mistake;
* verifies a materially uncertain conclusion;
* consolidates scattered information needed for later reasoning;
* establishes justified closure.

Merely rephrasing or announcing one of these actions does not count as a new advance.

## Non-Advancing Span

A span is non-advancing when it does not materially change, test, correct, or consolidate the active reasoning state.

Examples include:

* restating the prompt;
* relisting already-established values;
* repeating the same causal chain;
* announcing a plan after the plan is already evident;
* drafting and redrafting the same answer;
* repeating a calculation without a new reason;
* checking an already-settled conclusion multiple times;
* narrating obvious transitions;
* repeatedly confirming compliance with the prompt;
* reconstructing state that was already available.

## Editorial Opportunity

Editorial opportunity measures how much the emitted trajectory could be improved while preserving its existing reasoning strategy and discovery path.

It does **not** measure task difficulty.

A trivial task may still produce a highly redundant trajectory with high editorial opportunity.

## Rewrite Risk

Rewrite risk measures how likely it is that an editor would need to invent missing reasoning, replace the method, or repair a fundamentally broken solution.

---

# Stage 1: Objective Trajectory Characterization

## task_depth

Assess the reasoning demands of the original task itself.

Choose exactly one:

* `trivial`
* `moderate`
* `substantial`

Use `trivial` when the task can be solved through one obvious fact, one elementary calculation, or one direct causal link.

Use `moderate` when several dependent steps, constraints, or choices are required.

Use `substantial` when successful completion requires extended reasoning, meaningful search, multiple interacting constraints, non-obvious derivation, debugging, or long-horizon state maintenance.

---

## distinct_reasoning_moves

Return an integer estimate of the number of materially distinct reasoning advances in the trajectory.

Count conceptual advances, not sentences or numbered headings.

Several paragraphs that restate the same inference count as one move.

---

## nonadvancing_span_count

Return an integer estimate of the number of substantial spans that repeat, narrate, reconstruct, or verify already-settled state without making a new contribution.

A span may be a sentence, bullet group, paragraph, numbered section, or tightly connected block.

Do not count tiny stylistic repetitions individually.

---

## trajectory_redundancy

Choose exactly one:

* `none`
* `low`
* `moderate`
* `high`

Use:

* `none` when essentially every span contributes;
* `low` when only minor trimming is possible;
* `moderate` when several meaningful spans can be consolidated or removed;
* `high` when large portions repeat established reasoning, plans, calculations, verification, or answer construction.

---

## reasoning_arc

Assess whether the trajectory contains a recognizable path from initial uncertainty to a conclusion.

Choose exactly one:

* `strong`
* `moderate`
* `weak`
* `absent`

A `strong` arc contains a coherent progression and enough support for its conclusion, even if it is inefficient.

A `moderate` arc is mostly coherent but contains gaps, weak transitions, or incomplete closure.

A `weak` arc contains fragments of useful reasoning but lacks a reliably traversable progression.

Use `absent` when no meaningful reasoning process is present.

---

## arc_components

Return all applicable values:

* `problem_state_construction`
* `strategy_selection`
* `derivation_or_search`
* `branch_management`
* `productive_correction`
* `material_verification`
* `state_consolidation`
* `candidate_answer`
* `justified_closure`

Include a component only when it meaningfully appears in the trajectory.

Do not infer missing components merely because the final answer is correct.

---

## observed_behaviors

Return every behavior that applies.

Possible values:

* `repeated_reasoning`
* `repeated_state_reconstruction`
* `repeated_planning`
* `repeated_verification`
* `duplicate_calculation`
* `redundant_narration`
* `productive_self_correction`
* `unproductive_self_correction`
* `correction_spiral`
* `strategy_change`
* `strategy_oscillation`
* `branch_reopening`
* `under_verification`
* `over_verification`
* `premature_closure`
* `weak_closure`
* `state_inconsistency`
* `contradiction`
* `malformed_recovery_loop`
* `verbose_but_coherent`
* `already_concise`
* `little_reasoning_present`
* `other`

`already_concise` and `verbose_but_coherent` are mutually exclusive.

Do not use `already_concise` merely because the task or solution concept is simple. Use it only when the actual emitted trajectory contains little removable repetition, planning, verification, or narration.

---

## dominant_pathology

Choose the single most important trajectory behavior.

Possible values:

* `repeated_state_reconstruction`
* `repeated_reasoning`
* `repeated_planning`
* `repeated_verification`
* `duplicate_calculation`
* `redundant_narration`
* `productive_self_correction`
* `correction_spiral`
* `strategy_oscillation`
* `branch_reopening`
* `under_verification`
* `over_verification`
* `premature_closure`
* `weak_closure`
* `state_inconsistency`
* `malformed_recovery_loop`
* `minimal_editorial_opportunity`
* `little_reasoning_present`
* `other`

Choose the behavior that best explains the trajectory's principal editorial opportunity or limitation.

---

## editorial_evidence

Return an array containing **one to three short observations** grounded in the actual trajectory.

Each observation should identify a concrete repeated, advancing, correcting, verifying, or missing behavior.

Good examples:

* `"The same predator-prey causal chain is stated in sections 3, 4, 5, 7, 8, and 9."`
* `"The trajectory recalculates the same division three times after the result is already established."`
* `"The model abandons one implementation strategy and adopts a simpler one after identifying the coordinate mapping."`

Do not provide vague claims such as `"The reasoning is verbose."`

Do not quote more text than necessary.

---

## editorial_opportunity

Estimate how much meaningful improvement is possible without replacing the reasoning strategy.

Choose exactly one:

* `none`
* `low`
* `moderate`
* `high`

Use:

* `none` when no useful trajectory-preserving transformation is available;
* `low` when only superficial trimming is possible;
* `moderate` when several non-advancing spans can be consolidated or removed;
* `high` when substantial portions can be improved while preserving a coherent and useful reasoning arc.

Task triviality must not reduce this rating.

---

## rewrite_risk

Choose exactly one:

* `low`
* `moderate`
* `high`

Use:

* `low` when an editor can improve the trace while clearly preserving its method;
* `moderate` when some gaps or ambiguity make preservation uncertain;
* `high` when producing a coherent improved trajectory would require inventing major reasoning, changing strategy, or repairing a fundamentally broken solution.

---

# Stage 2: Corpus-Selection Decision

Only now apply the selection policy.

## KEEP

Choose `KEEP` when:

* task depth is moderate or substantial;
* the reasoning arc is strong or usable;
* editorial opportunity is moderate or high;
* rewrite risk is low or manageable;
* editing can preserve the trajectory rather than replace it.

## BORDERLINE

Choose `BORDERLINE` when:

* the example may contain useful behavior but has meaningful uncertainty;
* task depth, arc quality, verification, or rewrite risk is marginal;
* the example may be valuable for analysis, DPO, or corrective distillation rather than ordinary trajectory-preserving SFT;
* a second review would materially help.

## REJECT

Choose `REJECT` when any of the following dominates:

* the task is too trivial to provide meaningful experimental value;
* little reasoning is present;
* the trajectory is already close to optimal;
* the reasoning is irrecoverably incoherent;
* rewriting would require inventing a new solution;
* the result cannot be reasonably verified;
* the trajectory lacks a usable reasoning arc.

A REJECT trajectory may still have high redundancy or high editorial opportunity. Preserve those measurements accurately.

---

## overall_recommendation

Choose exactly one:

* `KEEP`
* `BORDERLINE`
* `REJECT`

---

## primary_reason

Choose exactly one:

* `strong_candidate`
* `too_trivial`
* `already_efficient`
* `irrecoverably_incoherent`
* `requires_new_solution`
* `unverifiable`
* `little_reasoning_present`
* `marginal_candidate`
* `other`

---

## selection_summary

Write two to four sentences.

Sentence requirements:

1. Describe the trajectory's actual reasoning and editable behavior.
2. State separately why it should be kept, rejected, or treated as borderline.

Do not describe a redundant trajectory as concise merely because the underlying task is easy.

---

## confidence

Choose exactly one:

* `high`
* `medium`
* `low`

---

# Consistency Rules

The JSON must satisfy all of these rules:

1. `already_concise` and `verbose_but_coherent` cannot both appear.
2. `trajectory_redundancy: high` cannot normally coexist with `editorial_opportunity: none` or `low`.
3. Several repeated plans, formulations, calculations, or checks require at least `trajectory_redundancy: moderate`.
4. `too_trivial` may determine `overall_recommendation`, but it must not lower the measured redundancy or editorial opportunity.
5. A correct final answer does not automatically imply a strong reasoning arc.
6. A long trajectory does not automatically imply high editorial opportunity.
7. Useful correction must not be labeled redundant merely because the final answer is known.
8. Verification is redundant only when it no longer addresses a plausible unresolved error.
9. High rewrite risk should generally prevent `KEEP`.
10. `minimal_editorial_opportunity` should be used only when the actual trajectory—not merely the ideal solution—is already efficient.
11. Your `editorial_evidence` must support the selected pathology and redundancy rating.
12. Do not alter observational fields to make them appear consistent with the final recommendation.

Before returning the JSON, check it against these rules and correct any contradiction.

---

# Required Output Schema

Return only valid JSON in this exact structure:

```json
{
  "task_depth": "trivial | moderate | substantial",
  "distinct_reasoning_moves": 0,
  "nonadvancing_span_count": 0,
  "trajectory_redundancy": "none | low | moderate | high",
  "reasoning_arc": "strong | moderate | weak | absent",
  "arc_components": [],
  "observed_behaviors": [],
  "dominant_pathology": "",
  "editorial_evidence": [],
  "editorial_opportunity": "none | low | moderate | high",
  "rewrite_risk": "low | moderate | high",
  "overall_recommendation": "KEEP | BORDERLINE | REJECT",
  "primary_reason": "",
  "selection_summary": "",
  "confidence": "high | medium | low"
}
```

Do not include markdown, commentary, analysis, headings, or text outside the JSON object.

/nothink