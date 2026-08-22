# ACE Trajectory Stratification

## Context

You are annotating one raw reasoning trajectory.

The annotation answers:

> What kinds of trajectory-preserving transformation does this example support?

A trajectory-preserving transformation improves how an existing reasoning path is expressed while retaining its substantive development: strategy, order of discovery, meaningful uncertainty, useful struggle, productive corrections, important branch changes, necessary verification, and conclusion.

Your output is a compact annotation record describing the trajectory and the kinds of transformation it could support.

## Definitions

A **reasoning advance** materially changes the active solution state by establishing a fact or constraint, deriving a result, eliminating an alternative, selecting or changing strategy, incorporating evidence, detecting or correcting an error, performing meaningful verification, consolidating information needed later, or establishing justified closure.

**Useful struggle** is difficulty that materially shapes the solution, such as detecting a real mistake, discovering a constraint, eliminating an approach, motivating a better representation, producing a successful correction, or triggering a productive strategy change.

**Non-advancing behavior** repeats, narrates, or reconstructs the current state without materially changing it. Examples include restating the problem, relisting settled facts, announcing an already evident plan, repeating a calculation, reconstructing the same conclusion, reopening a rejected branch without new evidence, repeatedly drafting the answer, or repeatedly checking an already established result.

**Premature commitment** occurs when a candidate answer, strategy, diagnosis, or conclusion is treated as settled before the relevant evidence or verification is complete.

**Closure** is the point where the trajectory has established the answer and resolved the material uncertainty required by the task.

**New reasoning** is a derivation, fact, check, strategy, correction, or conclusion that is absent from and unsupported by the source trajectory.

## Positive attribute labels

Select every positive attribute visibly present in the trajectory:

- `coherent_arc`: recognizable progression from initial interpretation toward a conclusion.
- `nontrivial_reasoning`: multiple meaningful reasoning advances.
- `useful_struggle`: difficulty, uncertainty, or a false start materially contributes to later progress.
- `productive_self_correction`: detects and successfully repairs a meaningful mistake.
- `useful_branch_exploration`: alternatives help eliminate possibilities or select an approach.
- `productive_strategy_change`: changes methods in response to a meaningful discovery, failure, or constraint.
- `meaningful_verification`: a check materially increases confidence, catches an error, or establishes closure.
- `effective_state_consolidation`: creates a useful summary, representation, or intermediate state supporting later reasoning.
- `stable_representation`: adopts and maintains notation, terminology, or a representation supporting continued progress.
- `appropriate_uncertainty`: candidates remain provisional until sufficient evidence is available.
- `evidence_before_commitment`: relevant evidence is gathered and evaluated before settling on a conclusion.
- `justified_closure`: concludes after necessary reasoning and verification are complete.
- `effective_action_state_integration`: a test, tool, execution, or simulated state is incorporated into the next reasoning step.
- `efficient_progression`: most spans make a distinct contribution.

## Negative attribute labels

Select every negative attribute visibly present in the trajectory:

- `redundant_narration`: narrates or restates reasoning without advancing it.
- `repeated_planning`: repeatedly announces or reconstructs an established plan.
- `repeated_state_reconstruction`: repeatedly rebuilds facts, constraints, calculations, or conclusions.
- `duplicate_calculation`: repeats a calculation or derivation without a new purpose.
- `redundant_verification`: equivalent checks continue after material uncertainty is resolved.
- `under_verification`: commits before completing a materially necessary check.
- `branch_reopening`: a rejected or settled branch returns without new evidence.
- `strategy_oscillation`: moves repeatedly between approaches without stable evidentiary reason.
- `correction_spiral`: circles through repair attempts after an effective correction is available.
- `premature_commitment`: treats a candidate answer or strategy as settled before adequate support.
- `weak_closure`: necessary evidence is present but not consolidated into a clear, justified conclusion.
- `overextended_closure`: continues reasoning after the answer is justified.
- `action_state_disconnect`: a tool, test, execution, or simulation result is not incorporated before the next action.
- `representation_churn`: repeatedly changes equivalent representations in a way that causes unnecessary reconstruction.
- `state_inconsistency`: later reasoning conflicts with an established fact, constraint, result, or decision.
- `unresolved_material_error`: a meaningful error remains active or supports the final conclusion.
- `incomplete_arc`: ends before completing the substantive reasoning arc.

## Transformation-support labels

Select every label accurately describing a trajectory-preserving transformation supported by the source. These are annotations, not editing instructions.

- `SPAN_REMOVAL`: complete non-advancing spans could be removed while leaving the substantive arc intact.
- `STATE_CONSOLIDATION`: repeatedly reconstructed state could be represented as stable working state.
- `VERIFICATION_CALIBRATION`: verification could be reduced, consolidated, or positioned more effectively while retaining needed checks.
- `CORRECTION_PRESERVATION`: a productive mistake-and-correction sequence could retain useful struggle while reducing surrounding repetition or confusion.
- `BRANCH_RESOLUTION`: meaningful exploration exists alongside rejected branches that repeatedly return without new evidence.
- `STRATEGY_TRANSITION`: a productive method change has a trigger and transition that could be clearer or more efficient.
- `COMMITMENT_SEQUENCING`: a candidate is treated as settled before evidence already present later supports commitment.
- `CLOSURE_CALIBRATION`: closure could be reached more directly using present reasoning, or reasoning could stop once justified.
- `ACTION_STATE_INTEGRATION`: external results could be incorporated more directly into subsequent reasoning state.
- `REPRESENTATION_NORMALIZATION`: shifting or equivalent representations could be stabilized without removing a productive representation change.

## Attribute consistency

Positive and negative attributes are independent and may describe the same portion of a trajectory.

A sequence may contain useful struggle while also containing repetition, narration, reconstruction, or unnecessary verification.

Examples:

* `productive_self_correction` may coexist with `repeated_state_reconstruction` or `correction_spiral`.
* `meaningful_verification` may coexist with `redundant_verification`.
* `productive_strategy_change` may coexist with `strategy_oscillation` or `repeated_planning`.
* `coherent_arc` may coexist with any negative attribute.
* A correct conclusion does not imply `efficient_progression`.
* A supported positive attribute does not cancel a supported negative attribute.

After writing the chronological summary, assess the positive labels and negative labels independently. Record every label directly supported by the trajectory.

## Annotation sequence

Complete fields in this order:

1. Describe the reasoning trajectory chronologically.
2. Record all positive attributes visibly present.
3. Record all negative attributes visibly present.
4. Record all transformation-support labels justified by those observations.
5. Indicate whether fully repairing the trajectory would require reasoning absent from the source.
6. Report confidence.

The chronological summary should describe major approaches, discoveries, mistakes, corrections, branch changes, verification, and conclusion in order.

## Output shape

Return one valid JSON object using this exact structure:

```json
{
  "trajectory_arc_summary": "A concise chronological description of what happened across the reasoning trajectory.",
  "positive_attributes_observed": [],
  "negative_attributes_observed": [],
  "transformation_support_labels": [],
  "requires_new_reasoning_for_full_repair": false,
  "confidence": "high"
}
```

Use only labels defined above. Use only confidence values `high`, `medium`, or `low`. Each array may contain any number of labels, including zero.

Set `requires_new_reasoning_for_full_repair` to `true` when complete repair would require an absent derivation, check, strategy, correction, continuation, or conclusion.

Return JSON only.

## Additional instructions

{extra_instructions}

## Input

### Problem

{{PROBLEM}}

### Raw reasoning trajectory

{{TRAJECTORY}}
