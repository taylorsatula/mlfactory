# ACE Trajectory Stratification — Negative Pass

## Context

You are annotating one raw reasoning trajectory for the negative behaviors and supported transformations it visibly contains. Do not record positive attributes in this pass; a separate pass will handle those.

**Non-advancing behavior** repeats, narrates, or reconstructs the current state without materially changing it. Examples include restating the problem, relisting settled facts, announcing an already evident plan, repeating a calculation, reconstructing the same conclusion, reopening a rejected branch without new evidence, repeatedly drafting the answer, or repeatedly checking an already established result.

**Premature commitment** occurs when a candidate answer, strategy, diagnosis, or conclusion is treated as settled before the relevant evidence or verification is complete.

**Closure** is the point where the trajectory has established the answer and resolved the material uncertainty required by the task.

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

## Transformation-support labels (negative basis)

Select every label accurately describing a trajectory-preserving transformation justified by the negative attributes above:

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

## Output shape

Return one valid JSON object using this exact structure:

```json
{
  "trajectory_arc_summary": "A concise chronological description of what happened across the reasoning trajectory.",
  "negative_attributes_observed": [],
  "transformation_support_labels": [],
  "requires_new_reasoning_for_full_repair": false,
  "confidence": "high"
}
```

Use only labels defined above. Use only confidence values `high`, `medium`, or `low`. Each array may contain any number of labels, including zero.

Set `requires_new_reasoning_for_full_repair` to `true` when complete repair would require an absent derivation, check, strategy, correction, continuation, or conclusion.

Return JSON only.

## Input

### Problem

{{PROBLEM}}

### Raw reasoning trajectory

{{TRAJECTORY}}
