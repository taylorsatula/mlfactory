# ACE Trajectory Stratification — Positive Pass

## Context

You are annotating one raw reasoning trajectory for the positive behaviors it visibly contains. Do not record negative behaviors in this pass; a separate pass will handle those.

A **reasoning advance** materially changes the active solution state by establishing a fact or constraint, deriving a result, eliminating an alternative, selecting or changing strategy, incorporating evidence, detecting or correcting an error, performing meaningful verification, consolidating information needed later, or establishing justified closure.

**Useful struggle** is difficulty that materially shapes the solution, such as detecting a real mistake, discovering a constraint, eliminating an approach, motivating a better representation, producing a successful correction, or triggering a productive strategy change.

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

## Transformation-support labels (positive basis)

Select every label accurately describing a trajectory-preserving transformation justified by the positive attributes above:

- `CORRECTION_PRESERVATION`: a productive mistake-and-correction sequence could retain useful struggle while reducing surrounding repetition or confusion.
- `BRANCH_RESOLUTION`: meaningful exploration exists alongside rejected branches.
- `STRATEGY_TRANSITION`: a productive method change has a trigger and transition that could be clearer or more efficient.
- `ACTION_STATE_INTEGRATION`: external results could be incorporated more directly into subsequent reasoning state.
- `REPRESENTATION_NORMALIZATION`: shifting or equivalent representations could be stabilized without removing a productive representation change.

## Output shape

Return one valid JSON object using this exact structure:

```json
{
  "trajectory_arc_summary": "A concise chronological description of what happened across the reasoning trajectory.",
  "positive_attributes_observed": [],
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
