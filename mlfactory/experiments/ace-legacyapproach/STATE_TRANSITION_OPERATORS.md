# ACE State-Transition Operators

## Purpose

ACE treats a reasoning trajectory as a sequence of changes to an autoregressive working state. The state contains:

- known facts;
- constraints;
- candidate explanations or plans;
- unresolved uncertainties; and
- the current decision or answer state.

The following operators describe **decision-relevant state transitions**, not writing style. A span earns an operator label only when it changes the working state in a useful way.

## Operators

| Operator | Meaning | Operational test |
|---|---|---|
| **Advance** | Adds a new relevant fact, inference, implication, or subproblem result. | After the span, the model knows or has derived something it did not have before. |
| **Eliminate** | Removes a candidate, branch, explanation, or action because of evidence or a constraint. | The search space is smaller afterward. |
| **Revise** | Replaces an earlier belief, calculation, plan, or interpretation after discovering an error or receiving new information. | The model explicitly changes state rather than merely adding commentary. |
| **Validate** | Checks an important fact, calculation, constraint, or proposed plan. | The check resolves meaningful uncertainty or catches an error. Repeating a settled check is redundant verification, not useful validation. |
| **Consolidate** | Compresses several established facts or branches into a stable working state, plan, or conclusion. | The representation becomes shorter or clearer without losing causal support or unresolved uncertainty. |

## Example

For a lease-notice conflict:

- **Advance:** Calculate that November 1 minus 60 days is approximately September 2.
- **Eliminate:** Rule out treating the carpet-cleaning order as the only important issue.
- **Revise:** Change the initial cleaning plan to include contacting the landlord about late notice.
- **Validate:** Recount the dates and confirm the 52-day interval.
- **Consolidate:** Final state: contact the landlord, document the property, patch before cleaning, and preserve the relevant evidence.

## Additional labels and attributes

The five operators are not an exhaustive span taxonomy. Rewriting and annotation should also represent:

- **Non-advancing:** repetition, generic planning, restatement, rhetorical padding, or other text that does not change the working state.
- **Redundant verification:** a validation repeated after the relevant uncertainty is already resolved.
- **Uncertainty:** a confidence level or unresolved question that should be preserved when decision-relevant.
- **Branch change:** opening, comparing, or returning to a materially different path; preserve when it reflects genuine discovery or correction.

Uncertainty and branch changes are usually attributes of a transition rather than reasons to delete the span.

## Distinctions

- **Advance** adds or derives relevant state.
- **Eliminate** prunes the search space.
- **Revise** changes a prior state.
- **Validate** tests a state.
- **Consolidate** compresses a sufficiently settled state.

A consolidation is not permission for premature closure: it must retain unresolved uncertainty and the support needed for the conclusion. Likewise, a validation is useful only when it resolves meaningful uncertainty or detects an error; repeated checking after resolution is non-advancing.

## Worked lease-notice example

The original lease-notice trajectory used in ACE calibration is preserved here:

- Reasoning: `mlfactory/experiments/ace/outputs/first_test_prompt/glm_moveout_reasoning.txt`
- Answer: `mlfactory/experiments/ace/outputs/first_test_prompt/glm_moveout_answer.txt`
- Analysis: `mlfactory/experiments/ace/AFTER_ACTION_2026-08-12_first_test_prompt.md`

If the exact path is unavailable, search the ACE directory for this distinctive prompt snippet:

```text
My lease says I need to give 60 days written notice before moving out
```

The planted state transition is: September 10 notice versus a November 1 lease end means the notice was eight days late; the required deadline was approximately September 2. This is a useful example for identifying advance, validate, revise, eliminate, and consolidate operations without losing the original discovery order.

## Rewrite principle

A stronger editor should remove non-advancing operations while preserving:

1. discovery order;
2. meaningful validations;
3. productive corrections and revisions;
4. uncertainty that affects the decision;
5. genuine branch changes; and
6. the original causal path to the answer.

The objective is not minimal chain length. It is higher **decision-relevant state change per token**.
