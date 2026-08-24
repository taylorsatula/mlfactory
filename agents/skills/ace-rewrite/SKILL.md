# ACE Rewrite Editing

## Purpose

Edit sampled reasoning traces for **Autoregressive Context Engineering (ACE)**. ACE treats the model's emitted tokens as its working state: later computation proceeds from them, so their quality determines effective reasoning capability. The editor's job is to remove non-advancing state and strengthen the causal path from prompt to answer, while preserving the trajectory that actually occurred.

The output must remain plausible as an autoregressive generation. It should not read like a polished expert memo assembled after seeing the answer.

## Core principle

Optimize for **task-relevant reasoning-state progress per token**, not for brevity, polish, or hindsight-optimal organization.

A substantive reasoning span should contribute to one or more of the following:

- **Advance** the state: add a new fact, inference, implication, subproblem result, or calculation.
- **Eliminate** a branch: remove a candidate, explanation, or action because of evidence or a constraint.
- **Revise** the state: replace an earlier belief, calculation, plan, or interpretation after discovering an error or receiving new information.
- **Validate** the state: check an important fact, calculation, constraint, or proposed plan.
- **Consolidate** the state: compress several settled facts or branches into a stable working conclusion or plan.

These are editorial descriptors only. **Do not emit them as labels** in the rewritten trace.

## Source boundary

The source trace is authoritative. You may understand the problem deeply, but you must not use outside knowledge to:

- repair factual errors,
- complete missing reasoning,
- resolve unresolved uncertainty,
- improve the solution,
- or change the conclusion.

Upstream verification decides whether the trace is admissible. ACE editing serializes what is actually there.

If producing a coherent result would require reasoning absent from the source, a different solution strategy, correction of an unresolved material error, or a changed conclusion, **mark the record as not safely trajectory-preserving** and stop. Do not repair it.

## Hard rules

1. **No invented reasoning.** Do not add mistakes, realizations, corrections, strategy changes, or uncertainties that did not occur in the source. If the source begins with a framing that it does not later revise, preserve that framing. If the source later revises it, preserve both the original framing and the actual revision sequence.

2. **No operator labels.** Do not write `Advance:`, `Eliminate:`, `Revise:`, `Validate:`, or `Consolidate:` in the rewritten text.

3. **Preserve temporal placement.** Edit within the existing structure. Do not relocate non-adjacent reasoning to organize it more cleanly. Non-adjacent material may be deleted if it merely repeats already-settled state and no intervening evidence materially changes its justification.

4. **Local compression.** Combine short spans that state the same settled point. Tighten wordy framing. Stop when compression would remove a calculation, threshold, conditional branch, caveat, source of uncertainty, or step in the causal chain.

5. **Preserve load-bearing quantitative reasoning.** Keep every distinct quantitative fact, calculation, threshold, comparison, or scenario distinction that materially affects later reasoning. Repeated mentions of the same settled quantity may be consolidated.

6. **Preserve layered uncertainty.** Do not collapse distinct states into one just because they lead to the same action. Let epistemic reconstruction, safety/operational assumptions, and unresolved gaps coexist when the source keeps them separate.

7. **Preserve useful struggle.** Keep unsuccessful exploration when it materially changes what happens next: it eliminates an approach, reveals a constraint, exposes an error, motivates a representation change, or causes a strategy transition. Remove failed exploration that produces no downstream state change.

8. **Do not normalize voice or structure.** Edit each trace in its own register. Do not force domains into a single template.

## Common pathologies to remove or consolidate

- Repeated state reconstruction
- Repeated planning or narration without new content
- Duplicate calculation of an already-settled quantity
- Redundant verification after uncertainty is resolved
- Reopening a branch after it has been eliminated
- Strategy oscillation without new evidence
- Correction spirals that do not converge
- Representation churn without downstream effect
- Repeated or overextended closure

## Deletion test

Before deleting or merging a span, ask: **does a later part of the source depend on this span having occurred when it did?**

- If yes, preserve it or compress it in place.
- If no, and it only reconstructs already-available state, it is a candidate for removal.

## Preserving vs. removing repeated state

When the source repeats a conclusion, apply this rule:

- **Preserve the earliest occurrence needed to establish the state.**
- **Later repetitions may be removed** unless intervening evidence materially strengthens, qualifies, or changes that state.

Do not keep the “most informative” occurrence if doing so would erase the earlier state and the trajectory that produced it.

## Workflow

1. Read the prompt and full source trace.
2. Identify state transitions and non-advancing spans.
3. Apply the deletion test and local compression.
4. Verify against the hard rules.
5. If the source is not safely trajectory-preserving, abstain.
6. Save the edited trace with minimal transformation metadata.

## Output metadata

Use a small fixed schema. Do not write an essay about the editing process.

```json
{
  "rewrite_status": "rewritten",
  "transformations_applied": ["span_removal", "state_consolidation"],
  "source_reasoning_added": false,
  "conclusion_changed": false,
  "branch_order_changed": false,
  "judgment_flags": []
}
```

Allowed `transformations_applied` values include: `span_removal`, `state_consolidation`, `verification_calibration`, `pathology_removal`, `voice_preservation`, `abstain_repair_needed`.

Provenance (source ID, model ID, timestamp, hashes, paths) is supplied by the harness, not generated by the editor.

## Do not optimize for compression ratio

Edit magnitude should follow the amount of non-advancing state, not a target length. A well-structured source may require only small edits; a pathological source may permit large reductions.

## Abstract examples

### What to compress

Source writes four bullet points:

- Warfarin effect accumulates slowly.
- A missed dose does not immediately correct the INR.
- A dose given on Day 6 evening may not have fully expressed its effect by the Day 7 6 AM draw.
- The INR may continue to rise for another 24–48 hours.

These all express one consequential inference about delayed expression of warfarin effect under a long half-life. They can be combined into one sentence without losing state.

### What not to move

Source has:

1. What to tell the family
2. Improving vs. worsening criteria
3. Practical recommendations for the family

Section 3 repeats some items from Section 1, but Section 2 sits between them. Do **not** move Section 3 forward to merge it with Section 1. Compress each section locally, or remove the repeated items from Section 3 if they add nothing after Section 2.

### What not to collapse

Source maintains:

- Best-supported reconstruction: 5 doses most likely, 6 possible.
- Safety/operational assumption: behave as though 6 dose-equivalents may have occurred.

These are distinct states. Do not collapse them into one sentence, even though both support holding the next dose.
