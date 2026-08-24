# ACE Rewrite Examples

This file preserves concrete cases from pilot work that informed the rules in `SKILL.md`. Use it as a regression reference when evaluating edits or refining the skill.

---

## 1. Do not invent correction arcs

**Source:** seed 5011, `manufacturing_defect_analysis`

The source opens by framing the problem as a material-change-driven interfacial stress/compaction issue, not a lamination-vs-sintering blame dispute. There is no prior wrong framing to correct.

**Bad rewrite:**

> “My first instinct is to raise pressure and lower sintering temperature, but that sequence is backwards. The upstream suspect is the material, so the first lever should be material, not process.”

**Why it is bad:** The rewrite invents a mistaken first instinct that did not occur in the source. It makes the trajectory look more pedagogically satisfying but falsifies the reasoning history.

**Good rewrite:** Preserve the source’s opening framing. Do not manufacture a mistake-and-correction arc.

---

## 2. Do not emit ACE operator labels

**Source:** seed 5011

**Bad rewrite:**

> **Advance:** The evidence points to the material change as the upstream driver.
> **Eliminate:** Carbon fiber is not a defensible primary option for Monday.
> **Validate:** The 30/40/64-kip values are vehicle loads; the 141-kip value is a girder-level effect.

**Why it is bad:** These labels are editorial descriptors, not tokens the rewritten reasoning should contain. Training on labeled text risks teaching the model to LARP the taxonomy rather than perform the transitions.

**Good rewrite:** State the same content in natural language without the labels.

---

## 3. Preserve load-bearing quantitative reasoning

**Source:** seed 5011

The source notes that the 24 MPa lamination test pushed capacitance +4.2% against a ±5% customer tolerance, leaving only ~0.8% margin, and derives that Cpk ≥ 1.33 would require a process sigma around ≤0.2%.

**Bad rewrite:**

> The 24 MPa setting pushed capacitance near the customer limit, leaving little margin.

**Why it is bad:** It removes the quantitative argument that justifies rejecting 24 MPa as a permanent setting. The calculation is reasoning, not padding.

**Good rewrite:** Keep the +4.2%, ±5%, 0.8% margin, and Cpk/sigma reasoning intact.

---

## 4. Preserve temporal placement; do not relocate non-adjacent reasoning

**Source:** seed 5048, `clinical_case_review`

Original order:

1. What to tell Sarah
2. Improving vs. worsening criteria
3. Practical recommendations for the family
4. If the daughter becomes confrontational

**Bad rewrite:**

Move sections 3 and 4 forward to merge them with section 1, producing one large family-communication block before the improving/worsening criteria.

**Why it is bad:** It reorganizes the reasoning trajectory for thematic cleanliness. The improving/worsening criteria sit between the two family-communication sections in the source; relocating them erases the actual order of thought.

**Good rewrite:** Keep the original order. Compress each section locally, or remove repeated items from the later section if they add no new state after the intervening criteria.

---

## 5. Do not collapse layered uncertainty

**Source:** seed 5046, `pharmacokinetic_dosing`

The source maintains two simultaneous states:

- Epistemic reconstruction: best-supported count is **5 doses most likely, 6 possible**.
- Safety/operational assumption: **behave as though 6 dose-equivalents may have occurred**.

**Bad rewrite:**

> We should assume the patient received about 5–6 doses and hold the next warfarin dose.

**Why it is bad:** It collapses the distinction between what the evidence best supports and what the clinical decision must conservatively assume. These are separate reasoning states.

**Good rewrite:** Preserve both states explicitly, as the source does.

---

## 6. Quantitative reasoning must be load-bearing

**Source:** seed 5046

The source walks through an arithmetic contradiction:

- 4 doses → 3 tablets remaining, but tray shows 1.
- 5 doses → 2 tablets remaining, but tray shows 1.
- 6 doses → 1 tablet remaining, matching the tray count.

**Bad rewrite:**

> The tray count of 1 tablet remaining suggests 6 doses were given, but the notes are inconsistent.

**Why it is bad:** It skips the step-by-step contradiction that makes the reconstruction meaningful. The arithmetic is the reasoning.

**Good rewrite:** Keep the 4/5/6-dose scenarios and their expected tablet counts.

---

## 7. Useful struggle vs. useless struggle

**Source:** seed 5011

The source explores several possible physical mechanisms for the defect pattern (packing change, decomposition change, thermal stress, trace impurities) before concluding that material change is the upstream driver and pressure/temperature are compensating levers.

**Useful struggle:** Keep the mechanism exploration because it eliminates alternative root causes and motivates the material-first recommendation.

**Useless struggle:** If the source had proposed a mechanism, immediately accepted it, then restated the same mechanism three more times without changing the conclusion, the repetitions could be removed.

---

## 8. Abstain rather than repair

**Hypothetical case:**

A source trace reaches an intermediate step, identifies a contradiction, and then stops without resolving it. The correct answer is obvious to you as the editor, but the source does not derive it.

**Bad rewrite:**

> The contradiction implies X, so the next step is Y.

**Why it is bad:** You are completing the reasoning from outside knowledge. That is corrective distillation, not ACE trajectory editing.

**Good rewrite:** Mark the record as `not_trajectory_preserving` and stop. Upstream verification should decide whether to exclude the trace or handle it differently.

---

## 9. Softening invented premises

**Source:** seed 5048

The source has a section titled “If the daughter becomes confrontational.” The prompt says Sarah is pushing for transfer, not that she is confrontational.

**Bad rewrite:** Keep the “confrontational” framing as if it were a stated fact.

**Good rewrite:** Preserve the branch but soften the premise: “If disagreement persists or escalates.” This removes the invented premise without deleting the conditional branch.

---

## 10. Preserve earliest occurrence, not most informative occurrence

**Source:** seed 5046

The source states early that the MAR should not be treated as proof of 6 doses. Later, after the full evidence analysis, it repeats the same point in stronger terms.

**Bad rewrite:** Delete the early statement and keep only the later, more fully argued version.

**Why it is bad:** It erases the original state and the trajectory that produced it. The later statement is downstream of the evidence analysis; the earlier statement establishes the initial epistemic stance.

**Good rewrite:** Keep the early statement. Remove or shorten the later verbatim repetition unless intervening evidence has materially changed the claim.
