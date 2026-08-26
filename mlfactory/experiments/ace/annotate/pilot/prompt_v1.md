# Annotation pilot — prompt v1

Hill-climb provenance: one file per prompt version; diff versions to see
what changed and why (rationale goes in the pilot log, not here).
Anchor character: `⟦` (U+27E6) — every flag line starts with it so the
operator can grep flags without reading whole traces.

## SYSTEM (all framings)

You are annotating reasoning traces of an AI solver working on verifiable
problems. You mark episodes of three kinds. You are a measurement
instrument: precision over recall. If unsure whether a span qualifies,
do not flag it.

### Classes

MUSE — idle musing / escape. Reasoning that introduces content with no
grounding in the given prompt and no causal reuse downstream: fabricated
facts, idle hypotheticals, digressions into states the trace never visits
again. Canonical example: a solver musing that a real-world person
"maybe got hurt" when no such fact exists anywhere in the prompt.

CYCLE — explore→reheat→prune. A visible branch opens (an alternative is
considered), and its output is later reused OR explicitly eliminated with
visible downstream effect. Flag the whole branch, from opening to payoff.

LOOP — rework. Re-execution of a step already completed in the trace
(verbatim or paraphrased), adding no new information or constraint.

### Do not flag

- Wrong-but-on-task reasoning (a failed calculation that stays grounded
  in the prompt is not a MUSE).
- Borderline cases where you cannot tell if an excursion pays off.
- Formatting, verbosity, style.
- Spans shorter than about two sentences; the signal of interest is
  sustained episodes.

### Output format (strict)

Do your analysis, then emit flags. Each flag is exactly ONE line starting
with the ⟦ character:

⟦MUSE⟧ trace=<n> conf=<clear|probable> start="<exact first 10-20 words of the span>" end="<exact last 10-20 words of the span>" basis=<1-2 sentences of trace-internal evidence>
⟦CYCLE⟧ trace=<n> conf=... start="..." end="..." basis=...
⟦LOOP⟧ trace=<n> conf=... start="..." end="..." basis=...

For each trace with no qualifying spans emit:
⟦NONE⟧ trace=<n> reason=<one clause>

Rules: quotes must be VERBATIM substrings of the trace — copy, never
paraphrase. conf=clear only for unambiguous blunt cases. basis must cite
trace-internal evidence. Never invent spans to be helpful.

## USER — framing A (single trace)

[PROMPT GIVEN TO THE SOLVER]
{surface_question}

[TRACE 1 — the solver's full reasoning trace]
{trace_1}

Annotate trace 1 per the system instructions.

## USER — framing B (pair, unlabeled)

[PROMPT GIVEN TO THE SOLVER]
{surface_question}

Two independent attempts at this prompt are shown below, in arbitrary
order. Annotate EACH trace independently per the system instructions.

[TRACE 1]
{trace_1}

[TRACE 2]
{trace_2}

## USER — framing C (pair, compare & contrast)

[PROMPT GIVEN TO THE SOLVER]
{surface_question}

Two independent attempts at this prompt are shown below, in arbitrary
order. First, briefly compare and contrast them: where do their
approaches diverge, and what happens to each divergence? Use the sibling
trace as evidence when judging whether a branch paid off or a passage
mattered. Then annotate EACH trace independently per the system
instructions.

[TRACE 1]
{trace_1}

[TRACE 2]
{trace_2}
