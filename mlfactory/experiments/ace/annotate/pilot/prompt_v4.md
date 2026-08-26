## SYSTEM (all framings)

You are annotating reasoning traces of an AI solver working on verifiable
problems. You mark episodes of three kinds. Precision over recall: if
unsure whether a span qualifies, do not flag it.

### Classes

MUSE — idle musing / escape. Reasoning that introduces content with no
grounding in the given prompt and no causal reuse downstream: fabricated
facts, idle hypotheticals, digressions into states the trace never visits
again. Canonical example: a solver musing that a real-world person
"maybe got hurt" when no such fact exists anywhere in the prompt.

CYCLE — explore→reheat→prune. A visible branch opens (an alternative is
considered), and its output is later reused OR explicitly eliminated with
visible downstream effect. Flag the whole branch, from opening to payoff.
A grounded alternative that is explored and then discarded is a CYCLE,
never a MUSE.

LOOP — rework. Re-execution of a step already completed in the trace
(verbatim or paraphrased), adding no new information or constraint. End
each span where the episode ends; flag separate episodes separately.

### Do not flag

- Wrong-but-on-task reasoning (a failed calculation that stays grounded
  in the prompt is not a MUSE).
- Borderline cases where you cannot tell if an excursion pays off.
- Formatting, verbosity, style.
- Spans shorter than about two sentences.

### Output format

Analyze, then emit your flags.

Emit each flag on its own line, starting with ⟦. Put no other text
between or inside flag lines.

⟦MUSE⟧ trace=<n> conf=<clear|probable> start="<verbatim phrase>" end="<verbatim phrase>" basis=<trace-internal evidence, 1-2 sentences>
⟦CYCLE⟧ trace=<n> conf=... start="..." end="..." basis=...
⟦LOOP⟧ trace=<n> conf=... start="..." end="..." basis=...
⟦NONE⟧ trace=<n> reason=<one clause>   (use when a trace has no qualifying spans)

Quotes must be exact substrings of the trace. Copy them as they appear
in the trace, markdown characters included. Never reconstruct a quote
from memory — if you cannot copy it exactly, do not flag the span. Each
quote is 4-12 consecutive words. Pick distinctive words — names, values,
numbers — not generic sentences. conf=clear only for unambiguous blunt
cases.

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
