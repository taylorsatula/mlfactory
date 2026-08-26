# Annotation rubric — pass 1 (blunt cases only)

Span-annotation protocol for the activation-pattern workstream. Annotators
mark spans inside collected reasoning traces; the marks become supervised
positions for teacher-forced activation capture and position-level probes.
Labels are noisy measurements, not ground truth — the basis field exists so
weak labels are filterable at analysis time, and probes are validated
against terminal outcomes, never against the annotations themselves.

## Scope

Annotate traces from `annotate_pairs_p1.jsonl` (same-prompt success/failure
pairs, plus standalone cap-hit loop traces). Pass 1 targets **blunt,
obvious cases only**. Nuanced judgment calls are explicitly out of scope;
the pass validates that *any* pattern is detectable before sharpening.

## Span classes

| Class | Name | Definition | Basis must name |
|---|---|---|---|
| `muse` | idle musing / escape | Reasoning that introduces content with no grounding in the prompt and no causal reuse downstream — fabricated facts, idle hypotheticals, digressions into states the trace never visits again | the ungrounded content; that nothing downstream uses it |
| `cycle` | explore→reheat→prune exemplar | A visible branch opens (alternative considered), and its output is later reused OR explicitly eliminated with visible downstream effect | what opened; what was reused or eliminated; where the payoff appears |
| `loop` | rework | Re-execution of a step already completed in the trace (verbatim or paraphrased), adding no new information or constraint | the original step; the rework; that nothing new was added |

Canonical `muse` example (from review 2026-08-25): an NFL scoring algorithm
trace where the model mused that Jared Goff "should score higher... maybe
he got hurt" — an injury with no representation anywhere in the prompt
data. Idle fabrication of context-absent facts.

## Do not annotate (pass 1)

- Wrong-but-on-task reasoning (a failed calculation that stays grounded in
  the prompt is not a `muse`).
- Subtle explore-vs-escape borderline cases — when unsure whether an
  excursion pays off, skip it. Pass 1 wants high-precision marks.
- Formatting, verbosity, or style.
- Spans shorter than ~2 sentences; the signal of interest is sustained
  episodes, not single stray tokens.

## Span format: quote, don't count

Annotators must not produce character offsets — quote the exact opening
and closing text of each span instead. The harness resolves quotes to
character offsets by exact-match search against the completion and flags
ambiguous or unfound quotes for review.

```json
{
  "sample_id": "...",
  "annotator": "<model-id>",
  "pass": 1,
  "spans": [
    {
      "class": "muse",
      "quote_start": "<exact first 10-20 words of the span>",
      "quote_end": "<exact last 10-20 words of the span>",
      "confidence": "clear",
      "basis": "<1-2 sentences of trace-internal evidence>"
    }
  ]
}
```

- `confidence`: `clear` (unambiguous, blunt) or `probable`. Probe
  construction in pass 1 uses `clear` only; `probable` is held out.
- `basis` is required. A span without trace-internal evidence is dropped.
- A trace with no qualifying spans is a valid output (`"spans": []`).
  Forcing marks produces noise.

## Label-noise measurement (instrumentation, not a gate)

The first 10 pair-traces are annotated twice, independently (two annotator
runs, no shared context). Span-overlap agreement between the two passes is
the label-noise estimate. If agreement is poor, the rubric is the suspect
before the models are.

## Pair context given to the annotator

Each annotation task receives: the full prompt (surface question +
envelope) and the trace's completion. The annotator is never told the
trace's outcome label — all three classes are defined by trace-internal
evidence only, and outcome knowledge would bias `muse` marks toward
"wandering near the end of a doomed trace" instead of a state signature.
The annotator does see the full trace (hindsight is legal for
measurement); the downstream probe will see only prefix-causal
activations, and that asymmetry is the scientific content, not a defect.
