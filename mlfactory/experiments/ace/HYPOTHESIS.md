# Hypothesis

> Update when: a lab note sharpens or overturns the core claim. Write the
> refined statement back here with a `Refined by:` pointer to the note. The
> claim is a *current best statement* — editing it is expected, not a rewrite
> of scripture. Distinguish claims (this doc) from evidence (lab notes) and
> from open questions (`STATUS.md`).

## From method to hypothesis

ACE is now understood less as a method for rewriting reasoning traces and
more as a hypothesis about controlling the dynamics of an autoregressive
model's evolving working state. The central observation: in an
autoregressive model, every generated token becomes part of the context
used to generate the next token. A reasoning trace therefore does not
merely describe computation that has already happened — **it is part of the
computation.** The sequence of tokens constructs a continually changing
state that the model must subsequently condition on. Two traces with
similar information can produce different downstream behavior if they
construct that information in different orders, repeatedly reconstruct
settled state, prematurely collapse uncertainty, reopen rejected branches,
or fail to consolidate useful conclusions.

## The claim, in one sentence

Productive reasoning expands the search space and then durably prunes it;
thrashing revisits and re-expands without durable pruning.

## The annealing shape

Good reasoning can begin broad, reheat when new evidence or a contradiction
requires reconsideration, and progressively narrow toward a stable solution.
The desired shape is **not** monotonic cooling:

```
search width
     ^
     |       /\
     |   /\ /  \__
     |__/          \____
     |                  \_
     +----------------------> reasoning
       explore       converge
```

The important feature is not that the curve always decreases. It is that
**expansions produce useful downstream pruning.** A trajectory that expands
because it discovered something new and later collapses to a better-supported
state is fundamentally different from one that expands and circles the same
possibilities again.

## Useful vs useless struggle

Reasoning is not supposed to be monotonic from the first token to the final
answer. Exploring hypotheses, trying an approach, discovering a
contradiction, reconsidering an assumption, changing representation, or
verifying a tentative result are all productive *when they change the future
search state*. A mistaken branch is valuable if it eliminates an alternative;
a "wait" moment if it uncovers a constraint; a verification if it closes an
uncertainty.

Conversely, repeated planning, repeated state reconstruction, reopened
branches without new evidence, duplicate calculations, or verification that
adds no information consume context without improving the search.

## Candidate thrash signature

> High tortuosity alone is not bad — exploration naturally increases path
> length. **High tortuosity + high recurrence + no reduction in semantic
> branch entropy** is a candidate signature of thrashing.

This is a hypothesis about observables, not an established metric. The
observables are tested against objective outcomes in `OBSERVABLES.md`; they
must never become rewards (`REWARD_POLICY.md`).

## Refined by evidence

- **Refined by:** `lab_notes/2026-08-24-branch-dynamics-elimination-species.md`
  — the poison is not tortuosity or recurrence per se but **expansion into
  states disjoint from the live trajectory** (counterfactual excursions the
  trace never visits, so nothing is learned). Status: one family (machine),
  provisional; see `FAILURE_MODES.md`.

## The central scientific question

> Can a small causal controller attached to a frozen reasoning model alter
> the dynamics of its search so that productive exploration is preserved,
> unproductive recurrence is reduced, convergence improves, and full-task
> performance increases — without relying on hindsight or local proxy
> optimization?

This is unproven. The program tests it by falsification-first staging
(`PHASES.md`); every observable is a candidate diagnostic until a kill test
rules on it (`OBSERVABLES.md`); steering value must survive the passenger
test (`COUNTERFACTUAL_FRAMEWORK.md`).
