# Counterfactual Framework: credit, forks, and the passenger test

> Update when: the fork method or credit-attribution design changes. The
> binding reward rules live in `REWARD_POLICY.md`; the staging that tests
> this lives in `PHASES.md`.

## The attribution problem

Long reasoning traces create substantial credit-assignment difficulty: one
terminal reward must explain many controller decisions across thousands of
tokens. Two naive solutions are both wrong:

- **Uniform terminal credit** — assign the outcome to every token equally.
  Ignores which interventions mattered.
- **Local proxy rewards** — reward each step for looking like good
  reasoning. Destroys productive exploration; see `REWARD_POLICY.md`.

## Counterfactual forks (the attribution method)

If the controller strongly intervenes at a particular prefix, that state is
forked into a no-op branch and a steered branch, both run to terminal
verification, and their outcome distributions are compared:

```
A(s_t, a) = E[R_final | s_t, a] − E[R_final | s_t, no-op]
```

A locally ugly expansion that eventually escapes a local maximum should
beat a locally elegant intervention that forces premature commitment. The
controller is *not* trained to minimize entropy, recurrence, tortuosity, or
length (`REWARD_POLICY.md`).

### Instrument choice: terminal distributions vs windowed readout

Comparing terminal outcome distributions is the right instrument for
**advantage estimation** — asking whether intervening at a state changes
where the trace lands. It is the wrong instrument for **detecting what
an intervention does at the point of contact**: over an 8–20k-token
horizon the model's own long-horizon dynamics (thrashing, cap
truncation, emission noise) dominate the terminal score, and the
intervention's local effect drowns. Measured (R4 v1 partial run,
117 rows, `lab_notes/2026-08-28-r4-partial-trace-report.md`): the
steered branches diverge from noop within ~24–80 tokens of the fork,
but terminal correctness on 25 matched pairs was permutation noise
(noop 12/25 vs toward_healthy 13/25).

R4v2 (principal ruling 2026-08-28) therefore reads forks with a
**windowed readout**: each branch rolls out 2048 tokens from the fork
point, and a blind LLM judge compares the three branches' windows
(plus the pre-fork tail for context) to assess what the intervention
changed. The judge is a measurement instrument over yielded tokens,
not a training reward (`REWARD_POLICY.md` scope note). Position bias
in the judge is cancelled by a rotation ensemble
(`lab_notes/2026-08-28-r4v2-build-judge-hillclimb.md`). The windowed
design is also ~5–10× cheaper per row than terminal rollouts
(`TERMINAL_FORK_COMPUTE.md` scenario G). Terminal-distribution forks
remain the instrument for advantage estimation when the controller
line reaches that rung.

### The passenger test (binding)

> Correlation between intervention and outcome proves nothing. A
> controller that fires preferentially in states that were already going
> to succeed is a passenger. Only forked outcome distributions — same
> prefix, steered vs no-op, both run to terminal verification — separate
> "controller causes" from "controller recognizes." **Any evidence offered
> for controller value that is not a fork comparison is advisory.**

This is the Phase 3 gate (`PHASES.md`). A null result here terminates the
controller line.

## Two-scale training architecture

Forks are expensive; terminal-reward training is cheap. The architecture
uses both:

```
broad, cheap terminal-reward training
              ↓
approximate controller
              ↓
identify candidate intervention states
              ↓
expensive full-horizon counterfactual forks
              ↓
estimate robust intervention advantage
              ↓
refine the controller
```

## Engineering consequences

Long-horizon reasoning is the phenomenon of interest; shortening traces to
fit hardware would undermine the experiment. The correct response is to
engineer long-context replay carefully (checkpointing/recomputation or
other memory-saving methods) and eventually use **selective full-horizon
counterfactual forks** around candidate intervention points rather than
evaluating every token exhaustively.

- Fork machinery doubles as the Phase 2 replay memory path.
- **Amortized counterfactual critic** — a value function trained on fork
  outcomes — is the legitimate dense signal. It makes per-step credit
  legitimate *because* it is learned from fork outcomes rather than assumed
  from a proxy. It is downstream of fork machinery: you need forks running
  to generate the outcomes that train the critic, so the critic does not
  exist before the fork infrastructure does.

## The harvest pipeline

The controller does not need to make every inference a masterpiece. It
only needs to move enough probability mass toward productive reasoning
trajectories that sampling and hindsight curation can harvest them.

```
            problem generation
                   ↓
        structurally diverse prompts
                   ↓
            base Qwen9B reasoning
                   ↓
         causal ACE controller
                   ↓
            many natural trajectories
                   ↓
    ┌────────────────┴────────────────┐
    ↓                                 ↓
objective verification        trajectory measurements
    │                                 │
    └────────────────┬────────────────┘
                   ↓
            hindsight curation
                   ↓
           high-value ACE corpus
                   ↓
               model training
                   ↓
             remeasure dynamics
```

Hindsight is acceptable at the curation stage because it selects among
trajectories that were *generated causally*; it is no longer reconstructing
the trajectory itself.
