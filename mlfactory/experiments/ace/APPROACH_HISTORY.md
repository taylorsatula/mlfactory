# Approach History: retrospective rewriting → prospective steering

> Update when: the methodology changes (e.g. a non-controller approach is
  considered, or the controller design is replaced). Records *why* the
  current approach won; the current design lives in `COUNTERFACTUAL_FRAMEWORK.md`.

## The first ACE approach (retrospective, abandoned)

A long raw trajectory was reviewed; an editor identified redundant or
non-advancing reasoning; a stronger model rewrote the trace while
attempting to preserve strategy, discovery order, uncertainty,
corrections, branch structure, and conclusion. The intention was to
create training targets with more information per token.

### What it produced (durable — the new approach inherits these)

A practical vocabulary for reasoning behaviors:

- state reconstruction
- redundant verification
- branch reopening
- correction spirals
- representation churn
- premature commitment
- overextended closure

And design constraints that carry forward to the prospective approach:

- do not invent reasoning
- do not move a branch merely because a cleaner organization is available
- preserve quantitative checks and conditional branches
- do not convert an actual trajectory into a polished hindsight solution

These survived the pivot because they constrain *any* trajectory-level
intervention, not just retrospective editing.

Two further durable deliverables:

- **The "measure first, decide second" methodology.** Characterize the
  actual trajectory first; only after those observations are fixed may you
  apply a keep/kill policy. Two questions must remain independent: *what
  editable behavior exists in the emitted trajectory?* and *is that
  behavior experimentally valuable?* A trajectory may be highly redundant
  even when the underlying problem is trivial — mark the redundancy
  accurately, then reject separately because the task is too trivial. Do
  not lower the measurement to justify the decision. The new experiment's
  kill-test protocol is a narrower instance of this; the general
  principle is the ruling. (Source: legacy classifier system prompt.)
- **Cross-model pathology diversity is cheap to harvest.** One hidden
  anomaly produced the full detection spectrum across seven models —
  clean detection, dismissal, inversion, blindness, parroting,
  re-litigation, and degeneration. **Surface competence and anomaly
  detection are independent axes** (the most elaborate plan came from the
  blindest model). This is empirical prior for substrate choice: different
  models fail differently, which matters if the experiment ever compares
  models or harvests cross-model. (Source:
  `ace-legacyapproach/AFTER_ACTION_2026-08-12_first_test_prompt.md`.)

### The fundamental weakness

The editor sees the completed trajectory and therefore knows the answer
and the eventual path. It is trying to infer, after the fact, which parts
of the trace mattered computationally. Even an extremely capable editor
can therefore make a lossy intervention: delete an apparently redundant
excursion that actually changed later search behavior, preserve elegant
reasoning that contributed little, or reorganize to a hindsight-optimal
structure. **A rewritten trace can be textually superior while producing
no improvement in the student's actual search dynamics.**

## The pivot (prospective, current)

> ACE should steer reasoning prospectively rather than reconstruct it
> retrospectively.

The desired training data is generated under a causal policy that
influences the model *while it reasons*. At generation time the controller
sees only the prefix and the model state available at that point. It
cannot inspect future tokens or the final answer. Hindsight is used later
only for evaluation and selection of trajectories, not for deciding how a
token should have been generated.

### Why this is stronger

Individual generations no longer need to be perfect. The controller merely
needs to shift some probability mass toward productive reasoning
trajectories. Generate enough samples, objectively verify them,
characterize their behavior, and retain the strongest intact examples.
The retained traces are genuine products of the causal generation process
rather than teacher-generated reconstructions.

```
prompt
  ↓
base model + causal ACE controller
  ↓
many on-policy trajectories
  ↓
objective verification
  ↓
measurement / hindsight curation
  ↓
high-quality corpus
```

## What survives from the legacy experiment

The rewritten traces are not useless. They become **controls and
perturbations**: evidence about what humans and stronger models think a
better trajectory looks like. They may eventually reveal which textual
transformations correspond to real improvements in search dynamics. But
they are no longer assumed to be the ground-truth training target.

The legacy experiment is archived in `mlfactory/experiments/ace-legacyapproach/`
(read-only; never write into it). See `LEGACY.md`.
