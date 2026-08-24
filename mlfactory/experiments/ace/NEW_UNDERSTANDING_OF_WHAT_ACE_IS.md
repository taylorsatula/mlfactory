# ACE: Current Understanding

Autoregressive Context Engineering (ACE) is now understood less as a method for rewriting reasoning traces and more as a hypothesis about controlling the dynamics of an autoregressive model’s evolving working state.

The central observation is simple but consequential: in an autoregressive model, every generated token becomes part of the context used to generate the next token. A reasoning trace therefore does not merely describe computation that has already happened. It is part of the computation. The sequence of tokens constructs a continually changing state that the model must subsequently condition on. Consequently, two traces containing broadly similar information can produce different downstream behavior if they construct that information in different orders, repeatedly reconstruct settled state, prematurely collapse uncertainty, reopen rejected branches, or fail to consolidate useful conclusions.

The original formulation was that reasoning quality depends not only on the information contained in a reasoning trace, but also on the quality of the evolving autoregressive working state that the trace constructs. The practical objective was consequently framed as increasing the value of each reasoning token rather than simply minimizing token count. A good trace should make meaningful progress per token, but “meaningful progress” includes exploration, correction, verification, elimination, and consolidation. Brevity is therefore a possible consequence of good context engineering, not the objective itself.

This led to an important distinction between useful and useless struggle. Reasoning is not supposed to be monotonic from the first token to the final answer. A model may need to explore several hypotheses, try an approach, discover a contradiction, reconsider an assumption, change representation, or verify a tentative result. Such apparent inefficiency is productive when it changes the future search state. A mistaken branch can be valuable if it eliminates an alternative. A “wait” moment can be valuable if it uncovers a constraint. A verification step can be valuable if it changes confidence or closes an important uncertainty. Conversely, repeated planning, repeated state reconstruction, reopened branches without new evidence, duplicate calculations, or verification that adds no new information can consume context without improving the search.

The most useful conceptual framing that emerged is therefore:

> **Productive exploration expands the search space and then successfully prunes it.**

Thrashing is different. A model can repeatedly expand, revisit, and reconstruct possibilities without producing durable pruning. This creates a useful distinction between exploration and pathological repetition.

The annealing analogy captures the desired trajectory. Good reasoning can begin with relatively broad search, temporarily “reheat” when new evidence or a contradiction requires reconsideration, and progressively narrow toward a stable solution. The desired shape is not monotonic cooling:

```text
search width
     ^
     |       /\
     |   /\ /  \__
     |__/          \____
     |                  \_
     +----------------------> reasoning
       explore       converge
```

The important feature is not that the curve always decreases. It is that expansions produce useful downstream pruning. A trajectory that expands because it discovered something new and later collapses to a better-supported state is fundamentally different from one that expands and then simply circles the same possibilities again.

This led to another useful candidate characterization:

> **High tortuosity alone is not bad—exploration naturally increases path length. High tortuosity + high recurrence + no reduction in semantic branch entropy is a candidate signature of thrashing.**

This is still a hypothesis rather than an established metric. The distinction is important. The research should not decide in advance that entropy, recurrence, or tortuosity are the correct definitions of reasoning quality. Those are proposed observables that can be tested against objective outcomes.

That realization changed the role of the original trace-rewriting experiment.

The first ACE approach was effectively retrospective. A long raw trajectory was reviewed, an editor was asked to identify redundant or non-advancing reasoning, and a stronger model rewrote the trace while attempting to preserve its strategy, discovery order, uncertainty, corrections, branch structure, and conclusion. The intention was to create training targets with more information per token.

This produced genuinely useful design insights. It established a practical vocabulary for behaviors such as state reconstruction, redundant verification, branch reopening, correction spirals, representation churn, premature commitment, and overextended closure. It also established important constraints: do not invent reasoning, do not move a branch merely because a cleaner organization is available, preserve quantitative checks and conditional branches, and do not convert an actual reasoning trajectory into a polished hindsight solution.

However, the retrospective nature of rewriting introduces a fundamental weakness. The editor sees the completed trajectory and therefore knows the answer and the eventual path. It is trying to infer, after the fact, which parts of the trace mattered computationally. Even an extremely capable editor can therefore make a lossy intervention. It may delete an apparently redundant excursion that actually changed the model’s later search behavior, preserve elegant reasoning that contributed little, or reorganize the material according to a hindsight-optimal structure. A rewritten trace can be textually superior while producing no improvement in the student’s actual search dynamics.

This led to the current conceptual pivot:

> **ACE should ultimately steer reasoning prospectively rather than reconstruct it retrospectively.**

The desired training data should be generated under a causal policy that influences the model while it is reasoning. At generation time, the controller sees only the prefix and the model state available at that point. It cannot inspect future tokens or the final answer. Hindsight is used later only for evaluation and selection of trajectories, not for deciding how a token should have been generated.

The resulting conceptual pipeline is:

```text
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

This is a much stronger experimental setup because individual generations no longer need to be perfect. The controller merely needs to shift some probability mass toward productive reasoning trajectories. Generate enough samples, objectively verify them, characterize their behavior, and retain the strongest intact examples. The retained traces are genuine products of the causal generation process rather than teacher-generated reconstructions.

The proposed implementation is a small adapter/router attached to a frozen language model. At an intermediate transformer layer, the controller reads the current residual representation (h_t), produces a bounded intervention (\Delta h_t), and modifies the residual state:

[
h'_t = h_t + g_t\Delta h_t
]

where (g_t) is a learned gate. The controller is evaluated independently at each generation step. The base model remains frozen. A zero-initialized output projection makes the controller initially equivalent to the unmodified model. The intended controller therefore does not replace the reasoner; it lightly perturbs the reasoner’s trajectory.

The router is not supposed to be given a hard-coded semantic command such as “explore now” or “prune now.” Ideally, those behaviors emerge from training. At a particular internal state, the controller may learn that a small intervention improves expected terminal performance. At another state, it may learn that the best intervention is effectively zero. The learned function is conceptually:

[
h_t \rightarrow g_t,\Delta h_t
]

rather than a static adjustment applied uniformly to the whole generation.

The adapter architecture is deliberately small relative to the base model. The current implementation uses a mid-depth residual hook in Qwen3.5-9B, with a 4096-dimensional residual passing through a small bottleneck and back to 4096 dimensions, with a bounded residual scale and learned gate. The model weights remain frozen. Initial smoke tests demonstrated that this control surface works: zero initialization reproduced the baseline exactly, controller-only updates left the base weights unchanged, nonzero intervention changed logits and generation, interventions were prefix-causal, controller save/load was exact, and the perturbation remained bounded.

Those smoke tests established only plumbing. They did not establish that the controller improves reasoning.

A subsequent learnability experiment demonstrated that terminal correctness can produce gradients into the controller while keeping it close to the base model. However, that run was scientifically inadequate for testing ACE because it disabled Qwen’s reasoning mode, used relatively easy multi-step arithmetic problems, and limited replay to the first 192 completion tokens because of memory constraints. The controller learned a small, weakly state-dependent residual bias, but not an obvious explore/prune policy. This failure is informative: the substrate was not exercising the phenomenon we care about. A model solving simple problems in a few hundred tokens is unlikely to reveal meaningful search dynamics.

The correct experimental substrate therefore needs to place the model near its productive reasoning frontier. The problems should be difficult enough to induce genuine search pressure but not so difficult that the model simply collapses into unrecoverable failure. The goal is for Qwen to naturally produce traces in which it sometimes succeeds directly, sometimes explores several branches, sometimes makes and repairs mistakes, sometimes verifies, and sometimes struggles or fails. The exact success percentage is not sacred, but a broad mixture of outcomes is far more useful than a near-ceiling or near-zero task set.

This also clarified what “hard” means for ACE. The first corpus included some questions that were so far beyond the 4B model’s capability that the resulting traces effectively represented reasoning stack overflow: thousands of tokens of speculative reconstruction, repeated branches, and eventual truncation. Those traces are interesting as capacity-boundary failures, but they are poor ordinary trajectory-preserving training targets because the model never established a coherent reasoning arc. The useful region is closer to the productive frontier: tasks where the model has enough capacity to form a real search process, but enough difficulty that the search is not trivial.

This is why prompt generation has become an important experimental component. Madlibz can deterministically construct task envelopes specifying structural properties such as domain, analytical load, delayed constraint conflicts, competing hypotheses, deceptive local optima, representation shifts, bookkeeping interactions, adversarial edge cases, and other search topologies. The external authoring model turns these envelopes into natural problems, while the envelope and resulting problem remain reproducible and frozen. The authoring system does not itself decide whether a problem is good for ACE; the model’s actual sampled behavior determines that downstream.

The current prompt-generation strategy therefore separates proposal from calibration:

```text
Madlibz envelope
        ↓
authored verifiable problem
        ↓
many untouched Qwen9B rollouts
        ↓
empirical difficulty/search behavior
        ↓
retain useful frontier problems
```

This is deliberately different from asking an authoring model to make problems that “cause the model to say Wait.” The prompt should create structural reasons for reconsideration, not script the solver’s language. Examples include initially plausible routes that later consume scarce capacity, hypotheses that fit early evidence but fail under a later observation, local algorithmic improvements that break a downstream invariant, bookkeeping systems in which early assumptions affect later state, or representations that become tractable only after transformation.

The first 30 Madlibz candidates produced broad coverage across domains, task types, verifiers, and search topologies. The next empirical question is how Qwen3.5-9B actually behaves on those problems. The desired output is not simply “30 hard problems.” It is a calibrated population of problems that reliably induces the search behavior ACE is meant to influence.

In parallel, the project has begun developing direct measurements of reasoning dynamics.

Teacher-forcing a full-precision model through an existing trajectory allows access to next-token distributions and hidden states at arbitrary reasoning positions. This makes several candidate observables possible.

The simplest is token entropy:

[
H_{\text{tok}}(t)
=================

-\sum_v p(v\mid x_{\le t})\log p(v\mid x_{\le t})
]

which measures next-token uncertainty. Token entropy is useful as a baseline but is not equivalent to semantic search width. Many lexical alternatives can express the same reasoning move.

A more relevant target is semantic branch entropy. At a reasoning prefix, multiple short continuations can be sampled and clustered according to the distinct reasoning strategies they pursue. The entropy of those clusters would approximate the number of materially different future reasoning directions currently alive:

[
H_{\text{branch}}(t)
====================

-\sum_c P(c)\log P(c)
]

with effective branch count (e^{H_{\text{branch}}}). This is conceptually closer to “search space” than raw next-token entropy, although the measurement infrastructure and clustering method remain experimental.

Hidden-state geometry may provide another view. Given multiple continuations from the same prefix, their hidden representations can be compared through dispersion, covariance spectrum, or effective rank. This could indicate how broadly the model is exploring representational space, but it must be tested rather than assumed to have an intrinsically semantic interpretation.

Latent recurrence may expose repeated reconstruction of prior working states. A trajectory’s hidden representations can be compared against earlier states to form a recurrence matrix. Repeated returns to similar regions of representation space may correspond to branch reopening, correction spirals, or state reconstruction.

Trajectory tortuosity provides another candidate measure:

[
T=
\frac{\sum_t d(h_t,h_{t+1})}
{d(h_0,h_T)}
]

Again, high tortuosity is not inherently bad. Productive search can naturally produce long paths. The hypothesis is specifically that high tortuosity combined with recurrence and failure to reduce semantic branch entropy may characterize thrashing.

These measurements are intentionally being kept separate from controller training initially. Token entropy, branch entropy, recurrence, and tortuosity are diagnostics, not rewards. Otherwise the experiment risks defining “good reasoning” by its own assumptions and then congratulating itself for optimizing those assumptions.

The current research posture is therefore conservative:

> **Treat the ACE hypothesis as unproven and use measurements to determine whether the proposed observables actually track productive exploration and convergence.**

This is why each experimental stage is deliberately narrow. First establish controller plumbing. Then establish that a controller can learn any nontrivial state-dependent intervention. Then establish full reasoning-mode, long-horizon training on appropriately difficult problems. Then perform causal counterfactual analysis of where interventions actually improve the distribution of complete futures. Only after that should the controller be scaled and used to harvest a large corpus.

The most important protection against local optimization is that steering value must be defined by downstream outcome, not by local cleanliness. An intervention at state (s_t) is useful only to the extent that it improves expected terminal return relative to no intervention:

[
A(s_t,a)
========

## \mathbb{E}[R_{\text{final}}\mid s_t,a]

\mathbb{E}[R_{\text{final}}\mid s_t,\text{no-op}]
]

A locally ugly expansion that eventually escapes a local maximum should beat a locally elegant intervention that forces premature commitment. The controller therefore should not initially be trained to minimize entropy, recurrence, tortuosity, or trace length.

Long reasoning traces create substantial engineering difficulties here. Thousands of tokens of genuine reasoning make rollout generation more expensive, increase memory requirements during replay, and worsen credit assignment because one terminal reward must explain many controller decisions. Shortening the traces simply to fit the hardware would undermine the experiment, because long-horizon reasoning is the phenomenon of interest. The correct solution is to engineer long-context replay carefully, use checkpointing/recomputation or other memory-saving methods, and eventually use selective full-horizon counterfactual forks around candidate intervention points rather than evaluating every token exhaustively.

The counterfactual framework is particularly important. If the controller strongly intervenes at a particular prefix, that state can be forked into a no-op branch and a steered branch, both run to completion, and their outcome distributions compared. This provides a direct estimate of whether the intervention at that state improved future potential. It is much stronger than judging the next few dozen tokens.

This suggests a two-scale training architecture:

```text
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

That architecture avoids both extremes: blindly assigning terminal credit uniformly across thousands of tokens and inventing local proxy rewards that may destroy productive exploration.

The ultimate corpus-generation process follows naturally from this. The controller does not need to make every inference a masterpiece. It only needs to move enough probability mass toward productive reasoning trajectories that sampling and hindsight curation can harvest them. Generate many trajectories under the frozen controller, verify the outcomes objectively, measure their dynamics, and retain the strongest intact trajectories. Hindsight is acceptable at this stage because it is being used to select among trajectories that were generated causally; it is no longer being used to reconstruct the trajectory itself.

That gives the current ACE program a clean conceptual architecture:

```text
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
 objective verification          trajectory measurements
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

The deepest change from the original ACE idea is therefore philosophical as well as technical.

Originally, the hypothesis was attacked through **textual reconstruction**: inspect a completed reasoning trace, identify waste, and create a better serialization.

The new hypothesis is about **causal trajectory control**: observe the evolving internal state of a model while it reasons, learn a small controller that can nudge that state prospectively, generate many resulting trajectories, and determine whether those trajectories exhibit more productive search dynamics and better outcomes.

The rewritten traces already created are not useless. They become controls and perturbations: evidence about what humans and stronger models think a better trajectory looks like. They may eventually reveal which textual transformations correspond to real improvements in search dynamics. But they are no longer assumed to be the ground-truth training target.

The central scientific question is now much sharper:

> **Can a small causal controller attached to a frozen reasoning model alter the dynamics of its search so that productive exploration is preserved, unproductive recurrence is reduced, convergence improves, and full-task performance increases—without relying on hindsight or local proxy optimization?**

That is the current ACE hypothesis.
