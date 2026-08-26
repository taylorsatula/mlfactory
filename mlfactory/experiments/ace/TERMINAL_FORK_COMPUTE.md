# Terminal-fork compute — the problem statement

> Update when: the fork cost model, the statistical floor, the hardware
> budget, or the §9 constraint set changes; or when this problem is solved
> or sidestepped (the doc then records what was assumed when). A
> self-contained brief on the compute problem at the center of ACE, written
> so a reader needs no other experiment document. Every cost figure is
> measured on the actual substrate, not estimated; sources are named
> inline. **No solutions are proposed**; §9 states the constraint set any
> resolution must respect.

## 1. What ACE is trying to achieve

**Setup.** A frozen Qwen3.5-9B reasoning model (8.95B params, bf16; hybrid
architecture: 24 gated-delta-net linear-attention blocks + 8 full-attention
blocks) solves verifiable problems in native thinking mode (temperature
0.8, top-p 0.95). Traces are long: median **22.3k tokens** (bf16), under a
**26,000-token backstop cap**; ~25% of traces hit the cap. Problems come
from a 46-prompt pool calibrated so the base model's per-prompt success
rate lies strictly between 0 and 1 (measured bands 1/8–7/8 over 8 samples
per prompt — `CALIBRATION.md`). Every trace terminates in a strict
deterministic verifier: each problem is solver-built, so a reference
answer exists by construction and scoring is a binary `check()` with no
partial credit (`gen/`).

**Hypothesis (unproven).** Productive reasoning expands the search space
and then durably prunes it; thrashing revisits and re-expands without
durable pruning. The desired trace shape is annealing with reheats, not
monotonic cooling: expansions are valuable only insofar as they produce
downstream pruning (`HYPOTHESIS.md`).

**Artifact under test.** A small prefix-causal controller attached to the
residual stream at one intermediate layer (L15 of 32 in the current
plumbing; which layer carries causal leverage is itself an open Phase-3
question — `LAYER_HYPOTHESES.md`). Zero-initialized,
so at step 0 it is bit-exactly the base model; interventions are bounded
(‖Δh‖ < 0.1·‖h‖) and are a pointwise function of the current residual
state, which is causally derived from tokens ≤ t only. The base model
stays frozen; only the controller trains. The controller can never see
future tokens, completed trajectories, or the answer
(`core/steering_controller.py`).

**Scientific question.** Can that controller, trained from terminal
verified outcome alone, alter the search dynamics so that productive
exploration is preserved, unproductive recurrence is reduced, and
full-task performance increases — without hindsight or local-proxy
optimization?

**Staging.** Falsification-first: cheap diagnostics kill what can be
killed cheaply; the final gate is causal. A null result at any gate
terminates the line (`PHASES.md`).

**The success standard (binding).** Controller value must be proven on
**forked outcome distributions**: same prefix, steered vs no-op, both run
to terminal verification. This standard — the passenger test — is what
makes terminal forks simultaneously the unit of evidence and the unit of
dense training signal, and it is the origin of the compute problem this
document states (`COUNTERFACTUAL_FRAMEWORK.md`).

## 2. Why terminal forks are the attribution mechanism

### 2.1 The credit problem

One terminal reward must explain controller decisions spread over 20k+
tokens. The controller fires (or declines to fire) at every token
position; a trace contains thousands of candidate intervention states;
only the final verifier output is observed.

### 2.2 Two shortcuts, both inadmissible

- **Uniform terminal credit** — assign the outcome to every position
  equally. Ignores which interventions mattered; cannot distinguish a
  decisive mid-trace nudge from noise.
- **Local proxy rewards** — score each step for *looking like* good
  reasoning (entropy, recurrence, tortuosity, length, judge/RLAIF
  scores, PRM-style step scorers). All banned, permanently
  (`REWARD_POLICY.md`). The reason is the **local-proxy trap**: any
  step-level scorer trained on what good reasoning looks like penalizes
  productive struggle and rewards premature convergence. A locally ugly
  expansion that eventually escapes a local maximum must beat a locally
  elegant intervention that forces premature commitment — and no local
  scorer can tell them apart by construction. This is the failure mode
  the entire fork architecture exists to avoid.

### 2.3 The fork and the passenger test

The attribution method is counterfactual:

```
A(s_t, a) = E[R_final | s_t, a] − E[R_final | s_t, no-op]
```

If the controller strongly intervenes at state s_t, that state is forked
into a no-op branch and a steered branch, both run to terminal
verification, and their outcome distributions are compared. The no-op
branch is exactly the frozen base model (zero-init controller).

> Correlation between intervention and outcome proves nothing. A
> controller that fires preferentially in states that were already going
> to succeed is a passenger. Only forked outcome distributions separate
> "controller causes" from "controller recognizes." **Any evidence
> offered for controller value that is not a fork comparison is
> advisory.** (The passenger test, `COUNTERFACTUAL_FRAMEWORK.md`.)

This is the Phase-3 gate; a null result there kills the controller line.

**Consequence — the load-bearing one.** Every admissible unit of evidence
is a pair of terminal executions, and every admissible unit of dense
(per-state) training signal is a batch of them. There is no admissible
way to value a prefix short of executing it to terminal state: the
verifier is binary and terminal, partial credit does not exist, and every
cheaper valuation is a banned proxy by §2.2.

Note the asymmetry with the cheap training phase: GRPO with a
group-relative baseline also uses terminal reward only and is admissible
— but it normalizes *prompt difficulty*, not *per-intervention credit*.
It can move the controller in aggregate; it cannot tell the controller
*which states* to intervene in. Forks are needed for (a) the evidence
standard and (b) per-state credit.

## 3. Unit cost: one terminal rollout (measured)

All costs below are counted in **terminal rollouts × their remaining
horizon**. The measured unit economics (bf16 HF, thinking on, cap 26k —
`STATUS.md` R9, `lab_notes/2026-08-25-grpo-h200-setup.md`,
`...-grpo-h200-smoke-results.md`):

| Quantity | Measured value | Conditions |
|---|---|---|
| Model load | 16.7 GB, 8.4 s | bf16, H200 |
| Generation, short context (≤1k) | ~150 tok/s aggregate | batch 4, HF SDPA |
| Generation, deep context (21–26k) | ~73 tok/s aggregate (worst group ~56) | batch 4 |
| Per-rollout wall time | ~4–6 min | median 22.3k tokens |
| Group of 4 rollouts | 15–25 min | 20k+ traces |
| Median trace length | 22.3k (bf16); 18.4k (q8, n=720) | thinking on |
| Cap-hit rate | 25% (6/24 bf16); 22% (q8) | scored ≈ 0 |
| Kernel route | flash-attn 2.8.3 and FLA+causal-conv1d: **zero gain**; current throughput is ~15% of the batch-4 bandwidth ceiling (~1066 tok/s) | kernel avenue exhausted (`STATUS.md` Q11) |
| Determinism | default SDPA backend is call-to-call non-deterministic (identical seed/weights diverge at 422–1764 tokens); MATH backend verified bit-stable | R11 |
| Gradient replay | windowed (cached-prefix) replay **killed** — boundary corruption up to 11.8 nats and a crashed backward; gradient-checkpointed full-trace replay is bit-exact, fits a 140 GB card | R10 |

**Hardware.** Two tiers: local 2× RTX 3090 (24 GB; one sample at a time;
collection-only by ruling — the fork regime does not fit: bf16 9B +
long-context generation + replay memory) and a rented Vast.ai box,
2× H200 140 GB (PCIe interconnect), billed hourly with the lifecycle
actively managed from local. **Fork-phase GPU-hours are therefore
directly money.** Useful conversion at deep-context throughput: **1
GPU-hour ≈ 2.6×10⁵ tokens ≈ ~12 median rollouts.**

## 4. One fork comparison: the design floor

A single fork comparison at one state s_t, at minimum viable design:

- **Two arms** (steered, no-op) — irreducible; the passenger test is
  defined on the pair.
- **Distributions, not samples.** Each arm needs m terminal samples,
  because the advantage is a difference of binomial proportions on a
  mid-band pool (base rates concentrated around p ≈ 0.5 — exactly where
  variance is largest). For m samples per arm:

  | base rate p | m per arm | SD of Â | Δ detectable at ≈2 SD |
  |---|---|---|---|
  | 0.5 | 8 | 0.25 | ~0.5 |
  | 0.5 | 32 | 0.125 | ~0.25 |
  | 0.5 | 80 | ~0.08 | ~0.16 |

  Detecting a realistic advantage of Δ = 0.2 (e.g. moving a 3/8 prompt
  toward 5/8) with ~80% power needs **≈ 80 samples per arm** (~160
  terminal continuations per fork state). Underpowered forks are worse
  than no forks: they manufacture spurious advantages, and the problem
  compounds across fork states (multiple comparisons).

- **The floor is already scraped.** Because s_t lies on an already-
  collected trajectory, one no-op sample exists for free (the original
  suffix), and the shared prefix is prefilled once and reused across the
  batched continuations. Pair-matched seeds remove shared-prefix noise.
  The irreducible cost is therefore ≈ **(2m − 1) continuations from the
  fork point**, each generated to terminal state. There is no valid
  design below this floor.

- **A structural tension on fork depth.** Early fork points leave a long
  remaining horizon (expensive; maximal room for the intervention to
  matter; low correlation between arms). Late fork points are cheap but
  the outcome is largely determined, leaving little advantage to detect.

- **Cap-hits are uninformative compute.** A fork continuation that hits
  the 26k backstop scores ≈ 0 by construction and carries the known
  truncation-backdoor confound (`REWARD_POLICY.md` §backdoor). At the
  measured 22–25% cap-hit rate, roughly a quarter of fork compute buys
  bounded information.

## 5. Forks of forks: where the exponential enters

A single fork comparison is affordable (§7, scenario B). The exponential
appears as soon as interventions are chained, and chaining is not
optional — it is the phenomenon itself.

1. **Interventions move the state.** A steered branch is a different
   trajectory with different downstream states. Every candidate
   intervention state downstream of s_t inside the steered branch is a
   *different state* from its counterpart in the no-op branch; its
   advantage cannot be reused, it must be re-measured inside each branch
   that reaches it. Evaluating "intervene at s_t, then at s′" requires
   forks inside the branches of the first fork — forks of forks.

2. **Credit over k decision points distinguishes 2^k patterns.** To know
   which *combination* of interventions caused the outcome, the naive
   enumeration runs every subset of the k intervention decisions to
   terminal state, each with m samples: (2m)^d terminal paths at tree
   depth d. Marginal (one-at-a-time) forks cost O(2m·k) paths instead —
   but marginals assume **additivity of interventions**, which the
   hypothesis itself argues against: the value of a durable prune plausibly
   depends on the earlier expansion it prunes, and a closure nudge only
   pays off given the preceding search. Path dependence is the thing
   under study; additive credit is precisely the approximation that
   erases it.

3. **The tree depth is not bounded by 1.** The annealing shape
   (explore → reheat → prune) posits multiple expansion/prune cycles per
   trace; the observed failure modes (terminal loops, premature
   commitment, counterfactual escape) sit at different trace depths. The
   controller's gate fires per token, so candidate states number in the
   thousands per 22k trace; which subset becomes decision points is part
   of what the controller learns, and cannot be known before forking.

4. **Token cost of the tree.** With branching ≈ 2m continuations per
   level, depth d, and remaining horizon L, full enumeration costs
   O((2m)^d · L) tokens versus O(2m·k·L) for marginals. At the measured
   unit cost, the multiplicative gap between the two is computed in §7
   (scenario D): at d = 3, m = 8 it is already a factor of ~85 and it
   grows as (2m)^(d−1)/(d).

## 6. Multipliers on the tree

Each of the following is a separate multiplicative factor, and each is
load-bearing (removing it changes the experiment rather than cheapening
it):

- **On-policy re-evaluation.** The controller updates at every training
  iteration. Fork advantages are measured for the state distribution
  *under the current controller*; after an update, the controller visits
  different states and fires differently, so earlier fork estimates no
  longer describe the policy. The tree is not explored once — it is
  re-explored per controller version.
- **Coverage.** A state-dependent policy needs advantage estimates over
  the state distribution it will encounter: 46 prompts × ~8 traces ×
  k candidate states = O(10³) fork states per controller version before
  any depth is added.
- **Constant-factor headroom is bounded and does not touch the
  exponential.** The kernel avenue is exhausted (zero gain from
  flash-attn and FLA; Q11); throughput sits at ~15% of the bandwidth
  ceiling, so engineering may buy a small constant — ×2–5 at best. Every
  figure in §7 divides by that constant; none of the exponents move.
- **Substrate and determinism bind the serving options.** Anything the
  bf16 training stack consumes must be generated on bf16 HF
  (`OPERATIONS.md` §substrate policy), and rollout generation must run
  under a deterministic SDPA backend (default backend is call-to-call
  non-deterministic; R11). Fork comparisons need matched, reproducible
  arms; quantized-API or speculative-decoding shortcuts are unavailable
  for the steered arm by those rulings.
- **Long horizon is the phenomenon.** Shortening traces to fit hardware
  would undermine the experiment: the pool was calibrated thinking-on at
  the 26k cap, and the hypothesis locates the learnable decision points
  (reheat, durable pruning) mid-to-late trace — measured: replay of a
  ≤8k prefix window cannot reach them. The trace length that makes each
  rollout expensive is the reason the experiment exists.
- **Hobbyist economics.** Owned hardware (2× 3090, 24 GB, one of which
  carries ~1.9 GB desktop overhead) cannot host the fork regime at all;
  the fork phase exists only on rented H200-hours. Compute demand grows
  exponentially in tree depth and linearly in coverage and iterations;
  the budget is fixed.

## 7. Worked example at measured numbers

Assumptions (each stated so they can be varied): 46-prompt pool × 8
traces = 368 traces; k = 3 candidate fork states per trace (deliberately
conservative); fork points mid-trace, remaining horizon ~10k tokens;
deep-context throughput 73 tok/s per H200 (1 GPU-h ≈ 2.6×10⁵ tokens).

| Scenario | Computation | Tokens | GPU-hours | Wall on 2× H200 |
|---|---|---|---|---|
| **A. Cheap phase (current GRPO, no forks)** — 12 iters × 4 prompts × group 4 × 2 arms | 384 terminal rollouts | ~8.6×10⁶ | ~33 | ~16 h |
| **B. Marginal forks, one controller version, underpowered (m = 8/arm)** — 368 traces × 3 states × 15 continuations | ~1.7×10⁴ continuations | ~1.7×10⁸ | ~630 | ~13 days |
| **C. Marginal forks, powered (m ≈ 80/arm, §4)** — same coverage, 159 continuations/state | ~1.8×10⁵ continuations | ~1.8×10⁹ | ~6,700 | ~9 months |
| **D. Interaction-resolving tree for ONE trace** — d = 3 chained decisions, m = 8: (2m)³ = 4,096 terminal paths × ~8k tokens | 1 trace | ~3.3×10⁷ | ~125 | ~3 days |
| **E. Training loop** — B or C re-explored per controller update, ~12 iterations | B×12 / C×12 | — | ~7,500 / ~80,000 | years / decades |

Reading: the cheap phase (A) is affordable and is what runs today.
Marginal forks for a *single snapshot* of one controller version already
cost 13 days of continuous rental at a sample size too small to detect a
realistic advantage (B), or 9 months at a powered one (C). Resolving
intervention interactions — which the hypothesis says matter — costs
~125 GPU-hours *per trace* (D). The training loop multiplies any of
these by the iteration count (E). A ×2–5 engineering constant (§6) moves
the table entries; it does not move any exponent, and it leaves every
entry beyond hobbyist budget except A.

## 8. Containment in the current design (description, not proposals)

The existing framework already concentrates fork demand rather than
eliminating it (`COUNTERFACTUAL_FRAMEWORK.md` §two-scale):

- **Two-scale architecture.** Broad, cheap terminal-reward GRPO (scenario
  A) trains an approximate controller and identifies *candidate*
  intervention states; expensive full-horizon counterfactual forks are
  then spent selectively around those candidates, and their outcomes
  refine the controller. Forks are never spent exhaustively per token.
- **Candidate sources that survive Phase 1.** Fork placement cannot use
  an entropy tripwire (loop onset gives no prospective entropy warning —
  measured, `FAILURE_MODES.md`); candidates come from surviving
  within-prompt observables (`OBSERVABLES.md`: `ent_early`, `ent_trend`,
  tortuosity, step-cosine at L6/L23/L25, recurrent-state norms) and from
  the controller's own intervention magnitude.
- **The amortized counterfactual critic.** A value function trained on
  fork outcomes is the designated legitimate dense signal — legitimate
  *because* it is learned from fork outcomes rather than assumed from a
  proxy. It is strictly downstream of fork machinery: it cannot reduce
  the cost of the forks that would train it. The first fork data must be
  bought at full price.
- **Staging principle.** Cheap hardware kills ideas; expensive hardware
  exploits survivors. The fork tier is entered only by a controller that
  survived the cheap phases.

What containment does **not** change: every selected fork still costs
terminal executions at the §4 floor, the selection heuristic itself is
only validated by fork comparisons, and on-policy re-evaluation (§6)
still applies to whatever subset of the tree is selected.

## 9. Constraint set any resolution must respect

Binding, from the experiment's own rulings — a resolution that violates
any of these changes the experiment rather than solving it:

1. **Reward is terminal verified outcome only.** No entropy, tortuosity,
   recurrence, length, judge/RLAIF, or PRM step-scoring terms, ever —
   diagnostics only. (`REWARD_POLICY.md`)
2. **The evidence standard is the passenger test.** Forked outcome
   distributions — same prefix, steered vs no-op, both run to terminal
   verification. Any substitute must separate "controller causes" from
   "controller recognizes"; correlation-based evidence is advisory.
3. **Long horizon is the phenomenon.** Thinking on, 26k backstop,
   decision points mid-to-late trace. Shortening the substrate voids the
   calibration and the hypothesis.
4. **Substrate identity.** Anything consumed by bf16 training is measured
   on bf16 HF; rollout generation under a deterministic SDPA backend.
5. **Statistical honesty.** Within-prompt variance is the signal; an
   underpowered fork comparison is worse than none (spurious advantages),
   and error must be controlled across the fork-state family, not per
   fork.
6. **Artifacts are immutable.** Existing rollout rows are evidence —
   never regenerated or truncated; new compute produces new rows.
7. **The bar is real.** The controller must beat a well-tuned
   constant-λ fixed-direction steering baseline judged on the same
   forked outcomes (`STATUS.md` Q10) — fixed-direction baselines already
   cut tokens 41–71% with accuracy held in external work, so the fork
   machinery must evaluate competitors too, not only the controller.
8. **Budget.** Hobbyist: 2× RTX 3090 owned (cannot host the fork
   regime), 2× H200 rented hourly.

## 10. The problem in one paragraph

ACE trains a causal controller whose only admissible reward is the
terminal verified outcome of ~22k-token reasoning traces, and whose only
admissible evidence and per-state credit signal is the counterfactual
fork: same prefix, steered vs no-op, both executed to terminal state,
compared as distributions (~160 terminal continuations per fork state to
detect a realistic effect). Because each intervention changes every
downstream state, evaluating *sequences* of interventions grows a tree —
forks of forks — whose leaves are terminal executions; the tree's depth
is the number of chained interventions a trace contains, and the
hypothesis under test (explore → reheat → prune) posits several. Total
compute is O((2m)^d · L) tokens in the tree's branching, depth, and
horizon, re-explored at every controller update because fork estimates
are on-policy. At measured throughput (≈12 median rollouts per H200-hour)
even one underpowered marginal-fork snapshot of one controller version is
~630 GPU-hours of rented hardware; a powered one is ~6,700; and
interaction resolution costs ~125 GPU-hours per trace. Every standard
escape hatch — step-level rewards, value bootstrapping from prefixes,
shorter traces, cheaper serving — is either banned as a local proxy,
downstream of the fork machinery itself, or barred by the substrate and
trace-length rulings. The exponential is structural, not engineering
slack; the budget is hobbyist. That gap is the open problem.
