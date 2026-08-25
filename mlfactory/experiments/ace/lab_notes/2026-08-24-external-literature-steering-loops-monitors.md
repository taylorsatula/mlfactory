# Lab note — 2026-08-24 — External literature: steering, loops, and decoding monitors

Scope: external literature review (five papers + one activation-steering
survey) mapped onto ACE's open questions and binding rulings. This records
what was known from the literature as of this date and which ACE open
questions it bears on — evidence *external to ACE's own experiments*, not
a finding from ACE's runs. It is the evidence for pending write-backs,
not the write-backs themselves (those go to the named topical docs per
the Decisions manifest below). Papers: Manifold Steering (2505.22411),
Reasoning Strength Planning (2506.08390), Activation Steering survey
(Emergent Mind), Circular Reasoning / Self-Reinforcing Loops
(2601.05693), CoT Dynamics (2608.03291), e-CUSUM Decoding (2607.11317).

## The upshot in one paragraph

The papers collectively **validate ACE's substrate and central premise** —
a steer-able "overthinking / reasoning-effort" direction exists in the
residual stream of reasoning models, is low-dimensional, and can be
intervened on to cut tokens 41–71% while holding or improving accuracy
(Manifold, RSP). They **validate ACE's two hardest methodological
commitments** with independent evidence: that loop onset has a latent
precursor in hidden states *before* verbatim repetition, so a
hidden-state signal — not token entropy — is the right trigger (Loops,
e-CUSUM); and that observable failure signatures are model-family
dependent and can invert sign, so pooled/within-prompt caution is
mandatory (CoT Dynamics). They **raise the bar for ACE** in one specific
way: several cheap, training-free baselines (a fixed manifold-projected
direction, a constant-λ additive direction, a logit penalty on hedge
markers, a CUSUM early-stop) already do much of what a naive controller
might do — so ACE's learned, state-dependent, terminal-reward controller
must prove, on forked outcomes, that it does something a fixed direction
cannot. The sharpest single tension: every steering paper uses **one
global, fixed-sign** intervention, but ACE's own failure-mode taxonomy
contains **opposite** pathologies (over-extended loops vs premature
commitment); a global "reduce overthinking" direction would help one and
worsen the other. That is the precise argument for a state-dependent
controller, and it is the argument the external literature does not make
for itself.

## Model-family caveat (read first)

ACE runs **Qwen3.5-9B** (hybrid DeltaNet/full-attention). None of the
papers test Qwen3.5. The model-family proximity, best→worst:

| Paper | Models | Proximity to Qwen3.5-9B |
|---|---|---|
| Circular Reasoning (Loops) | **Qwen3-8B, Qwen3-32B** + R1-distills + Phi + GPT | **closest** (same generation, hybrid arch) |
| CoT Dynamics | **Qwen3-8B, Qwen3-14B** + Llama3 + OLMo2 | **close** (same generation) |
| Manifold Steering | R1-Distill-**Qwen** (Qwen2.5 backbone) + R1-Distill-Llama-8B | one generation older (Qwen2.5) |
| Reasoning Strength Planning | R1-Distill-**Qwen** (Qwen2.5) + QwQ-32B | one generation older (Qwen2.5) |
| e-CUSUM | R1-Distill-**Qwen**-1.5B, FP16/INT4 | oldest and smallest; quantization focus |

Implication: the **loop/observable** evidence transfers best (Qwen3-8B is
ACE's nearest sibling and the Loops paper validated semantic-circularity
and V-shaped attention on it specifically). The **steering-feasibility**
evidence (a direction exists, is low-dim, is steer-able) is on Qwen2.5-distills;
the existence of a residual-stream steering axis almost certainly transfers
(the residual convention is shared, and ACE's own b1 map already found the
signal in Qwen3.5 at L6/L17/L25 + rec_2), but the *specific* direction and
the layer it lives at do not transfer by fiat. Treat all "direction at
layer 27" / "α=0.3" numbers as Qwen2.5-distill-specific, not ACE's.

---

## 1. Activation/latent steering of reasoning models

### 1a. Manifold Steering (2505.22411) — the key validation

**What it shows.** A single "overthinking direction" in residual-stream
activation space, extracted by **difference-in-means** between a redundant
set (responses >16k tokens **with hesitation keywords** "wait",
"alternatively") and a concise set (<1k tokens, none). Ablating the
projected component `h' = h − α r r^T h` across **all layers and positions**
cuts tokens up to **71%** (R1-1.5B GSM8K; R1-7B GSM8K −62%, MATH500 −42%,
AMC −24%, AIME −22%) while holding or improving Pass@1. Reverse steering
makes the model *more* hesitant/verbose — causal control of the behavior.
Cross-domain: the math-extracted direction transfers to LiveCodeBench
and GPQA (12–27% token cut, accuracy held).

**The interference-noise plateau.** The naive difference-in-means direction
in high-dim space = overthinking direction + an **orthogonal interference
component** `r_other` in `M^⊥`. As α grows past ~1.5, token count
**rebounds** because `r_other` disrupts unrelated capabilities and
**amplifies through layers** (Theorem 4.2: `‖Δμ^(l+1)‖ ≥ γ‖Δμ^(l)‖`).
Fix: project the direction onto the low-dim activation manifold (PCA,
top-10 components = 70% variance), nulling `r_other`. Their stated
future work, verbatim: *"dynamic steering strategies, where the strength
adapts to task complexity in real-time."*

**How it maps to ACE.**
- **Validates the substrate and premise** (residual stream carries a
  steer-able overthinking signal; mid-to-late layers separate, early
  overlap — matches ACE's L0 negative, mid positive, L31-inverts map).
- **Gives ACE the theoretical frame for the bottleneck.** ACE's
  `SteeringController` confines the intervention to a 4096→512→4096
  bottleneck — the *learned* analog of Manifold's PCA projection onto the
  manifold. This is the mechanism that addresses interference noise; the
  magnitude bound `‖δ‖ < α‖h‖` and the per-token gate are complementary
  safety on *how much* and *when*, not on *which subspace*. (The user's
  "bounded-perturbation design implicitly addresses interference noise" is
  right in spirit; precisely, it is the **bottleneck** that maps to the
  manifold projection, and the **relative bound** that maps to Manifold's
  layer-wise-amplification concern — `‖h‖` grows ~7 (L0) → ~48 (L15) → ~263
  (L31), so a *relative* bound is depth-aware where an absolute one is
  meaningless.)
- **Sharpens the forbidden-proxy ruling.** Manifold's direction is
  *defined by* hesitation keywords + token length — exactly the surface
  proxies ACE forbids as rewards (`REWARD_POLICY.md`). Manifold shows a
  direction *correlated* with those proxies exists and is steer-able; it
  does **not** pass anything like the passenger test (no forked outcomes;
  the direction is applied globally with no counterfactual). ACE's bet is
  that a terminal-reward controller finds a *causally useful* direction
  without the proxy. Manifold is therefore evidence *for* "the signal is
  there" and a *baseline to beat*, not a method to adopt.

**Tension ACE must resolve (the central one).** Manifold's intervention
is **one global direction, fixed sign, applied everywhere**. ACE's
`FAILURE_MODES.md` contains **opposite** pathologies: **emission
paralysis** (over-extended, loop, "I'll write the response now" ×217–905)
**and premature commitment** (under-explored, locked-in early). A global
"reduce overthinking" direction would help emission paralysis and
**worsen** premature commitment (make an overconfident wrong trace more
decisive). This is the precise argument for a state-dependent controller
that can intervene in **opposite directions** by detected mode — exactly
ACE's design (gated, state-conditioned `d = tanh(up(SiLU(down(LN(h)))))`),
and exactly what the global-sign steering literature does not provide for
itself.

**Write-back target:** `HYPOTHESIS.md` (premise validated; the
global-fixed-sign-vs-heterogeneous-failures tension sharpens *why* the
controller must be state-dependent), `COUNTERFACTUAL_FRAMEWORK.md`
(Manifold fixed-direction steering is a Phase-3 passenger-test baseline),
`LAYER_HYPOTHESES.md` (mid-to-late separation converges with ACE's map;
their late-layer choice is Qwen2.5-distill-specific).

### 1b. Reasoning Strength Planning (2506.08390) — the pre-allocated direction

**What it shows.** LRMs **pre-plan reasoning strength in the residual
stream before generation**: a Lasso linear probe on the activation at the
`think` token predicts the subsequent reasoning-token count with
**R > 0.8** across sizes/families; prediction improves with depth. The
encoding is a **pre-allocated, shared direction** — four
difficulty-contrast vectors collapse to cosine **~0.99**, only the L2 norm
scales with difficulty. Additive steering `h' = h + λr` (constant λ,
global, all layers/positions) **causally controls token count** by moving
the end-of-reasoning logit; positive λ **improves accuracy** (R1-Distill-Qwen-7B
AIME +4.58; -14B AIME +5.83) up to a ceiling, then degrades. On **easy**
tasks, negative λ cuts tokens **without harming accuracy** — direct
evidence LRMs misallocate effort and overthink easy questions. They
explicitly note temperature 0.6 is set "to prevent endless repetitions."

**How it maps to ACE.**
- **Supports feasibility and raises the bar simultaneously.** The
  reasoning-length axis is real, low-dim (one shared direction), and
  *already present* in the model before generation. A well-tuned constant
  λ on it improves accuracy on hard math. So ACE's learned controller must
  show it beats **picking a good constant λ** — otherwise the learned
  complexity is unjustified. This **directly sharpens Q10** ("did the
  controller learn a nontrivial state-dependent intervention, or just a
  weak bias?"): RSP's pre-allocated direction explains why a constant bias
  is the easy thing to learn and a state-dependent policy is the hard
  thing. ACE's first run "learned a weak bias, not an explore/prune policy
  — substrate was too easy"; RSP suggests the substrate *having* a
  pre-allocated length axis is part of why.
- **Same global-fixed-sign limitation as Manifold** (constant λ, one
  direction, no state dependence). The lever is **wholesale CoT length**,
  not **unproductive recurrence** — it shortens *all* reasoning, including
  productive exploration. A terminal-outcome-trained state-dependent
  controller could in principle discriminate "productive expansion" from
  "thrashing," which a constant λ cannot.
- **Reinforces calibration.** "Overthinking more pronounced on simpler
  problems" (Manifold) + "overthink easy questions, steering-without-harm
  on easy tasks" (RSP) both back ACE's `CALIBRATION.md` logic: tune to the
  productive frontier, send DEAD-EASY → hard preset (the certify
  44/48→needs-hard-preset result is the same phenomenon).

**Write-back target:** `STATUS.md` Q10 (sharpened: the bar is "beat a
well-tuned constant λ on the pre-allocated axis, on forked outcomes"),
`CALIBRATION.md` (external corroboration of the easy/hard frontier
logic), `COUNTERFACTUAL_FRAMEWORK.md` (constant-λ additive steering is a
Phase-3 baseline).

### 1c. Activation Steering survey (Emergent Mind) — positions ACE's design

The paradigm: `h' = h + c·v` at a chosen layer, `v` from contrastive pairs
(ActAdd/CAA) or mean-centering or SAE features. The variants closest to
ACE's design are the **dynamic/adaptive** family: SADI (per-input
semantic masks), DyAC (KL-constrained dynamic intensity — a
perturbation *bound in probability space*), FASB (classifier-probe-gated
intervention + backtracking). **None of the surveyed methods train a
controller on outcome reward.** ACE's controller is a learned,
state-dependent, gated, bounded residual adapter trained by RL on
terminal verified outcome — a point in the design space the survey does
not catalog. The survey's stated limitations — hyperparameter sensitivity
(layer + coefficient → off-target degradation), fluency-conditioning
trade-off (excessive early-layer strength degrades fluency; dynamic/late
better), transferability not guaranteed, multi-property direct-summing
fails — are exactly the problems ACE's bottleneck+gate+relative-bound
address architecturally. Position: ACE is the "RL-trained
terminal-reward controller" in the activation-steering family;
Manifold/RSP are its "computed, fixed-direction" cousins it must beat.

**Write-back target:** no single doc; this is positioning context for
`APPROACH_HISTORY.md` (why the prospective controller, not a fixed
steering vector).

---

## 2. Loop / thrashing pathology

### 2a. Circular Reasoning / Self-Reinforcing Loops (2601.05693) — the most relevant single paper

**What it shows (on Qwen3-8B specifically, ACE's nearest sibling).**
- **Semantic circularity precedes textual repetition.** Via reasoning
  graphs (PCA/t-SNE on sentence-cluster nodes), the trajectory collapses
  into a periodic orbit **before** verbatim repetition begins — "the
  model is trapped in a 'semantic attractor' prior to generating verbatim
  repetitive text." Validated on **Qwen3-8B** (Figs 25–27), not only
  DS-Qwen-14B.
- **Determinism surge / state collapse.** At loop onset, max logit
  spikes and **entropy drops toward 0**; cosine similarity of identical
  tokens across cycles saturates to ~1.0 and norm differences vanish.
- **Statement-loop triggers.** Dense **high-entropy minority tokens**
  ("But", "Wait", "Alternatively", "However", "Therefore") act as
  semantic pivots at an impasse — "unable to derive a solution yet
  unwilling to terminate, resorts to recursive retrials." Their attention
  share spikes immediately before onset.
- **Persistence: V-shaped attention.** Attention concentrates on
  attention-sinks + recent tokens; once repetition saturates the local
  window, attention to recent tokens rises abnormally → self-reinforcing.
  Validated on Qwen3-8B (Figs 29, 31).
- **CUSUM early prediction.** A linear classifier on
  **sentence-averaged last-layer hidden states** gives a score
  `x_i = w^T h_i + b`; CUSUM `S_i = max(0, S_{i-1} + (x_i − r))` with a
  **persistence window** (p consecutive steps above threshold) detects
  onset. On **Qwen3-8B: EDR 0.64, FPR 0.30, ~44 sentences / ~1815 tokens
  before onset.** Robust across Qwen/Llama/Phi/GPT. Early intervention
  beats standard intervention (DS-Qwen-7B statement-loop completion
  0.80 → 0.88).

**How it maps to ACE — this is the highest-density correspondence.**
- **Independent confirmation of ACE's latent-precursor + representation-space
  claims, on Qwen3-8B.** ACE found reopening is **paraphrastic, not
  lexical** ("zero verbatim returns in 21 machine traces despite visible
  semantic returns"; "string-space instruments are blind; the instrument
  must live in representation space"). The Loops paper's
  semantic-circularity-before-textual-repetition is the same claim at a
  coarser granularity, on the same model family. This validates the next
  instrument ACE queued (semantic return-after-elimination in
  representation space; on-path/off-path detector).
- **Resolves the "loop onset gives no prospective entropy warning"
  finding — ACE was looking at the wrong observable.** ACE's tripwire used
  **token entropy** (H_tok), which collapses only *at* onset and is
  bimodal pre-onset → "cannot use an entropy tripwire." The Loops paper
  agrees token entropy is wrong (their determinism surge = entropy→0
  *post*-onset) and supplies the working alternative: a **CUSUM on a
  hidden-state linear-probe score**, with a persistence window. ACE's
  onset probe failed at **n=4, AUROC≈0.25** using a *one-shot* probe on
  *pre-onset vs matched-healthy* states. The Loops paper succeeded with
  (a) ≥50 loop cases, (b) a loop-vs-nonloop classifier (easier
  separation than pre-onset-vs-healthy), (c) CUSUM accumulation, (d) a
  persistence window to kill false positives (without it, FPR > 50%).
  ACE's underpowered result is a **data + method** shortfall, not
  no-signal.
- **Directly actionable for Q2 and Q5.** Regenerate hard-preset families
  to get ≥50 loop traces (ACE already plans this); replace the one-shot
  probe with a CUSUM + persistence formulation; the input can be ACE's
  *stronger* sites (L6/L17/L25 linear, rec_2 recurrent at +0.759) rather
  than last-layer. The persistence window is the specific trick that
  makes a loop detector usable as a fork trigger without false-positive
  flooding.
- **Hedge-marker nuance, not contradiction.** Loops says But/Wait trigger
  statement loops; ACE found `wait` neutral and `but`/`unless`/`again`
  failure-leaning while `actually`/`however`/`just-to` success-leaning.
  Reconciliation (ACE's own, validated): the *same* token (e.g.
  "However") is a loop trigger (Loops' high-entropy pivot) **or** a
  decisive success-leaning replacer (ACE) depending on context — whether
  it spawns a recursive retrial or a state replacement. This is exactly
  ACE's "what follows the wait matters" and "two species of elimination"
  (deductive vs counterfactual). It is also why a **global** hedge
  penalty (the Lotfi baseline) is a blunt instrument and a
  state-dependent one is sharper.
- **Emission paralysis = Loops' closure-intent statement loops.** ACE's
  loops carry closure intent ("I'll write the response now" ×217–905) at
  near-zero post-onset entropy — the Loops taxonomy's statement loops
  with determinism surge. The Loops mechanism (semantic attractor →
  V-shaped attention lock-in) is the *cause* ACE's `FAILURE_MODES.md`
  describes observationally. Early intervention beating late (0.80→0.88)
  supports ACE's plan to fork at loop-onset states and the "closure nudge
  is highest-headroom" bet (Q3).

**Caveat.** Qwen3-8B's EDR is the *lowest* of the tested models (0.64),
partly because Qwen3-8B has the fewest loop cases in LoopBench (15.43%
total loop rate). The signal is real but imperfect; do not treat CUSUM
as a solved trigger.

**Write-back target:** `STATUS.md` Q2 & Q5 (reopen with a concrete
method: CUSUM+persistence, not one-shot probe; gating data need ≥50 loop
traces), `FAILURE_MODES.md` (emission-paralysis mechanism now
externally characterized; "success is the exit" converges with Loops'
early-intervention-beats-late), `OBSERVABLES.md` (token-entropy tripwire
confirmed dead; hidden-state CUSUM is the candidate successor),
`LAYER_HYPOTHESES.md` (V-shaped attention / state collapse may be most
visible in the recurrent channel — rec_2 is ACE's strongest site).

### 2b. CoT Dynamics (2608.03291) — the generalization caution, validated

**What it shows.** On SAT problems, failing (complete-but-wrong) traces
show **premature verification collapse**: more backtracking, more cycling,
**lower transition entropy** (role-level next-role diversity, not token
entropy), earlier finalization (77.1% vs 89.8%), despite **equal clause
coverage** — "they mention much of the same clauses while organizing
reasoning around narrower, repeated checking and earlier commitment."

**The flagged caveat — and it is stronger than the flag suggests.** Only
**backtracking density** is universal (higher in wrong traces across
cross-family, within-family, and within-model for all three models).
**Transition entropy and cycle rate invert sign within-model for
OLMo2-13B** (wrong traces have *higher* entropy, *less* cycling —
opposite to the cross-family pattern). Length and verification density
are **cross-family only** and partly reflect verbosity/style. Verbatim:
*"premature verification collapse is better understood as a recurring
pattern than a universal signature"*; *"the broad cross-family collapse
pattern is not solely an outcome effect that reproduces identically
within every model."*

**How it maps to ACE.**
- **Direct empirical support for ACE's within-prompt / cross-family
  caution.** ACE killed `ent_late` within-prompt as a **composition
  artifact** (pooled +0.63 → within-machine +0.037, sign-flips across
  mixed prompts) and insists survivors must replicate on
  grid/hypothesis/adversary. CoT Dynamics shows the same phenomenon at
  the *model* level: a signature that holds cross-family can **invert**
  within-model. ACE's caution is not prudence-for-its-own-sake; it is
  empirically motivated, and the cost of ignoring it is reading a
  composition artifact as a signal (exactly ACE's `ent_late` mistake).
- **Scope boundary: CoT Dynamics excludes looping traces.** It labels
  `truncated_or_looping` as "failures of generation rather than of
  reasoning" and **removes them from the failure analysis.** So its
  signatures speak to ACE's **premature-commitment** failure mode
  (complete-but-wrong), **not** to ACE's **emission-paralysis** loops
  (which CoT Dynamics would have excluded). The two papers (Loops vs
  CoT Dynamics) cover **disjoint** failure classes: Loops owns the
  looping/termination pathology; CoT Dynamics owns the complete-but-wrong
  pathology. ACE has both, so ACE needs both lenses.
- **A length-direction warning.** CoT Dynamics' failing SAT traces are
  **shorter** (premature commitment); ACE's thrashing implies **longer**
  traces (revisits without pruning). "Lower transition entropy + high
  cycling" is a shared observable, but in CoT Dynamics it points to
  *under-exploration* and in ACE to *unproductive over-exploration*. Do
  not assume the observable means the same thing in both regimes — test
  the sign within-prompt, which is exactly ACE's keeper test.
- **Transition entropy is a candidate ACE observable, with the family
  caveat built in.** It is a role-level trajectory quantity (entropy of
  the next-role transition matrix), not next-token entropy — closer to
  ACE's `ent_trend` (which survived the machine kill test) than to the
  dead `ent_late`. Worth measuring in ACE's representation-space
  segmentation, but only with the within-prompt sign-consistency test,
  and only after grid/hypothesis/adversary land.

**Write-back target:** `OBSERVABLES.md` (generalization caution now has
external teeth — add the CoT Dynamics sign-flip as the cited reason
cross-family replication is mandatory, not optional), `FAILURE_MODES.md`
(premature commitment = CoT Dynamics' premature verification collapse;
note the length-direction difference from thrashing so the two are not
conflated), `STATUS.md` Q4 (cross-family replication of the surviving
machine observables is now externally motivated).

---

## 3. Hedge markers and decoding-time monitors

### 3a. e-CUSUM Decoding (2607.11317) — the cheap baseline, named

**What it shows.** Token log-probability is the **wrong observable** for a
decoding monitor: under the model's own sampling law,
`D_t = log p(w_t) + H_t` is a **mean-zero martingale** — it measures
sampling self-consistency, not trajectory health. In a confident
repetition loop `p≈1, H_t≈0 ⇒ D_t≈0`: the monitor is silent exactly when
the model is confidently wrong; Azuma's band reacts only to
high-variance (high-entropy) stretches, i.e. to uncertainty, not error.
Replacement: a **calibrated e-CUSUM** on an alarm score `a_t =
0.7·r_t + 0.3·u_t` fusing **verbatim n-gram repetition** `r_t` (the part
that sees confident loops) and **entropy spikes** `u_t`; intervention is
online backtracking + min-p modulation (+ a "Wait" marker). On R1-Distill-
Qwen-1.5B (FP16/INT4), GSM8K n=100: **accuracy gains not statistically
significant** (FP16 +4pp p=0.48; INT4 +6pp p=0.18); +28% tokens, +30%
wall-time; severe loops INT4 1%→0%. Stated as a methodological/preliminary
study; the dominant failure on this regime is **non-termination, not
looping** (verbatim loops ≤4% of tokens even on failing traces).

**The Lotfi et al. (2026) summary (verbatim, the load-bearing baseline
claim):** *"quantized models over-sample 'overthinking' markers (wait,
but, alternatively) at high-entropy positions and, in up to 52% of
failures, reach the correct answer in intermediate steps yet never commit
to it; they mitigate this with a training-free logit penalty on those
markers."* The paper flags a tension with its own "Wait"-marker injection
(marker penalty vs marker injection is "an open empirical question").

**How it maps to ACE.**
- **Sharpens the forbidden-proxy ruling from a different angle.** ACE
  forbids token entropy / hedge markers as *rewards* (reward-poisoning
  rationale: a local proxy penalizes productive struggle). e-CUSUM shows
  they are the wrong observable even for *monitoring* (mean-zero martingale
  blind to confident repetition). Two independent reasons converge on the
  same ruling: do not use token log-prob/entropy as the steering signal.
  ACE's `ent_late` death and "entropy tripwire fails" are special cases.
- **Names the cheapest baseline ACE must beat.** A **training-free logit
  penalty on wait/but/alternatively at high-entropy positions** (Lotfi)
  is the simplest intervention that targets overthinking markers. ACE's
  controller, trained on terminal reward, must beat it on forked
  outcomes — and the argument for beating it is the two-species point:
  a global marker penalty also penalizes ACE's success-leaning
  `actually`/`however`/`just-to` decisive replacers, while a
  state-dependent controller need not.
- **The "non-termination vs looping" regime difference — do not
  over-generalize either way.** e-CUSUM's regime (R1-Distill-Qwen-1.5B,
  INT4, GSM8K) has rare verbatim loops and dominant non-termination.
  ACE's regime (Qwen3.5-9B, bf16, reasoning-shaped machine/certify tasks)
  has ~20% of cap-truncated traces in verbatim closure-intent loops. The
  difference is plausibly model size + precision + task type, not a
  contradiction. ACE's emission-paralysis is real in its regime; e-CUSUM
  tells ACE not to assume verbatim looping is the *only* termination
  pathology — non-termination ("reaches the answer but never commits,"
  52% per Lotfi) is a sibling failure worth checking in ACE's traces
  (this is the "weak closure" / "emission paralysis" boundary in
  `TRAJECTORY_VOCABULARY.md`).
- **Convergence on "success is the exit."** e-CUSUM backtracks +
  re-explores rather than terminates; Loops' early intervention beats
  late; ACE found correct traces re-verify *more* and the distinguishing
  feature is the exit. Three independent works converge: the fix is not
  "stop early" but "break the self-reinforcing attractor and resume
  productive search." This is ACE's closure-nudge / rescue-the-search
  class (Q3), now externally corroborated.

**Write-back target:** `REWARD_POLICY.md` (forbidden-proxy list now has
a second, independent rationale; add Lotfi logit-penalty + Manifold
fixed-direction + RSP constant-λ + CUSUM early-stop to the **baselines**
table as Phase-3 passenger-test competitors), `FAILURE_MODES.md`
(non-termination / "reaches answer but never commits" as a sibling to
verbatim loops — check whether ACE's emission-paralysis splits into
loop vs non-commit), `COUNTERFACTUAL_FRAMEWORK.md` (the cheap baselines
are no-op-equivalent competitors the fork must beat).

---

## Cross-cutting synthesis

**1. Substrate and premise: validated, externally.** A residual-stream
steering axis for reasoning effort/overthinking exists (Manifold, RSP),
is low-dimensional (Manifold: top-10 PCA = 70% variance; RSP: cosine-0.99
single direction), and is steer-able to large token reductions with
accuracy held or improved. ACE's choice of a small residual-stream
controller at the architecture-agnostic hook point is on well-supported
ground. The low dimensionality is independent support for ACE's 512-wide
bottleneck.

**2. Interference noise: the bottleneck is the relevant mechanism.**
Manifold's diagnosis (a fixed high-dim direction accumulates an orthogonal
interference component that amplifies through layers and disrupts other
capabilities) is the failure mode ACE's **bottleneck** (subspace
confinement) plus **state-dependent direction** plus **per-token gate**
plus **relative magnitude bound** collectively avoid. Mapping precisely:
bottleneck ↔ manifold projection; relative bound ↔ layer-wise
amplification; gate ↔ "intervene only where needed." Calling the whole
package "bounded perturbation" is right in spirit; the bottleneck is the
piece that actually addresses *interference noise* specifically.

**3. The central tension: global fixed-sign steering vs ACE's
heterogeneous failure modes.** Manifold and RSP both use **one global
direction with a fixed sign**. ACE has **opposite** pathologies
(over-extended loops vs premature commitment; CoT Dynamics confirms the
latter produces *shorter* failing traces). A global "reduce overthinking"
intervention helps one and worsens the other. This is the strongest
external-derived argument for ACE's state-dependent, gated controller:
the literature validates that *a* direction works, and simultaneously
shows why *one fixed* direction is insufficient for a model with
heterogeneous failure modes. It also sharpens Q10: the bar is not
"learn a direction" (Manifold/RSP compute one for free) but "learn a
state-dependent policy that a fixed direction cannot replicate."

**4. Loop onset: the trigger is a hidden-state CUSUM, not token entropy.**
Three independent sources agree token entropy is the wrong trigger
(ACE's tripwire failure, e-CUSUM's mean-zero martingale, Loops'
determinism-surge-only-post-onset). The Loops paper supplies the working
replacement — CUSUM on a hidden-state score with a persistence window —
validated on Qwen3-8B. ACE's Q5 "infeasible at n=4" is a data+method
shortfall, not no-signal; the fix is concrete (≥50 loop traces, CUSUM
not one-shot probe, persistence window, use ACE's stronger layers/recurrent
channel). This is the single most actionable item in the brief.

**5. Hedge markers: forbidden reward, useful diagnostic, cheap baseline.**
The papers do not contradict ACE's ban on hedge-frequency as a reward;
they reinforce it (a global marker penalty is the cheap baseline, and it
is blunt — it hits productive decisive replacers too). The useful
diagnostic is hedge markers **at high-entropy positions within a semantic
attractor** (Loops), i.e. context-conditional — which is exactly ACE's
"what follows the wait matters" and two-species-of-elimination finding.
The state-dependent controller's edge over the Lotfi logit penalty is
discriminating the two species.

**6. Generalization caution: validated with teeth.** CoT Dynamics shows
a cross-family signature can **invert** within-model. ACE's within-prompt
kill-test (which killed `ent_late` as a composition artifact) is the
within-problem analog. Cross-family replication of the surviving machine
observables is now externally mandated, not optional — and the tested
Qwen3-8B/14B (closest to ACE) still show family/size dependence.

**7. The baselines ACE must now account for (Phase-3 passenger test).**
All training-free or cheap; a learned controller that only matches these
has no value claim:
- Lotfi training-free logit penalty on hedge markers (cheapest).
- Manifold manifold-projected fixed-direction steering (71% token cut).
- RSP constant-λ additive steering on the pre-allocated direction
  (accuracy gains on hard math).
- Loops CUSUM early-stop / early-intervention (decode-time).
- e-CUSUM backtrack + re-explore (decode-time).
The fork gate (`COUNTERFACTUAL_FRAMEWORK.md`) should run the controller
against at least the Manifold fixed direction and the Lotfi logit
penalty as no-op-equivalent competitors. If a fixed direction already
matches on forked outcomes, the controller line's value claim collapses
— which is exactly what the passenger test is designed to detect.

**8. "Success is the exit" — three-way convergence.** ACE (correct
traces re-verify more; the exit distinguishes success), Loops (early
intervention before entrenchment beats late, 0.80→0.88), and e-CUSUM
(backtrack + re-explore beats terminate) all say the fix is breaking the
self-reinforcing attractor and resuming productive search, not stopping
early. This corroborates ACE's closure-nudge / rescue-the-search class
(Q3) as the highest-headroom intervention.

## What these papers do NOT give ACE

- No Qwen3.5 evidence (closest is Qwen3-8B). Steering-direction and
  layer numbers are Qwen2.5-distill-specific; re-verify on ACE's model.
- No forked-outcome / passenger-test evidence in any steering paper.
  Manifold's reverse-steering is suggestive of causality but is not the
  fork test; none of these papers isolate "controller causes" from
  "controller recognizes." ACE's Phase-3 gate remains unreplaced.
- No treatment of heterogeneous/opposite failure modes with a single
  intervention — that is ACE's contribution to make, not the literature's.
- CoT Dynamics explicitly excludes looping traces, so its signatures do
  not speak to ACE's emission-paralysis loops; Loops owns those.
- e-CUSUM's own accuracy gains are not statistically significant (n=100,
  single model); the load-bearing claim is the Lotfi summary, not
  e-CUSUM's results.

## Decisions (write-back manifest)

Per pattern 7, this section names every doc the evidence changes — the
write-back manifest. Resolving a row updates the `STATUS.md` row + the
named topical doc; this note is the evidence, not the write-back itself:

| ACE question/doc | What the evidence says to write back |
|---|---|
| `STATUS.md` Q2 (terminal-loop early-stop) | CUSUM + persistence on hidden states is the method; token-entropy tripwire is confirmed dead by three sources |
| `STATUS.md` Q5 (loop-onset separability) | Re-open with concrete method: ≥50 loop traces, CUSUM not one-shot probe; underpowered, not no-signal |
| `STATUS.md` Q10 (state-dependent vs bias) | Bar sharpened: beat a well-tuned constant λ on RSP's pre-allocated axis, on forked outcomes |
| `REWARD_POLICY.md` baselines table | Add Lotfi logit penalty, Manifold fixed direction, RSP constant-λ, CUSUM early-stop, e-CUSUM backtrack |
| `OBSERVABLES.md` | Token-entropy tripwire dead (3 sources); hidden-state CUSUM candidate successor; transition entropy (role-level) candidate with family caveat; cross-family replication now externally mandated (CoT Dynamics sign-flip) |
| `FAILURE_MODES.md` | Emission-paralysis mechanism = Loops' semantic-attractor + V-shaped attention; non-termination/"reaches answer never commits" as a sibling (Lotfi 52%); premature commitment = CoT Dynamics' premature verification collapse (note length-direction difference from thrashing) |
| `LAYER_HYPOTHESES.md` | V-shaped attention / state collapse may be most visible in the recurrent channel (rec_2 +0.759); Manifold/RSP late-layer choices are Qwen2.5-distill-specific |
| `HYPOTHESIS.md` | Premise (residual steering axis exists, low-dim) externally validated; global-fixed-sign-vs-heterogeneous-failures tension sharpens *why* state-dependence is required |
| `CALIBRATION.md` | "Overthinking easy questions" (Manifold + RSP) corroborates the productive-frontier / DEAD-EASY→hard-preset logic |
