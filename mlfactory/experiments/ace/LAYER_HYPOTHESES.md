# Layer Hypotheses: where each phenomenon is readable

> Update when: a layer map confirms or kills a layer hypothesis, or a new
> phenomenon gets a layer prior. Each hypothesis carries a status. This is
> the *theory* half; the pure architecture facts it rests on live in
> `QWEN35_ARCHITECTURE.md`. **Measurement site ≠ steering site** — this
> doc maps readability; the steering layer is chosen later on causal-
> leverage criteria from Phase-3 forks (`PHASES.md`).

## Methodological ruling (on record)

Measurement site and steering site are two independent questions. Letting
the steering choice (or a current hunch) pre-decide the measurement site is
the error this pass corrects. **Readability ≠ causal leverage; those can live
at different layers.** The steering layer gets chosen later on its own
criteria.

## Status key

- `prior` — from architecture + interpretability, untested
- `supported` — measured separability in b1 map
- `killed` — measured and weak/inverted
- `underpowered` — test ran but n too thin to rule

## Per-phenomenon layer hypotheses

| Phenomenon | Candidate layers | Status | Evidence |
|---|---|---|---|
| Globally integrated working state | L15, L19 | **L15 killed** as measurement site (+0.149 pooled, weakest mid-layer); L19 not separately tested | b1 map |
| Branch open/close change-points | L6, L16–L19 | **supported (position-level)** — cycle-onset separability, merged LOO AUROC 0.989–0.992 L16–L19 (best L18), n=291 onsets | annotation probe 2026-08-26, merged corpus 2026-08-27 |
| Idle-muse onset | L16–L19 (merged), L23–L29 (xsub-only) | **supported (position-level, small n)** — merged n=13 clear / 29 all onsets, LOO AUROC 0.947–0.952 clear / 0.967–0.968 all; best-layer set not settled | annotation probe 2026-08-26, merged corpus 2026-08-27 |
| Return-after-elimination / thrash | L19, L23; DeltaNet recurrent state | prior (untested); semantic return-after-elimination instrument not yet built | — |
| Loop-onset early warning | L2, L17–L20 | **supported (position-level)** — merged n=285 loop onsets vs depth-matched controls, LOO AUROC 0.975–0.978 (best L2); pre-onset (one token before the span) also ~0.98. The b1 probe's AUROC≈0.25 reading was deflated by the auroc() normalization bug (2026-08-26); its n=4 shortage was real, that number is not | b1 map (flawed number) + annotation probe 2026-08-26, merged corpus 2026-08-27 |
| Solution consolidated but emission blocked | L19, L23, L27; final layer | prior (untested) | — |

The table above is the **prior** (from architecture + interpretability).
The map below is the **evidence** that confirmed or killed entries in it.
They are keyed differently (phenomenon vs layer) because the prior asks
"where might X be readable?" and the map answers "what is readable at
layer Y?" — read the prior top-down by phenomenon, the map top-down by
separability.

## What the b1 multi-layer map established

Outcome separation (rank-biserial of mean step-cosine-distance, correct vs
wrong, clean finishes) by layer. Top layers:

| layer | type | rb | note |
|---|---|---|---|
| rec_2 | recurrent | **+0.759** | strongest single separability number in the map; recurrent channel |
| L25 | lin | +0.588 | |
| L17 | lin | +0.564 | |
| L6 | lin | +0.561 | |
| rec_12 | recurrent | +0.635 | |
| rec_9 | recurrent | +0.603 | |
| L24 | lin | +0.518 | |
| L13 | lin | +0.490 | |
| L23 | FULL | +0.443 | |
| **L15 (current hook)** | FULL | **+0.149** | **weakest mid-layer — wrong place to measure** |
| L31 | FULL | −0.273 | final layer inverts |
| L0 | lin | −0.747 | early layer negative |

- **Outcome separation is broadly distributed across mid layers, strongest
  at linear-attention blocks, not full-attention blocks.** The architecture
  prior — that linear-attention blocks carry state-tracking content the
  full-attention residuals read out less directly — is **supported** at the
  measurement level.
- **The recurrent channel carries outcome information the residual does
  not.** rec_2 (+0.759) is stronger than any residual layer. But signs
  flip by layer: rec_8 (−0.574), rec_20 (−0.686). The recurrent channel is
  worth the v1 chunked-trajectory investment.
- **Only the *final* recurrent state is captured so far.** "Did the
  eliminated branch persist in compressed memory at time t" (the
  latent-persistence claim) needs the deferred v1 chunked-trajectory pass;
  this pass validates only that the channel is worth that pass.

## What the map does NOT establish (honest scope)

- **Causality.** All measurements are teacher-forced and observational.
  Separability is not steering-effectiveness. The Phase-3 passenger test
  still has to run before any of this licenses an intervention.
- **Where to steer.** Readability ≠ causal leverage. The steering layer
  is a separate decision; this pass deliberately does not make it.
- **Generalization.** Three families only (assign/machine/certify). Any
  layer finding is provisional until grid/hypothesis/adversary land.
- **AUROC numbers from the b1 era.** `analyze_map.py`'s `auroc()` divided
  by len(pos) twice, deflating every AUROC it printed by a factor of
  len(pos) (fixed 2026-08-26). Any b1-era AUROC reading — including the
  loop-onset probe's ≈0.25 — is unreliable by that factor; rank-biserial
  readings were unaffected.

## What the position-level annotation probe established (2026-08-26, re-scored 2026-08-27)

The annotation sidestep (`ANNOTATION_SIDESTEP.md`) moved both sides of
the separation to the position level: LLM-flagged spans (classes
muse/cycle/loop) over the merged xsub+b2 corpus (280 traces, 231
captured), teacher-forced through bf16, onset positions vs
depth-matched within-trace controls, scored with leave-one-trace-out
AUROC (direction fit on all other traces).

| Class | n onsets (clear) | LOO AUROC | Layers |
|---|---|---|---|
| cycle | 291 | 0.989–0.992 | L16–L19, best L18 (xsub-only: n=128, same) |
| loop | 285 | 0.975–0.978 | L2–L6, best L2 (xsub-only: n=78, same) |
| muse | 13 (29 with probable) | 0.947–0.952 (0.967–0.968 all) | L16–L19 merged; xsub-only said L23–L29 — best-layer set unsettled at this n |

- **Pre-onset is separable too (0.95–0.99)**: the state one token
  before the episode's first token already encodes it. Divergence is a
  property of state dynamics, not (only) of the tokens that follow.
- **Escape vs reheat at onset**: same-class onsets in eventually-
  correct vs eventually-failed traces separate at AUROC 0.86 (loop,
  stable across corpora), 0.72 (cycle — deflated from xsub's 0.78 by
  the larger corpus), 0.957 (muse, conf=all, 23v6) — a nudge policy
  can in principle distinguish productive from dead-end excursions at
  onset.
- **Lookback (2026-08-27):** the pre-onset signal concentrates in the
  last ~8–16 tokens (LOO 0.97 at 2 tokens before onset, chance from 32),
  and rides mid-to-high layers (best L17–L30 at 2–4 tokens out) — the
  layer that carries "about to diverge" is not necessarily the layer
  that carries "diverging" (loop: L30 for lb_2 vs L2 at onset).
- **Recurrent channel (2026-08-27):** DeltaNet onset-vs-control LOO is
  real but weaker than residual — best rec_L2 (cycle 0.892, loop 0.859,
  muse 0.710 n=9); confirms the b1-era rec_2 hint; residual stream
  stays the primary steering substrate. **K5 transfer** (q8→bf16)
  passes at the focal layers (cycle L18 0.995, loop L2 0.985). Record:
  `lab_notes/2026-08-27-lookback-k5-rec-results.md`.
- **Scope honesty**: onset AUROCs are broadly high across most layers,
  which is either genuinely distributed signal or residual position/
  texture confound the depth-matching did not fully remove; the
  escape-vs-reheat comparison (annotated vs annotated) is the cleaner
  reading. All of this is observational — causal leverage is still
  the forks' question. Records: `lab_notes/2026-08-26-scale-annotation-
  to-r3.md` (xsub), `lab_notes/2026-08-27-b2-merge-capture-probe.md`
  (merged; STATUS.md R12).

## Recurrent vs residual: complementary channels

| Signal | Full-attention residual | DeltaNet recurrent state |
|---|---|---|
| Readable semantic state | Likely better | Less directly readable |
| Global integration | Strong at full-attn output | Compressed, indirect |
| Persistent latent memory | Limited to current residual readout | Likely stronger |
| Early drift detection | May lag | May lead |
| Exact retrieval/copy diagnostics | Stronger via attention patterns | Weaker / compressed |
| Intervention point | Natural | More complex but potentially useful |

```
Use full-attention residuals for readable state and intervention targets.
Use DeltaNet recurrent states for latent persistence, drift, and early-warning signals.
```

## Recommended empirical validation plan

### Phase 1: layer sweep (done — b1)

```
Full-attention residuals: 3, 7, 11, 15, 19, 23, 27, 31
DeltaNet recurrent states: especially 6, 10, 14, 18, 22, 26
```
For each layer: step-cosine-dist outcome separation, recurrent Frobenius
norm, loop-onset probe AUROC (underpowered).

### Phase 2: minimal hook set (pending — depends on causal validation)

If only one hook allowed: **not L15** — the map killed it. Candidates:
L6, L17, L25 (linear) + rec_2. If recurrent state is accessible: add
DeltaNet state from L14 or L18. **The final hook set is chosen on
causal-leverage criteria from Phase-3 forks, not these readability
numbers.**

### Phase 3: causal validation

For any selected layer:
```
Does reading the layer predict the event early enough?
Does ablating or steering the layer change the target behavior?
Does intervention reduce loops/thrash without harming valid reasoning?
Does the signal generalize across prompts and tasks?
```

## Current hook (transient)

The controller currently hooks L15 output (`core/steering_controller.py`,
`STEER_LAYER=15`). This was the architecture-doc default prior. The b1 map
shows L15 is a weak *measurement* site; it says nothing yet about L15 as a
*steering* site (causal leverage untested). A hook move is a Phase-3
decision, not a Phase-1 one.

## Head/block specialization (prior, untested)

No confirmed public head-level functional map for Qwen3.5-9B. Plausible
specialization classes to look for, when head-level analysis becomes the
question:

| Functional head type | Likely location | What to measure |
|---|---|---|
| Induction/copy heads | Early full-attn (3, 7) | Attention to prior matching n-grams |
| Retrieval heads | Mid full-attn (11, 15, 19) | Attention to relevant earlier branch content |
| State-tracking heads | Mid full-attn + DeltaNet state | Persistence of branch/goal variables |
| Answer-mover heads | Later full-attn (19, 23, 27) | Movement of answer content toward output position |
| Output/format heads | Late blocks (27, 31) | Formatting, final answer emission, instruction style |
| Momentum/fluency heads | DeltaNet layers | Local continuation and smooth discourse state |

Validate with: attention-pattern analysis, ablation, activation patching,
probe accuracy, logit-lens deltas. Status: `prior` — not yet tested; the b1
map was layer-level, not head-level.
