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
| Branch open/close change-points | L11, L15, L19 | prior (untested) | — |
| Return-after-elimination / thrash | L19, L23; DeltaNet recurrent state | prior (untested); semantic return-after-elimination instrument not yet built | — |
| Loop-onset early warning | L7, L11; DeltaNet state before L15 | **underpowered** — probe AUROC≈0.25, only 4 loop traces vs 128 healthy | b1 map |
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
