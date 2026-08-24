# Qwen3.5-9B Layer-Function Analysis for ACE Controllers

> Updated: 2026-07-26  
> Architecture: 32 text blocks; repeating 3x Gated DeltaNet linear attention followed by 1x full softmax attention  
> Full-attention blocks listed in config order: 3, 7, 11, 15, 19, 23, 27, 31  
> Current hook: block 15 output, stride 4

---

## 0. Source and confidence

This document should be read as a mixture of:

1. **Concrete architectural facts** obtained from Qwen3.5-9B configuration/documentation.
2. **General interpretability priors** from transformer research.
3. **Qwen3.5-specific hypotheses** that should be validated empirically.

The exact layer recommendations below are **not** established published facts about Qwen3.5-9B. They are priors intended to guide a measurement sweep.

---

## 1. Concrete architecture facts

From the Qwen3.5-9B Hugging Face config:

```json
"num_hidden_layers": 32,
"full_attention_interval": 4,
"layer_types": [
  "linear_attention",
  "linear_attention",
  "linear_attention",
  "full_attention",
  ...
]
```

This confirms a repeating pattern of:

```text
DeltaNet / linear attention
DeltaNet / linear attention
DeltaNet / linear attention
Full softmax attention
```

repeated 8 times.

Assuming 0-based indexing, the full-attention blocks are:

```text
3, 7, 11, 15, 19, 23, 27, 31
```

The linear-attention blocks are the intervening layers.

Relevant architectural implications:

- Full-attention blocks can attend directly over the visible token context.
- Linear-attention / Gated DeltaNet blocks maintain a compressed recurrent state.
- The recurrent state is updated over time and may retain information that is not explicitly visible in the residual stream at every token.
- Full-attention blocks are natural global integration/readout checkpoints.

---

## 2. General interpretability priors

These are not Qwen3.5-specific facts, but reasonable expectations from transformer interpretability work:

| Depth | Common observed role in dense transformers |
|---|---|
| Early layers | Local token relationships, syntax, copying, induction-style pattern completion |
| Middle layers | Semantic integration, factual recall, task-state representation |
| Late layers | Output preparation, formatting, instruction compliance, final token selection |

For a hybrid model, I would adapt this as:

| Component | Likely role |
|---|---|
| DeltaNet recurrent state | Persistent compressed context, drift, latent state, long-running thread memory |
| Full-attention residual outputs | More globally readable semantic state, retrieval, explicit context integration |
| Late full-attention blocks | Answer shaping, emission control, formatting, final policy |

---

## 3. Practical layer hypotheses for your target states

### Summary table

| Target phenomenon | Most likely measurement sites | Confidence |
|---|---:|---|
| Globally integrated working state | Block 15, Block 19 | Medium-high prior |
| Branch open/close change-points | Blocks 11, 15, 19 | Medium prior |
| Return-after-elimination / thrash | Blocks 19, 23; also DeltaNet recurrent state | Medium prior |
| Loop-onset early warning | Blocks 7, 11; DeltaNet state before 15 | Medium/low prior, needs testing |
| Solution consolidated but emission blocked | Blocks 19, 23, 27; compare with final logits | Medium prior |

---

## 4. Interpretation of block 15

Block 15 is a reasonable default hook because:

- It is a full-attention block.
- It is near the middle of the 32-layer stack.
- It is likely late enough to contain integrated task state.
- It is early enough to still allow intervention before final output shaping.

However, block 15 alone is probably insufficient for all three phenomena.

Recommended interpretation:

```text
Block 15:
  Good general-purpose working-state hook.

Block 11:
  Potentially better for early loop/repetition warning.

Block 19/23:
  Potentially better for branch re-entry and answer consolidation.

DeltaNet recurrent state:
  Potentially better for latent persistence of eliminated branches or slow drift.
```

---

## 5. Loop-onset states

Target phenomenon:

> The trace collapses into verbatim repetition, e.g. repeated phrases, with token entropy dropping near zero only after onset.

### Candidate signals

Before a verbatim loop begins, look for:

1. **Rising self-similarity** in hidden states.
2. **Reduced effective dimensionality** of the recent hidden-state trajectory.
3. **Increasing projection onto a learned repetition/loop direction.**
4. **Attention concentration on recent repeated spans** in full-attention blocks.
5. **DeltaNet recurrent-state contraction or excessive stability.**

### Candidate layers

Most plausible:

```text
Block 7 or 11: early repetition/induction signals
Block 15: readable confirmation of state collapse
DeltaNet state before block 15: possible early latent warning
```

### Measurement idea

For each candidate layer:

```text
pre_loop_states = hidden states from positions shortly before loop onset
matched_states  = hidden states from healthy matched positions
```

Train a simple linear probe:

```text
label = 1 if pre-loop window
label = 0 if matched healthy window
```

Evaluate:

```text
AUROC
AUPRC
lead time before onset
false positive rate
```

Important: match controls by position, task type, local text properties, and recent n-gram repetition so the probe is not merely detecting surface repetition.

---

## 6. Branch open/close events

Target phenomenon:

> The model opens a new reasoning branch or durably closes/prunes an existing one.

### Candidate layers

Most plausible:

```text
Blocks 11, 15, 19
```

Why:

- These are mid-depth full-attention blocks.
- They are likely involved in integrating context and updating the current working state.
- Branch changes are discourse-level events, so they likely become visible after full-attention integration.

### Measurement idea

Use change-point detection on the hidden-state trajectory:

```text
distance between pre-window representation and post-window representation
```

Prefer:

```text
Mahalanobis distance
whitened cosine distance
PCA-space change score
```

over raw cosine similarity, because raw cosine can be dominated by layer norm, scale, and high-variance directions.

Also train branch probes:

```text
current active branch
branch opened/closed
branch status: open, suspended, eliminated
```

The layer with best alignment to annotated branch events is the best candidate.

---

## 7. Return-after-elimination / thrash

Target phenomenon:

> The model semantically returns to a previously eliminated branch without verbatim repetition.

This is especially important in a hybrid architecture.

### Key distinction

There are two separate signals:

1. **Latent persistence**
   - The eliminated branch still exists in compressed memory.
   - Best candidate: DeltaNet recurrent state.

2. **Active semantic re-entry**
   - The current representation has moved back into the eliminated branch’s semantic region.
   - Best candidate: full-attention residual outputs, especially mid-to-late layers.

### Candidate layers

```text
DeltaNet recurrent state around blocks 12-22
Full-attention residuals at blocks 19 and 23
```

### Measurement idea

Create branch anchors while the branch is active:

```text
anchor_branch_l = prototype representation of branch at layer l
```

Then after elimination, compute:

```text
return_score_l(t) = similarity(current state, eliminated branch anchor)
```

But verify:

```text
- surface text is not verbatim repetition
- similarity is branch-specific, not generic topic similarity
- score rises before or at semantic re-entry
```

This is one area where the DeltaNet recurrent state may contain information that the full-attention residual stream does not yet expose.

---

## 8. Solution consolidated but emission blocked

Target phenomenon:

> The model internally has the answer, but continues reasoning, hedges, reformats, or delays emitting the answer.

### Candidate layers

```text
Blocks 19, 23, 27
Final layer / pre-logit state
```

### Measurement idea

Use logit lens or tuned lens at intermediate layers.

For each layer:

```text
P_l(answer) = probability assigned to final answer tokens at layer l
P_final(answer) = probability assigned by final output distribution
```

A blocked-emission signature could be:

```text
P_19(answer) high
P_23(answer) high
P_final(answer) low or delayed
final token continues thinking instead of answering
```

Or in representation space:

```text
projection onto answer-ready direction is high
but output continuation remains non-final
```

### Practical note

For this phenomenon, comparing intermediate layers with the final output is more informative than choosing a single layer.

---

## 9. Full-attention residuals vs. DeltaNet recurrent states

They should be treated as complementary.

| Signal | Full-attention residual | DeltaNet recurrent state |
|---|---|---|
| Readable semantic state | Likely better | Less directly readable |
| Global integration | Strong at full-attn output | Compressed, indirect |
| Persistent latent memory | Limited to current residual readout | Likely stronger |
| Early drift detection | May lag | May lead |
| Exact retrieval/copy diagnostics | Stronger via attention patterns | Weaker / compressed |
| Intervention point | Natural | More complex but potentially useful |

Therefore:

```text
Use full-attention residuals for readable state and intervention targets.
Use DeltaNet recurrent states for latent persistence, drift, and early-warning signals.
```

---

## 10. Head/block specialization expectations

There is no confirmed public head-level functional map for Qwen3.5-9B.

However, plausible specialization classes to look for:

| Functional head type | Likely location | What to measure |
|---|---|---|
| Induction/copy heads | Early full-attn blocks, e.g. 3, 7 | Attention to prior matching n-grams |
| Retrieval heads | Mid full-attn blocks, e.g. 11, 15, 19 | Attention to relevant earlier branch content |
| State-tracking heads | Mid full-attn + DeltaNet state | Persistence of branch/goal variables |
| Answer-mover heads | Later full-attn blocks, e.g. 19, 23, 27 | Movement of answer content toward output position |
| Output/format heads | Late blocks, e.g. 27, 31 | Formatting, final answer emission, instruction style |
| Momentum/fluency heads | DeltaNet layers | Local continuation and smooth discourse state |

These should be validated with:

```text
attention-pattern analysis
ablation
activation patching
probe accuracy
logit-lens deltas
```

---

## 11. Recommended empirical validation plan

### Phase 1: Layer sweep

Record from a representative set of reasoning traces:

```text
Full-attention residuals:
  3, 7, 11, 15, 19, 23, 27, 31

DeltaNet recurrent states:
  especially 6, 10, 14, 18, 22, 26
```

For each layer, compute:

```text
loop-onset probe AUROC
branch-state probe accuracy
change-point alignment
return-after-elimination similarity
answer-readiness logit lens score
```

### Phase 2: Select minimal hook set

Likely minimal set:

```text
Block 11 output: early loop/repetition warning
Block 15 output: general working state
Block 19 or 23 output: branch re-entry / answer consolidation
DeltaNet state near 14/18: latent persistence
```

If only one hook is allowed:

```text
Block 15 output
```

If two hooks are allowed:

```text
Block 11 + Block 19
```

If three hooks are allowed:

```text
Block 11 + Block 15 + Block 23
```

If recurrent state is accessible:

```text
Add DeltaNet state from block 14 or 18
```

### Phase 3: Causal validation

For any selected layer, verify causal usefulness:

```text
Does reading the layer predict the event early enough?
Does ablating or steering the layer change the target behavior?
Does intervention reduce loops/thrash without harming valid reasoning?
Does the signal generalize across prompts and tasks?
```

---

## 12. Bottom line

The strongest evidence-based statement is:

```text
Full-attention blocks are likely the best readable global-state checkpoints.
DeltaNet recurrent states are likely important for persistent latent state.
```

The practical prior for your ACE measurements is:

```text
Block 15: good central working-state hook.
Block 11: candidate early loop-warning hook.
Blocks 19/23: candidate branch re-entry and answer-consolidation hooks.
DeltaNet recurrent state: candidate latent persistence / eliminated-branch residue.
```

But the exact layer choices should be validated by a layer-wise probe and causal intervention sweep before being fixed in the controller.
