# Qwen3.5-9B Architecture (reference)

> Update when: the model is swapped, or a config field is confirmed/changed.
> Pure architecture facts from the Hugging Face config. **Theory stripped** —
> layer-function hypotheses and interpretive priors live in
> `LAYER_HYPOTHESES.md`. Measurement/steering-site findings live there too.

## Source

`/home/admin/models/hf/Qwen3.5-9B` — bf16 safetensors, 8.95B params.
Hugging Face `config.json`.

## Concrete architecture facts

```json
"num_hidden_layers": 32,
"full_attention_interval": 4,
"layer_types": [
  "linear_attention", "linear_attention", "linear_attention", "full_attention",
  ...  // repeats 8 times
]
```

Repeating pattern (0-based indexing):

```
DeltaNet / linear attention
DeltaNet / linear attention
DeltaNet / linear attention
Full softmax attention
```
× 8.

| Property | Value |
|---|---|
| Hidden size | 4096 |
| Text blocks | 32 |
| Vocab | 248,320 |
| Full-attention block indices | 3, 7, 11, 15, 19, 23, 27, 31 |
| Linear-attention block indices | all others (24 blocks) |
| EOS token id | 248044 |
| Chat turn end token id | 248046 (`<\|im_end\|>`) |
| Max positions | 262,144 |
| Precision loaded | bf16 |
| Param count | 8.95B |
| Fits on | one RTX 3090 (24 GB), with room for training-scale activations |

## Architectural implications

- Full-attention blocks attend directly over the visible token context.
- Linear-attention / Gated DeltaNet blocks maintain a compressed recurrent
  state, updated over time.
- The residual stream convention is shared by both block types — a
  layer-output hook sits below all cache machinery and is the only
  architecture-agnostic interception point.
- Linear blocks carry a fixed-size recurrent state instead of a KV cache,
  so any attention-internal hook would need two implementations; a residual
  hook needs one.

## Tokenization / generation facts

- The model ships no `generation_config.json`; `generate()` never stops at
  turn end without an explicit stop list. `core/steering_controller.py`
  sets `STOP_TOKEN_IDS = [248044, 248046]`.
- Chat template opens a live `äsident` block when `enable_thinking=True`
  (the model's native reasoning mode); `False` closes thinking immediately.

## Serving paths (what NOT to use for steering)

The GGUF/vLLM HTTP serving paths (llama-server, `models.yaml`) remain the
legacy/ops path — unsuitable for activation intervention. Do not use them
for steering. `OPERATIONS.md` covers the llama-server-off ruling.
