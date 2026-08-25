#!/usr/bin/env python3
"""One-off structural inspection of local Qwen3.5-9B under transformers 5.14.1.

Determines: top-level class AutoModelForCausalLM resolves to, the module path
to the text decoder layers, what a decoder layer returns (tensor vs tuple),
per-layer module types, and whether a forward hook on a layer output fires
during both prefill and cached decode steps.
"""
from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "/home/admin/models/hf/Qwen3.5-9B"

tok = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, dtype=torch.bfloat16, device_map="cuda")
model.eval()

print("top-level class:", type(model).__name__)
print("n params: %.3fB" % (sum(p.numel() for p in model.parameters()) / 1e9))

# walk to the decoder layer stack
core = model.model
print("model.model class:", type(core).__name__)
print("model.model children:", [n for n, _ in core.named_children()])
layers = core.layers
print("n layers:", len(layers))
for i in (0, 3, 15, 16, 31):
    print(f"  layers[{i}]: {type(layers[i]).__name__}")

# what does a decoder layer return? hook it and see.
seen = []

def hook(mod, args, output):
    seen.append((type(output).__name__,
                 tuple(output.shape) if torch.is_tensor(output)
                 else type(output)))
    return output

h15 = layers[15].register_forward_hook(hook)
h3 = layers[3].register_forward_hook(hook)

ids = tok("The capital of France is", return_tensors="pt").to("cuda")
with torch.no_grad():
    out = model(**ids)
print("forward logits:", tuple(out.logits.shape), out.logits.dtype)
print("hook saw (prefill):", seen)

seen.clear()
with torch.no_grad():
    gen = model.generate(**ids, max_new_tokens=5, do_sample=False)
print("generated ids:", gen[0].tolist())
print("hook saw (generate, 5 steps):", seen)
h15.remove()
h3.remove()

# hidden size sanity + layer output norm scale at a few depths
# NOTE: a forward hook's non-None return value REPLACES the module output.
# Observer hooks must return None explicitly.
acts = {}


def observer(mod, args, output, layer_idx):
    acts[layer_idx] = output.detach().float().norm(dim=-1).mean().item()
    return None


for i in (0, 15, 31):
    layers[i].register_forward_hook(
        lambda m, a, o, i=i: observer(m, a, o, i))
with torch.no_grad():
    model(**ids)
print("mean ||h|| per layer:", acts)
