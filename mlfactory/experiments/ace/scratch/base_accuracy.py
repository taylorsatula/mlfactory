#!/usr/bin/env python3
"""Base-policy correctness rate per family (thinking off, temp 0.9)."""
from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mlfactory.experiments.ace.core.problems import HOLDOUT, TRAIN, extract_answer, verify
from mlfactory.experiments.ace.core.steering_controller import (
    MODEL_PATH, build_prompt_ids, freeze_base_model, generate_batch)

tok = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, dtype=torch.bfloat16, device_map="cuda")
model.eval()
freeze_base_model(model)

# first instance of each family
seen = set()
items = []
for it in TRAIN + HOLDOUT:
    if it["family"] not in seen:
        seen.add(it["family"])
        items.append(it)

G = 4
for it in items:
    n_prompt = len(build_prompt_ids(tok, it["prompt"], enable_thinking=False))
    seqs, _ = generate_batch(model, tok, it["prompt"], n=G,
                             max_new_tokens=512, seed=11,
                             enable_thinking=False)
    rews, preds, lens = [], [], []
    for s in seqs:
        text = tok.decode(s[n_prompt:], skip_special_tokens=True)
        lens.append(len(s) - n_prompt)
        preds.append(extract_answer(text))
        rews.append(verify(text, it["gold"]))
    print(f"{it['id']:18s} gold={it['gold']:>8}  "
          f"acc={sum(rews)}/{G}  lens~{int(sum(lens)/G)}  preds={preds}")
