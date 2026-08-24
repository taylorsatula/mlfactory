#!/usr/bin/env python3
"""Measure completion lengths and base correctness with EOS stopping."""
from __future__ import annotations

import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from problems import HOLDOUT, TRAIN, extract_answer, verify
from steering_controller import (MODEL_PATH, build_prompt_ids,
                                 freeze_base_model, generate_batch)

tok = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, dtype=torch.bfloat16, device_map="cuda")
model.eval()
freeze_base_model(model)

# one train + one holdout family instance
items = [TRAIN[0], TRAIN[8], HOLDOUT[0]]
MAX_NEW = 768
G = 4
for it in items:
    n_prompt = len(build_prompt_ids(tok, it["prompt"]))
    t0 = time.time()
    seqs, _ = generate_batch(model, tok, it["prompt"], n=G,
                             max_new_tokens=MAX_NEW, seed=7,
                             enable_thinking=False)
    dt = time.time() - t0
    lens, rews, preds = [], [], []
    for s in seqs:
        text = tok.decode(s[n_prompt:], skip_special_tokens=True)
        lens.append(len(s) - n_prompt)
        rews.append(verify(text, it["gold"]))
        preds.append(extract_answer(text))
    print(f"{it['id']}: gold={it['gold']}  {dt:.1f}s  "
          f"lens={lens}  preds={preds}  rew={rews}")
