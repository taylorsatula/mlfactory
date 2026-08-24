#!/usr/bin/env python3
"""Throughput/memory probe for the controller-training loop design.

Measures: batched same-prompt rollout speed (steered + base), EOS trimming,
typical completion length for problems.py prompts at training sampling
settings, and peak memory of a grad-enabled replay forward+backward over a
prompt+completion at the expected sequence length.
"""
from __future__ import annotations

import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from problems import TRAIN, verify
from steering_controller import (MODEL_PATH, ResidualSteering,
                                 SteeringController, build_prompt_ids,
                                 freeze_base_model, generate_batch)

tok = AutoTokenizer.from_pretrained(MODEL_PATH)
print("eos:", tok.eos_token_id)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, dtype=torch.bfloat16, device_map="cuda")
model.eval()
freeze_base_model(model)

prompt = TRAIN[0]["prompt"]
gold = TRAIN[0]["gold"]
n_prompt = len(build_prompt_ids(tok, prompt))
print(f"prompt tokens: {n_prompt}")

ctrl = SteeringController().to(device="cuda", dtype=torch.float32)
# deliberately nonzero so hook does real work
g = torch.Generator().manual_seed(0)
with torch.no_grad():
    ctrl.up.weight.copy_(torch.randn(ctrl.up.weight.shape, generator=g) * 0.3)

for tag, c in (("base", None), ("steered", ctrl)):
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    seqs, recs = generate_batch(model, tok, prompt, n=6, max_new_tokens=256,
                                controller=c, seed=123)
    dt = time.time() - t0
    lens = [len(s) - n_prompt for s in seqs]
    texts = [tok.decode(s[n_prompt:], skip_special_tokens=True) for s in seqs]
    rew = [verify(t, gold) for t in texts]
    print(f"[{tag}] 6 rollouts x <=256 tok in {dt:.1f}s "
          f"({sum(lens)/dt:.0f} tok/s aggregate) lens={lens} rewards={rew}")
    print(f"  peak GPU mem: {torch.cuda.max_memory_allocated()/2**30:.1f} GiB")
    print(f"  sample tail: ...{texts[0][-120:]!r}")

# replay memory: grad-enabled teacher-forced pass over longest completion
longest = max(seqs, key=len)
T = len(longest)
x = torch.tensor([longest], device="cuda")
torch.cuda.reset_peak_memory_stats()
with ResidualSteering(model, ctrl, collect=True):
    logits = model(input_ids=x).logits
    loss = logits[:, -1].float().sum()
    loss.backward()
print(f"replay fwd+bwd at T={T}: peak {torch.cuda.max_memory_allocated()/2**30:.1f} GiB")
print("collected graph scalars:", len(_ := [1]) and None or None, end="")
print()
