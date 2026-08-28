#!/bin/bash
cd /workspace/mlfactory
export HF_HOME=/workspace/models ACE_MODEL_PATH=Qwen/Qwen3.5-9B CUDA_VISIBLE_DEVICES=1
export PYTHONFAULTHANDLER=1 PYTHONUNBUFFERED=1
/venv/main/bin/python - <<"PYEOF" 2>&1 | grep -v "Loading weights\|Fetching"
import torch
from mlfactory.experiments.ace.core.steering_controller import (
    generate_batch, SteeringController, freeze_base_model)
from mlfactory.experiments.ace.train import pool_adapter
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-9B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-9B",
    dtype=torch.bfloat16, device_map="cuda")
model.eval(); freeze_base_model(model)
ctrl = SteeringController().to(device=model.device, dtype=torch.bfloat16)

rows = pool_adapter.load_pool("/workspace/mlfactory/mlfactory/experiments/ace/data/acegen_live_b2.jsonl")
items = pool_adapter.make_items(rows, pids=[132])
it = items[0]
SEED, N, MAX = 82244, 4, 26000

seqs_s, _ = generate_batch(model, tok, it["prompt"], n=N, max_new_tokens=MAX,
    controller=ctrl, record=True, do_sample=True, temperature=0.8,
    top_p=0.95, seed=SEED, enable_thinking=True)
seqs_b, _ = generate_batch(model, tok, it["prompt"], n=N, max_new_tokens=MAX,
    controller=None, record=False, do_sample=True, temperature=0.8,
    top_p=0.95, seed=SEED, enable_thinking=True)
for j in range(N):
    a, b = seqs_s[j], seqs_b[j]
    flip = next((i for i in range(min(len(a), len(b))) if a[i] != b[i]), None)
    print(f"sample {j}: lens {len(a)} vs {len(b)} first_flip={flip}", flush=True)
print("DONE")
PYEOF
