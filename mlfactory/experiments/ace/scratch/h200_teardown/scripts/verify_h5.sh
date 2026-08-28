#!/bin/bash
cd /workspace/mlfactory
export HF_HOME=/workspace/models ACE_MODEL_PATH=Qwen/Qwen3.5-9B CUDA_VISIBLE_DEVICES=1
export PYTHONFAULTHANDLER=1 PYTHONUNBUFFERED=1
/venv/main/bin/python - <<"PYEOF" 2>&1 | grep -v "Loading weights\|Fetching"
import torch
from torch.nn.attention import sdpa_kernel, SDPBackend
from mlfactory.experiments.ace.core.steering_controller import (
    generate_batch, freeze_base_model)
from mlfactory.experiments.ace.train import pool_adapter
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-9B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-9B",
    dtype=torch.bfloat16, device_map="cuda")
model.eval(); freeze_base_model(model)

rows = pool_adapter.load_pool("/workspace/mlfactory/mlfactory/experiments/ace/data/acegen_live_b2.jsonl")
it = pool_adapter.make_items(rows, pids=[132])[0]
SEED, N, MAX = 82244, 4, 1600

def pair(label):
    def gen():
        torch.manual_seed(SEED)
        s, _ = generate_batch(model, tok, it["prompt"], n=N, max_new_tokens=MAX,
            controller=None, record=False, do_sample=True, temperature=0.8,
            top_p=0.95, seed=SEED, enable_thinking=True)
        return s
    a, b = gen(), gen()
    for j in range(N):
        x, y = a[j], b[j]
        m = min(len(x), len(y))
        flip = next((i for i in range(m) if x[i] != y[i]), None)
        print(f"{label} sample {j}: lens {len(x)} {len(y)} first_flip={flip}", flush=True)

with sdpa_kernel([SDPBackend.MATH]):
    pair("MATH-sdpa")
print("---")
torch.use_deterministic_algorithms(True)
try:
    with sdpa_kernel([SDPBackend.MATH]):
        pair("MATH+detalg")
except Exception as e:
    print("detalg FAILED:", type(e).__name__, str(e)[:200])
print("DONE")
PYEOF
