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
ctrl = SteeringController().to(device=model.device, dtype=torch.float32)

rows = pool_adapter.load_pool("/workspace/mlfactory/mlfactory/experiments/ace/data/acegen_live_b2.jsonl")
it = pool_adapter.make_items(rows, pids=[132])[0]
SEED, N, MAX = 82244, 4, 1600

def philox_offset(state):
    # CUDA RNG state: 8-byte seed followed by philox counter; try BOTH 8-byte words
    return (int.from_bytes(state[8:16], "little"), int.from_bytes(state[0:8], "little"))

def gen(arm):
    torch.manual_seed(SEED)
    s0 = torch.cuda.get_rng_state().clone()
    seqs, _ = generate_batch(model, tok, it["prompt"], n=N, max_new_tokens=MAX,
        controller=ctrl if arm == "steered" else None, record=arm == "steered",
        do_sample=True, temperature=0.8, top_p=0.95, seed=SEED, enable_thinking=True)
    s1 = torch.cuda.get_rng_state()
    o0, o1 = philox_offset(s0), philox_offset(s1)
    print(f"{arm}: rng_delta words=({o1[0]-o0[0]}, {o1[1]-o0[1]})", flush=True)
    return seqs

ba1 = gen("base"); ba2 = gen("base")
st1 = gen("steered"); st2 = gen("steered")
for name, a, b in (("base-vs-base", ba1, ba2), ("steered-vs-steered", st1, st2)):
    for j in range(N):
        x, y = a[j], b[j]
        m = min(len(x), len(y))
        flip = next((i for i in range(m) if x[i] != y[i]), None)
        print(f"{name} sample {j}: lens {len(x)} {len(y)} first_flip={flip}", flush=True)
print("DONE")
PYEOF
