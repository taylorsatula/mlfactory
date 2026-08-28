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
SEED, N, MAX = 82244, 4, 100

print("controller training mode:", ctrl.training)
# RNG consumption inside the hook? compare CUDA gen offset
def gen(arm):
    s0 = torch.cuda.get_rng_state()
    seqs, _ = generate_batch(model, tok, it["prompt"], n=N,
        max_new_tokens=MAX,
        controller=ctrl if arm == "steered" else None,
        record=arm == "steered", do_sample=True,
        temperature=0.8, top_p=0.95, seed=SEED, enable_thinking=True)
    return [s[:MAX] for s in seqs]

# exact dry-run order: steered first, then base (each reseeds internally)
st1 = gen("steered")
ba1 = gen("base")
st2 = gen("steered")  # repeatability check
for j in range(N):
    print(f"sample {j}: steered[:6]={st1[j][:6]} base[:6]={ba1[j][:6]} "
          f"same={st1[j]==ba1[j]} rep={st1[j]==st2[j]}")
