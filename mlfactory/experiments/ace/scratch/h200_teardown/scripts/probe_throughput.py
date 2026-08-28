#!/usr/bin/env python3
"""Throughput probe on a free GPU: measured tok/s for thinking-on decode,
plus the active attention implementation. Does not touch the running job.

Run:  ACE_MODEL_PATH=... python probe_throughput.py --device cuda:1
"""
import argparse, time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

p = argparse.ArgumentParser()
p.add_argument("--device", default="cuda:1")
p.add_argument("--model-path", default="Qwen/Qwen3.5-9B")
p.add_argument("--attn", default=None,
               help="force attn implementation (e.g. flash_attention_2)")
p.add_argument("--max-new", type=int, default=384)
p.add_argument("--batch", type=int, default=4)
args = p.parse_args()

tok = AutoTokenizer.from_pretrained(args.model_path)
load_kw = dict(dtype=torch.bfloat16, device_map={"": args.device})
if args.attn:
    load_kw["attn_implementation"] = args.attn
model = AutoModelForCausalLM.from_pretrained(args.model_path, **load_kw)
model.eval()

attn = getattr(model.config, "_attn_implementation", None) or \
       getattr(model.config, "attn_implementation", "unknown")
print(f"attn_implementation: {attn}", flush=True)
try:
    from transformers.utils import is_flash_attn_2_available as fa2
    print("flash_attn_2 available:", fa2(), flush=True)
except Exception as e:
    print("flash check n/a:", e, flush=True)

prompt = ("A courier drives 144 km at 48 km/h, waits 30 minutes, then "
          "drives 108 km at 54 km/h. What is the average speed for the "
          "whole elapsed time? Think carefully, step by step.")
ids = tok.apply_chat_template([{"role": "user", "content": prompt}],
                              add_generation_prompt=True, tokenize=True,
                              enable_thinking=True)
ids = list(ids.input_ids if hasattr(ids, "input_ids") else ids)
n_prompt = len(ids)
x = torch.tensor([ids] * args.batch, device=args.device)
mask = torch.ones_like(x)

# warmup (compile/cache) then timed run
for tag in ("warmup", "timed"):
    torch.manual_seed(0)
    t0 = time.time()
    out = model.generate(input_ids=x, attention_mask=mask,
                         max_new_tokens=args.max_new, do_sample=True,
                         temperature=0.8, top_p=0.95,
                         eos_token_id=[248044, 248046], pad_token_id=248044)
    dt = time.time() - t0
    n_new = sum(min(len(r) - n_prompt, args.max_new) for r in out.tolist())
    print(f"{tag}: {n_new} new tok over batch {args.batch} in {dt:.1f}s "
          f"= {n_new/dt:.0f} tok/s aggregate "
          f"({n_new/dt/args.batch:.0f} tok/s/seq) "
          f"mem={torch.cuda.max_memory_reserved()/2**30:.1f}GB", flush=True)
    torch.cuda.reset_peak_memory_stats()
