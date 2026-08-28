#!/usr/bin/env python3
"""R4 decode-throughput microbenchmark (diagnostic scratch — not imported).

Loads Qwen3.5-9B once, runs batched generate with the production settings
(bf16, sdpa, MATH backend forced, sampling on), and reports:
  - forward-call shapes (verifies cached seq_len=1 decode vs recompute)
  - prefill vs decode timing, per-forward host intervals
  - optional: fla->torch fallback swap (--no-fla), torch.compile (--compile),
    torch.profiler over decode steps (--profile)

Run:  CUDA_VISIBLE_DEVICES=<i> python bench_r4.py [--batch N] [--max-new N]
      [--prefix N] [--no-fla] [--compile] [--profile]
"""
from __future__ import annotations

import argparse
import statistics
import time

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "/home/admin/models/hf/Qwen3.5-9B"


def swap_to_torch_fallback(model):
    """Replicate the no-fla build: rebind per-module kernels to the torch
    fallbacks defined in modeling_qwen3_5 (they are bound at __init__)."""
    from transformers.models.qwen3_5 import modeling_qwen3_5 as M

    n = 0
    for m in model.modules():
        if type(m).__name__ == "Qwen3_5GatedDeltaNet":
            m.causal_conv1d_fn = None
            m.causal_conv1d_update = M.torch_causal_conv1d_update
            m.chunk_gated_delta_rule = M.torch_chunk_gated_delta_rule
            m.recurrent_gated_delta_rule = M.torch_recurrent_gated_delta_rule
            # fused norm -> torch norm with the same weights
            if type(m.norm).__name__ == "FusedRMSNormGated":
                tn = M.Qwen3_5RMSNormGated(m.norm.weight.shape[0],
                                          eps=getattr(m.norm, "variance_epsilon",
                                                      getattr(m.norm, "eps", 1e-6)))
                w = getattr(m.norm, "weight", None)
                if w is not None:
                    tn.weight = torch.nn.Parameter(w.detach().clone())
                tn.to(m.out_proj.weight.device)
                m.norm = tn
            n += 1
    print(f"[no-fla] swapped {n} GatedDeltaNet modules to torch fallback", flush=True)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--prefix", type=int, default=3900)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--seed", type=int, default=484100)
    ap.add_argument("--no-fla", action="store_true")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--runs", type=int, default=2,
                    help="repeat generate this many times (2nd+ are warm)")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    t0 = time.time()
    # AutoModelForCausalLM resolves qwen3_5 to Qwen3_5ForCausalLM (text class,
    # NOT the VL wrapper) — verified; matches production exactly.
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=torch.bfloat16, device_map="cuda",
        attn_implementation="sdpa")
    model.eval()
    print(f"[load] {time.time()-t0:.1f}s, "
          f"mem {torch.cuda.memory_allocated()/1e9:.1f} GB", flush=True)

    from transformers.models.qwen3_5 import modeling_qwen3_5 as M
    print(f"[env] fast_path_available={M.is_fast_path_available} "
          f"fla_chunk={M.chunk_gated_delta_rule} "
          f"conv1d_fn={M.causal_conv1d_fn}", flush=True)

    if args.no_fla:
        swap_to_torch_fallback(model)
    if args.compile:
        model = torch.compile(model)

    # hook the text-model embedding to log every forward call
    calls: list[tuple[int, float]] = []
    text_model = getattr(model.model, "language_model", model.model)
    emb = text_model.embed_tokens
    def pre_hook(_m, inp):
        calls.append((inp[0].shape[-1], time.perf_counter()))
    emb.register_forward_pre_hook(pre_hook)

    g = torch.Generator().manual_seed(args.seed)
    ids = torch.randint(1000, 200000, (args.batch, args.prefix), generator=g)
    x = ids.to(model.device)
    mask = torch.ones_like(x)

    for r in range(args.runs):
        calls.clear()
        torch.manual_seed(args.seed)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with sdpa_kernel(SDPBackend.MATH):
            if args.profile and r == args.runs - 1:
                from torch.profiler import profile as tprofile, ProfilerActivity
                with tprofile(activities=[ProfilerActivity.CPU,
                                          ProfilerActivity.CUDA]) as prof:
                    out = model.generate(
                        input_ids=x, attention_mask=mask,
                        max_new_tokens=args.max_new, do_sample=True,
                        temperature=0.8, top_p=0.95, eos_token_id=[],
                        pad_token_id=tok.pad_token_id or 0)
            else:
                out = model.generate(
                    input_ids=x, attention_mask=mask,
                    max_new_tokens=args.max_new, do_sample=True,
                    temperature=0.8, top_p=0.95, eos_token_id=[],
                    pad_token_id=tok.pad_token_id or 0)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        n_new = out.shape[-1] - args.prefix
        shapes = [c[0] for c in calls]
        uniq = sorted(set(shapes))
        print(f"[run{r}] {n_new} new tok in {dt:.2f}s = {n_new/dt:.2f} tok/s "
              f"aggregate ({args.batch*n_new/dt/args.batch:.2f}/stream); "
              f"forwards={len(calls)}, input-shape set={uniq}", flush=True)
        if len(calls) > 2:
            iv = [(calls[i+1][1] - calls[i][1]) * 1000
                  for i in range(1, len(calls) - 1)]  # decode-step intervals
            iv.sort()
            print(f"[run{r}] prefill {calls[1][1]-calls[0][1]:.2f}s; "
                  f"per-step forward interval ms: "
                  f"p10={iv[len(iv)//10]:.1f} p50={statistics.median(iv):.1f} "
                  f"p90={iv[len(iv)*9//10]:.1f} max={iv[-1]:.1f}", flush=True)
        if args.profile and r == args.runs - 1:
            prof.export_chrome_trace("/root/trace_r4.json"
                                     if __file__.startswith("/root")
                                     else "trace_r4.json")
            print(prof.key_averages().table(
                sort_by="self_cpu_time_total", row_limit=25), flush=True)

    print(f"[mem] peak {torch.cuda.max_memory_allocated()/1e9:.1f} GB", flush=True)


if __name__ == "__main__":
    main()
