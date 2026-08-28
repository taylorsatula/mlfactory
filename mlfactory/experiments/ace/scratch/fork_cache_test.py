#!/usr/bin/env python3
"""Scratch verification for the fork_r4 cached-decode path (2026-08-27).

Nothing imports this. Questions it answers before the R4 run uses the
cached path:
  T1  chunked prefill -> batch_repeat_interleave -> batched decode works
  T2  MATH-backend decode is deterministic (same seed -> same tokens)
  T3  the residual hook is live (hooked greedy diverges from unhooked)
  T4  chunked-prefix cached decode == monolithic full-forward decode
      (same seed, greedy) on a prefix short enough to fit a 24 GB card

Run: CUDA_VISIBLE_DEVICES=1 ace/.venv/bin/python -m mlfactory.experiments.ace.scratch.fork_cache_test
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

from mlfactory.experiments.ace.annotate.fork_r4 import FOCAL, load_plan
from mlfactory.experiments.ace.core.steering_controller import (
    MODEL_PATH, PAD_TOKEN_ID, STOP_TOKEN_IDS, build_prompt_ids)

CHUNK = 2048
T4_PREFIX = 6000  # monolithic-forward budget on a 24 GB card


@torch.no_grad()
def chunked_prefill(model, ids: list[int], upto: int, batch: int = 1):
    x = torch.tensor([ids[:upto]] * batch, dtype=torch.long,
                     device=model.device)
    cache = DynamicCache(config=model.config)
    for a in range(0, upto, CHUNK):
        model.model(input_ids=x[:, a:min(a + CHUNK, upto)],
                    past_key_values=cache, use_cache=True)
    return cache


@torch.no_grad()
def decode(model, cache, n_new: int, seed: int | None, greedy: bool):
    if seed is not None:
        torch.manual_seed(seed)
    kwargs = dict(
        input_ids=torch.empty((cache.batch_size, 0), dtype=torch.long,
                              device=model.device),
        attention_mask=torch.ones((cache.batch_size, cache.get_seq_length()),
                                  dtype=torch.long, device=model.device),
        past_key_values=cache, max_new_tokens=n_new,
        do_sample=not greedy, temperature=0.8, top_p=0.95,
        eos_token_id=STOP_TOKEN_IDS, pad_token_id=PAD_TOKEN_ID)
    out = model.generate(**kwargs)
    return out.tolist()


def main() -> None:
    import numpy as np
    plan = [r for r in load_plan() if r["state_id"] == "r4_loop_01"][0]
    dirs = np.load(Path(__file__).resolve().parents[1] / "data" /
                   "steering_directions" / "directions_annot_clear_merged.npz")
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    prefix = build_prompt_ids(tok, plan["prompt_text"], enable_thinking=True)
    c_ids = tok(plan["completion"][:26000 * 4],
                add_special_tokens=False)["input_ids"][:26000]
    ids = (prefix + c_ids)[: plan["fork_abs"]]
    f = plan["fork_abs"]
    print(f"state r4_loop_01: fork_abs={f} (prefix {f} tokens)")

    with sdpa_kernel(SDPBackend.MATH):
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, dtype=torch.bfloat16, device_map="cuda",
            attn_implementation="sdpa")
        model.eval()

        # T1+T2: batched chunked prefill, seeded decode twice
        big = chunked_prefill(model, ids, f, batch=2)
        a = decode(model, big, 40, seed=1234, greedy=False)
        del big
        torch.cuda.empty_cache()
        big2 = chunked_prefill(model, ids, f, batch=2)
        b = decode(model, big2, 40, seed=1234, greedy=False)
        del big2
        torch.cuda.empty_cache()
        print("T1 batched decode shapes:", len(a), [len(r) for r in a])
        print("T2 deterministic:", a == b)

        # T3: hook liveness — greedy 32 tokens with and without +lam d at L2
        delta = torch.tensor(dirs[f"dir_loop_L{FOCAL['LOOP']}"],
                             dtype=torch.float32, device=model.device)
        delta = plan["lam"] * delta
        g0 = decode(model, (c0 := chunked_prefill(model, ids, f)), 32, None, True)
        del c0
        torch.cuda.empty_cache()

        mode = {}
        def hook(_m, _inp, out):
            h = out[0] if isinstance(out, tuple) else out
            if h.shape[1] > 1:
                h = h.clone(); h[:, -1] = h[:, -1] + delta
            else:
                h = h + delta
            return (h,) + out[1:] if isinstance(out, tuple) else h
        handle = model.model.layers[FOCAL["LOOP"]].register_forward_hook(hook)
        c1 = chunked_prefill(model, ids, f)
        g1 = decode(model, c1, 32, None, True)
        handle.remove()
        del c0, c1
        torch.cuda.empty_cache()
        diff = sum(1 for x, y in zip(g0[0], g1[0]) if x != y)
        print(f"T3 hook live: {diff}/32 tokens differ "
              f"(lam={plan['lam']}, L{FOCAL['LOOP']})")

        # T4: cached chunked vs monolithic full forward, greedy 16 tokens
        del a, b, big, big2, g0, g1
        torch.cuda.empty_cache()
        cut = min(T4_PREFIX, f)
        gA = decode(model, chunked_prefill(model, ids, cut), 16, None, True)
        x = torch.tensor([ids[:cut]], device=model.device)
        outB = model.generate(input_ids=x, attention_mask=torch.ones_like(x),
                              max_new_tokens=16, do_sample=False,
                              eos_token_id=STOP_TOKEN_IDS,
                              pad_token_id=PAD_TOKEN_ID)
        gB = [outB[0, cut:].tolist()]
        print("T4 cached==monolithic (greedy 16):", gA == gB,
              "| cached:", gA[0][:8], "| mono:", gB[0][:8])


if __name__ == "__main__":
    main()
