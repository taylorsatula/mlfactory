#!/usr/bin/env python3
"""Diagnose the windowed-replay logprob drift: is it the cache mechanism,
the split point, or recurrence-rounding that grows with depth?

Passes (all zero-init controller == exact no-op, so any diff vs the
single-pass ref is pure mechanism drift):
  ref    single forward, no cache
  D      single forward WITH use_cache=True (no split) -> isolates cache
  A      split at n_prompt-1 (production window-1 shape)
  C1     split at ~1/3, C2 at ~2/3 -> drift vs split depth
  Also: position-binned |diff| for A, to see if drift accumulates.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mlfactory.experiments.ace.core.steering_controller import (
    SteeringController, build_prompt_ids, freeze_base_model, generate_batch)
from mlfactory.experiments.ace.train import pool_adapter
from mlfactory.experiments.ace.train.grpo import completion_logprobs


def fwd(model, ids, cache=None):
    x = torch.tensor([ids], device=model.device)
    with torch.no_grad():
        out = model(input_ids=x, past_key_values=cache,
                    use_cache=cache is not None)
    return out


def logp_from(logits, ids, np_eff):
    return completion_logprobs(logits, ids, np_eff).detach().float().cpu()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pool", type=Path, required=True)
    p.add_argument("--max-new", type=int, default=3000)
    cfg = p.parse_args()
    mp = os.environ.get("ACE_MODEL_PATH", "Qwen/Qwen3.5-9B")
    tok = AutoTokenizer.from_pretrained(mp)
    model = AutoModelForCausalLM.from_pretrained(
        mp, dtype=torch.bfloat16, device_map="cuda")
    model.eval(); freeze_base_model(model)
    items = pool_adapter.make_items(pool_adapter.load_pool(cfg.pool), [49])
    it = items[0]
    seqs, _ = generate_batch(model, tok, it["prompt"], n=1,
                             max_new_tokens=cfg.max_new, controller=None,
                             do_sample=True, temperature=0.8, top_p=0.95,
                             seed=80_000 + 17 * 49, enable_thinking=True)
    ids = list(seqs[0])
    n_prompt = len(build_prompt_ids(tok, it["prompt"], enable_thinking=True))
    n_comp = len(ids) - n_prompt
    out = {"id": it["id"], "n_prompt": n_prompt, "n_comp": n_comp}

    def split_at(sp):
        """prefix ids[:sp] cached; window ids[max(sp-1,0):] with cache."""
        if sp <= 0:
            o = fwd(model, ids, cache=None)
            return logp_from(o.logits, ids, n_prompt)
        o = fwd(model, ids[:sp], cache=None)
        cache = o.past_key_values
        del o
        ctx = ids[sp - 1:]
        o = fwd(model, ctx, cache=cache)
        lp = logp_from(o.logits, ctx, 1)
        del o, cache
        return lp

    # ref
    o = fwd(model, ids, cache=None)
    ref = logp_from(o.logits, ids, n_prompt)
    del o
    # D: single pass with cache on, no split
    o = model(input_ids=torch.tensor([ids], device=model.device),
              use_cache=True)
    lpD = logp_from(o.logits, ids, n_prompt)
    del o

    def cmp(name, lp, ref_slice):
        d = (lp - ref_slice).abs()
        row = {"n": int(d.numel()), "max": float(d.max()),
               "mean": float(d.mean()), "p99": float(d.quantile(0.99)),
               "argmax_pos": int(d.argmax())}
        out[name] = row
        print(name, json.dumps(row), flush=True)

    cmp("D_single_pass_cache_on", lpD, ref)
    lpA = split_at(n_prompt)
    cmp("A_split_at_prompt", lpA, ref)
    d = (lpA - ref).abs()
    bins = 6
    out["A_binned_mean_abs"] = [
        float(d[i * n_comp // bins:(i + 1) * n_comp // bins].mean())
        for i in range(bins)]
    print("A_binned", out["A_binned_mean_abs"], flush=True)
    lpC1 = split_at(n_prompt + n_comp // 3)
    cmp("C1_split_at_third", lpC1, ref[n_comp // 3:])
    lpC2 = split_at(n_prompt + 2 * n_comp // 3)
    cmp("C2_split_at_two_thirds", lpC2, ref[2 * n_comp // 3:])
    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()
