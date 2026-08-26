#!/usr/bin/env python3
"""Step-1 ladder probe: ONE 26k-cap thinking-on trace, then the real
replay cost on it — gradient-checkpointed full-trace replay with one
backward step, plus the windowed path for comparison. Measures memory
and grad finiteness at production trace length; checks base fingerprint.

Run (remote):
  HF_HOME=/workspace/models PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  ACE_MODEL_PATH=Qwen/Qwen3.5-9B CUDA_VISIBLE_DEVICES=0 \
  /venv/main/bin/python /workspace/probe_long_replay.py \
      --pool /workspace/mlfactory/mlfactory/experiments/ace/data/acegen_live_b2.jsonl \
      --pid 49 --out /workspace/s1_long
"""
from __future__ import annotations

import argparse
import json
import time
from argparse import Namespace
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mlfactory.experiments.ace.core.steering_controller import (
    SteeringController, build_prompt_ids, freeze_base_model, generate_batch)
from mlfactory.experiments.ace.train import pool_adapter
from mlfactory.experiments.ace.train.grpo import (
    Replay, check_base, iter_replay_windows, ref_logprobs, replay_backward,
    replay_full, snapshot_base)


def mem_gb() -> float:
    return round(torch.cuda.max_memory_reserved() / 2**30, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pool", type=Path, required=True)
    p.add_argument("--pid", type=int, default=49)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--max-new", type=int, default=26000)
    p.add_argument("--window", type=int, default=8192)
    p.add_argument("--from-trace", action="store_true",
                   help="reuse OUT/trace.pt from a previous run (skips gen)")
    cfg = p.parse_args()
    cfg.out.mkdir(parents=True, exist_ok=True)
    import os
    mp = os.environ.get("ACE_MODEL_PATH", "Qwen/Qwen3.5-9B")

    rep = {}
    tok = AutoTokenizer.from_pretrained(mp)
    model = AutoModelForCausalLM.from_pretrained(
        mp, dtype=torch.bfloat16, device_map="cuda")
    model.eval(); freeze_base_model(model)
    snap = snapshot_base(model)
    items = pool_adapter.make_items(pool_adapter.load_pool(cfg.pool),
                                    [cfg.pid])
    it = items[0]
    rep["item"] = it["id"]

    if cfg.from_trace:
        tr = torch.load(cfg.out / "trace.pt", weights_only=False)
        ids, n_prompt = tr["ids"], tr["n_prompt"]
        assert tr["id"] == it["id"]
        rep["gen"] = {"reused_trace": True, "n_new": len(ids) - n_prompt}
        print(json.dumps(rep["gen"]), flush=True)
    else:
        # 1. one long thinking-on rollout
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        seqs, _ = generate_batch(model, tok, it["prompt"], n=1,
                                 max_new_tokens=cfg.max_new, controller=None,
                                 do_sample=True, temperature=0.8, top_p=0.95,
                                 seed=123_456, enable_thinking=True)
        ids = list(seqs[0])
        n_prompt = len(build_prompt_ids(tok, it["prompt"],
                                        enable_thinking=True))
        rep["gen"] = {"n_new": len(ids) - n_prompt,
                      "truncated": seqs[0][-1] not in (248044, 248046),
                      "dt_s": round(time.time() - t0, 1),
                      "tok_per_s": round((len(ids) - n_prompt)
                                          / max(time.time() - t0, 1e-6), 1),
                      "peak_mem": mem_gb()}
        print(json.dumps(rep["gen"]), flush=True)

    ctrl = SteeringController().to(device="cuda", dtype=torch.float32)
    tcfg = Namespace(beta_kl=0.02, lambda_mag=0.1)
    eng = Replay(model, "full", cfg.window)

    # 2. CLEAN equivalence first, at zero-init (exact no-op): any diff here
    #    is mechanism drift, uncontaminated by weight updates.
    torch.save({"ids": ids, "n_prompt": n_prompt, "id": it["id"]},
               cfg.out / "trace.pt")
    ref = ref_logprobs(model, ids, n_prompt)
    logp_full0, _ = replay_full(model, ids, n_prompt, ctrl)
    d_full0 = (logp_full0.detach().float().cpu() - ref.float().cpu()).abs()
    rep["eq_full_zeroinit"] = {"max_abs_diff": float(d_full0.max()),
                               "mean_abs_diff": float(d_full0.mean())}
    del logp_full0
    parts, t0 = [], time.time()
    torch.cuda.reset_peak_memory_stats()
    for a, b, logp_w, rel_w in iter_replay_windows(model, ids, n_prompt,
                                                   ctrl, cfg.window):
        parts.append(logp_w.detach().float().cpu())
        del logp_w
    logp_win = torch.cat(parts)
    d_win0 = (logp_win - ref.float().cpu()).abs()
    rep["eq_window_zeroinit"] = {"max_abs_diff": float(d_win0.max()),
                                 "mean_abs_diff": float(d_win0.mean()),
                                 "dt_s": round(time.time() - t0, 1),
                                 "peak_mem": mem_gb()}
    del logp_win
    print(json.dumps({k: rep[k] for k in
                      ("eq_full_zeroinit", "eq_window_zeroinit")}),
          flush=True)

    # 3. full-trace checkpointed replay + one backward step (weights move
    #    here; everything after includes the controller's effect)
    opt = torch.optim.AdamW(ctrl.parameters(), lr=1e-3)
    opt.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    st = replay_backward(model, ids, n_prompt, ctrl, tcfg, eng,
                         advantage=1.0, grad_scale=1.0)
    gn = torch.nn.utils.clip_grad_norm_(ctrl.parameters(), 1.0)
    opt.step()
    rep["replay_full"] = {**st, "grad_norm": float(gn),
                          "dt_s": round(time.time() - t0, 1),
                          "peak_mem": mem_gb()}
    print(json.dumps(rep["replay_full"], default=str), flush=True)

    # 4. post-step diffs (controller now nonzero: effect + drift mixed;
    #    diagnostic only, not an equivalence verdict)
    logp_full, _ = replay_full(model, ids, n_prompt, ctrl)
    d2 = (logp_full.detach().float().cpu() - ref.float().cpu()).abs()
    rep["replay_full"]["max_abs_diff_vs_ref_poststep"] = float(d2.max())
    rep["fingerprint_ok"] = check_base(model, snap)
    (cfg.out / "probe.json").write_text(json.dumps(rep, indent=2,
                                                   default=str))
    print(json.dumps(rep, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
