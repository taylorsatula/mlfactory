#!/usr/bin/env python3
"""Step-1 guard test: a planted NaN in the controller must trip the
objective guard (exit 9), and the same path with finite weights must not.

Run (remote):
  HF_HOME=/workspace/models PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  ACE_MODEL_PATH=Qwen/Qwen3.5-9B CUDA_VISIBLE_DEVICES=0 \
  /venv/main/bin/python /workspace/test_nan_guard.py \
      --pool /workspace/mlfactory/mlfactory/experiments/ace/data/acegen_live_b2.jsonl
"""
from __future__ import annotations

import argparse
import os
from argparse import Namespace
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mlfactory.experiments.ace.core.steering_controller import (
    SteeringController, build_prompt_ids, freeze_base_model, generate_batch)
from mlfactory.experiments.ace.train import pool_adapter
from mlfactory.experiments.ace.train.grpo import Replay, replay_backward


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pool", type=Path, required=True)
    cfg = p.parse_args()
    mp = os.environ.get("ACE_MODEL_PATH", "Qwen/Qwen3.5-9B")

    tok = AutoTokenizer.from_pretrained(mp)
    model = AutoModelForCausalLM.from_pretrained(
        mp, dtype=torch.bfloat16, device_map="cuda")
    model.eval(); freeze_base_model(model)
    items = pool_adapter.make_items(pool_adapter.load_pool(cfg.pool), [49])
    it = items[0]
    seqs, _ = generate_batch(model, tok, it["prompt"], n=1,
                             max_new_tokens=300, controller=None,
                             do_sample=True, temperature=0.8, top_p=0.95,
                             seed=7, enable_thinking=True)
    ids = list(seqs[0])
    n_prompt = len(build_prompt_ids(tok, it["prompt"], enable_thinking=True))

    ctrl = SteeringController().to(device="cuda", dtype=torch.float32)
    # make it provably nonzero so the NaN actually flows through
    with torch.no_grad():
        ctrl.up.weight.normal_(0, 0.02)
        ctrl.gate.bias.fill_(1.0)
    tcfg = Namespace(beta_kl=0.02, lambda_mag=0.1)

    for mode in ("full",):
        # window mode is excluded: it is known-broken twice over
        # (diag_window_drift boundary corruption; fla chunk-backward
        # crashes on the mutated cache state). The guard protects the
        # production path only.
        eng = Replay(model, mode, window=128)
        # positive control: finite weights -> finite loss
        st = replay_backward(model, ids, n_prompt, ctrl, tcfg, eng,
                             advantage=1.0, grad_scale=1.0)
        assert st["n_tok"] > 0
        print(f"[{mode}] positive control ok: n_tok={st['n_tok']} "
              f"kl={st['kl']:.4f}", flush=True)
        ctrl.zero_grad(set_to_none=True)
        # plant NaN, expect SystemExit(9)
        with torch.no_grad():
            ctrl.up.weight.fill_(float("nan"))
        try:
            replay_backward(model, ids, n_prompt, ctrl, tcfg, eng,
                            advantage=1.0, grad_scale=1.0)
        except SystemExit as e:
            assert e.code == 9, f"unexpected exit code {e.code}"
            print(f"[{mode}] NaN guard TRIPPED (exit 9) as required",
                  flush=True)
        else:
            raise AssertionError(f"[{mode}] NaN did NOT trip the guard")
        with torch.no_grad():
            ctrl.up.weight.normal_(0, 0.02)
    print("NAN-GUARD-TEST PASS", flush=True)


if __name__ == "__main__":
    main()
