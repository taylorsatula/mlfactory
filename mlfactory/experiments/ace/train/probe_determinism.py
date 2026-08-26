#!/usr/bin/env python3
"""Cross-process sampling determinism probe (Step-1 anomaly, Finding 5).

Two FRESH processes, same seed/settings: are the generated token ids
bit-identical? If yes, the resume contract holds across crashes; if no,
long-trace regeneration is only statistical and rows files are the
evidence of record. Run twice, one process per launch:

  ... python -m mlfactory.experiments.ace.train.probe_determinism \
      --pool ... --pid 132 --tag A
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mlfactory.experiments.ace.core.steering_controller import (
    freeze_base_model, generate_batch)
from mlfactory.experiments.ace.train import pool_adapter


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pool", type=Path, required=True)
    p.add_argument("--pid", type=int, default=132)
    p.add_argument("--tag", type=str, required=True)
    p.add_argument("--max-new", type=int, default=2000)
    p.add_argument("--seed", type=int, default=80_000 + 17 * 132)
    p.add_argument("--out", type=Path, default=Path("/workspace/det"))
    cfg = p.parse_args()
    cfg.out.mkdir(parents=True, exist_ok=True)
    mp = os.environ.get("ACE_MODEL_PATH", "Qwen/Qwen3.5-9B")
    tok = AutoTokenizer.from_pretrained(mp)
    model = AutoModelForCausalLM.from_pretrained(
        mp, dtype=torch.bfloat16, device_map="cuda")
    model.eval(); freeze_base_model(model)
    it = pool_adapter.make_items(pool_adapter.load_pool(cfg.pool),
                                 [cfg.pid])[0]
    seqs, _ = generate_batch(model, tok, it["prompt"], n=4,
                             max_new_tokens=cfg.max_new, controller=None,
                             do_sample=True, temperature=0.8, top_p=0.95,
                             seed=cfg.seed, enable_thinking=True)
    torch.save({"ids": seqs, "seed": cfg.seed, "pid": cfg.pid,
                "max_new": cfg.max_new}, cfg.out / f"run_{cfg.tag}.pt")
    lens = [len(s) for s in seqs]
    print(f"[{cfg.tag}] saved {cfg.out}/run_{cfg.tag}.pt lens={lens}",
          flush=True)


if __name__ == "__main__":
    main()
