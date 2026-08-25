#!/usr/bin/env python3
"""H200 forecasting smoke for the GRPO attempt on the b2 pool.

Target regime (user rulings 2026-08-25): thinking ON, bf16, b2 pool.
This script measures — it does not train:

  1. model load        baseline VRAM (bf16, single GPU)
  2. generation        thinking-on group rollouts on real pool prompts:
                       tok/s, trace-length distribution, truncation rate,
                       and per-prompt strict correctness — which doubles
                       as the first pass of the pool's bf16
                       re-verification-by-regeneration (OPERATIONS.md)
  3. replay curve      replay-with-gradient cost vs completion length ->
                       sets the replay cap for the production run
  4. one backward step controller gradients end-to-end: finiteness,
                       grad norm, peak memory, base-fingerprint check

Exercises the REAL production paths (steering_controller.generate_batch,
grpo.replay_logprobs, frontier solver_prompt, gen.calibrate CHECK) so the
numbers forecast the actual run, not a replica.

Run (on the remote):
  HF_HOME=/workspace/models \\
  PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \\
  /venv/main/bin/python -m mlfactory.experiments.ace.train.smoke_h200 \\
      --pool mlfactory/experiments/ace/data/acegen_live_b2.jsonl \\
      --out /workspace/smoke1 --model-path Qwen/Qwen3.5-9B
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch

from mlfactory.experiments.ace.core.steering_controller import (
    STEER_LAYER, SteeringController, build_prompt_ids, freeze_base_model,
    generate_batch,
)
from mlfactory.experiments.ace.frontier.collect_rollouts import solver_prompt
from mlfactory.experiments.ace.gen.calibrate import CHECK
from mlfactory.experiments.ace.train.grpo import (
    check_base, replay_logprobs, snapshot_base,
)


def mem_gb() -> dict:
    return {"alloc": round(torch.cuda.memory_allocated() / 2**30, 2),
            "reserved": round(torch.cuda.memory_reserved() / 2**30, 2),
            "peak_alloc": round(torch.cuda.max_memory_allocated() / 2**30, 2),
            "peak_reserved":
                round(torch.cuda.max_memory_reserved() / 2**30, 2)}


def load_pool(path: Path, pids: list[int] | None,
              per_family: int) -> list[dict]:
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    items = []
    for r in rows:
        pid = r["provenance"]["proposal_id"]
        if pids and pid not in pids:
            continue
        items.append({
            "id": f"{r['domain']}-p{pid}", "pid": pid,
            "family": r["domain"],
            "prompt": solver_prompt(r),
            "reference": r["problem"]["reference_answer"],
            "knobs": r.get("knobs", {}),
        })
    if pids:
        return items
    # stratified: first `per_family` rows per family, file order
    seen, out = {}, []
    for it in items:
        seen[it["family"]] = seen.get(it["family"], 0) + 1
        if seen[it["family"]] <= per_family:
            out.append(it)
    return out


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pool", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model-path", type=str,
                   default=os.environ.get("ACE_MODEL_PATH",
                                          "Qwen/Qwen3.5-9B"))
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--per-family", type=int, default=1)
    p.add_argument("--pids", type=str, default=None,
                   help="comma-separated proposal ids (overrides stratify)")
    p.add_argument("--group", type=int, default=4)
    p.add_argument("--max-new", type=int, default=26000)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--replay-caps", type=str,
                   default="512,1024,2048,4096,8192,16384")
    return p.parse_args()


def main():
    cfg = parse_args()
    cfg.out.mkdir(parents=True, exist_ok=True)
    report = {"config": vars(cfg) | {"torch": torch.__version__,
                                      "cuda": torch.version.cuda,
                                      "device_name":
                                          torch.cuda.get_device_name(0)},
              "steps": []}

    def log(step: str, **kw):
        row = {"step": step, **kw}
        report["steps"].append(row)
        print(json.dumps(row), flush=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    # ---- 1. model load -------------------------------------------------
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(cfg.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_path, dtype=torch.bfloat16, device_map={"": cfg.device})
    model.eval()
    freeze_base_model(model)
    snap = snapshot_base(model)
    log("load", dt_s=round(time.time() - t0, 1), mem=mem_gb(),
        n_params=sum(p.numel() for p in model.parameters()))

    # ---- 2. thinking-on generation on pool prompts ----------------------
    pids = ([int(x) for x in cfg.pids.split(",")] if cfg.pids else None)
    items = load_pool(cfg.pool, pids, cfg.per_family)
    log("pool", n=len(items),
        families=sorted({it["family"] for it in items}))
    ctrl = SteeringController().to(device=cfg.device, dtype=torch.float32)

    gen_rows, traces = [], []
    for it in items:
        n_prompt = len(build_prompt_ids(tok, it["prompt"],
                                        enable_thinking=True))
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        t0 = time.time()
        seqs, _ = generate_batch(model, tok, it["prompt"], n=cfg.group,
                                 max_new_tokens=cfg.max_new,
                                 controller=None, do_sample=True,
                                 temperature=cfg.temperature,
                                 top_p=cfg.top_p,
                                 seed=70_000 + it["pid"],
                                 enable_thinking=True)
        dt = time.time() - t0
        n_tok = sum(len(s) - n_prompt for s in seqs)
        oks, lens = [], []
        for s in seqs:
            text = tok.decode(s[n_prompt:], skip_special_tokens=True)
            lens.append(len(s) - n_prompt)
            oks.append(int(CHECK[it["family"]](text, it["reference"],
                                               it["knobs"])))
            traces.append({"id": it["id"], "ids": s, "n_prompt": n_prompt,
                           "ok": oks[-1]})
        gen_rows.append({
            "id": it["id"], "family": it["family"],
            "group": cfg.group, "strict_correct": f"{sum(oks)}/{len(oks)}",
            "lens": lens, "truncated": [l >= cfg.max_new for l in lens],
            "tok_per_s": round(n_tok / dt, 1), "dt_s": round(dt, 1),
            "peak_mem": mem_gb()["peak_reserved"],
        })
        log("generate", **gen_rows[-1] | {"ids": "elided"})
        torch.cuda.empty_cache()

    (cfg.out / "gen_rows.json").write_text(json.dumps(gen_rows, indent=2))

    # ---- 3. replay memory curve (with gradients, as in production) -----
    traces.sort(key=lambda t: len(t["ids"]))
    probe = traces[-1] if traces else None
    caps = [int(c) for c in cfg.replay_caps.split(",")]
    if probe is not None:
        n_comp = len(probe["ids"]) - probe["n_prompt"]
        opt = torch.optim.AdamW(ctrl.parameters(), lr=1e-3)
        for cap in caps:
            if cap > n_comp:
                continue
            opt.zero_grad(set_to_none=True)
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
            try:
                t0 = time.time()
                logp, rel = replay_logprobs(
                    model, probe["ids"], probe["n_prompt"],
                    controller=ctrl, collect=True, token_cap=cap)
                loss = -logp.mean() + 0.1 * (
                    rel if rel is not None else logp.new_zeros(()))
                loss.backward()
                dt = time.time() - t0
                gn = torch.nn.utils.clip_grad_norm_(ctrl.parameters(), 1.0)
                log("replay", cap=cap, comp_len=n_comp, dt_s=round(dt, 1),
                    peak_mem=mem_gb()["peak_reserved"],
                    grad_norm=float(gn),
                    grad_finite=all(torch.isfinite(p.grad).all()
                                    for p in ctrl.parameters()
                                    if p.grad is not None))
            except torch.cuda.OutOfMemoryError:
                log("replay", cap=cap, comp_len=n_comp, oom=True,
                    peak_mem=mem_gb()["peak_reserved"])
                torch.cuda.empty_cache()
                break

    # ---- 4. summary: fingerprint + report -------------------------------
    log("fingerprint", base_unchanged=check_base(model, snap))
    (cfg.out / "report.json").write_text(json.dumps(report, indent=2,
                                                    default=str))
    print("\n=== summary ===", flush=True)
    tot_tok = sum(sum(g["lens"]) for g in gen_rows)
    tot_dt = sum(g["dt_s"] for g in gen_rows)
    if tot_dt:
        print(f"aggregate generation: {tot_tok} tok / {tot_dt:.0f}s "
              f"= {tot_tok/tot_dt:.0f} tok/s "
              f"(group={cfg.group}, thinking on)")
    all_lens = [l for g in gen_rows for l in g["lens"]]
    if all_lens:
        all_lens.sort()
        print(f"trace lengths: min={all_lens[0]} "
              f"median={all_lens[len(all_lens)//2]} max={all_lens[-1]}")
    print(f"report: {cfg.out}/report.json")


if __name__ == "__main__":
    main()
