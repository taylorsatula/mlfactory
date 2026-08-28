#!/usr/bin/env python3
"""R4v2 — windowed forks on detector-nominated onset states.

Redesign of R4 (principal ruling 2026-08-28,
lab_notes/2026-08-28-r4-attendance-stopped-design-change.md): the v1
design rolled each fork out to the terminal state and scored terminal
correctness, but the intervention's effect lives in the tokens right
after the fork — terminal correctness over an 8-20k horizon is noise
relative to it (117-row partial run: noop 12/25 vs toward_healthy
13/25 on matched pairs). R4v2 rolls each fork out for WINDOW tokens
and saves the window text plus the pre-fork tail; an LLM judge reads
the three-branch windows and assesses WHAT the intervention changed.
Terminal correctness is deliberately out of scope — not scored, not
stored, not a metric here.

Everything the v1 run proved is inherited unchanged:
  Plan      the frozen fork_plan_r4.jsonl (27 states x 3 arms x m=24).
  Arms      noop / toward_healthy (+lam d) / toward_diverge (-lam d).
  Layers    focal per class: CYCLE L18, LOOP L2, MUSE L17.
  Seeds     paired across arms; batch seeds derive from sha256(state_id)
            so they are stable across processes and resumes.
  Model     bf16 HF Qwen3.5-9B, sdpa with FLASH_ATTENTION forced
            (bit-deterministic under concurrency, verified 2026-08-27).
  Guards    model-load and batch-1 OOM retries, periodic + post-generate
            empty_cache, resume by (state, arm, seed).

Row schema (one per continuation):
  identity:   state_id, class, substrate, pid, sample_i, arm, lam,
              layer, fork_token, seed_i, seed_batch
  evidence:   prefix_tail (last TAIL tokens before the fork, text),
              window (generated text, up to WINDOW tokens),
              n_new, window_capped, elapsed_s

Run as a module from the repo root:
  CUDA_VISIBLE_DEVICES=0 python -m mlfactory.experiments.ace.annotate.fork_r4v2 --run
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoModelForCausalLM, AutoTokenizer

from mlfactory.experiments.ace.core.steering_controller import (
    MODEL_PATH, PAD_TOKEN_ID, STOP_TOKEN_IDS, build_prompt_ids)

HERE = Path(__file__).resolve().parent
ACE = HERE.parent
DATA = ACE / "data"
PLAN_PATH = Path(__file__).resolve().parents[4] / "artifacts" / "fork_plan_r4.jsonl"

FOCAL = {"CYCLE": 18, "LOOP": 2, "MUSE": 17}
ARMS = ("noop", "toward_healthy", "toward_diverge")
M_PER_ARM = 24
SUB_BATCH = 1
CAP_ABS = 26000          # absolute-trace bound the plan was built under
WINDOW = 2048            # post-fork rollout the judge reads
TAIL = 512               # pre-fork context tokens saved for the judge
TEMPERATURE = 0.8
TOP_P = 0.95
RUN_SEED_BASE = 484100   # shared with fork_r4.py: same state -> same seeds
DIRECTIONS = DATA / "steering_directions" / "directions_annot_clear_merged.npz"


def load_plan() -> list[dict]:
    return [json.loads(l) for l in PLAN_PATH.open()]


_DECODE_STEPS = [0]


def _empty_cache_hook(_m, _inp, out):
    """Periodic empty_cache during decode (every 128 steps). Keeps the
    steady footprint near weights+KV under concurrent processes; frees
    only unused cached blocks, no effect on numerics."""
    h = out[0] if isinstance(out, tuple) else out
    if h.shape[1] == 1:
        _DECODE_STEPS[0] += 1
        if _DECODE_STEPS[0] % 128 == 0:
            torch.cuda.empty_cache()


def make_hook(model, layer: int, delta: torch.Tensor, st: dict):
    """Steer the residual stream from the fork position onward.
    Prefill chunk: add delta at the fork position's index if the chunk
    contains it. Decode step: every generated position is >= fork, so
    add delta unconditionally."""
    def hook(_m, _inp, out):
        h = out[0] if isinstance(out, tuple) else out
        d = delta.to(h.dtype)
        if h.shape[1] > 1:
            idx = st["fork"] - 1 - st["start"]
            if 0 <= idx < h.shape[1]:
                h = h.clone()
                h[:, idx] = h[:, idx] + d
        else:
            h = h + d
        return (h,) + out[1:] if isinstance(out, tuple) else h
    return model.model.layers[layer].register_forward_hook(hook)


@torch.no_grad()
def gen_one(model, arm_delta, layer, prefix_ids: list[int], fork_abs: int,
            max_new: int, seed: int):
    """One continuation from the fork point (batch 1).

    Full-forward generation over the shared prefix — the GRPO-proven
    pattern (no cache surgery). The hook adds delta at the fork
    position during prefill and at every decode step, so steering
    starts exactly at the fork. Returns the new-token id list
    (positions >= fork), trimmed at the first stop token."""
    torch.manual_seed(seed)
    st = {"start": 0, "end": fork_abs, "fork": fork_abs}
    handle = None
    if arm_delta is not None:
        handle = make_hook(model, layer, arm_delta, st)
    gc_handle = model.model.layers[0].register_forward_hook(
        _empty_cache_hook)
    stop = set(STOP_TOKEN_IDS)
    try:
        x = torch.tensor([prefix_ids], dtype=torch.long, device=model.device)
        kwargs = dict(input_ids=x, attention_mask=torch.ones_like(x),
                      max_new_tokens=max_new, do_sample=True,
                      temperature=TEMPERATURE, top_p=TOP_P,
                      eos_token_id=STOP_TOKEN_IDS, pad_token_id=PAD_TOKEN_ID)
        out = model.generate(**kwargs)
        row = out.tolist()[0][fork_abs:]
        for i, t in enumerate(row):
            if t in stop:
                row = row[:i + 1]
                break
        return row
    finally:
        gc_handle.remove()
        if handle is not None:
            handle.remove()


def gen_with_retry(model, arm_delta, layer, prefix_ids, fork_abs, max_new,
                   seed, what: str) -> list[int]:
    """Batch-1 generation with backoff retries on OOM (loud raise if
    persistent — a silent skip drops the row)."""
    for wait_s in (0, 25, 60):
        if wait_s:
            print(f"OOM at {what}, retrying after {wait_s}s", flush=True)
            torch.cuda.empty_cache()
            time.sleep(wait_s)
        try:
            return gen_one(model, arm_delta, layer, prefix_ids, fork_abs,
                           max_new, seed)
        except torch.cuda.OutOfMemoryError:
            continue
    raise torch.cuda.OutOfMemoryError(f"persistent OOM: {what}")


def run(args) -> None:
    plan = load_plan()
    if args.only:
        keep = set(args.only.split(","))
        plan = [p for p in plan if p["state_id"] in keep]
    m = args.m or M_PER_ARM
    seed_filter = ({int(s) for s in args.seeds.split(",")}
                   if args.seeds else None)
    arms_run = (tuple(a for a in ARMS if a in set(args.arms.split(",")))
                if args.arms else ARMS)
    out_path = Path(args.out)
    done: set[tuple] = set()
    if out_path.exists():
        for l in out_path.open():
            r = json.loads(l)
            done.add((r["state_id"], r["arm"], r["seed_i"]))
    total_rows = len(plan) * len(arms_run) * m
    print(f"states: {len(plan)} | arms: {len(arms_run)} | m: {m} | "
          f"window: {WINDOW} | rows target: {total_rows} | "
          f"already done: {len(done)}", flush=True)

    dirs = np.load(DIRECTIONS)
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=torch.bfloat16, device_map="cuda",
        attn_implementation="sdpa")
    model.eval()

    directions = {cls: torch.tensor(
        dirs[f"dir_{cls.lower()}_L{FOCAL[cls]}"], dtype=torch.float32,
        device=model.device).to(model.dtype) for cls in FOCAL}

    fout = out_path.open("a")
    rows_written = 0
    for si, st in enumerate(plan):
        prefix = build_prompt_ids(tok, st["prompt_text"], enable_thinking=True)
        c_ids = tok(st["completion"][:CAP_ABS * 4],
                    add_special_tokens=False)["input_ids"][:CAP_ABS]
        prefix_ids = (prefix + c_ids)[: st["fork_abs"]]
        assert len(prefix_ids) == st["fork_abs"], \
            f"prefix rebuild mismatch for {st['state_id']}"
        max_new = min(WINDOW, CAP_ABS - st["fork_abs"])
        prefix_tail = tok.decode(prefix_ids[max(0, len(prefix_ids) - TAIL):],
                                 skip_special_tokens=False)
        d = directions[st["class"]]
        arm_deltas = {"noop": None,
                      "toward_healthy": st["lam"] * d,
                      "toward_diverge": -st["lam"] * d}
        for arm in arms_run:
            pending = [s for s in range(m)
                       if (st["state_id"], arm, s) not in done
                       and (seed_filter is None or s in seed_filter)]
            if not pending:
                continue
            for seed_i in pending:
                # paired seeds: base_hash + seed_i — identical across
                # arms (each arm iterates seeds afresh) and stable
                # under resume/seed-subsets because the torch seed keys
                # on seed_i, not on the pending list's position (v1
                # keyed on the batch index, which shifts on resume)
                seed = (RUN_SEED_BASE
                        + int.from_bytes(hashlib.sha256(
                              st["state_id"].encode()).digest()[:4],
                              "big") % 100000 * 97 + seed_i)
                what = f"{st['state_id']} {arm} seed {seed_i}"
                t0 = time.time()
                new_ids = gen_with_retry(model, arm_deltas[arm], st["layer"],
                                         prefix_ids, st["fork_abs"], max_new,
                                         seed, what)
                torch.cuda.empty_cache()
                row = {
                    "state_id": st["state_id"], "class": st["class"],
                    "substrate": st["substrate"], "pid": st["pid"],
                    "sample_i": st["sample_i"], "arm": arm,
                    "lam": st["lam"], "layer": st["layer"],
                    "fork_token": st["fork_abs"],
                    "seed_i": seed_i, "seed_batch": seed,
                    "n_new": len(new_ids),
                    "window_capped": len(new_ids) >= max_new,
                    "elapsed_s": round(time.time() - t0, 1),
                    "prefix_tail": prefix_tail,
                    "window": tok.decode(new_ids, skip_special_tokens=False),
                }
                fout.write(json.dumps(row) + "\n")
                fout.flush()
                rows_written += 1
                print(f"[{si+1}/{len(plan)}] {what}: {len(new_ids)} tok, "
                      f"{time.time()-t0:.0f}s, "
                      f"done {len(done)+rows_written}/{total_rows}",
                      flush=True)
    fout.close()
    print(f"run complete: {rows_written} new rows -> {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--out", default=str(DATA / "fork_r4v2_results.jsonl"))
    ap.add_argument("--m", type=int, default=0, help="samples per arm (0=24)")
    ap.add_argument("--only", default="", help="comma-sep state_ids")
    ap.add_argument("--arms", default="", help="comma-sep arm subset")
    ap.add_argument("--seeds", default="",
                    help="comma-sep seed_i subset (empty=all)")
    args = ap.parse_args()
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        if args.run:
            run(args)
        else:
            ap.error("pass --run")


if __name__ == "__main__":
    main()
