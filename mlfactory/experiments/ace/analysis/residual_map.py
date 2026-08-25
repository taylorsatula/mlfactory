#!/usr/bin/env python3
"""Decoupled multi-layer mapping scan of collected ACE traces.

Measurement site is NOT chosen here. This captures broadly so that where each
phenomenon lives can be decided offline from the data:

  * residual output of ALL 32 layers (stride 16, fp16) — no assumption that
    full-attention blocks are the right place; linear-block residuals are
    captured too
  * final recurrent state of ALL 24 linear-attention layers (raw) — the
    channel the architecture doc flags as potentially carrying what the
    residual readout does not
  * per-position next-token entropy (chunked lm_head on the final layer) —
    the one GPU-expensive quantity that stays in-scan

Logit-lens, probes, change-point alignment, per-layer outcome separation,
and return-after-elimination are all OFFLINE steps over stored residuals,
so analysis can be re-cut without re-running the GPU.

Reconstruction caveats (same as entropy_scan.py): served stream rebuilt
via solver_prompt + build_prompt_ids; rows store text not ids so
re-tokenization may diverge at a few positions — fine for diagnostics, not
for exact fork placement.

Run (two shards, one per GPU, from the ace dir):
  CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m mlfactory.experiments.ace.analysis.residual_map --shard 0:2
  CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      .venv/bin/python -m mlfactory.experiments.ace.analysis.residual_map --shard 1:2
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch

from mlfactory.experiments.ace.core.steering_controller import MODEL_PATH, build_prompt_ids
from mlfactory.experiments.ace.frontier.collect_rollouts import solver_prompt
from mlfactory.experiments.ace.core.trace_diagnostics import find_terminal_loop
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

ACE = Path(__file__).resolve().parent.parent
CANDIDATES = ACE / "data" / "acegen_probe_b1.jsonl"
ROWS_DEFAULT = [ACE / "data" / "acegen_probe_b1_gpu0.jsonl",
                ACE / "data" / "acegen_probe_b1_gpu1.jsonl"]
LOGIT_CHUNK = 256
N_LAYERS = 32
LINEAR_LAYERS = [i for i in range(N_LAYERS) if i % 4 != 3]   # 24 linear-attn
FULL_LAYERS = [i for i in range(N_LAYERS) if i % 4 == 3]     # 8 full-attn


def load_cands() -> dict[int, dict]:
    out = {}
    for line in CANDIDATES.read_text().splitlines():
        if line.strip():
            rec = json.loads(line)
            out[int(rec["provenance"]["proposal_id"])] = rec
    return out


@torch.no_grad()
def scan_row(model, tok, rec, row, stride):
    prompt = solver_prompt(rec)
    p_ids = build_prompt_ids(tok, prompt, enable_thinking=True)
    comp = row["completion"]
    c_enc = tok(comp, add_special_tokens=False, return_offsets_mapping=True)
    c_ids = list(c_enc.input_ids)[:26000]   # doctrine suspended for map forward: cap at 26k
    ids = p_ids + c_ids
    n_prompt = len(p_ids)

    x = torch.tensor([ids], dtype=torch.long, device=model.device)
    layers = model.model.layers
    captured = [None] * N_LAYERS       # subsampled (stride) per layer
    final_full = {}                    # layer 31 at full resolution for entropy

    def make_hook(i, full=False):
        def hook(_m, _inp, out):
            h = out[0] if isinstance(out, tuple) else out
            if full:
                # copy bf16 to CPU first, then cast to fp32 there — avoids a
                # Tc x 4096 x 4B fp32 transient on GPU (the OOM driver on the
                # desktop-shared GPU0).
                final_full["h"] = h.detach()[0, n_prompt:].cpu().float()
            else:
                captured[i] = h.detach()[0, n_prompt::stride].half().cpu()
        return hook

    handles = [layers[i].register_forward_hook(make_hook(i)) for i in range(N_LAYERS)]
    handles.append(layers[N_LAYERS - 1].register_forward_hook(make_hook(N_LAYERS - 1, full=True)))
    cache = DynamicCache(config=model.config)
    try:
        model.model(input_ids=x, past_key_values=cache)  # base only: no 13 GB logits tensor
    finally:
        for h in handles:
            h.remove()

    # entropy from the final layer's full-resolution residual (kept on CPU as
    # fp32); lm_head matmul runs on small GPU slices to bound transient memory.
    h_final_cpu = final_full["h"]
    Tc = h_final_cpu.shape[0]
    W = model.lm_head.weight  # (vocab, 4096) on GPU, bf16
    ent = np.full(Tc, np.nan, dtype=np.float32)
    for s in range(0, Tc - 1, LOGIT_CHUNK):
        e = min(s + LOGIT_CHUNK, Tc - 1)
        h_slice = h_final_cpu[s:e].to(model.device)        # (c, 4096) bf16
        z = h_slice @ W.float().T                            # (c, 248k) fp32 transient
        logz = torch.logsumexp(z, dim=-1)
        p = torch.softmax(z, dim=-1)
        ent[s:e] = (logz - (p * z).sum(-1)).cpu().numpy()
        del z, p, h_slice
    ent[Tc - 1] = np.nan

    # final recurrent states for linear-attention layers
    rec_states = {}
    for i in LINEAR_LAYERS:
        try:
            st = cache.layers[i].recurrent_states[0]
        except (AttributeError, IndexError, KeyError):
            st = None
        if st is not None:
            rec_states[str(i)] = st.detach().cpu().numpy().astype(np.float16)
        else:
            rec_states[str(i)] = np.zeros(0, dtype=np.float16)
    del cache
    gc.collect()

    # loop onset char -> token
    onset_tok = -1
    loop = find_terminal_loop(comp)
    if loop is not None:
        oc = loop["onset_char"]
        offs = c_enc.offset_mapping
        onset_tok = next((i for i, (a, b) in enumerate(offs) if a <= oc < b), -1)

    # stack residuals: (N_LAYERS, Tc/stride, 4096)
    resid = np.stack([captured[i].numpy() for i in range(N_LAYERS)], axis=0) \
        if all(c is not None for c in captured) else None
    return {
        "residuals": resid,
        "n_layers": N_LAYERS,
        "stride": stride,
        "entropy": ent.astype(np.float16),
        "onset_tok": onset_tok,
        **{f"rec_{i}": rec_states[str(i)] for i in LINEAR_LAYERS},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", default="0:1")
    ap.add_argument("--stride", type=int, default=16)
    ap.add_argument("--out-dir", default=str(ACE / "data" / "map_b1"))
    ap.add_argument("--rows", nargs="*", default=[str(p) for p in ROWS_DEFAULT])
    ap.add_argument("--max-tokens", type=int, default=None,
                    help="skip traces with n_new_tokens above this (GPU0 desktop limit)")
    ap.add_argument("--min-tokens", type=int, default=None,
                    help="skip traces with n_new_tokens below this")
    args = ap.parse_args()
    si, ns = map(int, args.shard.split(":"))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cands = load_cands()
    rows, seen = [], set()
    for f in args.rows:
        for line in Path(f).read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            k = (r["proposal_id"], r["sample_i"])
            if k not in seen:
                seen.add(k)
                rows.append(r)
    rows.sort(key=lambda r: (r["proposal_id"], r["sample_i"]))
    rows = [r for i, r in enumerate(rows) if i % ns == si]
    if args.max_tokens is not None:
        rows = [r for r in rows if r["n_new_tokens"] <= args.max_tokens]
    if args.min_tokens is not None:
        rows = [r for r in rows if r["n_new_tokens"] >= args.min_tokens]
    todo = [r for r in rows
            if not (out_dir / f"p{r['proposal_id']:02d}_s{r['sample_i']}.npz").exists()]
    print(f"shard {si}/{ns}: {len(rows)} rows, {len(todo)} to do", flush=True)
    if not todo:
        return

    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=torch.bfloat16, device_map="cuda",
        attn_implementation="sdpa")
    model.eval()

    for j, row in enumerate(todo):
        t0 = time.time()
        rec = cands[int(row["proposal_id"])]
        try:
            out = scan_row(model, tok, rec, row, args.stride)
        except Exception as e:  # noqa: BLE001
            print(f"FAIL p{row['proposal_id']:02d}_s{row['sample_i']}: {e}", flush=True)
            torch.cuda.empty_cache()
            continue
        meta = {k: row.get(k) for k in
                ("proposal_id", "sample_i", "domain", "correct", "truncated",
                 "n_new_tokens", "match_mode")}
        np.savez(
            out_dir / f"p{row['proposal_id']:02d}_s{row['sample_i']}.npz",
            **out, meta=json.dumps(meta))
        print(f"[{j+1}/{len(todo)}] p{row['proposal_id']:02d}_s{row['sample_i']} "
              f"tok={row['n_new_tokens']} correct={row['correct']} "
              f"onset={out['onset_tok']} {time.time()-t0:.1f}s", flush=True)
        del out
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
