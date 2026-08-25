#!/usr/bin/env python3
"""Teacher-forced logit/hidden-state scan of collected ACE traces.

One forward pass per collected trace under the frozen base model
(= the zero-init controller's step-0 substrate). Captures, per trace:

  * per-position next-token entropy H(t) over the completion, computed via
    chunked lm_head (logits are never materialized: 20k x 248k vocab ~ 10GB)
  * residual hidden states at the controller hook layer (STEER_LAYER=15),
    subsampled with --stride, fp16
  * terminal-loop onset mapped char->token (if the trace loops)

Outputs one .npz per sample into --out-dir, plus nothing else. Analysis
(outcome separation, onset signatures) is a separate CPU-side step.

Reconstruction caveats:
  * the served stream is rebuilt via solver_prompt + build_prompt_ids
    (identical to the collector), completion appended with
    add_special_tokens=False;
  * rows store decoded text, not token ids — re-tokenization may differ
    from the sampled tokenization at a few positions. Acceptable for
    diagnostics; do NOT use these offsets for exact fork placement without
    a token-exact re-derivation.

Run (two shards, one per GPU, from the ace dir):
  CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m mlfactory.experiments.ace.analysis.entropy_scan --shard 0:2
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m mlfactory.experiments.ace.analysis.entropy_scan --shard 1:2
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from mlfactory.experiments.ace.core.steering_controller import MODEL_PATH, build_prompt_ids
from mlfactory.experiments.ace.frontier.collect_rollouts import solver_prompt
from mlfactory.experiments.ace.core.trace_diagnostics import find_terminal_loop

ACE = Path(__file__).resolve().parent.parent
CANDIDATES = ACE / "data" / "acegen_probe_b1.jsonl"
ROWS_DEFAULT = [ACE / "data" / "acegen_probe_b1_gpu0.jsonl",
                ACE / "data" / "acegen_probe_b1_gpu1.jsonl"]
LOGIT_CHUNK = 256


def load_cands() -> dict[int, dict]:
    out = {}
    for line in CANDIDATES.read_text().splitlines():
        if line.strip():
            rec = json.loads(line)
            out[int(rec["provenance"]["proposal_id"])] = rec
    return out


@torch.no_grad()
def scan_row(model, tok, rec: dict, row: dict, stride: int):
    prompt = solver_prompt(rec)
    p_ids = build_prompt_ids(tok, prompt, enable_thinking=True)
    comp = row["completion"]
    c_enc = tok(comp, add_special_tokens=False, return_offsets_mapping=True)
    c_ids = list(c_enc.input_ids)
    ids = p_ids + c_ids
    n_prompt = len(p_ids)

    x = torch.tensor([ids], dtype=torch.long, device=model.device)
    captured = {}
    layer = model.model.layers[15]  # STEER_LAYER

    def hook(_mod, _inp, out):
        h = out[0] if isinstance(out, tuple) else out
        captured["h"] = h.detach()

    handle = layer.register_forward_hook(hook)
    try:
        hidden = model.model(input_ids=x).last_hidden_state  # (1, T, 4096)
    finally:
        handle.remove()

    h_hook = captured["h"][0, n_prompt:]          # (Tc, 4096) completion-only
    h_last = hidden[0, n_prompt:]                 # (Tc, 4096)

    # chunked entropy over completion positions: position t predicts token t+1
    Tc = h_last.shape[0]
    ent = np.empty(Tc, dtype=np.float32)
    W = model.lm_head.weight
    for s in range(0, Tc - 1, LOGIT_CHUNK):
        e = min(s + LOGIT_CHUNK, Tc - 1)
        z = h_last[s:e].float() @ W.float().T     # (c, 248k)
        logz = torch.logsumexp(z, dim=-1)
        p = torch.softmax(z, dim=-1)
        ent[s:e] = (logz - (p * z).sum(-1)).cpu().numpy()
        del z, p
    ent[Tc - 1] = np.nan  # no next token

    # loop onset: char offset -> token offset via completion offset mapping
    onset_tok = -1
    loop = find_terminal_loop(comp)
    if loop is not None:
        oc = loop["onset_char"]
        offs = c_enc.offset_mapping
        onset_tok = next((i for i, (a, b) in enumerate(offs) if a <= oc < b), -1)

    return {
        "entropy": ent.astype(np.float16),
        "hidden15": h_hook[::stride].half().cpu().numpy(),
        "stride": stride,
        "onset_tok": onset_tok,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", default="0:1", help="i:n — process rows[i::n]")
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--out-dir", default=str(ACE / "data" / "scan_b1"))
    ap.add_argument("--rows", nargs="*", default=[str(p) for p in ROWS_DEFAULT])
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
    todo = [r for r in rows
            if not (out_dir / f"p{r['proposal_id']:02d}_s{r['sample_i']}.npz").exists()]
    print(f"shard {si}/{ns}: {len(rows)} rows, {len(todo)} to do", flush=True)
    if not todo:
        return

    from transformers import AutoModelForCausalLM, AutoTokenizer
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
        except Exception as e:  # noqa: BLE001 — log and continue the cohort
            print(f"FAIL p{row['proposal_id']:02d}_s{row['sample_i']}: {e}", flush=True)
            continue
        meta = {k: row.get(k) for k in
                ("proposal_id", "sample_i", "domain", "correct", "truncated",
                 "n_new_tokens", "match_mode")}
        np.savez_compressed(
            out_dir / f"p{row['proposal_id']:02d}_s{row['sample_i']}.npz",
            **out, meta=json.dumps(meta))
        print(f"[{j+1}/{len(todo)}] p{row['proposal_id']:02d}_s{row['sample_i']} "
              f"tok={row['n_new_tokens']} correct={row['correct']} "
              f"onset={out['onset_tok']} {time.time()-t0:.1f}s", flush=True)
        del out
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
