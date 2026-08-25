#!/usr/bin/env python3
"""Probe 2: semantically aligned raw/rewrite waypoints for seed 5046.

The ACE rewrite of seed 5046 preserved the section skeleton byte-identically
(verified: 30/30 markdown headings match). Those headings are used as matched
reasoning waypoints between the raw trace and the rewrite.

For each text (raw `trace.content`, ACE rewrite v2) we rebuild a controlled
stream — identical chat-templated problem prompt + empty think block + text —
and run ONE teacher-forced pass under full-precision Qwen3.5-9B (bf16):

  * per-position next-token entropy over the text region (full 248,320-way
    logits, chunked lm_head);
  * final-layer hidden state at each waypoint (mean-pooled over the heading's
    token span).

Sensible-behavior tests:
  1. alignment: matched waypoints should be latent-nearest (cosine diagonal
     dominates off-diagonal; argmax over rewrite waypoints recovers identity);
  2. entropy: waypoint entropy series should co-vary between raw and rewrite
     (same semantic junctures carry the uncertainty), with aggregate levels
     comparing the two documents.

Caveats: instrument = Qwen3.5-9B; author = Qwen3.8-27B (trace) / GLM (rewrite
editor). The edited surface here is the answer document, not hidden thinking.

Run:
  CUDA_VISIBLE_DEVICES=1 /home/admin/look_and_review/mtp-lab/venv/bin/python \
      probe_5046_waypoints.py
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ACE_DIR = Path(__file__).resolve().parent
# Immutable source data lives in the archived legacy experiment.
DATA = ACE_DIR.parent.parent / "ace-legacyapproach" / "data"
RECORD_PATH = DATA / "thrash_5046_ace_rewrite_v2.json"   # provides `prose`
RAW_PATH = DATA / "thrash_5046_original_trace.txt"
REWRITE_PATH = DATA / "thrash_5046_ace_rewrite_v2.txt"
MODEL_PATH = "/home/admin/models/hf/Qwen3.5-9B"
THINK_STUB = "<think>\n</think>\n\n"   # standard no-thinking turn boundary
LOGIT_CHUNK = 256
H_WINDOW = 12                          # tokens from waypoint start

HEADING_RE = re.compile(r"(?m)^(#{1,3})[ ](.+)$")


def find_headings(text: str) -> list[dict]:
    return [
        {"level": len(m.group(1)), "title": m.group(2).strip(),
         "start": m.start(), "end": m.end()}
        for m in HEADING_RE.finditer(text)
    ]


@torch.no_grad()
def measure(model, tok, prompt_ids: list[int], text: str):
    """Teacher-force prompt+text; return per-position entropy, surprisal,
    heading token spans (absolute positions), and final-layer hiddens."""
    completion = THINK_STUB + text
    enc = tok(completion, add_special_tokens=False, return_offsets_mapping=True)
    comp_ids = list(enc.input_ids)
    offsets = enc.offset_mapping
    ids = prompt_ids + comp_ids
    cs = len(prompt_ids)
    T = len(ids)

    x = torch.tensor([ids], device="cuda")
    out = model(input_ids=x, output_hidden_states=True)
    h = out.hidden_states[-1][0]                     # [T, H] bf16
    lm = model.get_output_embeddings()

    positions = list(range(cs - 1, T - 1))           # h_t predicts token t+1
    ent, nll = [], []
    for s in range(0, len(positions), LOGIT_CHUNK):
        idx = positions[s:s + LOGIT_CHUNK]
        z = lm(h[idx]).float()
        logp = z - torch.logsumexp(z, dim=-1, keepdim=True)
        p = logp.exp()
        ent.extend((-(p * logp).sum(-1)).tolist())
        tgt = torch.tensor(ids[cs + s: cs + s + len(idx)], device=z.device)
        nll.extend((-logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)).tolist())
        del z, logp, p

    # entropy position i corresponds to absolute token position positions[i]
    pos_index = {p: i for i, p in enumerate(positions)}

    waypoints = []
    for hd in find_headings(completion):
        span_tokens = [cs + i for i, (a, b) in enumerate(offsets)
                       if b > hd["start"] and a < hd["end"]]
        if not span_tokens:
            continue
        t0 = span_tokens[0]
        window = [pos_index[p] for p in range(t0, min(t0 + H_WINDOW, T - 1))
                  if p in pos_index]
        vec = h[span_tokens].float().mean(0)
        vec = vec / vec.norm()
        waypoints.append({
            "level": hd["level"], "title": hd["title"],
            "tok_start": t0,
            "entropy": sum(ent[i] for i in window) / max(1, len(window)),
            "vec": vec,
        })
    mean_ent = sum(ent) / len(ent)
    mean_nll = sum(nll) / len(nll)
    return {"tokens": T, "text_tokens": T - cs, "entropy": ent,
            "mean_entropy": mean_ent, "mean_nll": mean_nll,
            "waypoints": waypoints, "h": h}


def main() -> None:
    record = json.loads(RECORD_PATH.read_text())
    prose = record["prose"]
    raw = RAW_PATH.read_text()
    rewrite = REWRITE_PATH.read_text()
    print(f"seed=5046 domain={record['domain']} "
          f"raw_chars={len(raw)} rewrite_chars={len(rewrite)}")

    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    enc = tok.apply_chat_template(
        [{"role": "user", "content": prose}],
        add_generation_prompt=True, tokenize=True)
    prompt_ids = list(enc.input_ids if hasattr(enc, "input_ids") else enc)
    print(f"prompt tokens: {len(prompt_ids)}")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=torch.bfloat16, device_map="cuda",
        trust_remote_code=True)
    model.eval()

    R = measure(model, tok, prompt_ids, raw)
    W = measure(model, tok, prompt_ids, rewrite)
    ln2 = math.log(2)
    print(f"\nraw:      {R['text_tokens']} text tokens, {len(R['waypoints'])} "
          f"waypoints, meanH={R['mean_entropy']/ln2:.3f} bits, "
          f"surprisal={R['mean_nll']/ln2:.3f} bits")
    print(f"rewrite:  {W['text_tokens']} text tokens, {len(W['waypoints'])} "
          f"waypoints, meanH={W['mean_entropy']/ln2:.3f} bits, "
          f"surprisal={W['mean_nll']/ln2:.3f} bits")

    # ---- alignment check: cross-text cosine matrix over waypoints ----------
    rw_vecs = torch.stack([w["vec"] for w in R["waypoints"]])
    wr_vecs = torch.stack([w["vec"] for w in W["waypoints"]])
    sim = rw_vecs @ wr_vecs.T                          # [n_raw, n_rw]
    n = min(len(R["waypoints"]), len(W["waypoints"]))
    diag = sim.diag()[:n]
    off = sim[:n, :n][~torch.eye(n, dtype=torch.bool)]
    argmax_j = sim[:n, :n].argmax(dim=1)
    recovered = int((argmax_j == torch.arange(n, device=sim.device)).sum())
    print(f"\n=== latent alignment over {n} matched waypoints ===")
    print(f"diag (matched)   cosine: mean={diag.mean():+.3f} "
          f"min={diag.min():+.3f} max={diag.max():+.3f}")
    print(f"off-diag (other) cosine: mean={off.mean():+.3f} "
          f"max={off.max():+.3f}")
    print(f"argmax recovers identity: {recovered}/{n}")

    # ---- entropy at matched waypoints --------------------------------------
    e_r = [w["entropy"] for w in R["waypoints"][:n]]
    e_w = [w["entropy"] for w in W["waypoints"][:n]]
    mr = sum(e_r) / n
    mw = sum(e_w) / n
    cov = sum((a - mr) * (b - mw) for a, b in zip(e_r, e_w)) / n
    sr = math.sqrt(sum((a - mr) ** 2 for a in e_r) / n)
    sw = math.sqrt(sum((b - mw) ** 2 for b in e_w) / n)
    pearson = cov / (sr * sw) if sr > 0 and sw > 0 else float("nan")
    print(f"\n=== waypoint entropy (12-token window, bits) ===")
    print(f"Pearson r between matched series: {pearson:+.3f}")
    print(f"{'waypoint':52s} {'raw':>6s} {'rewrite':>6s} {'cos':>7s}")
    for i in range(n):
        w = R["waypoints"][i]
        if w["level"] <= 2:      # major sections for the printed table
            print(f"{'#' * w['level'] + ' ' + w['title'][:46]:52s} "
                  f"{e_r[i]/ln2:6.2f} {e_w[i]/ln2:6.2f} {diag[i]:+7.3f}")
    print(f"{'ALL-30 mean':52s} {mr/ln2:6.2f} {mw/ln2:6.2f} "
          f"{diag.mean():+7.3f}")


if __name__ == "__main__":
    main()
