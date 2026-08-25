#!/usr/bin/env python3
"""Recon probe: teacher-forced replay of one ACE trace under a full-precision model.

Validates that search-dynamics instrumentation is feasible with the existing
local stack (no architecture changes):

  * loads the local HF Qwen3.5-9B (full precision bf16) on a free GPU;
  * rebuilds the served token stream (chat template + <think> wrapper), the
    same pattern as mtp-lab/extract.py::render_ids;
  * runs ONE forward pass over the prompt + a reasoning prefix;
  * extracts (a) per-position next-token entropy from full-vocab logits and
    (b) final-layer hidden-state snapshots at several trace positions.

Measurement model caveat: the trace was authored by Qwen3.8-27B (see
trace_source in the record); entropy here is "surprisal under Qwen3.5-9B",
which is the instrument we would standardize on, not the author.

Run:
  CUDA_VISIBLE_DEVICES=1 /home/admin/look_and_review/mtp-lab/venv/bin/python \
      probe_teacher_forced.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ACE_DIR = Path(__file__).resolve().parent
# Immutable source data lives in the archived legacy experiment.
RECORD_PATH = (ACE_DIR.parent.parent / "ace-legacyapproach" / "data"
             / "thrash_5003_ace_rewrite.json")
MODEL_PATH = "/home/admin/models/hf/Qwen3.5-9B"
REASONING_PREFIX_TOKENS = 2048  # cap for a fast probe; full trace is ~30k tok
LOGIT_CHUNK = 256               # positions per lm_head chunk (vocab = 248,320)


def build_ids(tok, prose: str, reasoning_prefix: str):
    """Same served-stream reconstruction as mtp-lab/extract.py::render_ids."""
    msgs = [{"role": "user", "content": prose}]
    enc = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True)
    prompt_ids = list(enc.input_ids if hasattr(enc, "input_ids") else enc)
    completion = "<think>\n" + reasoning_prefix
    comp_ids = tok(completion, add_special_tokens=False).input_ids
    return prompt_ids + comp_ids, len(prompt_ids)


@torch.no_grad()
def main() -> None:
    record = json.loads(RECORD_PATH.read_text())
    prose = record["prose"]
    reasoning = record["trace"]["reasoning"]
    print(f"record: seed={record['seed']} domain={record['domain']} "
          f"trace_source={record.get('trace_source')} "
          f"reasoning_chars={len(reasoning)}")

    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    # Truncate the reasoning to a token budget *before* building the stream.
    reasoning_ids = tok(reasoning, add_special_tokens=False).input_ids
    prefix_ids = reasoning_ids[:REASONING_PREFIX_TOKENS]
    reasoning_prefix = tok.decode(prefix_ids)
    ids, cs = build_ids(tok, prose, reasoning_prefix)
    T = len(ids)
    print(f"stream: {T} tokens total, prompt={cs}, reasoning_prefix={T - cs}")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=torch.bfloat16, device_map="cuda",
        trust_remote_code=True,
    )
    model.eval()
    print(f"model loaded: {MODEL_PATH} "
          f"({sum(p.numel() for p in model.parameters()) / 1e9:.2f}B params)")

    x = torch.tensor([ids], device="cuda")
    out = model(input_ids=x, output_hidden_states=True)
    hs = out.hidden_states  # tuple len n_layers+1, each [1, T, H]
    print(f"hidden_states: {len(hs)} layers x shape {tuple(hs[-1].shape)} "
          f"dtype={hs[-1].dtype}")

    # --- measurement 1: next-token entropy over reasoning positions --------
    # position t holds h_t predicting token t+1; valid t in [cs-1, T-2]
    h = hs[-1][0]  # [T, H]
    lm = model.get_output_embeddings()
    positions = list(range(cs - 1, T - 1))
    entropies: list[float] = []
    top1: list[int] = []
    for s in range(0, len(positions), LOGIT_CHUNK):
        chunk = h[positions[s]:positions[s] + LOGIT_CHUNK]
        z = lm(chunk).float()                     # [n, vocab]
        logp = z - torch.logsumexp(z, dim=-1, keepdim=True)
        p = logp.exp()
        ent = -(p * logp).sum(-1)                 # nats
        entropies.extend(ent.tolist())
        top1.extend(z.argmax(-1).tolist())
        del z, logp, p

    actual_ids = ids[cs:T]  # token at position t+1 for each measured t
    mean_ent = sum(entropies) / len(entropies)
    print(f"\n=== token entropy over {len(entropies)} reasoning positions ===")
    print(f"mean H = {mean_ent:.3f} nats ({mean_ent / math.log(2):.3f} bits)")

    # decile profile: does uncertainty change along the trajectory?
    k = len(entropies) // 10
    profile = [sum(entropies[i * k:(i + 1) * k]) / k for i in range(10)]
    print("decile profile (bits):",
          " ".join(f"{v / math.log(2):.2f}" for v in profile))

    # highest-entropy positions: do they land on branch/revision markers?
    order = sorted(range(len(entropies)), key=lambda i: -entropies[i])
    print("\ntop-8 highest-entropy positions:")
    for i in order[:8]:
        tok_text = tok.decode([actual_ids[i]])
        ctx = tok.decode(actual_ids[max(0, i - 6):i + 1]).replace("\n", " ")
        print(f"  pos {i:5d} H={entropies[i] / math.log(2):5.2f} bits "
              f"next_tok={tok_text!r:>22}  ctx=...{ctx!r}")

    # surprisal of the actually-emitted token (NLL of the realized trajectory)
    import torch as _t
    # recompute quickly in chunks to get logp of realized token
    nlls: list[float] = []
    for s in range(0, len(positions), LOGIT_CHUNK):
        idx = positions[s:s + LOGIT_CHUNK]
        z = lm(h[idx]).float()
        logp = z - _t.logsumexp(z, dim=-1, keepdim=True)
        tgt = _t.tensor(actual_ids[s:s + len(idx)], device=z.device)
        nlls.extend((-logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)).tolist())
        del z, logp
    mean_nll = sum(nlls) / len(nlls)
    print(f"\nmean realized-token surprisal = {mean_nll / math.log(2):.3f} bits "
          f"(vs mean entropy {mean_ent / math.log(2):.3f} bits)")

    # --- measurement 2: latent trajectory snapshots -------------------------
    snaps = [cs - 1 + int((len(positions)) * f) for f in (0.0, 0.25, 0.5, 0.75, 0.99)]
    vecs = torch.stack([hs[-1][0, p] for p in snaps]).float()
    norm = vecs / vecs.norm(dim=-1, keepdim=True)
    sim = norm @ norm.T
    print("\n=== final-layer hidden snapshots (feasibility for dispersion/"
          "recurrence/tortuosity) ===")
    print("positions:", snaps)
    print("pairwise cosine similarity:")
    for r in range(len(snaps)):
        print("  " + " ".join(f"{sim[r, c]:+.3f}" for c in range(len(snaps))))


if __name__ == "__main__":
    main()
