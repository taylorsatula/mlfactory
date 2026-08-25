#!/usr/bin/env python3
"""Smallest GRPO-style training run for the residual steering controller.

Question (only): can a frozen-base, prefix-causal controller learn a
nontrivial state-dependent intervention from *terminal objective
correctness* alone?

Reward = final-answer correctness (0/1). No entropy / recurrence /
tortuosity / length / "explore" / "prune" terms. KL vs the frozen Qwen
policy (k3 estimator) + a small relative-intervention regularizer.

Matched unsteered Qwen rollouts use the same prompts, sampling settings,
and per-prompt seeds so sampling variance is visible. The ACE hypothesis
is unproven; this run also checks mundane alternatives (constant
intervention, gate collapse, prompt memorization, verbosity, luck).

Run:
  CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m mlfactory.experiments.ace.train.grpo
  CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m mlfactory.experiments.ace.train.grpo --smoke
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mlfactory.experiments.ace.core.problems import HOLDOUT, TRAIN, extract_answer, verify
from mlfactory.experiments.ace.core.steering_controller import (
    ALPHA, MODEL_PATH, STEER_LAYER, ResidualSteering, SteeringController,
    build_prompt_ids, freeze_base_model, generate_batch,
)

ACE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ACE_DIR / "data" / "controller_train"


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson interval for a binomial proportion."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, center - half), min(1.0, center + half))


def group_advantages(rewards: list[float], eps: float = 1e-4) -> list[float]:
    """GRPO-style group-relative advantages. All-equal -> all zero."""
    r = torch.tensor(rewards, dtype=torch.float32)
    return ((r - r.mean()) / (r.std(unbiased=False) + eps)).tolist()


def k3_kl(logp_pol: torch.Tensor, logp_ref: torch.Tensor) -> torch.Tensor:
    """Non-negative unbiased KL(pol || ref) estimator, x ~ pol."""
    logratio = logp_ref - logp_pol
    return logratio.exp() - logratio - 1


def completion_logprobs(logits: torch.Tensor, ids: list[int],
                        n_prompt: int, chunk: int = 64) -> torch.Tensor:
    """Per-completion-token log p(token_t | prefix_<t) from full logits."""
    T = len(ids)
    pos = list(range(n_prompt - 1, T - 1))
    tgt = torch.tensor(ids[n_prompt:], device=logits.device)
    parts = []
    for s in range(0, len(pos), chunk):
        idx = pos[s:s + chunk]
        z = logits[0, idx].float()
        logp = z - torch.logsumexp(z, dim=-1, keepdim=True)
        parts.append(logp.gather(-1, tgt[s:s + len(idx)].unsqueeze(-1))
                     .squeeze(-1))
    return torch.cat(parts) if parts else logits.new_zeros((0,))


def record_stats(records: list[dict], n: int) -> list[dict]:
    """Fold batched hook records into per-rollout gate / rel-norm / mean-Δh."""
    per = [{"gates": [], "rel": [], "dsum": None, "dn": 0} for _ in range(n)]
    for rec in records:
        g = rec["gate"].reshape(n, -1)          # [n, T_call]
        dn = rec["delta_norm"].reshape(n, -1)
        hn = rec["h_norm"].reshape(n, -1)
        delta = rec["delta"]                    # [n, T_call, H]
        for i in range(n):
            per[i]["gates"].extend(g[i].tolist())
            rel = (dn[i] / hn[i].clamp_min(1e-12)).tolist()
            per[i]["rel"].extend(rel)
            dmean = delta[i].float().mean(0)
            if per[i]["dsum"] is None:
                per[i]["dsum"] = dmean
            else:
                per[i]["dsum"] = per[i]["dsum"] + dmean
            per[i]["dn"] += 1
    out = []
    for p in per:
        g = torch.tensor(p["gates"], dtype=torch.float32)
        r = torch.tensor(p["rel"], dtype=torch.float32)
        mean_d = (p["dsum"] / max(p["dn"], 1)) if p["dsum"] is not None else None
        out.append({
            "gate_mean": float(g.mean()) if g.numel() else 0.0,
            "gate_std": float(g.std(unbiased=False)) if g.numel() else 0.0,
            "gate_sat": float(((g <= 0.02) | (g >= 0.98)).float().mean())
            if g.numel() else 0.0,
            "rel_mean": float(r.mean()) if r.numel() else 0.0,
            "rel_max": float(r.max()) if r.numel() else 0.0,
            "mean_delta": mean_d,
        })
    return out


def decode_completion(tok, ids: list[int], n_prompt: int) -> str:
    return tok.decode(ids[n_prompt:], skip_special_tokens=True)


def roll_group(model, tok, prompt: str, n: int, max_new: int, seed: int,
               controller=None, record=False, temperature=0.9, top_p=0.95):
    return generate_batch(
        model, tok, prompt, n=n, max_new_tokens=max_new,
        controller=controller, record=record, do_sample=True,
        temperature=temperature, top_p=top_p, seed=seed,
        enable_thinking=False)


def replay_logprobs(model, ids: list[int], n_prompt: int, controller=None,
                    collect=False, token_cap: int | None = 192):
    """Teacher-forced pass. With controller+collect, returns graph tensors.

    ``token_cap`` limits completion tokens in the replay graph. T=331
    replay used 19.9 GiB; T≈580 OOMs a 24 GB card (cublas handle).
    """
    if token_cap is not None and len(ids) > n_prompt + token_cap:
        ids = ids[: n_prompt + token_cap]
    x = torch.tensor([ids], device=model.device)
    ctx = (ResidualSteering(model, controller, collect=collect)
           if controller is not None else nullcontext())
    enable_grad = controller is not None
    with ctx as active, torch.set_grad_enabled(enable_grad):
        logits = model(input_ids=x).logits
        logp = completion_logprobs(logits, ids, n_prompt)
        rel = None
        if controller is not None and collect and active.collected:
            rel = torch.stack(active.collected).mean()
    del logits
    return logp, rel


def evaluate(model, tok, items, controller, group_size, max_new, seed_base,
             tag: str) -> dict:
    """Matched steered + base groups on a split. Returns summary + per-item."""
    per = []
    s_ok = b_ok = n_tot = 0
    s_len = b_len = 0
    traces = []
    for i, it in enumerate(items):
        n_prompt = len(build_prompt_ids(tok, it["prompt"],
                                        enable_thinking=False))
        seed = seed_base + i * 17
        s_seqs, s_rec = roll_group(model, tok, it["prompt"], group_size,
                                   max_new, seed, controller=controller,
                                   record=controller is not None)
        b_seqs, _ = roll_group(model, tok, it["prompt"], group_size,
                               max_new, seed, controller=None)
        s_stats = record_stats(s_rec, group_size) if s_rec else [{}] * group_size
        s_rews, b_rews = [], []
        for j, (ss, bs) in enumerate(zip(s_seqs, b_seqs)):
            st = decode_completion(tok, ss, n_prompt)
            bt = decode_completion(tok, bs, n_prompt)
            sr, br = verify(st, it["gold"]), verify(bt, it["gold"])
            s_rews.append(float(sr))
            b_rews.append(float(br))
            s_ok += int(sr)
            b_ok += int(br)
            n_tot += 1
            s_len += len(ss) - n_prompt
            b_len += len(bs) - n_prompt
            if j == 0:
                traces.append({
                    "id": it["id"], "gold": it["gold"],
                    "steered_ok": sr, "base_ok": br,
                    "steered_pred": extract_answer(st),
                    "base_pred": extract_answer(bt),
                    "steered_len": len(ss) - n_prompt,
                    "base_len": len(bs) - n_prompt,
                    "steered_tail": st[-400:],
                    "base_tail": bt[-400:],
                })
        gmean = (sum(s["gate_mean"] for s in s_stats) / len(s_stats)
                 if s_stats and "gate_mean" in s_stats[0] else 0.0)
        gstd = (sum(s["gate_std"] for s in s_stats) / len(s_stats)
                if s_stats and "gate_std" in s_stats[0] else 0.0)
        rmean = (sum(s["rel_mean"] for s in s_stats) / len(s_stats)
                 if s_stats and "rel_mean" in s_stats[0] else 0.0)
        per.append({
            "id": it["id"], "family": it["family"],
            "steered_mean": sum(s_rews) / len(s_rews),
            "base_mean": sum(b_rews) / len(b_rews),
            "steered_rews": s_rews, "base_rews": b_rews,
            "gate_mean": gmean, "gate_std": gstd, "rel_mean": rmean,
        })
        print(f"  [{tag}] {it['id']:32s}  "
              f"steer={sum(s_rews):.0f}/{len(s_rews)}  "
              f"base={sum(b_rews):.0f}/{len(b_rews)}  "
              f"gμ={gmean:.3f} gσ={gstd:.3f} rel={rmean:.4f}")
    s_lo, s_hi = wilson_ci(s_ok, n_tot)
    b_lo, b_hi = wilson_ci(b_ok, n_tot)
    return {
        "tag": tag, "n": n_tot,
        "steered_acc": s_ok / n_tot, "base_acc": b_ok / n_tot,
        "steered_ci": [s_lo, s_hi], "base_ci": [b_lo, b_hi],
        "steered_mean_len": s_len / n_tot, "base_mean_len": b_len / n_tot,
        "per_item": per, "traces": traces,
    }


def train_iteration(model, tok, items, controller, opt, cfg, iter_i: int):
    """One GRPO step over ``items`` (each with a steered + matched base group)."""
    t0 = time.time()
    opt.zero_grad(set_to_none=True)
    recs = []
    loss_pg = loss_kl = loss_mag = 0.0
    n_used = 0
    grad_scale = 1.0 / max(len(items), 1)

    for pi, it in enumerate(items):
        n_prompt = len(build_prompt_ids(tok, it["prompt"],
                                        enable_thinking=False))
        seed = cfg.seed + iter_i * 1009 + pi * 17
        s_seqs, s_rec = roll_group(
            model, tok, it["prompt"], cfg.group_size, cfg.max_new, seed,
            controller=controller, record=True,
            temperature=cfg.temperature, top_p=cfg.top_p)
        b_seqs, _ = roll_group(
            model, tok, it["prompt"], cfg.group_size, cfg.max_new, seed,
            controller=None,
            temperature=cfg.temperature, top_p=cfg.top_p)
        s_stats = record_stats(s_rec, cfg.group_size)
        s_rews, b_rews, s_lens, b_lens = [], [], [], []
        for ss, bs in zip(s_seqs, b_seqs):
            st = decode_completion(tok, ss, n_prompt)
            bt = decode_completion(tok, bs, n_prompt)
            s_rews.append(float(verify(st, it["gold"])))
            b_rews.append(float(verify(bt, it["gold"])))
            s_lens.append(len(ss) - n_prompt)
            b_lens.append(len(bs) - n_prompt)
        adv = group_advantages(s_rews)
        # Replay each steered rollout (B=1) for on-policy logprobs + mag.
        for j, (ss, A) in enumerate(zip(s_seqs, adv)):
            torch.cuda.empty_cache()
            logp_ref, _ = replay_logprobs(model, ss, n_prompt, controller=None)
            logp_pol, rel = replay_logprobs(model, ss, n_prompt,
                                            controller=controller, collect=True)
            if logp_pol.numel() == 0:
                continue
            logp_ref = logp_ref.detach()
            kl = k3_kl(logp_pol, logp_ref)
            pg = -(A * logp_pol)
            mag = rel if rel is not None else logp_pol.new_zeros(())
            loss_j = ((pg + cfg.beta_kl * kl).mean()
                      + cfg.lambda_mag * mag) * grad_scale
            loss_j.backward()
            loss_pg += float(pg.mean().detach()) * grad_scale
            loss_kl += float(kl.mean().detach()) * grad_scale
            loss_mag += float(mag.detach()) * grad_scale
            n_used += 1
        recs.append({
            "id": it["id"], "family": it["family"],
            "steered_mean": sum(s_rews) / len(s_rews),
            "base_mean": sum(b_rews) / len(b_rews),
            "steered_rews": s_rews, "base_rews": b_rews,
            "advantages": adv,
            "steered_mean_len": sum(s_lens) / len(s_lens),
            "base_mean_len": sum(b_lens) / len(b_lens),
            "gate_mean": sum(s["gate_mean"] for s in s_stats) / len(s_stats),
            "gate_std": sum(s["gate_std"] for s in s_stats) / len(s_stats),
            "gate_sat": sum(s["gate_sat"] for s in s_stats) / len(s_stats),
            "rel_mean": sum(s["rel_mean"] for s in s_stats) / len(s_stats),
            "rel_max": max(s["rel_max"] for s in s_stats),
        })

    gn = torch.nn.utils.clip_grad_norm_(controller.parameters(), 1.0)
    opt.step()
    dt = time.time() - t0
    s_acc = sum(r["steered_mean"] for r in recs) / len(recs)
    b_acc = sum(r["base_mean"] for r in recs) / len(recs)
    print(f"iter {iter_i:02d}  steer={s_acc:.2f}  base={b_acc:.2f}  "
          f"pg={loss_pg:.3f} kl={loss_kl:.4f} mag={loss_mag:.4f}  "
          f"|g|={float(gn):.3f}  {dt:.0f}s")
    return {
        "iter": iter_i, "dt_s": dt,
        "steered_acc": s_acc, "base_acc": b_acc,
        "loss_pg": loss_pg, "loss_kl": loss_kl, "loss_mag": loss_mag,
        "grad_norm": float(gn), "n_replayed": n_used,
        "prompts": recs,
    }


def summarize(log_rows: list[dict], evals: list[dict],
              fingerprints_ok: bool) -> dict:
    """Degeneracy / learnability diagnostics from the run log."""
    last = log_rows[-1] if log_rows else {}
    first = log_rows[0] if log_rows else {}
    gate_stds = [p["gate_std"] for r in log_rows for p in r.get("prompts", [])]
    gate_means = [p["gate_mean"] for r in log_rows for p in r.get("prompts", [])]
    rels = [p["rel_mean"] for r in log_rows for p in r.get("prompts", [])]
    train_final = next((e for e in reversed(evals) if e["tag"].startswith("train")),
                       None)
    hold_final = next((e for e in reversed(evals) if e["tag"].startswith("hold")),
                      None)
    hold_init = next((e for e in evals if e["tag"].startswith("hold")), None)
    return {
        "n_iters": len(log_rows),
        "base_fingerprints_unchanged": fingerprints_ok,
        "train_batch_acc_first": first.get("steered_acc"),
        "train_batch_acc_last": last.get("steered_acc"),
        "train_batch_base_acc_last": last.get("base_acc"),
        "mean_within_rollout_gate_std":
            (sum(gate_stds) / len(gate_stds)) if gate_stds else 0.0,
        "std_across_prompt_gate_mean":
            (float(torch.tensor(gate_means).std(unbiased=False))
             if len(gate_means) > 1 else 0.0),
        "mean_rel_norm": (sum(rels) / len(rels)) if rels else 0.0,
        "max_rel_norm_seen": max((p["rel_max"] for r in log_rows
                                  for p in r.get("prompts", [])), default=0.0),
        "alpha": ALPHA,
        "final_kl": last.get("loss_kl"),
        "final_grad_norm": last.get("grad_norm"),
        "holdout_init": ({k: hold_init[k] for k in
                          ("steered_acc", "base_acc", "steered_ci", "base_ci",
                           "steered_mean_len", "base_mean_len")}
                         if hold_init else None),
        "holdout_final": ({k: hold_final[k] for k in
                           ("steered_acc", "base_acc", "steered_ci", "base_ci",
                            "steered_mean_len", "base_mean_len")}
                          if hold_final else None),
        "train_eval_final": ({k: train_final[k] for k in
                              ("steered_acc", "base_acc", "steered_ci",
                               "base_ci", "steered_mean_len", "base_mean_len")}
                             if train_final else None),
    }


def snapshot_base(model):
    named = dict(model.named_parameters())
    keys = [k for k in ("model.embed_tokens.weight",
                        "model.layers.15.mlp.down_proj.weight",
                        "lm_head.weight") if k in named]
    return {k: named[k].detach().view(-1)[:128].clone() for k in keys}


def check_base(model, snap) -> bool:
    named = dict(model.named_parameters())
    return all(torch.equal(named[k].detach().view(-1)[:128], v)
               for k, v in snap.items())


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--iters", type=int, default=16)
    p.add_argument("--prompts-per-iter", type=int, default=4)
    p.add_argument("--group-size", type=int, default=6)
    p.add_argument("--max-new", type=int, default=640)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--beta-kl", type=float, default=0.02)
    p.add_argument("--lambda-mag", type=float, default=0.1)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-every", type=int, default=8)
    p.add_argument("--eval-group", type=int, default=4)
    p.add_argument("--out", type=str, default=str(DATA_DIR))
    p.add_argument("--smoke", action="store_true",
                   help="Tiny 1-iter / 2-prompt / G=2 / 96-token run.")
    return p.parse_args()


def main():
    cfg = parse_args()
    if cfg.smoke:
        cfg.iters = 1
        cfg.prompts_per_iter = 2
        cfg.group_size = 2
        cfg.max_new = 96
        cfg.eval_every = 1
        cfg.eval_group = 2
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    freeze_base_model(model)
    snap = snapshot_base(model)

    # fp32 controller: optimizer precision; hook casts delta back to bf16.
    ctrl = SteeringController().to(device="cuda", dtype=torch.float32)
    n_params = sum(p.numel() for p in ctrl.parameters())
    opt = torch.optim.AdamW(ctrl.parameters(), lr=cfg.lr,
                            betas=(0.9, 0.95), weight_decay=0.0)
    print(f"controller params={n_params:,}  frozen base  "
          f"layer={STEER_LAYER}  iters={cfg.iters}  G={cfg.group_size}  "
          f"max_new={cfg.max_new}  thinking=off")

    rng = random.Random(cfg.seed)
    train_pool = list(TRAIN)
    log_rows, evals = [], []

    def run_eval(tag):
        print(f"\n=== eval {tag} ===")
        # holdout always; train eval on a fixed 8-prompt slice
        hold = evaluate(model, tok, HOLDOUT, ctrl, cfg.eval_group,
                        cfg.max_new, seed_base=90_000, tag=f"hold-{tag}")
        tr_slice = TRAIN[::3][:8]
        trn = evaluate(model, tok, tr_slice, ctrl, cfg.eval_group,
                       cfg.max_new, seed_base=80_000, tag=f"train-{tag}")
        evals.append(hold)
        evals.append(trn)
        (out / f"eval_{tag}.json").write_text(
            json.dumps({"holdout": hold, "train": trn}, indent=2, default=str))
        return hold, trn

    run_eval("init")

    for it in range(cfg.iters):
        rng.shuffle(train_pool)
        batch = train_pool[:cfg.prompts_per_iter]
        row = train_iteration(model, tok, batch, ctrl, opt, cfg, it)
        log_rows.append(row)
        with (out / "train.jsonl").open("a") as fh:
            fh.write(json.dumps(row, default=str) + "\n")
        if (it + 1) % cfg.eval_every == 0 or it + 1 == cfg.iters:
            run_eval(f"iter{it + 1:02d}")
            ctrl.save(out / f"ckpt_iter{it + 1:02d}",
                      extra_meta={"iter": it + 1, "layer_idx": STEER_LAYER})

    fingerprints_ok = check_base(model, snap)
    summary = summarize(log_rows, evals, fingerprints_ok)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n=== summary ===")
    print(json.dumps(summary, indent=2))
    print(f"base fingerprints unchanged: {fingerprints_ok}")
    print(f"artifacts: {out}")


if __name__ == "__main__":
    main()
