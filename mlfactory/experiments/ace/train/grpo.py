#!/usr/bin/env python3
"""GRPO training for the residual steering controller on the b2 pool.

Question (only): can a frozen-base, prefix-causal controller learn a
nontrivial state-dependent intervention from *terminal verified outcome*
alone? Reward = strict per-family gen check (REWARD_POLICY: no entropy /
length / recurrence / judge / PRM terms — and truncation is reported as a
backdoor length term, per group, every batch).

Substrate rulings (2026-08-25): thinking ON, bf16 HF, max_new 26000
backstop, b2 pool with CHECK-based verify, replay windows <= 8k tokens
(measured ceiling) OR gradient-checkpointed full-trace replay.

Two replay modes (``--replay-mode auto`` = full):

  full    One teacher-forced pass over the whole trace with per-layer
          torch.utils.checkpoint (non-reentrant). The steering hook sits
          at module level, so it fires exactly once; recomputation re-runs
          only the wrapped layer forward. Measured bit-exact vs the
          no-grad single pass (equivalence check, 2026-08-25).
  window  Tiled replay via cached detached prefixes. KNOWN-BROKEN:
          cache continuation corrupts the first ~50 tokens after each
          split boundary by up to 11.8 nats (diag_window_drift,
          2026-08-25). Investigation only; no silent OOM fallback to it —
          a full-mode OOM is guard trip exit 8.

Per-sample rollout rows (prompt id, sample_i, seed, length, truncated,
eos, reward, gate stats) are append-safe resume-keyed JSONL: the output
file IS the resume state. Resume semantics (cross-process sampling is
NOT bit-stable on this stack — probe_determinism 2026-08-25: same seed,
fresh processes, first token flip at 354–736 — so nothing is ever
regenerated over existing evidence): completed groups are reused
verbatim; PARTIAL groups are frozen from disk (rewards from rows,
replay from the persisted sequences of the samples that made it);
only groups with zero rows are generated fresh.

Seed spaces (fresh, disjoint): train batches cfg.seed (default 80_000)
+ iter*1009 + 17*pid; eval holdout 90_000; eval train-slice 95_000;
revalidation 100_000 + 17*pid. The smoke consumed the 70_000 space.

Run (remote H200):
  HF_HOME=/workspace/models PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  ACE_MODEL_PATH=Qwen/Qwen3.5-9B CUDA_VISIBLE_DEVICES=0 \
  /venv/main/bin/python -m mlfactory.experiments.ace.train.grpo \
      --pool mlfactory/experiments/ace/data/acegen_live_b2.jsonl \
      --out /workspace/grpo1
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from pathlib import Path

import torch
from torch.utils.checkpoint import checkpoint
from transformers import AutoModelForCausalLM, AutoTokenizer

from mlfactory.experiments.ace.core.steering_controller import (
    ALPHA, MODEL_PATH, STOP_TOKEN_IDS, STEER_LAYER, ResidualSteering,
    SteeringController, build_prompt_ids, freeze_base_model, generate_batch,
)
from mlfactory.experiments.ace.train import pool_adapter

ACE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ACE_DIR / "data" / "controller_train"
STOP_SET = set(STOP_TOKEN_IDS)


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


class _TokenLogprobs(torch.autograd.Function):
    """Per-token log p(target | row) from a logits matrix.

    Saves only the bf16 logits reference + targets; softmax is recomputed
    in backward. The naive chunked float32 upcast retains ~2x63 MB of
    vocab-wide intermediates per 64-token chunk — ~52 GB of autograd
    graph on a 26k trace, which OOMs a 140 GB H200 (measured, Step 1).
    """

    @staticmethod
    def forward(ctx, logits, targets, chunk: int = 256):
        Tp = logits.shape[0]
        out = logits.new_empty((Tp,), dtype=torch.float32)
        for s in range(0, Tp, chunk):
            e = min(s + chunk, Tp)
            z = logits[s:e].float()
            out[s:e] = (z.gather(-1, targets[s:e].unsqueeze(-1)).squeeze(-1)
                        - torch.logsumexp(z, dim=-1))
        ctx.save_for_backward(logits, targets)
        ctx.chunk = chunk
        return out

    @staticmethod
    def backward(ctx, grad_out):
        logits, targets = ctx.saved_tensors
        Tp = logits.shape[0]
        grad_logits = torch.empty_like(logits)
        for s in range(0, Tp, ctx.chunk):
            e = min(s + ctx.chunk, Tp)
            p = logits[s:e].float().softmax(-1)
            g = grad_out[s:e].float().unsqueeze(-1)
            gl = (-p) * g                      # d/dz = g*(onehot - softmax)
            gl.scatter_add_(-1, targets[s:e].unsqueeze(-1), g)
            grad_logits[s:e] = gl.to(logits.dtype)
        return grad_logits, None, None


def completion_logprobs(logits: torch.Tensor, ids: list[int],
                        n_prompt: int, chunk: int = 256) -> torch.Tensor:
    """Per-token log p(ids[t] | prefix_<t) for t >= n_prompt, from the
    rows of ``logits`` (shape [1, T, V]). Graph-cheap: see _TokenLogprobs."""
    rows = logits[0, n_prompt - 1:len(ids) - 1]
    tgt = torch.tensor(ids[n_prompt:], device=logits.device)
    if rows.shape[0] == 0:
        return logits.new_zeros((0,))
    return _TokenLogprobs.apply(rows, tgt, chunk)


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


# ---------------------------------------------------------------------------
# Replay primitives
# ---------------------------------------------------------------------------

def replay_logprobs(model, ids: list[int], n_prompt: int, controller=None,
                    collect=False, token_cap: int | None = 192):
    """Single-pass teacher-forced replay (the smoke's measurement primitive).

    No checkpointing; ``token_cap`` limits completion tokens in the graph.
    Kept for ``smoke_h200.py`` compatibility and equivalence checks — the
    production paths are ``replay_full`` / ``iter_replay_windows``.
    """
    if token_cap is not None and len(ids) > n_prompt + token_cap:
        ids = ids[: n_prompt + token_cap]
    x = torch.tensor([ids], device=model.device)
    ctx = (ResidualSteering(model, controller, collect=collect)
           if controller is not None else nullcontext())
    enable_grad = controller is not None
    with ctx as active, torch.set_grad_enabled(enable_grad):
        logits = model(input_ids=x, use_cache=False).logits
        logp = completion_logprobs(logits, ids, n_prompt)
        rel = None
        if controller is not None and collect and active.collected:
            rel = torch.stack(active.collected).mean()
    del logits
    return logp, rel


def ref_logprobs(model, ids: list[int], n_prompt: int) -> torch.Tensor:
    """No-controller, no-grad single pass: the KL reference logprobs.

    Also the ground truth both replay modes must match in equivalence
    checks (a zero-init controller is an exact no-op).
    """
    x = torch.tensor([ids], device=model.device)
    with torch.no_grad():
        logits = model(input_ids=x, use_cache=False).logits
    logp = completion_logprobs(logits, ids, n_prompt).detach()
    del logits
    return logp


@contextmanager
def checkpointed_layers(model):
    """Wrap every decoder layer's forward in non-reentrant checkpointing.

    HF's built-in gradient_checkpointing only activates while
    ``model.training``; the base model stays ``eval()`` (frozen, and no
    dropout stochasticity in the replay), so layer forwards are wrapped
    directly. The steering hook is registered at module level: it fires
    exactly once per layer, and checkpoint recomputation re-runs only the
    wrapped forward — no double-counted ``collect`` entries, no re-hooked
    deltas. Saved activations shrink to one hidden state per layer
    boundary (~7 GB at T=26k), recomputation peak is one layer.
    """
    layers = model.model.layers
    orig = [l.forward for l in layers]
    for l in layers:
        of = l.forward
        l.forward = (lambda *a, _f=of, **k:
                     checkpoint(_f, *a, use_reentrant=False, **k))
    try:
        yield
    finally:
        for l, f in zip(layers, orig):
            l.forward = f


def replay_full(model, ids: list[int], n_prompt: int, controller,
                collect: bool = True):
    """Gradient-checkpointed full-trace replay. Returns (logp, rel)."""
    x = torch.tensor([ids], device=model.device)
    with checkpointed_layers(model), \
            ResidualSteering(model, controller, collect=collect) as active:
        logits = model(input_ids=x, use_cache=False).logits
    logp = completion_logprobs(logits, ids, n_prompt)
    rel = (torch.stack(active.collected).mean()
           if active.collected else logp.new_zeros(()))
    del logits
    return logp, rel


def iter_replay_windows(model, ids: list[int], n_prompt: int, controller,
                        window: int):
    """Tiled replay with detached prefixes. Yields (a, b, logp_w, rel_w):
    completion-token range [a, b) and its grad-carrying logprobs.

    For each window the prefix is re-run UNDER THE CONTROLLER (on-policy
    states) with no grad into a cache; the window — one extra context
    token prepended — is then forwarded with grad. The caller must
    backward the window loss before advancing (generator keeps the graph
    alive only for the current window).
    """
    T_comp = len(ids) - n_prompt
    a = 0
    while a < T_comp:
        b = min(a + window, T_comp)
        s = n_prompt + a                       # ids index of window start
        ctx_start = 0 if s == 0 else s - 1     # +1 left context token
        np_eff = n_prompt if s == 0 else 1
        cache = None
        if ctx_start > 0:
            with ResidualSteering(model, controller, collect=False), \
                    torch.no_grad():
                out = model(input_ids=torch.tensor([ids[:ctx_start]],
                                                   device=model.device),
                            use_cache=True)
            cache = out.past_key_values
            del out
        with ResidualSteering(model, controller, collect=True) as active:
            out = model(input_ids=torch.tensor([ids[ctx_start:n_prompt + b]],
                                               device=model.device),
                        past_key_values=cache, use_cache=False)
        logits = out.logits
        logp_w = completion_logprobs(logits, ids[ctx_start:n_prompt + b],
                                     np_eff)
        rel_w = (torch.stack(active.collected).mean()
                 if active.collected else logp_w.new_zeros(()))
        del logits, out, cache
        yield a, b, logp_w, rel_w
        torch.cuda.empty_cache()
        a = b


class Replay:
    """Mode dispatcher.

    ``auto`` = full (checkpointed). ``window`` exists for investigation
    only: diag_window_drift (2026-08-25) showed cache-continuation
    corrupts the first ~50 tokens after every split boundary by up to
    11.8 nats (likelihood ratios off by >1e5), growing with split depth.
    There is therefore NO silent OOM fallback to windowed replay — a
    full-mode OOM is a guard trip (exit 8), not a degradation.

    On OOM the partially accumulated gradients of the interrupted sample
    are restored from a snapshot before any retry, so a mid-backward OOM
    cannot poison the batch.
    """

    def __init__(self, model, mode: str, window: int):
        assert mode in ("auto", "full", "window")
        if mode == "window":
            print("WARNING: --replay-mode window is KNOWN-BROKEN "
                  "(boundary corruption, diag_window_drift). Investigation "
                  "use only; never for production gradients.", flush=True)
        self.model = model
        self.mode = mode
        self.window = window
        self.sticky: str | None = None
        self.oom_fallbacks = 0

    def current(self) -> str:
        if self.sticky is not None:
            return self.sticky
        return "window" if self.mode == "window" else "full"

    def note_oom(self) -> bool:
        """False = no permitted fallback; caller must guard-trip."""
        return False


# ---------------------------------------------------------------------------
# Per-sample rollout rows — append-safe, resume-keyed
# ---------------------------------------------------------------------------

class RowWriter:
    """Resume state lives in the output file.

    Key: (tag, iter, arm, id, sample_i). Completed groups are reused
    verbatim on restart; PARTIAL groups are frozen from disk (never
    regenerated — cross-process sampling is not bit-stable on this
    stack, probe_determinism 2026-08-25); only zero-row groups are
    generated fresh. An existing row is never rewritten or duplicated.

    Sequences are persisted alongside the rows (``seqs/`` dir): a
    crash-resumed group must replay the SAME tokens, otherwise the
    resumed iteration silently loses that group's gradient.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self.seqs_dir = self.path.parent / "seqs"
        self.seqs_dir.mkdir(parents=True, exist_ok=True)
        self.keys: set[tuple] = set()
        self.groups: dict[tuple, dict[int, dict]] = defaultdict(dict)
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                if line.strip():
                    self._index(json.loads(line))

    def seqs_path(self, tag: str, it: int, arm: str, iid: str) -> Path:
        safe = iid.replace("/", "_")
        return self.seqs_dir / f"{tag}_it{it}_{arm}_{safe}.pt"

    def load_seqs(self, tag: str, it: int, arm: str, iid: str):
        p = self.seqs_path(tag, it, arm, iid)
        return torch.load(p, weights_only=False) if p.exists() else None

    def save_seqs(self, tag: str, it: int, arm: str, iid: str, seqs) -> None:
        torch.save(seqs, self.seqs_path(tag, it, arm, iid))

    @staticmethod
    def _key(r: dict) -> tuple:
        return (r["tag"], r["iter"], r["arm"], r["id"], r["sample_i"])

    def _index(self, r: dict) -> None:
        self.keys.add(self._key(r))
        self.groups[(r["tag"], r["iter"], r["arm"],
                     r["id"])][r["sample_i"]] = r

    def group_rows(self, tag: str, it: int, arm: str, iid: str,
                   n: int) -> list[dict] | None:
        g = self.groups.get((tag, it, arm, iid), {})
        if all(i in g for i in range(n)):
            return [g[i] for i in range(n)]
        return None

    def group_partial(self, tag: str, it: int, arm: str,
                      iid: str) -> dict[int, dict]:
        """{sample_i: row} for a group with some (possibly zero) rows."""
        return dict(self.groups.get((tag, it, arm, iid), {}))

    def add(self, row: dict) -> bool:
        k = self._key(row)
        if k in self.keys:
            return False
        with self.path.open("a") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._index(row)
        return True


def rollout_group(model, tok, item, n: int, max_new: int, seed: int,
                  arm: str, tag: str, it: int, rows: RowWriter, cfg,
                  controller=None):
    """One prompt's rollout group, resume-aware, rows appended.

    Returns (seqs, rewards, lens, truncated, stats, present):
      seqs     list of n id-lists (fresh gen or persisted), or None if
               the group resumed from rows without a sequences file
      present  length-n mask: which samples have on-disk evidence.
               Absent samples carry placeholder zeros in the other lists
               and MUST be excluded from rewards/stats/replay.
    Partial groups are frozen from disk — never regenerated (cross-
    process sampling is not bit-stable; a regenerated group would mix
    lineages across the immutable rows).
    """
    iid, fam = item["id"], item["family"]
    present = [True] * n
    have = rows.group_rows(tag, it, arm, iid, n)
    if have is not None:
        rewards = [r["reward"] for r in have]
        lens = [r["n_new"] for r in have]
        trunc = [r["truncated"] for r in have]
        stats = [{"gate_mean": r.get("gate_mean", 0.0),
                  "gate_std": r.get("gate_std", 0.0),
                  "gate_sat": r.get("gate_sat", 0.0),
                  "rel_mean": r.get("rel_mean", 0.0),
                  "rel_max": r.get("rel_max", 0.0)} for r in have]
        # persisted sequences let a resumed iteration replay the same tokens
        seqs = rows.load_seqs(tag, it, arm, iid)
        return seqs, rewards, lens, trunc, stats, present
    partial = rows.group_partial(tag, it, arm, iid)
    if partial:
        present = [i in partial for i in range(n)]
        seqs = rows.load_seqs(tag, it, arm, iid)
        rewards = [partial[i]["reward"] if i in partial else 0.0
                   for i in range(n)]
        lens = [partial[i]["n_new"] if i in partial else 0
                for i in range(n)]
        trunc = [partial[i]["truncated"] if i in partial else False
                 for i in range(n)]
        stats = [{"gate_mean": partial[i].get("gate_mean", 0.0),
                  "gate_std": partial[i].get("gate_std", 0.0),
                  "gate_sat": partial[i].get("gate_sat", 0.0),
                  "rel_mean": partial[i].get("rel_mean", 0.0),
                  "rel_max": partial[i].get("rel_max", 0.0)}
                 if i in partial else
                 {"gate_mean": 0.0, "gate_std": 0.0, "gate_sat": 0.0,
                  "rel_mean": 0.0, "rel_max": 0.0} for i in range(n)]
        print(f"  [resume] {tag}/{it}/{arm} {iid}: partial group frozen "
              f"from disk ({sum(present)}/{n} samples)"
              + ("" if seqs is not None else " [seqs file missing -> no replay]"),
              flush=True)
        return seqs, rewards, lens, trunc, stats, present

    n_prompt = len(build_prompt_ids(tok, item["prompt"],
                                    enable_thinking=cfg.thinking))
    seqs, recs = generate_batch(
        model, tok, item["prompt"], n=n, max_new_tokens=max_new,
        controller=controller, record=controller is not None,
        do_sample=True, temperature=cfg.temperature, top_p=cfg.top_p,
        seed=seed, enable_thinking=cfg.thinking)
    stats = record_stats(recs, n) if recs else [{}] * n
    rows.save_seqs(tag, it, arm, iid, seqs)
    rewards, lens, trunc = [], [], []
    for j, s in enumerate(seqs):
        text = decode_completion(tok, s, n_prompt)
        rew = float(pool_adapter.verify_item(item, text))
        n_new = len(s) - n_prompt
        truncated = s[-1] not in STOP_SET
        row = {
            "tag": tag, "iter": it, "arm": arm, "id": iid, "pid": item.get("pid"),
            "family": fam, "sample_i": j, "seed": seed,
            "n_prompt": n_prompt, "n_new": n_new,
            "truncated": truncated, "eos": not truncated, "reward": rew,
        }
        if controller is not None and j < len(stats) and "gate_mean" in stats[j]:
            row.update({"gate_mean": stats[j]["gate_mean"],
                        "gate_std": stats[j]["gate_std"],
                        "gate_sat": stats[j]["gate_sat"],
                        "rel_mean": stats[j]["rel_mean"],
                        "rel_max": stats[j]["rel_max"]})
        rows.add(row)
        rewards.append(rew)
        lens.append(n_new)
        trunc.append(truncated)
    return seqs, rewards, lens, trunc, stats, present


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _grad_snapshot(ctrl) -> list:
    return [None if p.grad is None else p.grad.detach().clone()
            for p in ctrl.parameters()]


def _grad_restore(ctrl, snap) -> None:
    for p, g in zip(ctrl.parameters(), snap):
        if g is None:
            p.grad = None
        elif p.grad is None:
            p.grad = g.clone()
        else:
            p.grad.copy_(g)


def replay_backward(model, ids, n_prompt, ctrl, cfg, eng: Replay,
                    advantage: float, grad_scale: float) -> dict:
    """One steered trajectory's pg+KL+mag, backward accumulated.

    Returns per-sample stats (pg/kl/mag means, n_tok, mode). NaN/Inf in
    loss or grads is a hard stop (exit 9) — a poisoned objective
    invalidates everything downstream.
    """
    ref = ref_logprobs(model, ids, n_prompt)
    mode = eng.current()
    pg_sum = kl_sum = mag = 0.0
    n_tok = 0

    def _check(t: torch.Tensor, what: str):
        if not torch.isfinite(t).all():
            print(f"GUARD-TRIP: non-finite {what}; stopping.", flush=True)
            sys.exit(9)

    if mode == "full":
        logp, rel = replay_full(model, ids, n_prompt, ctrl, collect=True)
        if logp.numel() == 0:
            return {"n_tok": 0, "mode": mode}
        kl = k3_kl(logp, ref)
        pg = -(advantage * logp)
        _check(pg, "pg"); _check(kl, "kl")
        loss = ((pg + cfg.beta_kl * kl).mean()
                + cfg.lambda_mag * rel) * grad_scale
        loss.backward()
        pg_sum, kl_sum = float(pg.mean()), float(kl.mean())
        mag = float(rel.detach())
        n_tok = int(logp.numel())
    else:
        T_comp = len(ids) - n_prompt
        for a, b, logp_w, rel_w in iter_replay_windows(
                model, ids, n_prompt, ctrl, eng.window):
            kl_w = k3_kl(logp_w, ref[a:b])
            pg_w = -(advantage * logp_w)
            _check(pg_w, "pg"); _check(kl_w, "kl")
            w = logp_w.numel() / T_comp
            loss_w = ((pg_w + cfg.beta_kl * kl_w).sum() / T_comp
                      + cfg.lambda_mag * rel_w * w) * grad_scale
            loss_w.backward()
            pg_sum += float(pg_w.sum()); kl_sum += float(kl_w.sum())
            mag += float(rel_w.detach()) * w
            n_tok += int(logp_w.numel())
        if n_tok:
            pg_sum, kl_sum = pg_sum / n_tok, kl_sum / n_tok

    for p in ctrl.parameters():
        if p.grad is not None:
            _check(p.grad, "controller grad")
    return {"pg": pg_sum, "kl": kl_sum, "mag": mag,
            "n_tok": n_tok, "mode": mode}


def train_iteration(model, tok, items, ctrl, opt, cfg, iter_i: int,
                    rows: RowWriter, eng: Replay):
    """One GRPO step: steered + matched-base groups, replayed steered arm."""
    t0 = time.time()
    opt.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    recs = []
    loss_pg = loss_kl = loss_mag = 0.0
    n_used = 0
    grad_scale = 1.0 / max(len(items), 1)
    cap_hits = eos_hits = n_samp = zero_var = 0
    replay_modes = set()

    def _pm(vals, mask):
        v = [x for x, p in zip(vals, mask) if p]
        return (sum(v) / len(v)) if v else 0.0

    for pi, it in enumerate(items):
        seed = cfg.seed + iter_i * 1009 + 17 * it.get("pid", pi)
        s_seqs, s_rews, s_lens, s_trunc, s_stats, s_pres = rollout_group(
            model, tok, it, cfg.group_size, cfg.max_new, seed,
            "steered", "train", iter_i, rows, cfg, controller=ctrl)
        _, b_rews, b_lens, b_trunc, _, b_pres = rollout_group(
            model, tok, it, cfg.group_size, cfg.max_new, seed,
            "base", "train", iter_i, rows, cfg, controller=None)
        cap_hits += sum(t for t, p in zip(s_trunc, s_pres) if p) + \
            sum(t for t, p in zip(b_trunc, b_pres) if p)
        n_pres_grp = sum(s_pres) + sum(b_pres)
        eos_hits += n_pres_grp - (sum(t for t, p in zip(s_trunc, s_pres) if p)
                                  + sum(t for t, p in zip(b_trunc, b_pres) if p))
        n_samp += n_pres_grp
        s_rews_pres = [r for r, p in zip(s_rews, s_pres) if p]
        adv_pres = group_advantages(s_rews_pres) if len(s_rews_pres) >= 2 \
            else [0.0] * len(s_rews_pres)
        adv = []
        k = 0
        for p in s_pres:
            if p:
                adv.append(adv_pres[k]); k += 1
            else:
                adv.append(None)
        if s_rews_pres and max(s_rews_pres) == min(s_rews_pres):
            zero_var += 1
        replayed = 0
        if s_seqs is not None:
            n_prompt = len(build_prompt_ids(tok, it["prompt"],
                                            enable_thinking=cfg.thinking))
            for j, (ss, A) in enumerate(zip(s_seqs, adv)):
                if A is None:
                    continue
                snap = _grad_snapshot(ctrl)
                try:
                    st = replay_backward(model, ss, n_prompt, ctrl, cfg,
                                         eng, A, grad_scale)
                except torch.cuda.OutOfMemoryError:
                    if eng.note_oom():
                        _grad_restore(ctrl, snap)
                        st = replay_backward(model, ss, n_prompt, ctrl, cfg,
                                             eng, A, grad_scale)
                    else:
                        print("GUARD-TRIP: full-replay OOM. The windowed "
                              "fallback is known-broken (diag_window_drift: "
                              "boundary corruption up to 11.8 nats); "
                              "refusing silent degradation. Stop and "
                              "rethink.", flush=True)
                        sys.exit(8)
                if st["n_tok"] == 0:
                    continue
                replay_modes.add(st["mode"])
                loss_pg += st["pg"] * grad_scale
                loss_kl += st["kl"] * grad_scale
                loss_mag += st["mag"] * grad_scale
                n_used += 1
                replayed += 1
        elif any(s_pres) and not all(s_pres):
            print(f"  [resume] {it['id']}: partial steered group without "
                  f"sequences -> rewards counted, replay skipped", flush=True)
        recs.append({
            "id": it["id"], "family": it["family"],
            "steered_mean": _pm(s_rews, s_pres),
            "base_mean": _pm(b_rews, b_pres),
            "steered_rews": s_rews_pres,
            "base_rews": [r for r, p in zip(b_rews, b_pres) if p],
            "advantages": [a for a in adv if a is not None],
            "steered_mean_len": _pm(s_lens, s_pres),
            "base_mean_len": _pm(b_lens, b_pres),
            "steered_cap_hits": sum(t for t, p in zip(s_trunc, s_pres) if p),
            "base_cap_hits": sum(t for t, p in zip(b_trunc, b_pres) if p),
            "replayed": replayed,
            "gate_mean": _pm([s["gate_mean"] for s in s_stats], s_pres),
            "gate_std": _pm([s["gate_std"] for s in s_stats], s_pres),
            "gate_sat": _pm([s.get("gate_sat", 0.0) for s in s_stats],
                            s_pres),
            "rel_mean": _pm([s.get("rel_mean", 0.0) for s in s_stats],
                            s_pres),
            "rel_max": max((s.get("rel_max", 0.0) for s, p in
                            zip(s_stats, s_pres) if p), default=0.0),
        })

    gn = torch.nn.utils.clip_grad_norm_(ctrl.parameters(), 1.0)
    if not torch.isfinite(gn):
        print(f"GUARD-TRIP: non-finite grad norm {float(gn)}; stopping.",
              flush=True)
        sys.exit(9)
    opt.step()
    dt = time.time() - t0
    s_acc = sum(r["steered_mean"] for r in recs) / len(recs)
    b_acc = sum(r["base_mean"] for r in recs) / len(recs)
    peak = torch.cuda.max_memory_reserved() / 2**30
    cap_rate = cap_hits / max(n_samp, 1)
    print(f"iter {iter_i:02d}  steer={s_acc:.2f}  base={b_acc:.2f}  "
          f"pg={loss_pg:.3f} kl={loss_kl:.4f} mag={loss_mag:.4f}  "
          f"|g|={float(gn):.3f}  cap={cap_hits}/{n_samp} "
          f"zerovar={zero_var}/{len(items)}  "
          f"replay={'+'.join(sorted(replay_modes)) or '-'}  "
          f"peak={peak:.1f}GB  {dt:.0f}s", flush=True)
    if zero_var == len(items) and cfg.stop_on_zero_var:
        print("GUARD-TRIP: zero reward variance in every group this "
              "iteration; stopping per Step-3 criterion.", flush=True)
        sys.exit(6)
    return {
        "iter": iter_i, "dt_s": dt,
        "steered_acc": s_acc, "base_acc": b_acc,
        "loss_pg": loss_pg, "loss_kl": loss_kl, "loss_mag": loss_mag,
        "grad_norm": float(gn), "n_replayed": n_used,
        "cap_hit_rate": cap_rate, "zero_var_groups": zero_var,
        "peak_mem_gb": peak, "replay_modes": sorted(replay_modes),
        "prompts": recs,
    }


def evaluate(model, tok, items, ctrl, group_size, max_new, seed_base,
             tag: str, it: int, rows: RowWriter, cfg) -> dict:
    """Matched steered + base groups (same seed) on a split."""
    per = []
    s_ok = b_ok = n_tot = 0
    s_len = b_len = 0
    s_cap = b_cap = 0
    for i, itm in enumerate(items):
        seed = seed_base + i * 17
        s_seqs, s_rews, s_lens, s_trunc, s_stats, s_pres = rollout_group(
            model, tok, itm, group_size, max_new, seed,
            "steered", f"eval-{tag}", it, rows, cfg, controller=ctrl)
        _, b_rews, b_lens, b_trunc, _, b_pres = rollout_group(
            model, tok, itm, group_size, max_new, seed,
            "base", f"eval-{tag}", it, rows, cfg, controller=None)
        s_rews_p = [r for r, p in zip(s_rews, s_pres) if p]
        b_rews_p = [r for r, p in zip(b_rews, b_pres) if p]
        s_stats_p = [s for s, p in zip(s_stats, s_pres) if p] or [{}]
        s_ok += int(sum(s_rews_p)); b_ok += int(sum(b_rews_p))
        n_tot += len(s_rews_p)
        s_len += sum(l for l, p in zip(s_lens, s_pres) if p)
        b_len += sum(l for l, p in zip(b_lens, b_pres) if p)
        s_cap += sum(t for t, p in zip(s_trunc, s_pres) if p)
        b_cap += sum(t for t, p in zip(b_trunc, b_pres) if p)
        gmean = sum(s.get("gate_mean", 0.0) for s in s_stats_p) / len(s_stats_p)
        gstd = sum(s.get("gate_std", 0.0) for s in s_stats_p) / len(s_stats_p)
        rmean = sum(s.get("rel_mean", 0.0) for s in s_stats_p) / len(s_stats_p)
        per.append({
            "id": itm["id"], "family": itm["family"],
            "steered_mean": (sum(s_rews_p) / len(s_rews_p))
            if s_rews_p else 0.0,
            "base_mean": (sum(b_rews_p) / len(b_rews_p))
            if b_rews_p else 0.0,
            "steered_rews": s_rews_p, "base_rews": b_rews_p,
            "steered_cap_hits": sum(t for t, p in zip(s_trunc, s_pres) if p),
            "base_cap_hits": sum(t for t, p in zip(b_trunc, b_pres) if p),
            "gate_mean": gmean, "gate_std": gstd, "rel_mean": rmean,
        })
        print(f"  [{tag}] {itm['id']:32s}  "
              f"steer={sum(s_rews_p):.0f}/{len(s_rews_p)}  "
              f"base={sum(b_rews_p):.0f}/{len(b_rews_p)}  "
              f"gμ={gmean:.3f} gσ={gstd:.3f} rel={rmean:.4f}", flush=True)
    s_lo, s_hi = wilson_ci(s_ok, n_tot)
    b_lo, b_hi = wilson_ci(b_ok, n_tot)
    return {
        "tag": tag, "n": n_tot,
        "steered_acc": (s_ok / n_tot) if n_tot else 0.0,
        "base_acc": (b_ok / n_tot) if n_tot else 0.0,
        "steered_ci": [s_lo, s_hi], "base_ci": [b_lo, b_hi],
        "steered_mean_len": (s_len / n_tot) if n_tot else 0.0,
        "base_mean_len": (b_len / n_tot) if n_tot else 0.0,
        "steered_cap_hits": s_cap, "base_cap_hits": b_cap,
        "per_item": per,
    }


def check_equivalence(model, tok, item, cfg, window: int) -> dict:
    """Replay-mode correctness gate (Step 1).

    Rolls one short thinking-on trace, then compares three logprob paths:
      ref     no-grad single pass (ground truth; zero-init controller no-op)
      full    gradient-checkpointed full-trace replay under the controller
      window  detached-prefix tiled replay under the controller
    Both replay paths must agree with ref to bf16 precision (~1e-2 abs);
    anything larger means the segmentation changed the math, and the
    policy gradient would be training on wrong numbers.
    """
    torch.manual_seed(cfg.seed)
    n_prompt = len(build_prompt_ids(tok, item["prompt"],
                                    enable_thinking=cfg.thinking))
    seqs, _ = generate_batch(model, tok, item["prompt"], n=1,
                             max_new_tokens=cfg.eq_max_new, controller=None,
                             do_sample=True, temperature=cfg.temperature,
                             top_p=cfg.top_p, seed=cfg.seed,
                             enable_thinking=cfg.thinking)
    ids = list(seqs[0])
    n_comp = len(ids) - n_prompt
    ctrl = SteeringController().to(device=model.device, dtype=torch.float32)
    ctrl.eval()

    def stats(lp: torch.Tensor) -> dict:
        with torch.no_grad():
            lp = lp.detach().float()
            return {"n": int(lp.numel()), "mean": float(lp.mean()),
                    "min": float(lp.min()), "max": float(lp.max())}

    ref = ref_logprobs(model, ids, n_prompt)
    out = {"id": item["id"], "n_prompt": n_prompt, "n_comp": n_comp,
           "window": window, "ref": stats(ref)}

    logp_full, rel_full = replay_full(model, ids, n_prompt, ctrl)
    d_full = (logp_full.detach().float() - ref.float()).abs()
    out["full"] = {"max_abs_diff": float(d_full.max()),
                   "mean_abs_diff": float(d_full.mean()),
                   "rel": float(rel_full.detach()),
                   "grad_ok": None}
    g = torch.autograd.grad(logp_full.sum(), ctrl.parameters(),
                            allow_unused=True)
    out["full"]["grad_ok"] = bool(any(x is not None and
                                        torch.isfinite(x).all() for x in g))
    del logp_full, g

    parts, rels, ws = [], [], []
    for a, b, logp_w, rel_w in iter_replay_windows(
            model, ids, n_prompt, ctrl, window):
        parts.append(logp_w.detach().float().cpu())
        rels.append(float(rel_w.detach())); ws.append(b - a)
        del logp_w
    logp_win = torch.cat(parts)
    d_win = (logp_win - ref.float().cpu()).abs()
    out["window"] = {"max_abs_diff": float(d_win.max()),
                     "mean_abs_diff": float(d_win.mean()),
                     "rel_weighted": float(sum(r * w for r, w in
                                                 zip(rels, ws)) / sum(ws)),
                     "n_windows": len(ws)}
    out["pass"] = (out["full"]["max_abs_diff"] < 1e-2
                   and out["full"]["grad_ok"] and n_comp > 0)
    out["window_ok"] = out["window"]["max_abs_diff"] < 1e-2
    return out


def revalidate(model, tok, items, cfg, rows: RowWriter) -> dict:
    """Step 2: unsteered n-rollouts on the FULL pool, thinking on.

    Re-bands the pool on the policy it will train against (the first
    unsteered bf16 batch is the pool's re-verification-by-regeneration).
    Seeds live in the fresh 100_000 space.
    """
    base_seed = 100_000
    results = []
    for itm in items:
        seed = base_seed + 17 * itm.get("pid", 0)
        _, rews, lens, trunc, _, pres = rollout_group(
            model, tok, itm, cfg.revalidate_n, cfg.max_new, seed,
            "base", "revalidate", 0, rows, cfg, controller=None)
        rews_p = [r for r, p in zip(rews, pres) if p]
        trunc_p = [t for t, p in zip(trunc, pres) if p]
        lens_p = [l for l, p in zip(lens, pres) if p]
        k = int(sum(rews_p)); n = len(rews_p)
        band = ("DEAD-HARD" if k == 0
                else "DEAD-EASY" if k == n else "LIVE")
        results.append({"id": itm["id"], "pid": itm.get("pid"),
                        "family": itm["family"], "strict": f"{k}/{n}",
                        "band": band, "lens": lens_p, "truncated": trunc_p})
        print(f"  [revalidate] {itm['id']:32s} {k}/{n}  {band}  "
              f"cap={sum(trunc_p)}/{n}", flush=True)
    n_live = sum(1 for r in results if r["band"] == "LIVE")
    summary = {
        "n_prompts": len(results), "n_live": n_live,
        "by_family": {
            f: {"live": sum(1 for r in results
                            if r["family"] == f and r["band"] == "LIVE"),
                "total": sum(1 for r in results if r["family"] == f)}
            for f in sorted({r["family"] for r in results})
        },
        "results": results,
    }
    print(f"\nREVALIDATION: {n_live}/{len(results)} LIVE", flush=True)
    if n_live < 15:
        print("STOP-CRITERION: re-banded pool collapsed below ~15 LIVE "
              "prompts -> pool-texture verdict, not controller verdict. "
              "Stop and rethink (Step-2 rule).", flush=True)
    return summary


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

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
    pick = ("steered_acc", "base_acc", "steered_ci", "base_ci",
            "steered_mean_len", "base_mean_len",
            "steered_cap_hits", "base_cap_hits")
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
        "cap_hit_rate_last": last.get("cap_hit_rate"),
        "replay_oom_fallbacks": None,  # filled by main (eng not in scope)
        "holdout_init": ({k: hold_init[k] for k in pick} if hold_init else None),
        "holdout_final": ({k: hold_final[k] for k in pick} if hold_final else None),
        "train_eval_final": ({k: train_final[k] for k in pick}
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
    p.add_argument("--pool", type=str, default=None,
                   help="Calibrated pool JSONL (b2). Without it: legacy "
                        "arithmetic problems, thinking off (old shape).")
    p.add_argument("--only-pids", type=str, default=None,
                   help="comma-separated proposal ids; restrict the pool "
                        "BEFORE the split (dry-run slices).")
    p.add_argument("--check-equivalence", action="store_true",
                   help="Step-1 gate: one rollout, then compare ref vs "
                        "full-checkpointed vs windowed logprobs. No training.")
    p.add_argument("--eq-max-new", type=int, default=3000)
    p.add_argument("--revalidate", action="store_true",
                   help="Step-2 mode: unsteered re-band of the full pool, "
                        "no training.")
    p.add_argument("--revalidate-n", type=int, default=8)
    p.add_argument("--iters", type=int, default=12)
    p.add_argument("--prompts-per-iter", type=int, default=4)
    p.add_argument("--group-size", type=int, default=4)
    p.add_argument("--max-new", type=int, default=26000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--beta-kl", type=float, default=0.02)
    p.add_argument("--lambda-mag", type=float, default=0.1)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=80_000,
                   help="train-batch seed base (fresh 80k space; smoke "
                        "consumed 70k). Eval: hold 90k, train-slice 95k; "
                        "revalidate 100k.")
    p.add_argument("--thinking", action=argparse.BooleanOptionalAction,
                   default=True)
    p.add_argument("--replay-mode", choices=("auto", "full", "window"),
                   default="auto",
                   help="auto=full. window is known-broken (boundary "
                        "corruption) — investigation only.")
    p.add_argument("--replay-window", type=int, default=8192,
                   help="measured ceiling: 8k window = 111.6 GB peak")
    p.add_argument("--eval-every", type=int, default=4)
    p.add_argument("--skip-eval", action="store_true",
                   help="training iterations only (dry runs)")
    p.add_argument("--eval-group", type=int, default=4)
    p.add_argument("--eval-train-slice", type=int, default=8,
                   help="train-split prompts per train eval (every 3rd)")
    p.add_argument("--stop-on-zero-var", action="store_true",
                   help="hard-stop (exit 6) when every group in an "
                        "iteration has zero reward variance")
    p.add_argument("--out", type=str, default=str(DATA_DIR))
    p.add_argument("--smoke", action="store_true",
                   help="Legacy tiny arithmetic run (thinking off).")
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
        cfg.thinking = False
        cfg.pool = None
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)

    model_path = os.environ.get("ACE_MODEL_PATH", MODEL_PATH)
    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    freeze_base_model(model)
    snap = snapshot_base(model)

    # fp32 controller: optimizer precision; hook casts delta back to bf16.
    ctrl = SteeringController().to(device="cuda", dtype=torch.float32)
    n_params = sum(p.numel() for p in ctrl.parameters())
    opt = torch.optim.AdamW(ctrl.parameters(), lr=cfg.lr,
                            betas=(0.9, 0.95), weight_decay=0.0)
    eng = Replay(model, cfg.replay_mode, cfg.replay_window)
    rows = RowWriter(out / "rollout_rows.jsonl")
    print(f"controller params={n_params:,}  frozen base  layer={STEER_LAYER}  "
          f"iters={cfg.iters}  G={cfg.group_size}  max_new={cfg.max_new}  "
          f"thinking={cfg.thinking}  replay={cfg.replay_mode} "
          f"(window={cfg.replay_window})  model={model_path}  "
          f"rows_resume={len(rows.keys)}", flush=True)

    if cfg.pool:
        pool_rows = pool_adapter.load_pool(Path(cfg.pool))
        pids = ([int(x) for x in cfg.only_pids.split(",")]
                if cfg.only_pids else None)
        items = pool_adapter.make_items(pool_rows, pids)
        train_items, hold_items = pool_adapter.stratified_split(items)
        pool_adapter.write_split_manifest(
            out / "split_manifest.json", train_items, hold_items,
            Path(cfg.pool), extra={
                "seed_spaces": {"train": cfg.seed, "eval_hold": 90_000,
                                "eval_train": 95_000, "revalidate": 100_000},
                "replay_mode": cfg.replay_mode,
                "replay_window": cfg.replay_window})
        print(f"pool: {len(items)} items -> train={len(train_items)} "
              f"holdout={len(hold_items)} (manifest in {out})", flush=True)
        legacy = False
    else:
        from mlfactory.experiments.ace.core.problems import HOLDOUT, TRAIN
        train_items, hold_items = list(TRAIN), list(HOLDOUT)
        legacy = True

    if cfg.check_equivalence:
        if not cfg.pool:
            print("--check-equivalence requires --pool", flush=True)
            sys.exit(2)
        res = check_equivalence(model, tok, items[0], cfg, cfg.replay_window)
        (out / "equivalence.json").write_text(json.dumps(res, indent=2))
        print(json.dumps(res, indent=2), flush=True)
        print(f"EQUIVALENCE(full) {'PASS' if res['pass'] else 'FAIL'}  "
              f"window_ok={res['window_ok']}", flush=True)
        return

    if cfg.revalidate:
        if not cfg.pool:
            print("--revalidate requires --pool", flush=True)
            sys.exit(2)
        summary = revalidate(model, tok, items, cfg, rows)
        (out / "revalidate_summary.json").write_text(
            json.dumps(summary, indent=2))
        print(f"artifacts: {out}", flush=True)
        return

    rng = random.Random(cfg.seed)
    log_rows, evals = [], []
    done_iters: dict[int, dict] = {}          # resume: train.jsonl is the
    if (out / "train.jsonl").exists():       # iteration-level resume state
        for line in (out / "train.jsonl").read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done_iters[int(r["iter"])] = r
    if done_iters:
        print(f"resuming: {len(done_iters)} iterations already in "
              f"train.jsonl ({sorted(done_iters)})", flush=True)

    def run_eval(tag, it):
        print(f"\n=== eval {tag} ===", flush=True)
        hold = evaluate(model, tok, hold_items, ctrl, cfg.eval_group,
                        cfg.max_new, seed_base=90_000, tag=f"hold-{tag}",
                        it=it, rows=rows, cfg=cfg)
        tr_slice = (train_items[::3][:cfg.eval_train_slice]
                    if not legacy else train_items[::3][:8])
        trn = evaluate(model, tok, tr_slice, ctrl, cfg.eval_group,
                       cfg.max_new, seed_base=95_000, tag=f"train-{tag}",
                       it=it, rows=rows, cfg=cfg)
        evals.append(hold)
        evals.append(trn)
        (out / f"eval_{tag}.json").write_text(
            json.dumps({"holdout": hold, "train": trn}, indent=2,
                       default=str))
        return hold, trn

    run_eval("init", it=-1) if not cfg.skip_eval else None

    pool = list(train_items)
    for it in range(cfg.iters):
        rng.shuffle(pool)
        if it in done_iters:
            log_rows.append(done_iters[it])
            print(f"iter {it:02d}  already in train.jsonl — skipped",
                  flush=True)
            if not cfg.skip_eval and ((it + 1) % cfg.eval_every == 0
                                      or it + 1 == cfg.iters):
                run_eval(f"iter{it + 1:02d}", it=it)
            continue
        batch = pool[:cfg.prompts_per_iter]
        row = train_iteration(model, tok, batch, ctrl, opt, cfg, it,
                              rows, eng)
        log_rows.append(row)
        with (out / "train.jsonl").open("a") as fh:
            fh.write(json.dumps(row, default=str) + "\n")
        if not cfg.skip_eval and ((it + 1) % cfg.eval_every == 0
                                   or it + 1 == cfg.iters):
            run_eval(f"iter{it + 1:02d}", it=it)
            ctrl.save(out / f"ckpt_iter{it + 1:02d}",
                      extra_meta={"iter": it + 1, "layer_idx": STEER_LAYER})

    fingerprints_ok = check_base(model, snap)
    summary = summarize(log_rows, evals, fingerprints_ok)
    summary["replay_oom_fallbacks"] = eng.oom_fallbacks
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n=== summary ===", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"base fingerprints unchanged: {fingerprints_ok}", flush=True)
    print(f"artifacts: {out}", flush=True)


if __name__ == "__main__":
    main()
