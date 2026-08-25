#!/usr/bin/env python3
"""Rigorous Phase-1 kill test on MACHINE family (clean finishes only).

The doctrine's honest test is WITHIN-PROMPT: does the observable separate
correct from wrong samples of the SAME prompt? Pooled numbers confound with
prompt difficulty. We report both, but the within-prompt verdict is the one
that kills or keeps a metric.

Observables tested (all from stored residual_map captures):
  ent_early / ent_mid / ent_late / ent_trend : final-layer next-token entropy
  step_cos_L<l>  : mean cosine step-distance at each residual layer
  rec_frob_L<l>  : Frobenius norm of final recurrent state per linear layer
  recur_density  : fraction of positions returning to earlier state space
  tortuosity      : path length / displacement
  n_tokens        : completion length (known composition confound — sanity)

Within-prompt test: rank-biserial per prompt (correct vs wrong), then the
SIGN-CONSISTENCY across prompts — if signs flip, the metric does not
generalize even within one family, and is killed.

Usage: python3 -m mlfactory.experiments.ace.analysis.machine_kill_test
"""
from __future__ import annotations

import json
import zipfile
from pathlib import Path
from collections import defaultdict

import numpy as np

STRIDE = 16
THRESH = 0.92


def rank_biserial(a, b):
    if len(a) < 1 or len(b) < 1:
        return float("nan")
    x = sorted([(v, 1) for v in a] + [(v, 0) for v in b])
    pos = np.empty(len(x))
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and x[j + 1][0] == x[i][0]:
            j += 1
        pos[i:j + 1] = (i + j) / 2 + 1
        i = j + 1
    n1, n0 = len(a), len(b)
    r1 = sum(p for p, (_, g) in zip(pos, x) if g == 1)
    return 2 * (r1 - n1 * (n1 + 1) / 2) / (n1 * n0) - 1


def main():
    src = Path("data/map_b1")
    traces = []
    for p in sorted(src.glob("*.npz")):
        try:
            d = np.load(p, allow_pickle=True)
        except (EOFError, ValueError, OSError, zipfile.BadZipFile):
            continue
        meta = json.loads(str(d["meta"]))
        if meta["domain"] != "machine" or meta.get("truncated"):
            continue
        ent = d["entropy"].copy()
        ent = ent[~np.isnan(ent)]
        th = len(ent) // 3
        # per-layer step cosine distance
        step_cos = {}
        res = d["residuals"]
        if res is not None:
            for L in [6, 15, 17, 23, 25, 31]:  # key layers from the map
                H = res[L].astype(np.float32)
                if H.shape[0] >= 4:
                    Hn = H / (np.linalg.norm(H, axis=1, keepdims=True) + 1e-8)
                    step_cos[f"step_L{L}"] = float((1 - np.sum(Hn[1:] * Hn[:-1], axis=1)).mean())
                else:
                    step_cos[f"step_L{L}"] = 0.0
        # recurrent Frobenius norm at key layers
        rec_frob = {}
        for k in [f"rec_{i}" for i in [2, 9, 12, 14, 18]]:
            st = d.get(k)
            if st is not None and st.size > 0:
                rec_frob[f"frob_{k}"] = float(np.linalg.norm(st[0]))
        # recurrence density (layer 15, cheap proxy)
        recur_d = 0.0
        if res is not None:
            H = res[15].astype(np.float32)
            if H.shape[0] > 50:
                Hn = H / (np.linalg.norm(H, axis=1, keepdims=True) + 1e-8)
                sims = Hn @ Hn.T
                w = 200 // STRIDE
                hits = sum(1 for t in range(w + 1, Hn.shape[0])
                           if sims[t, :t - w].max() > THRESH)
                recur_d = hits / max(1, Hn.shape[0] - w - 1)
        # tortuosity (layer 15)
        tort = 0.0
        if res is not None:
            H = res[15].astype(np.float32)
            if H.shape[0] >= 2:
                steps = np.linalg.norm(np.diff(H, axis=0), axis=1).sum()
                end = np.linalg.norm(H[-1] - H[0]) + 1e-8
                tort = float(steps / end)
        traces.append({
            "pid": meta["proposal_id"], "correct": bool(meta["correct"]),
            "n_ent": len(ent),
            "ent_early": float(ent[:th].mean()) if th > 0 else 0,
            "ent_mid": float(ent[th:2*th].mean()) if th > 0 else 0,
            "ent_late": float(ent[2*th:].mean()) if th > 0 else 0,
            "ent_trend": float(ent[2*th:].mean() - ent[:th].mean()) if th > 0 else 0,
            "n_tokens": meta["n_new_tokens"],
            "recur_density": recur_d,
            "tortuosity": tort,
            **step_cos, **rec_frob,
        })

    keys = ["ent_early", "ent_mid", "ent_late", "ent_trend", "n_tokens",
            "recur_density", "tortuosity",
            "step_L6", "step_L15", "step_L17", "step_L23", "step_L25", "step_L31",
            "frob_rec_2", "frob_rec_9", "frob_rec_12", "frob_rec_14", "frob_rec_18"]

    print(f"machine clean: {len(traces)} traces\n")

    # === pooled ===
    print("== pooled (machine clean) ==")
    print(f"{'metric':16s} {'corr':>7s} {'wrong':>7s} {'rb':>7s}")
    for k in keys:
        a = [t[k] for t in traces if t["correct"]]
        b = [t[k] for t in traces if not t["correct"]]
        if a and b:
            print(f"{k:16s} {np.mean(a):7.4f} {np.mean(b):7.4f} {rank_biserial(a,b):7.3f}")

    # === within-prompt ===
    print("\n== within-prompt (the honest test) ==")
    byp = defaultdict(list)
    for t in traces:
        byp[t["pid"]].append(t)
    # only prompts with both outcomes
    mixed = {p: ts for p, ts in byp.items()
             if any(t["correct"] for t in ts) and any(not t["correct"] for t in ts)}
    print(f"prompts with mixed outcomes: {sorted(mixed)}\n")
    print(f"{'metric':16s} " + " ".join(f"p{p:02d}" for p in sorted(mixed)) + "  consistent?")
    for k in keys:
        rbs = []
        for p in sorted(mixed):
            ts = mixed[p]
            a = [t[k] for t in ts if t["correct"]]
            b = [t[k] for t in ts if not t["correct"]]
            rbs.append(rank_biserial(a, b))
        # sign consistency (ignore NaN)
        signs = [1 if r > 0 else (-1 if r < 0 else 0) for r in rbs if not np.isnan(r)]
        consistent = "KEEP" if len(set(signs)) == 1 and signs and signs[0] != 0 else "KILL"
        vals = " ".join(f"{r:+.2f}" for r in rbs)
        print(f"{k:16s} {vals}  {consistent}")


if __name__ == "__main__":
    main()
