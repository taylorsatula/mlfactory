#!/usr/bin/env python3
"""Phase-1 kill-test analysis over teacherforced_scan .npz outputs.

Per trace, from stored entropy + hook-layer hidden states:
  ent_early/mid/late : mean token entropy by trace thirds
  ent_trend          : late - early
  recur_density      : fraction of positions whose hidden state has cosine
                       sim > THRESH to any state >WINDOW tokens earlier
  tortuosity         : sum of step distances / endpoint distance

Tests:
  1. outcome separation (correct vs wrong), pooled + within-prompt
     (rank-biserial correlation; |r| < 0.2 => metric is a kill candidate)
  2. loop-onset signature: pre-onset window entropy/recurrence vs matched
     windows in non-loop sibling traces

Usage: python3 teacherforced_analyze.py [data/teacherforced_b1]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

THRESH = 0.92     # cosine similarity for "return to earlier state"
WINDOW = 200      # min token lag (subsampled positions are stride 4 apart)
STRIDE = 4
ONSET_WIN = 500   # tokens before onset for signature window


def feats(npz: Path) -> dict | None:
    d = np.load(npz, allow_pickle=False)
    meta = json.loads(str(d["meta"]))
    ent = d["entropy"].astype(np.float32)
    ent = ent[~np.isnan(ent)]
    if len(ent) < 300:
        return None
    H = d["hidden15"].astype(np.float32)          # (T/stride, 4096)
    Hn = H / (np.linalg.norm(H, axis=1, keepdims=True) + 1e-8)
    T = len(Hn)
    w = max(1, WINDOW // STRIDE)
    # recurrence: for each t, max sim to states < t-w
    recur_hits = 0
    counted = 0
    sims = Hn @ Hn.T                              # (T,T); T ~ 5k => 25M floats, ok
    for t in range(w + 1, T):
        counted += 1
        if sims[t, : t - w].max() > THRESH:
            recur_hits += 1
    steps = np.linalg.norm(np.diff(H, axis=0), axis=1).sum()
    end = np.linalg.norm(H[-1] - H[0]) + 1e-8
    th = len(ent) // 3
    return {
        **meta,
        "n_ent": len(ent),
        "ent_early": float(ent[:th].mean()), "ent_mid": float(ent[th:2*th].mean()),
        "ent_late": float(ent[2*th:].mean()),
        "ent_trend": float(ent[2*th:].mean() - ent[:th].mean()),
        "recur_density": recur_hits / max(1, counted),
        "tortuosity": float(steps / end),
        "onset_tok": int(d["onset_tok"]),
        "_ent": ent, "_sims": sims, "_w": w,
    }


def rank_biserial(a: list[float], b: list[float]) -> float:
    """a=correct, b=wrong. >0 means higher values on correct."""
    x = [(v, 1) for v in a] + [(v, 0) for v in b]
    x.sort(key=lambda t: t[0])
    # average ranks for ties
    pos = np.empty(len(x))
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and x[j + 1][0] == x[i][0]:
            j += 1
        pos[i:j + 1] = (i + j) / 2 + 1
        i = j + 1
    r1 = sum(p for p, (_, g) in zip(pos, x) if g == 1)
    n1, n0 = len(a), len(b)
    if not n1 or not n0:
        return float("nan")
    u1 = r1 - n1 * (n1 + 1) / 2
    return 2 * u1 / (n1 * n0) - 1


def main() -> None:
    src = Path(sys.argv[1] if len(sys.argv) > 1 else "data/teacherforced_b1")
    F = [f for p in sorted(src.glob("*.npz")) if (f := feats(p))]
    print(f"traces: {len(F)}")
    keys = ["ent_early", "ent_mid", "ent_late", "ent_trend",
            "recur_density", "tortuosity", "n_ent"]

    print("\n== outcome separation (pooled) ==")
    print(f"{'metric':14s} {'corr_mean':>9s} {'wrong_mean':>10s} {'rankbis':>8s}  verdict")
    for k in keys:
        a = [f[k] for f in F if f["correct"]]
        b = [f[k] for f in F if not f["correct"]]
        rb = rank_biserial(a, b)
        verdict = "KEEP?" if abs(rb) >= 0.2 else "KILL?"
        print(f"{k:14s} {np.mean(a):9.4f} {np.mean(b):10.4f} {rb:8.3f}  {verdict}")

    print("\n== within-prompt separation (prompts with both outcomes) ==")
    byp: dict[int, list] = {}
    for f in F:
        byp.setdefault(f["proposal_id"], []).append(f)
    for pid, fs in sorted(byp.items()):
        a = [f for f in fs if f["correct"]]
        b = [f for f in fs if not f["correct"]]
        if not a or not b:
            continue
        row = " ".join(f"{k}={rank_biserial([f[k] for f in a],[f[k] for f in b]):+.2f}"
                       for k in ["ent_late", "ent_trend", "recur_density", "tortuosity"])
        print(f"  p{pid:02d} {fs[0]['domain']:10s} {len(a)}c/{len(b)}w  {row}")

    print("\n== loop-onset signature ==")
    loops = [f for f in F if f["onset_tok"] >= 0]
    print(f"loop traces with onset: {len(loops)}")
    for f in loops:
        o = f["onset_tok"]
        ent = f["_ent"]
        if o >= ONSET_WIN and o + ONSET_WIN < len(ent):
            pre = ent[o - ONSET_WIN:o].mean()
            post = ent[o:o + ONSET_WIN].mean()
            base = ent[:max(1, o - ONSET_WIN)].mean()
            print(f"  p{f['proposal_id']:02d}_s{f['sample_i']} onset_tok={o} "
                  f"ent pre={pre:.3f} post={post:.3f} earlier={base:.3f}")


if __name__ == "__main__":
    main()
