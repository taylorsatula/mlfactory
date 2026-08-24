#!/usr/bin/env python3
"""Offline per-layer analysis — single streaming pass, low memory.

Reads one npz at a time, extracts scalar features per layer, discards the
array. Never holds more than one trace in RAM (~200 MB). No GPU.

Per layer (32 residual + 24 recurrent), per trace:
  - step_cos: mean cosine distance between consecutive residual states
  - ent_late: final-third next-token entropy (final layer only)
  - rec_frob: Frobenius norm of final recurrent state
  - onset: pre-onset vs mid-trace mean state projection (for loop traces)

Then rank-biserial per layer (correct vs wrong, clean finishes) and
AUROC for onset separability.

Usage: python3 teacherforced_map_analyze.py [data/teacherforced_map_b1]
"""
from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

import numpy as np

STRIDE = 16
ONSET_WIN = 512  # tokens


def rank_biserial(a, b):
    if len(a) < 2 or len(b) < 2:
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


def auroc(pos_scores, neg_scores):
    if len(pos_scores) < 2 or len(neg_scores) < 2:
        return float("nan")
    pos = np.array(pos_scores)
    neg = np.array(neg_scores)
    return float(np.mean([np.sum(neg < p) + 0.5 * np.sum(neg == p) for p in pos]) /
                 (len(pos) * len(neg)))


def main() -> None:
    src = Path(sys.argv[1] if len(sys.argv) > 1 else "data/teacherforced_map_b1")
    files = sorted(src.glob("*.npz"))

    # accumulators
    sep = {L: {"c": [], "w": []} for L in range(32)}          # step_cos per layer
    rec_sep = {f"rec_{i}": {"c": [], "w": []}
               for i in range(32) if i % 4 != 3}
    ent_late = {"c": [], "w": []}
    # onset: per-layer pre-onset mean state and mid-trace mean state
    onset_pos = {L: [] for L in range(32)}   # loop traces: pre-onset projected
    onset_neg = {L: [] for L in range(32)}   # healthy: mid-trace projected
    onset_dir = {}  # layer -> discriminant direction (computed in 2nd pass)

    n = 0
    for p in files:
        try:
            d = np.load(p, allow_pickle=True)
        except (EOFError, ValueError, OSError, zipfile.BadZipFile):
            continue
        meta = json.loads(str(d["meta"]))
        res = d["residuals"]
        ent = d["entropy"]
        onset = int(d["onset_tok"])
        n += 1
        trunc = bool(meta.get("truncated"))
        ok = bool(meta["correct"])

        # entropy (final layer)
        e = ent.copy()
        e = e[~np.isnan(e)]
        if len(e) > 9 and not trunc:
            ent_late["c" if ok else "w"].append(float(e[2 * len(e) // 3:].mean()))

        # per-layer step cosine distance
        if res is not None and not trunc:
            for L in range(32):
                H = res[L].astype(np.float32)
                if H.shape[0] >= 4:
                    Hn = H / (np.linalg.norm(H, axis=1, keepdims=True) + 1e-8)
                    diffs = 1 - np.sum(Hn[1:] * Hn[:-1], axis=1)
                    sep[L]["c" if ok else "w"].append(float(diffs.mean()))

        # recurrent state features
        for k in rec_sep:
            st = d[k]
            if st.size == 0:
                continue
            if not trunc:
                rec_sep[k]["c" if ok else "w"].append(float(np.linalg.norm(st[0])))

        # onset: collect pre-onset and mid-trace mean states per layer
        if res is not None:
            T = res.shape[1]
            ow = ONSET_WIN // STRIDE
            if onset >= ONSET_WIN and onset // STRIDE + ow < T:
                o = onset // STRIDE
                for L in range(32):
                    onset_pos[L].append(res[L][max(0, o - ow):o].mean(axis=0))
            elif not trunc and T > 2 * ow:
                mid = T // 2
                for L in range(32):
                    onset_neg[L].append(res[L][mid - ow:mid].mean(axis=0))

    print(f"traces loaded: {n}")

    # --- outcome separation: per-layer step cosine distance ---
    print("\n== per-layer outcome separation (mean step-cosine-dist, clean) ==")
    print(f"{'L':>3s} {'type':>4s} {'corr':>7s} {'wrong':>7s} {'rb':>7s}")
    for L in range(32):
        typ = "FULL" if L % 4 == 3 else "lin"
        a, b = sep[L]["c"], sep[L]["w"]
        if len(a) >= 2 and len(b) >= 2:
            print(f"{L:3d} {typ:>4s} {np.mean(a):7.4f} {np.mean(b):7.4f} "
                  f"{rank_biserial(a, b):7.3f}")

    # --- entropy ---
    print(f"\n== final-layer ent_late: corr={np.mean(ent_late['c']):.4f} "
          f"wrong={np.mean(ent_late['w']):.4f} rb={rank_biserial(ent_late['c'], ent_late['w']):.3f} "
          f"(n={len(ent_late['c'])}/{len(ent_late['w'])})")

    # --- onset separability ---
    print(f"\n== loop-onset probe (pre-onset vs matched healthy, AUROC) ==")
    print(f"  loop traces: {len(onset_pos[0])}, healthy: {len(onset_neg[0])}")
    if len(onset_pos[15]) >= 3 and len(onset_neg[15]) >= 3:
        for L in range(32):
            typ = "FULL" if L % 4 == 3 else "lin"
            pos = onset_pos[L]
            neg = onset_neg[L]
            if len(pos) < 3 or len(neg) < 3:
                continue
            pos = np.stack(pos).astype(np.float32)
            neg = np.stack(neg).astype(np.float32)
            dvec = pos.mean(0) - neg.mean(0)
            au = auroc(list(pos @ dvec), list(neg @ dvec))
            if abs(au - 0.5) > 0.1:
                print(f"  L{L:2d} ({typ:4s}): AUROC={au:.3f}")

    # --- recurrent state separation ---
    print(f"\n== recurrent-state Frobenius norm (final, clean) ==")
    print(f"{'L':>4s} {'corr':>7s} {'wrong':>7s} {'rb':>7s}")
    for k in sorted(rec_sep, key=lambda x: int(x.split("_")[1])):
        a, b = rec_sep[k]["c"], rec_sep[k]["w"]
        if len(a) >= 2 and len(b) >= 2:
            print(f"{k:>4s} {np.mean(a):7.2f} {np.mean(b):7.2f} "
                  f"{rank_biserial(a, b):7.3f}")


if __name__ == "__main__":
    main()
