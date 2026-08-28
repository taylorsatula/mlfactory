"""K5 transfer test (pre-registered kill condition 5, ANNOTATION_SIDESTEP.md):
does detection trained on q8 teacher-forced traces transfer to bf16 traces?

The probe/direction stack is substrate-agnostic by construction (capture is
always teacher-forced through the local HF bf16 model; the substrate only
decides which tokens get replayed). K5 tests whether that holds empirically:
fit the mean-difference onset-vs-controls direction on ONE substrate's
captures, evaluate it on the OTHER substrate's captures. Fit and eval data
never overlap, so the cross number is honest; the within number is pooled
(fit+eval same data, labeled as such).

CPU-only. Reads capture dirs (--cap-dirs), substrates taken from the npz
filename prefix (q8_* / bf16_*). Writes data/transfer_test.json.

Usage: .venv/bin/python -m mlfactory.experiments.ace.annotate.transfer_test
       [--cap-dirs D1 D2 ...] [--conf clear|all] [--out data/transfer_test.json]
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ACE = HERE.parent
DATA = ACE / "data"
N_LAYERS = 32
MIN_SIDE = 10


def auroc(pos_scores, neg_scores) -> float:
    if len(pos_scores) < 1 or len(neg_scores) < 1:
        return float("nan")
    pos = np.asarray(pos_scores, dtype=np.float64)
    neg = np.asarray(neg_scores, dtype=np.float64)
    return float(np.sum([np.sum(neg < p) + 0.5 * np.sum(neg == p) for p in pos])
                 / (len(pos) * len(neg)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap-dirs", nargs="+",
                    default=[str(DATA / "annot_captures")])
    ap.add_argument("--conf", default="clear", choices=["clear", "all"])
    ap.add_argument("--out", default=str(DATA / "transfer_test.json"))
    args = ap.parse_args()

    # (substrate, class, layer) -> {"pos": [vecs], "neg": [vecs]}
    acc = defaultdict(lambda: {"pos": [], "neg": []})
    n_traces = defaultdict(int)
    for d in [Path(p) for p in args.cap_dirs]:
        for f in sorted(d.glob("*.npz")):
            sub = f.stem.split("_")[0]
            if sub not in ("q8", "bf16"):
                continue
            try:
                cap = np.load(f, allow_pickle=True)
            except Exception:
                continue
            resid = cap["residuals"]
            pt = json.loads(bytes(cap["pos_table"]).decode())
            pos_slots: dict[str, list[int]] = defaultdict(list)
            neg_slots = []
            for rec in pt:
                if rec["kind"] == "control":
                    neg_slots.append(rec["pos_idx"])
                elif rec["kind"] == "onset":
                    if args.conf == "clear" and rec.get("conf") != "clear":
                        continue
                    pos_slots[(rec["class"] or "").lower()].append(rec["pos_idx"])
            if not neg_slots:
                continue
            n_traces[sub] += 1
            neg = {L: resid[L][neg_slots].astype(np.float32)
                   for L in range(N_LAYERS)}
            for cls, slots in pos_slots.items():
                for L in range(N_LAYERS):
                    acc[(sub, cls, L)]["pos"].append(
                        resid[L][slots].astype(np.float32))
                    acc[(sub, cls, L)]["neg"].append(neg[L])

    subs = sorted(n_traces)
    if len(subs) < 2:
        raise SystemExit(f"need two substrates in the capture dirs, got {subs}")
    print(f"traces: " + ", ".join(f"{s}={n_traces[s]}" for s in subs))

    results = {"within": {}, "cross": {}, "direction_cosine": {},
               "conf": args.conf, "cap_dirs": args.cap_dirs}
    for cls in ("muse", "cycle", "loop"):
        print(f"\n===== {cls.upper()} =====")
        print(f"  {'L':>3} | " + " | ".join(
            [f"{a}->{a} within" for a in subs] +
            [f"{a}->{b} cross" for a in subs for b in subs if a != b]))
        for L in range(N_LAYERS):
            cells = {}
            dirs = {}
            for s in subs:
                b = acc.get((s, cls, L))
                if not b or not b["pos"]:
                    continue
                pos = np.concatenate(b["pos"], axis=0)
                neg = np.concatenate(b["neg"], axis=0)
                if len(pos) < MIN_SIDE or len(neg) < MIN_SIDE:
                    continue
                dv = pos.mean(0) - neg.mean(0)
                if np.linalg.norm(dv) == 0:
                    continue
                dirs[s] = dv / np.linalg.norm(dv)
                cells[f"{s}->{s}"] = auroc(list(pos @ dv), list(neg @ dv))
            for a in subs:
                for b in subs:
                    if a == b or a not in dirs or b not in dirs:
                        continue
                    bp = acc[(b, cls, L)]
                    pos = np.concatenate(bp["pos"], axis=0)
                    neg = np.concatenate(bp["neg"], axis=0)
                    cells[f"{a}->{b}"] = auroc(list(pos @ dirs[a]),
                                               list(neg @ dirs[a]))
            if not cells:
                continue
            for a in subs:
                for b in subs:
                    if a != b and f"{a}->{b}" in cells:
                        results["cross"][f"{cls}:{a}->{b}:L{L}"] = {
                            "auroc": round(cells[f"{a}->{b}"], 3)}
                    elif a == b and f"{a}->{a}" in cells:
                        results["within"][f"{cls}:{a}:L{L}"] = {
                            "auroc": round(cells[f"{a}->{a}"], 3)}
            for a in subs:
                for b in subs:
                    if a < b and a in dirs and b in dirs:
                        results["direction_cosine"][f"{cls}:{a}x{b}:L{L}"] = \
                            round(float(dirs[a] @ dirs[b]), 3)
            row = " | ".join(f"{cells.get(k, float('nan')):.3f}"
                             for k in [f"{s}->{s}" for s in subs] +
                             [f"{a}->{b}" for a in subs for b in subs if a != b])
            mark = ""
            cross_vals = [cells.get(f"{a}->{b}", float("nan"))
                          for a in subs for b in subs if a != b]
            if any(v == v and abs(v - 0.5) > 0.15 for v in cross_vals):
                mark = " <<<"
            print(f"  L{L:2d} | {row}{mark}")

    # headline per class: best cross AUROC across layers and directions
    print("\n===== headline (best cross-substrate AUROC per class) =====")
    for cls in ("muse", "cycle", "loop"):
        best = None
        for k, v in results["cross"].items():
            if not k.startswith(cls + ":"):
                continue
            if best is None or v["auroc"] > best[1]:
                best = (k, v["auroc"])
        if best:
            print(f"  {cls}: {best[0]} AUROC={best[1]:.3f}")
            results[f"best_cross_{cls}"] = {"key": best[0], "auroc": best[1]}
        else:
            print(f"  {cls}: no usable cross reading (insufficient n)")

    with open(args.out, "w") as fh:
        json.dump(results, fh, indent=1)
    print(f"\nresults -> {args.out}")


if __name__ == "__main__":
    main()
