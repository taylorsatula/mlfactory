"""Rec-channel probe: onset-vs-controls separability in the DeltaNet
recurrent states (captured at REC_LAYERS [2, 8, 9, 12, 20], onset +
onset-anchor control positions only — see capture_activations.py).

The residual-channel probe (probe_positions.py) scores the 32 residual
layers; the rec channel was captured alongside but never scored. This is
that scoring: per class x rec layer, onset states vs same-trace control
states, with pooled, within-trace, and leave-one-trace-out AUROC.

Memory note: rec states are large (32x128x128 per row), so the corpus is
processed one rec layer at a time, per-trace rows held fp16.

CPU-only. Reads capture dirs (--cap-dirs). Writes data/probe_recurrent.json.

Usage: .venv/bin/python -m mlfactory.experiments.ace.annotate.probe_recurrent
       [--cap-dirs D1 D2 ...] [--conf clear|all] [--out data/probe_recurrent.json]
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
REC_LAYERS = [2, 8, 9, 12, 20]
MIN_NEG = 3


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
    ap.add_argument("--out", default=str(DATA / "probe_recurrent.json"))
    args = ap.parse_args()

    files = sorted(f for d in (Path(p) for p in args.cap_dirs)
                   for f in d.glob("*.npz"))
    if not files:
        raise SystemExit("no captures found")

    results = {"pooled": {}, "within": {}, "loo": {}, "conf": args.conf,
               "cap_dirs": args.cap_dirs}

    for li in REC_LAYERS:
        key = f"rec_L{li}"
        traces = []  # (pos rows flat fp16, neg rows flat fp16) per class
        for f in files:
            try:
                cap = np.load(f, allow_pickle=True)
            except Exception:
                continue
            if key not in cap.files:
                continue
            rec = cap[key]  # (n_rows, 32, 128, 128) fp16
            if rec.ndim != 4 or rec.shape[0] == 0:
                continue
            pt = json.loads(bytes(cap["pos_table"]).decode())
            ridx = cap["rec_pos_idx"]
            pos_by: dict[str, list[int]] = defaultdict(list)
            neg_rows: list[int] = []
            for row, pi in enumerate(ridx):
                p = pt[pi]
                if p["kind"] == "control":
                    neg_rows.append(row)
                elif p["kind"] == "onset":
                    if args.conf == "clear" and p.get("conf") != "clear":
                        continue
                    pos_by[(p["class"] or "").lower()].append(row)
            if not neg_rows:
                continue
            flat = rec.reshape(rec.shape[0], -1)  # fp16 view
            traces.append((flat, {c: r for c, r in pos_by.items()},
                           neg_rows))
        if not traces:
            continue

        print(f"\n===== rec_L{li} ({len(traces)} traces with material) =====")
        for cls in ("muse", "cycle", "loop"):
            entries = []  # (sum_pos, n_pos, sum_neg, n_neg, pos_flat, neg_flat)
            within = []
            for flat, pos_by, neg_rows in traces:
                rows = pos_by.get(cls)
                if not rows:
                    continue
                pos = flat[rows]
                neg = flat[neg_rows]
                if len(neg) < MIN_NEG:
                    continue
                pf = pos.astype(np.float32)
                nf = neg.astype(np.float32)
                entries.append((pf.sum(0), len(pf), nf.sum(0), len(nf),
                                pf, nf))
                dv = pf.mean(0) - nf.mean(0)
                if np.linalg.norm(dv) > 0:
                    within.append(auroc(list(pf @ dv), list(nf @ dv)))
            if len(entries) < 2:
                continue
            tot_p = np.sum([e[0] for e in entries], axis=0)
            tot_n = np.sum([e[2] for e in entries], axis=0)
            Np = sum(e[1] for e in entries)
            Nn = sum(e[3] for e in entries)
            # pooled (fit+eval same data)
            dv = tot_p / Np - tot_n / Nn
            pooled_au = float("nan")
            if np.linalg.norm(dv) > 0:
                all_p = np.concatenate([e[4] for e in entries])
                all_n = np.concatenate([e[5] for e in entries])
                pooled_au = auroc(list(all_p @ dv), list(all_n @ dv))
            # leave-one-trace-out
            loo_p, loo_n = [], []
            for sp, np_, sn, nn, pf, nf in entries:
                if Np - np_ < 1 or Nn - nn < 1:
                    continue
                dv = (tot_p - sp) / (Np - np_) - (tot_n - sn) / (Nn - nn)
                if np.linalg.norm(dv) == 0:
                    continue
                loo_p.extend((pf @ dv).tolist())
                loo_n.extend((nf @ dv).tolist())
            loo_au = auroc(loo_p, loo_n) if loo_p else float("nan")
            results["pooled"][f"{cls}:L{li}"] = {"auroc": round(pooled_au, 3),
                                                 "n_pos": int(Np), "n_neg": int(Nn)}
            results["within"][f"{cls}:L{li}"] = {
                "median": round(float(np.median(within)), 3) if within else None,
                "n_traces": len(within),
                "frac_above_0.5": round(float(np.mean(np.array(within) > 0.5)), 3)
                if within else None}
            results["loo"][f"{cls}:L{li}"] = {"auroc": round(loo_au, 3),
                                              "n_traces": len(entries)}
            mark = " <<<" if loo_au == loo_au and abs(loo_au - 0.5) > 0.15 else ""
            print(f"  {cls:5s}: pooled={pooled_au:.3f} (n_pos={Np}) | "
                  f"within median={np.median(within):.3f} "
                  f"({np.mean(np.array(within) > 0.5):.0%} of {len(within)} traces >0.5) | "
                  f"LOO={loo_au:.3f} ({len(entries)} traces){mark}")

    with open(args.out, "w") as fh:
        json.dump(results, fh, indent=1)
    print(f"\nresults -> {args.out}")


if __name__ == "__main__":
    main()
