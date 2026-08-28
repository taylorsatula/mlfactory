"""R3 directions: mean-difference steering directions per layer and class.

For every (class, layer) with enough data, the direction is
    d = mean(healthy control states) - mean(class onset states)
— the push that moves a divergence-onset state toward the healthy region.
These are the constant-lambda fixed-direction baselines that
TERMINAL_FORK_COMPUTE.md constraint 7 requires any learned controller to
beat; they are produced from the same R1 captures the probes scored.

CPU-only. Reads capture dirs (--cap-dirs, default data/annot_captures/)
plus probe_results.json for provenance, and writes
data/steering_directions/directions_annot_<conf>[_<tag>].npz with keys
dir_<class>_L<layer> plus a metadata json.

Usage: .venv/bin/python -m mlfactory.experiments.ace.annotate.compute_directions
       [--conf clear|all] [--cap-dirs D1 D2 ...] [--tag TAG] [--min-pos 5] [--min-auroc 0.6]
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
DEFAULT_CAP_DIRS = [DATA / "annot_captures"]
OUT_DIR = DATA / "steering_directions"
N_LAYERS = 32


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--conf", default="clear", choices=["clear", "all"])
    ap.add_argument("--cap-dirs", nargs="+", default=None,
                    help="capture dirs to pool (default: annot_captures only)")
    ap.add_argument("--tag", default="",
                    help="output suffix for pooled corpora "
                         "(directions_annot_<conf>_<tag>.npz)")
    ap.add_argument("--min-pos", type=int, default=5)
    ap.add_argument("--min-auroc", type=float, default=0.6,
                    help="only keep directions for layers the probe scored above this")
    ap.add_argument("--probe-results", default=str(DATA / "probe_results.json"))
    args = ap.parse_args()

    probe = json.loads(Path(args.probe_results).read_text()) if Path(args.probe_results).exists() else {"pooled": {}}

    # accumulate onset vectors per (class, layer); controls once per layer
    acc = {(cls, L): [] for cls in ("muse", "cycle", "loop") for L in range(N_LAYERS)}
    neg_by_layer: dict[int, list] = defaultdict(list)
    cap_dirs = [Path(d) for d in args.cap_dirs] if args.cap_dirs else DEFAULT_CAP_DIRS
    for f in sorted(f for d in cap_dirs for f in d.glob("*.npz")):
        try:
            d = np.load(f, allow_pickle=True)
        except Exception:
            continue
        resid = d["residuals"]
        pos_table = json.loads(bytes(d["pos_table"]).decode())
        for rec in pos_table:
            if rec["conf"] != "clear" and args.conf == "clear":
                continue
            cls = (rec["class"] or "").lower()
            v = resid[:, rec["pos_idx"]].astype(np.float32)
            if rec["kind"] == "onset" and cls in ("muse", "cycle", "loop"):
                for L in range(N_LAYERS):
                    acc[(cls, L)].append(v[L])
            elif rec["kind"] == "control":
                for L in range(N_LAYERS):
                    neg_by_layer[L].append(v[L])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    arrays = {}
    meta = {}
    n_kept = 0
    for (cls, L), pos_list in sorted(acc.items()):
        neg_list = neg_by_layer.get(L, [])
        if len(pos_list) < args.min_pos or len(neg_list) < args.min_pos:
            continue
        au = probe.get("pooled", {}).get(f"{cls}:onset:L{L}", {}).get("auroc")
        if au is not None and (1 - args.min_auroc) <= au <= args.min_auroc:
            continue  # probe showed no usable separation at this layer
        pos = np.stack(pos_list)
        neg = np.stack(neg_list)
        dvec = neg.mean(0) - pos.mean(0)
        nrm = np.linalg.norm(dvec)
        if nrm == 0:
            continue
        arrays[f"dir_{cls}_L{L}"] = (dvec / nrm).astype(np.float32)
        meta[f"{cls}:L{L}"] = {"n_pos": int(len(pos)), "n_neg": int(len(neg)),
                               "auroc": au, "norm": float(nrm),
                               "sign_convention": "direction points from divergence toward healthy"}
        n_kept += 1

    suffix = f"_{args.tag}" if args.tag else ""
    out = OUT_DIR / f"directions_annot_{args.conf}{suffix}.npz"
    np.savez(out, metadata=json.dumps(meta).encode(), **arrays)
    print(f"wrote {n_kept} directions -> {out}")
    for k in sorted(meta):
        m = meta[k]
        print(f"  {k}: n_pos={m['n_pos']} n_neg={m['n_neg']} auroc={m['auroc']}")


if __name__ == "__main__":
    main()
