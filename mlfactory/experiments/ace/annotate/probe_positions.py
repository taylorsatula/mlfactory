"""R2 probe: position-level separability of annotated reasoning episodes.

Loads the R1 captures (one or more capture dirs via --cap-dirs, default
data/annot_captures/) and tests, per layer and per class (muse/cycle/loop),
whether annotated span positions separate from matched healthy positions:

  1. pooled AUROC per (class, kind, layer) on the mean-difference direction
  2. within-trace AUROC distribution + sign consistency (the honest test —
     controls are depth-matched positions from the SAME trace, so prompt
     difficulty and trace-specific noise cancel)
  3. escape-vs-reheat at onset: do muse/cycle onsets in eventually-correct
     traces separate from onsets in eventually-failed traces at the same
     layer? (kill condition 3 of ANNOTATION_SIDESTEP.md)
  4. onset vs mid/end profile (kill condition 2: post-hoc-only separability)
  5. lead time: lookback kinds lb_<k> (state k tokens before onset) scored
     against decile-matched controls; how far back is divergence readable?
     (lb negs are depth-matched per position; core kinds use all controls,
     exactly as before the lookback extension)

Kill conditions adjudicated from the printed tables (see report at the end):
  K1 onset-null, K2 post-hoc-only, K3 escape==reheat, K4 label-mush
  (label-mush is judged from the pass2 agreement script, not here).

CPU-only, one capture in RAM at a time.

Usage: .venv/bin/python -m mlfactory.experiments.ace.annotate.probe_positions
       [--conf clear|all] [--cap-dirs D1 D2 ...] [--out data/probe_results.json]
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
N_LAYERS = 32
MIN_NEG = 3


def auroc(pos_scores, neg_scores) -> float:
    if len(pos_scores) < 1 or len(neg_scores) < 1:
        return float("nan")
    pos = np.asarray(pos_scores, dtype=np.float64)
    neg = np.asarray(neg_scores, dtype=np.float64)
    # Mann-Whitney: sum of win counts over all pairs, normalized once
    return float(np.sum([np.sum(neg < p) + 0.5 * np.sum(neg == p) for p in pos])
                 / (len(pos) * len(neg)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--conf", default="clear", choices=["clear", "all"])
    ap.add_argument("--cap-dirs", nargs="+", default=None,
                    help="capture dirs to pool (default: annot_captures only)")
    ap.add_argument("--out", default=str(DATA / "probe_results.json"))
    args = ap.parse_args()

    cap_dirs = [Path(d) for d in args.cap_dirs] if args.cap_dirs else DEFAULT_CAP_DIRS
    files = sorted(f for d in cap_dirs for f in d.glob("*.npz"))
    if not files:
        raise SystemExit(f"no captures in {cap_dirs} — run capture_activations first")

    # accumulators: (cls, kind, layer) -> {"pos": [...], "neg": [...]}
    pooled = defaultdict(lambda: {"pos": [], "neg": []})
    # per-trace vectors for leave-one-trace-out AUROC
    per_trace = defaultdict(lambda: {"pos": [], "neg": []})  # (cls,kind,L,trace) -> arrays
    # within-trace: (cls, kind, layer) -> list of per-trace (auroc, n_pos, n_neg)
    wtrace = defaultdict(list)
    # escape-vs-reheat: (cls, layer) -> {"succ": [...], "fail": [...]} onset vectors
    reh = defaultdict(lambda: {"succ": [], "fail": []})
    n_traces = 0
    n_pos_total = defaultdict(int)

    for f in files:
        try:
            d = np.load(f, allow_pickle=True)
        except Exception:
            continue
        resid = d["residuals"]  # (32, n_pos, hidden)
        pos_table = json.loads(bytes(d["pos_table"]).decode())
        n_traces += 1
        trace_pos_by: dict[tuple[str, str], list[int]] = defaultdict(list)  # (cls, kind) -> slot idx
        neg_slots: list[int] = []
        neg_by_decile: dict[int, list[int]] = defaultdict(list)
        for rec in pos_table:
            k = rec["kind"]
            cls = (rec["class"] or "").lower()
            if k == "control":
                neg_slots.append(rec["pos_idx"])
                if rec.get("decile") is not None:
                    neg_by_decile[rec["decile"]].append(rec["pos_idx"])
            elif k in ("onset", "mid", "end", "pre_onset") or k.startswith("lb_"):
                if args.conf == "clear" and rec.get("conf") != "clear":
                    continue
                trace_pos_by[(cls, k)].append(rec["pos_idx"])
        if not neg_slots:
            continue

        neg_cache: dict[tuple, dict] = {}
        for (cls, kind), slots in trace_pos_by.items():
            if kind.startswith("lb_"):
                # depth-matched negs: controls in the deciles of these
                # lookback positions (different anchor depths are not
                # comparable)
                decs = {pos_table[s].get("decile") for s in slots}
                kneg = sorted({i for dec in decs if dec is not None
                               for i in neg_by_decile.get(dec, [])})
                if len(kneg) < MIN_NEG:
                    continue
            else:
                kneg = neg_slots
            key = tuple(kneg)
            if key not in neg_cache:
                neg_cache[key] = {L: resid[L][kneg].astype(np.float32)
                                  for L in range(N_LAYERS)}
            neg_vecs = neg_cache[key]
            n_pos_total[(cls, kind)] += len(slots)
            for L in range(N_LAYERS):
                pv = resid[L][slots].astype(np.float32)
                pooled[(cls, kind, L)]["pos"].append(pv)
                pooled[(cls, kind, L)]["neg"].append(neg_vecs[L])
                per_trace[(cls, kind, L, n_traces)]["pos"].append(pv)
                per_trace[(cls, kind, L, n_traces)]["neg"].append(neg_vecs[L])
                # within-trace AUROC on mean-diff direction (same-trace fit)
                if len(slots) >= 1 and len(kneg) >= MIN_NEG:
                    dv = pv.mean(0) - neg_vecs[L].mean(0)
                    if np.linalg.norm(dv) > 0:
                        au = auroc(list(pv @ dv), list(neg_vecs[L] @ dv))
                        wtrace[(cls, kind, L)].append((au, len(slots), len(kneg)))

    # trace outcomes from each dir's manifest, keyed (dir name, npz stem):
    # (sub, pid, si) alone is ambiguous across corpora with shared pids
    outcomes = {}
    for d in cap_dirs:
        mf = d / "capture_manifest.jsonl"
        if mf.exists():
            for line in mf.open():
                m = json.loads(line)
                stem = Path(m["file"]).stem
                outcomes[(d.name, stem)] = m["outcome"]

    # escape-vs-reheat second pass (needs outcome)
    for f in files:
        try:
            d = np.load(f, allow_pickle=True)
        except Exception:
            continue
        resid = d["residuals"]
        pos_table = json.loads(bytes(d["pos_table"]).decode())
        outcome = outcomes.get((f.parent.name, f.stem))
        if outcome not in ("correct", "cap", "wrong"):
            continue
        side = "succ" if outcome == "correct" else "fail"
        for rec in pos_table:
            if rec["kind"] != "onset" or (rec["class"] or "").lower() not in ("muse", "cycle", "loop"):
                continue
            if args.conf == "clear" and rec.get("conf") != "clear":
                continue
            v = resid[:, rec["pos_idx"]].astype(np.float32)
            for L in range(N_LAYERS):
                reh[(rec["class"].lower(), L)][side].append(v[L])

    # ---------- report ----------
    print(f"captures: {n_traces} traces | conf={args.conf}")
    print("positive positions per (class, kind):",
          {f"{c}:{k}": n for (c, k), n in sorted(n_pos_total.items())})

    results = {"pooled": {}, "loo": {}, "escape_vs_reheat": {}}

    # leave-one-trace-out AUROC: direction from all OTHER traces, via
    # per-trace sums (O(traces x hidden), no re-concatenation)
    loo_scores = defaultdict(lambda: {"pos": [], "neg": []})
    by_ckl = defaultdict(list)
    for (cls, kind, L, tr), b in per_trace.items():
        if b["pos"] and b["neg"]:
            by_ckl[(cls, kind, L)].append((b["pos"][0], b["neg"][0]))
    for (cls, kind, L), entries in by_ckl.items():
        if len(entries) < 2:
            continue
        sum_p = np.stack([e[0].sum(0) for e in entries])
        n_p = np.array([e[0].shape[0] for e in entries])
        sum_n = np.stack([e[1].sum(0) for e in entries])
        n_n = np.array([e[1].shape[0] for e in entries])
        tot_p, tot_n = sum_p.sum(0), sum_n.sum(0)
        Np, Nn = n_p.sum(), n_n.sum()
        for i, (pv, nv) in enumerate(entries):
            if Np - n_p[i] < 1 or Nn - n_n[i] < 1:
                continue
            dv = (tot_p - sum_p[i]) / (Np - n_p[i]) - (tot_n - sum_n[i]) / (Nn - n_n[i])
            if np.linalg.norm(dv) == 0:
                continue
            loo_scores[(cls, kind, L)]["pos"].extend((pv @ dv).tolist())
            loo_scores[(cls, kind, L)]["neg"].extend((nv @ dv).tolist())
        if loo_scores[(cls, kind, L)]["pos"]:
            au_loo = auroc(loo_scores[(cls, kind, L)]["pos"],
                           loo_scores[(cls, kind, L)]["neg"])
            results["loo"][f"{cls}:{kind}:L{L}"] = {"auroc": round(au_loo, 3)}

    lb_kinds = sorted({k for (c, k) in n_pos_total if k.startswith("lb_")},
                      key=lambda s: int(s.split("_")[1]))
    for cls in ("muse", "cycle", "loop"):
        print(f"\n===== {cls.upper()} =====")
        for kind in ("pre_onset", "onset", "mid", "end", *lb_kinds):
            rows = []
            for L in range(N_LAYERS):
                b = pooled.get((cls, kind, L))
                if not b or not b["pos"]:
                    continue
                pos = np.concatenate(b["pos"], axis=0)
                neg = np.concatenate(b["neg"], axis=0)
                if len(pos) < 3 or len(neg) < MIN_NEG:
                    continue
                dv = pos.mean(0) - neg.mean(0)
                au = auroc(list(pos @ dv), list(neg @ dv))
                wt = wtrace.get((cls, kind, L), [])
                loo_au = results["loo"].get(f"{cls}:{kind}:L{L}", {}).get("auroc", float("nan"))
                rows.append((L, au, loo_au, len(wt)))
                results["pooled"][f"{cls}:{kind}:L{L}"] = {
                    "auroc": round(au, 3), "loo_auroc": loo_au,
                    "n_pos": int(len(pos)), "n_neg": int(len(neg)),
                }
            if rows:
                print(f"  -- {kind} (pooled AUROC | leave-one-trace-out AUROC | n_traces)")
                for L, au, loo_au, nt in rows:
                    typ = "FULL" if L % 4 == 3 else "lin "
                    cons = "" if loo_au != loo_au else (" <<<" if abs(loo_au - 0.5) > 0.15 else "")
                    print(f"    L{L:2d} ({typ}): {au:.3f} | {loo_au:.3f} | {nt}{cons}")

    print("\n===== escape vs reheat at onset (succ-trace onsets vs fail-trace onsets) =====")
    for cls in ("muse", "cycle", "loop"):
        rows = []
        for L in range(N_LAYERS):
            b = reh.get((cls, L))
            if not b or len(b["succ"]) < 3 or len(b["fail"]) < 3:
                continue
            s = np.stack(b["succ"]).astype(np.float32)
            fl = np.stack(b["fail"]).astype(np.float32)
            dv = s.mean(0) - fl.mean(0)
            au = auroc(list(s @ dv), list(fl @ dv))
            rows.append((L, au, len(s), len(fl)))
            results["escape_vs_reheat"][f"{cls}:L{L}"] = {
                "auroc": round(au, 3), "n_succ": int(len(s)), "n_fail": int(len(fl))}
        if rows:
            print(f"  -- {cls}")
            for L, au, ns, nf in rows:
                mark = " <<<" if abs(au - 0.5) > 0.15 else ""
                print(f"    L{L:2d}: AUROC={au:.3f} (n_succ={ns}, n_fail={nf}){mark}")

    if lb_kinds:
        print("\n===== lead time: separability k tokens BEFORE onset "
              "(LOO AUROC, decile-matched controls) =====")
        results["lead_time"] = {}
        focal = {"cycle": 18, "loop": 2, "muse": 17}
        for cls in ("muse", "cycle", "loop"):
            print(f"  -- {cls} (focal layer L{focal[cls]})")
            for kind in lb_kinds:
                cands = [(int(kk.split(":L")[1]), v["auroc"])
                         for kk, v in results["loo"].items()
                         if kk.startswith(f"{cls}:{kind}:L")
                         and v["auroc"] == v["auroc"]]
                if not cands:
                    continue
                L, au = max(cands, key=lambda t: t[1])
                n_pos = results["pooled"].get(f"{cls}:{kind}:L{L}", {}).get("n_pos")
                f_au = results["loo"].get(f"{cls}:{kind}:L{focal[cls]}",
                                          {}).get("auroc", float("nan"))
                results["lead_time"][f"{cls}:{kind}"] = {
                    "best_L": L, "loo_auroc": au, "n_pos": n_pos,
                    "focal_L": focal[cls], "focal_loo_auroc": f_au}
                print(f"    {kind:5s}: best L{L:2d} LOO={au:.3f} (n={n_pos})"
                      f" | focal L{focal[cls]} LOO={f_au:.3f}")

    with open(args.out, "w") as fh:
        json.dump(results, fh, indent=1)
    print(f"\nresults -> {args.out}")


if __name__ == "__main__":
    main()
