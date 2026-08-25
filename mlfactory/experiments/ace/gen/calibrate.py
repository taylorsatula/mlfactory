"""Calibration loop: score probe rollouts with STRICT family verifiers,
classify per-prompt band membership, emit the accepted pool.

The probe collector's `correct` field is a soft substring match and is
advisory only. This tool re-scores every completion with the family's
strict check(), then classifies:

    DEAD-HARD   0 of n correct        -> regenerate at easier knobs
    LIVE        1..n-1 correct        -> ACCEPT (gradient exists)
    DEAD-EASY   n of n correct        -> regenerate at harder knobs
    (preferred band: 2..5 of 8 — robust to upward controller drift)

Usage (from mlfactory/experiments/ace):
    .venv/bin/python -m gen.calibrate \
        --candidates data/acegen_probe_b1.jsonl \
        --probe data/acegen_probe_b1_rollouts.jsonl \
        --accept-out data/acegen_live_b1.jsonl
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from . import adversary, assign, certify, construct, grid, hypothesis, \
    machine, revise

CHECK = {
    "assign": assign.check,
    "machine": machine.check,
    "adversary": adversary.check,
    "certify": certify.check,
    "grid": grid.check,
    "hypothesis": hypothesis.check,
    "construct": construct.check,
    "revise": revise.check,
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--candidates", type=Path, required=True)
    p.add_argument("--probe", type=Path, required=True)
    p.add_argument("--accept-out", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    cfg = parse_args()
    cands = {json.loads(l)["provenance"]["proposal_id"]: json.loads(l)
             for l in cfg.candidates.read_text().splitlines() if l.strip()}
    rolls = [json.loads(l)
             for l in cfg.probe.read_text().splitlines() if l.strip()]

    by_pid = defaultdict(list)
    for r in rolls:
        by_pid[int(r["proposal_id"])].append(r)

    accepted = []
    stats = defaultdict(list)
    print(f'{"pid":>4} {"family":12} {"strict":>8} {"soft":>8} band')
    for pid in sorted(by_pid):
        group = by_pid[pid]
        cand = cands.get(pid)
        if cand is None:
            continue
        fam = cand["domain"]
        ref = cand["problem"]["reference_answer"]
        knobs = cand.get("knobs", {})
        strict = sum(1 for r in group
                     if CHECK[fam](r["completion"], ref, knobs))
        soft = sum(1 for r in group if r["correct"])
        n = len(group)
        if strict == 0:
            band = "DEAD-HARD"
        elif strict == n:
            band = "DEAD-EASY"
        else:
            band = "LIVE" + ("*" if 2 <= strict <= max(n - 3, 2) else "")
            accepted.append(cand)
        stats[fam].append((strict, n))
        print(f"{pid:>4} {fam:12} {strict:>4}/{n:<3} {soft:>4}/{n:<3} {band}")

    print("\nby family (strict):")
    for fam, pairs in sorted(stats.items()):
        k = sum(s for s, _ in pairs)
        n = sum(nn for _, nn in pairs)
        live = sum(1 for s, nn in pairs if 0 < s < nn)
        print(f"  {fam:12} {k:>3}/{n:<3} ({100*k/n:5.1f}%)  "
              f"live prompts: {live}/{len(pairs)}")

    if cfg.accept_out:
        cfg.accept_out.parent.mkdir(parents=True, exist_ok=True)
        with cfg.accept_out.open("w", encoding="utf-8") as fh:
            for c in accepted:
                fh.write(json.dumps(c, ensure_ascii=False) + "\n")
        print(f"\naccepted {len(accepted)} live prompts -> {cfg.accept_out}")


if __name__ == "__main__":
    main()
