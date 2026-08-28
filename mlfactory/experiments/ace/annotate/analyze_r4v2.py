#!/usr/bin/env python3
"""R4v2 verdict unblinding + aggregation.

Reads judge verdict JSONL, unblinds labels to arms via the recorded
label_to_arm map, and reports:
  * per-triplet arm ordering (rank groups)
  * aggregate: win rates, mean rank, mode distribution per arm
  * gold-anchor agreement (hand-read verdicts from the v1 partial
    trace report — the judge hillclimb's calibration target)

This is read-only analysis; it writes nothing but stdout (and an
optional per-triplet dump).

Run as a module from the repo root:
  python -m mlfactory.experiments.ace.annotate.analyze_r4v2 \
      --verdicts data/judge_r4v2_pilot.jsonl
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ACE = HERE.parent

# Calibration anchors — WINDOW-BASED. Original anchors came from
# full-trace reads (lab_notes/2026-08-28-r4-partial-trace-report.md);
# on 2026-08-28 both "tie" anchors were falsified by blind window
# reads (the 2048 windows diverge ~80-85% and the steered branch is
# more productive in-window — the terminal-similarity that made them
# "ties" is exactly the conflation R4v2 abolishes). Anchors below are
# verified by independent blind reads of the windows themselves.
GOLD = {
    ("r4_cycle_00", 0): ("toward_healthy", "pilot2 read: healthy commits "
                         "early + new analysis; noop re-derives tail "
                         "content, never commits"),
    ("r4_cycle_00", 1): ("toward_healthy", "CLOSE — healthy prunes with "
                         "the external constraint; genuinely borderline"),
    ("r4_cycle_00", 5): ("tie", "pilot2 read: healthy fixates on CMDS "
                         "format but adds M2-chain/length-5 structure; "
                         "noop stays on substance but only re-derives"),
    ("r4_cycle_03", 0): ("toward_healthy", "healthy converges clean; "
                         "noop lost in re-verification"),
    ("r4_cycle_03", 2): ("toward_healthy", "old 'tie' anchor falsified: "
                         "healthy adds a durable final-state snapshot; "
                         "noop only re-verifies"),
    ("r4_cycle_02", 0): ("toward_healthy", "BORDERLINE — pilot2 read: "
                         "healthy finds H3 early but backslides into the "
                         "flawed subtraction; noop converges late by "
                         "catching its own error"),
}


def _pass_ranks(label_to_arm: dict, parsed: dict) -> dict[str, float] | None:
    """arm -> rank position for one judge pass (0 = best; ties share
    the group's mean)."""
    if not parsed or not isinstance(parsed.get("ranking"), list):
        return None
    ranks: dict[str, float] = {}
    pos = 0
    for group in parsed["ranking"]:
        if not isinstance(group, list):
            group = [group]
        labs = [g for g in group if g in label_to_arm]
        if not labs:
            continue
        mean = pos + (len(labs) - 1) / 2
        for lab in labs:
            ranks[label_to_arm[lab]] = mean
        pos += len(labs)
    return ranks if ranks else None


def passes(verdict: dict) -> list[dict]:
    """Both schemas: ensemble rows carry 'passes'; rubric-v1 rows are a
    single pass inline."""
    if "passes" in verdict:
        return verdict["passes"]
    return [{"label_to_arm": verdict.get("label_to_arm", {}),
             "parsed": verdict.get("parsed"),
             "raw": verdict.get("raw", "")}]


def arm_ranks(verdict: dict) -> dict[str, float] | None:
    """arm -> rank averaged over the parseable passes of the ensemble."""
    acc: dict[str, list[float]] = {}
    for p in passes(verdict):
        r = _pass_ranks(p["label_to_arm"], p.get("parsed"))
        if not r:
            continue
        for arm, v in r.items():
            acc.setdefault(arm, []).append(v)
    if not acc:
        return None
    return {arm: sum(v) / len(v) for arm, v in acc.items()}


def pair_verdict(ranks: dict[str, float]) -> str:
    """healthy-vs-noop: which way does the judge say it went."""
    if "toward_healthy" not in ranks or "noop" not in ranks:
        return "missing"
    if ranks["toward_healthy"] < ranks["noop"]:
        return "toward_healthy"
    if ranks["noop"] < ranks["toward_healthy"]:
        return "noop"
    return "tie"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verdicts", required=True)
    ap.add_argument("--show", action="store_true",
                    help="print per-triplet characterizations")
    args = ap.parse_args()

    verdicts = [json.loads(l) for l in Path(args.verdicts).open()]
    parsed = [v for v in verdicts if arm_ranks(v)]
    print(f"verdicts: {len(verdicts)} | parseable: {len(parsed)}")
    if not parsed:
        return

    wins = Counter()
    pair_outcomes = Counter()
    modes: dict[str, Counter] = {a: Counter() for a in
                                 ("noop", "toward_healthy", "toward_diverge")}
    rank_sum: dict[str, list[float]] = {}
    pos_rank: dict[str, list[float]] = {lab: [] for lab in ("A", "B", "C")}
    for v in parsed:
        ranks = arm_ranks(v)
        best_rank = min(ranks.values())
        for arm, r in ranks.items():
            if r == best_rank:
                wins[arm] += 1
            rank_sum.setdefault(arm, []).append(r)
        pair_outcomes[pair_verdict(ranks)] += 1
        for p in passes(v):
            pr = _pass_ranks(p["label_to_arm"], p.get("parsed"))
            if not pr:
                continue
            for lab, arm in p["label_to_arm"].items():
                if arm in pr:
                    pos_rank[lab].append(pr[arm])
            for lab, info in (p.get("parsed") or {}).items():
                arm = p["label_to_arm"].get(lab)
                if arm and isinstance(info, dict):
                    modes[arm][info.get("mode", "?")] += 1
        if args.show:
            print(f"\n{v['state_id']} seed {v['seed_i']} "
                  f"(avg ranks: " +
                  ", ".join(f"{a[:4]}={r:.2f}" for a, r in
                             sorted(ranks.items(), key=lambda x: x[1])) + "):")
            for p in passes(v):
                pp = p.get("parsed") or {}
                print(f"  pass: " + "  ".join(
                    f"{lab}={p['label_to_arm'].get(lab, '?')[:4]}:" +
                    f"[{(pp.get(lab) or {}).get('mode', '?')}]"
                    for lab in ("A", "B", "C"))
                    + f"  ranking={pp.get('ranking')}")

    n = len(parsed)
    print(f"\nwins (share a best-rank group): "
          + "  ".join(f"{a}={wins[a]}/{n}" for a in
                      ("noop", "toward_healthy", "toward_diverge")))
    print("mean rank (0=best): "
          + "  ".join(f"{a}={sum(rank_sum[a])/len(rank_sum[a]):.2f}"
                      for a in ("noop", "toward_healthy", "toward_diverge")
                      if a in rank_sum))
    print(f"healthy-vs-noop outcomes: {dict(pair_outcomes)}")
    print("residual position bias (mean rank per label, ensemble-pass level; "
          "equal = cancelled): "
          + "  ".join(f"{lab}={sum(pos_rank[lab])/len(pos_rank[lab]):.2f}"
                      for lab in ("A", "B", "C") if pos_rank[lab]))
    print("modes per arm:")
    for arm in ("noop", "toward_healthy", "toward_diverge"):
        print(f"  {arm}: {dict(modes[arm])}")

    print("\ngold-anchor agreement:")
    agree = 0
    seen = 0
    for (state, seed), (expect, why) in GOLD.items():
        v = next((v for v in parsed
                  if v["state_id"] == state and v["seed_i"] == seed), None)
        if v is None:
            print(f"  {state} seed {seed}: (no verdict yet)")
            continue
        seen += 1
        got = pair_verdict(arm_ranks(v))
        ok = (got == expect)
        agree += ok
        print(f"  {state} seed {seed}: expect {expect} | judge {got} "
              f"| {'OK' if ok else 'MISMATCH'}  ({why})")
    if seen:
        print(f"agreement: {agree}/{seen}")


if __name__ == "__main__":
    main()
