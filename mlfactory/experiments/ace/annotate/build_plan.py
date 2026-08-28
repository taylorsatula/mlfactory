"""Build the trace-centric annotation plan for the xsub corpus.

Pairs every collected trace with exactly one sibling from the same
prompt + substrate, contrast-priority: correct<->cap first, then
correct<->wrong, cap<->wrong, then within-class leftovers. All lists
are consumed in sample_i order, so pairing is deterministic.

Marks the R0 double-annotation subset: five pairs spread across
families and substrates (one q8 pair per domain, two bf16 pairs).

Output: data/annotation_plan_xsub.jsonl — one row per pair.

Usage (from repo root):
    mlfactory/experiments/ace/.venv/bin/python \
        -m mlfactory.experiments.ace.annotate.build_plan
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ACE = HERE.parent
DATA = ACE / "data"

DEFAULT_CORPUS = {
    "q8": [DATA / "xsub_q8.jsonl"],
    "bf16": [DATA / "xsub_bf16_gpu0.jsonl", DATA / "xsub_bf16_gpu1.jsonl"],
}

# deterministic double-annotation subset for the default xsub corpus:
# first pair of each picked (substrate, pid) — five pairs, ten traces
DOUBLE_PICKS = [("q8", 53), ("q8", 140), ("q8", 150), ("bf16", 53), ("bf16", 140)]


def outcome(r: dict) -> str:
    if r["correct"]:
        return "correct"
    if r["truncated"]:
        return "cap"
    return "wrong"


def substrate_of(row: dict) -> str:
    """Substrate tag from the row's quant field (Q8* -> q8, else bf16)."""
    return "q8" if str(row.get("quant", "")).startswith("Q8") else "bf16"


def load_corpus_files(files: list[Path]) -> dict[str, list[dict]]:
    """Group rows by substrate tag, preserving file order."""
    by_sub: dict[str, list[dict]] = {}
    for f in files:
        for line in f.open():
            r = json.loads(line)
            by_sub.setdefault(substrate_of(r), []).append(r)
    return by_sub


def pair_samples(rows: list[dict]) -> list[tuple[dict, dict]]:
    """Contrast-priority deterministic pairing of one prompt's samples."""
    by_class = {"correct": [], "cap": [], "wrong": []}
    for r in sorted(rows, key=lambda r: r["sample_i"]):
        by_class[outcome(r)].append(r)

    pairs: list[tuple[dict, dict]] = []
    order = [("correct", "cap"), ("correct", "wrong"), ("cap", "wrong"),
             ("correct", "correct"), ("cap", "cap"), ("wrong", "wrong")]
    for a, b in order:
        while by_class[a] and by_class[b]:
            pairs.append((by_class[a].pop(0), by_class[b].pop(0)))
    leftovers = [r for cls in ("correct", "cap", "wrong") for r in by_class[cls]]
    assert not leftovers, f"unpaired samples: {[r['sample_id'] for r in leftovers]}"
    return pairs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", nargs="*", default=None,
                    help="corpus jsonl files (default: the xsub files)")
    ap.add_argument("--out", default=str(DATA / "annotation_plan_xsub.jsonl"))
    ap.add_argument("--double-per-domain", type=int, default=5,
                    help="R0 subset: first contrast pair of the first N "
                         "distinct domains (ignored when --double-picks given)")
    args = ap.parse_args()

    if args.corpus:
        corpus = load_corpus_files([Path(f) for f in args.corpus])
    else:
        corpus = {sub: [json.loads(l) for f in files for l in f.open()]
                  for sub, files in DEFAULT_CORPUS.items()}

    plan = []
    coverage = set()
    double_done = set()
    for sub, rows in corpus.items():
        by_pid: dict[int, list[dict]] = {}
        for r in rows:
            by_pid.setdefault(r["proposal_id"], []).append(r)
        for pid in sorted(by_pid):
            domain = by_pid[pid][0]["domain"]
            for i, (a, b) in enumerate(pair_samples(by_pid[pid])):
                sa, sb = sorted((a["sample_i"], b["sample_i"]))
                lo, hi = (a, b) if a["sample_i"] == sa else (b, a)
                pair_id = f"x{sub}_p{pid}_s{sa}s{sb}"
                plan.append({
                    "pair_id": pair_id,
                    "substrate": sub,
                    "pid": pid,
                    "domain": domain,
                    "task": lo["task"],
                    "samples": [
                        {"sample_i": lo["sample_i"], "outcome": outcome(lo),
                         "trace_chars": len(lo["completion"])},
                        {"sample_i": hi["sample_i"], "outcome": outcome(hi),
                         "trace_chars": len(hi["completion"])},
                    ],
                    "framing": "C",
                    "double": False,
                })
                coverage.update([(sub, pid, lo["sample_i"]), (sub, pid, hi["sample_i"])])

    # R0 double subset. xsub keeps its fixed picks; a generic corpus gets
    # one pair per domain (first mixed-outcome pair, else first pair).
    if args.corpus:
        seen = []
        for row in plan:
            if row["domain"] not in seen:
                seen.append(row["domain"])
        for dom in seen[:args.double_per_domain]:
            rows = [r for r in plan if r["domain"] == dom]
            mixed = [r for r in rows
                     if len({s["outcome"] for s in r["samples"]}) > 1]
            (mixed[0] if mixed else rows[0])["double"] = True
    else:
        for row in plan:
            if (row["substrate"], row["pid"]) in DOUBLE_PICKS:
                row["double"] = True
                double_done.add((row["substrate"], row["pid"]))
        # keep only the first marked pair per pick (pairs are in pid order)
        for pick in DOUBLE_PICKS:
            marked = [r for r in plan
                      if (r["substrate"], r["pid"]) == pick and r["double"]]
            for r in marked[1:]:
                r["double"] = False

    # guard: every corpus trace covered exactly once
    all_ids = {(substrate_of(r), r["proposal_id"], r["sample_i"])
               for rows in corpus.values() for r in rows}
    assert coverage == all_ids, f"coverage mismatch: {len(coverage)} vs {len(all_ids)}"
    assert len(plan) == len(all_ids) // 2
    if not args.corpus:
        assert double_done == set(DOUBLE_PICKS), "double subset incomplete"

    out = Path(args.out)
    with out.open("w") as f:
        for row in plan:
            f.write(json.dumps(row) + "\n")

    n_double = sum(1 for r in plan if r["double"])
    contrast = sum(1 for r in plan
                   if len({s["outcome"] for s in r["samples"]}) > 1)
    print(f"wrote {out.name}: {len(plan)} pairs covering {len(coverage)} traces")
    print(f"contrast pairs (mixed outcomes): {contrast}/{len(plan)}")
    print(f"double-annotation subset: {n_double} pairs "
          f"({[r['pair_id'] for r in plan if r['double']]})")


if __name__ == "__main__":
    main()
