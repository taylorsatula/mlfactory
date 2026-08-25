#!/usr/bin/env python3
"""Branch-ledger profiling of ACE traces (text-level, CPU).

Segments each completion into reasoning episodes via discourse markers and
computes the branch-level profile the doctrine actually asks about:

  opens        : "Alternatively" / "what if" / "another approach" — new branches
  waits        : "Wait" moments — reconsideration events
  verifies     : "verify" / "check" / "double check" / "confirm" episodes
  verify_dup   : fraction of verification episodes that are near-duplicates
                 (3-gram Jaccard >= 0.5) of an EARLIER verification episode
                 — the re-verification loop measure
  prune_durable: of explicit eliminations ("rule out", "eliminate", "cannot be",
                 "not possible"), fraction whose rejected option string never
                 recurs later in the trace
  closure_att  : closure-intent phrases ("final answer", "I'll write", ...)
  escaped      : trace ends shortly after last closure intent (emission worked)

Outcome separation is reported within-prompt where possible. Text markers are
a shallow proxy; the stored entropy/hidden states are the validation layer
(spike/collapse alignment with marker positions) for a later pass.

Usage: python3 -m mlfactory.experiments.ace.analysis.branch_ledger [rows.jsonl ...]
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

OPEN_RE = re.compile(r"\b(alternatively|another approach|what if|let me try|"
                     r"different (approach|route|way)|instead,)\b", re.I)
WAIT_RE = re.compile(r"\bwait\b", re.I)
VERIFY_RE = re.compile(r"\b(let'?s|let me|i'?ll|i will|we)\s+(double[- ]?check|verify|"
                       r"check|confirm|re-?check|re-?verify)\b|\bverification\b", re.I)
ELIM_RE = re.compile(r"\b(rule[sd]? out|eliminate[sd]?|cannot be|can'?t be|"
                     r"not possible|impossible|invalid)\b", re.I)
CLOSE_RE = re.compile(r"(final answer|i'?ll write|i will (now )?(write|output|present)|"
                      r"let me (now )?(write|output|present)|ready to (write|output)|"
                      r"i'?m (ready|confident)|formulate the (response|answer))", re.I)


def ngrams(text: str, n: int = 3) -> set[tuple]:
    w = text.lower().split()
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}


def ledger(text: str) -> dict:
    think = text.split("</think>")[0]
    opens = [m.start() for m in OPEN_RE.finditer(think)]
    waits = [m.start() for m in WAIT_RE.finditer(think)]
    vms = list(VERIFY_RE.finditer(think))
    # verification episode = 400 chars from marker
    eps = [think[m.start():m.start() + 400] for m in vms]
    grams = [ngrams(e) for e in eps]
    dup = 0
    for i, g in enumerate(grams):
        if not g:
            continue
        for j in range(i):
            if not grams[j]:
                continue
            jac = len(g & grams[j]) / len(g | grams[j])
            if jac >= 0.5:
                dup += 1
                break
    elims = [m.start() for m in ELIM_RE.finditer(think)]
    closes = [m.start() for m in CLOSE_RE.finditer(think)]
    # durable prune: after an elimination, does an elimination marker recur?
    # (proxy: eliminations whose 100-char context doesn't reappear later)
    durable = 0
    for pos in elims:
        frag = " ".join(think[max(0, pos - 60):pos + 60].lower().split())[:80]
        if frag and frag not in " ".join(think[pos + 400:].lower().split()):
            durable += 1
    last_close = closes[-1] if closes else -1
    escaped = bool(closes) and (len(text) - last_close) < 4000
    return {
        "opens": len(opens), "waits": len(waits), "verifies": len(vms),
        "verify_dup": dup / len(vms) if vms else 0.0,
        "elims": len(elims),
        "prune_durable": durable / len(elims) if elims else 0.0,
        "closure_att": len(closes), "escaped": escaped,
        "think_chars": len(think),
    }


def rank_biserial(a, b):
    x = sorted([(v, 1) for v in a] + [(v, 0) for v in b])
    pos, i = [0.0] * len(x), 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and x[j + 1][0] == x[i][0]:
            j += 1
        r = (i + j) / 2 + 1
        for k in range(i, j + 1):
            pos[k] = r
        i = j + 1
    n1, n0 = len(a), len(b)
    if not n1 or not n0:
        return float("nan")
    r1 = sum(p for p, (_, g) in zip(pos, x) if g == 1)
    return 2 * (r1 - n1 * (n1 + 1) / 2) / (n1 * n0) - 1


def main() -> None:
    files = sys.argv[1:] or ["data/acegen_probe_b1_gpu0.jsonl",
                             "data/acegen_probe_b1_gpu1.jsonl"]
    rows, seen = [], set()
    for f in files:
        for line in Path(f).read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            k = (r["proposal_id"], r["sample_i"])
            if k not in seen:
                seen.add(k)
                rows.append(r)
    # clean finishes only — crashouts excluded by instruction
    rows = [r for r in rows if not r["truncated"]]
    L = [{**ledger(r["completion"]), "correct": bool(r["correct"]),
          "pid": r["proposal_id"], "domain": r["domain"],
          "tokens": r["n_new_tokens"]} for r in rows]
    print(f"clean traces: {len(L)}")
    keys = ["opens", "waits", "verifies", "verify_dup", "elims",
            "prune_durable", "closure_att", "tokens"]

    print("\n== outcome separation, branch ledger (clean only, pooled) ==")
    print(f"{'feature':14s} {'corr':>7s} {'wrong':>7s} {'rankbis':>8s}")
    import statistics
    for k in keys:
        a = [f[k] for f in L if f["correct"]]
        b = [f[k] for f in L if not f["correct"]]
        print(f"{k:14s} {statistics.mean(a):7.3f} {statistics.mean(b):7.3f} "
              f"{rank_biserial(a, b):8.3f}")

    print("\n== within-prompt (clean, both outcomes present) ==")
    byp = defaultdict(list)
    for f in L:
        byp[f["pid"]].append(f)
    for pid, fs in sorted(byp.items()):
        a = [f for f in fs if f["correct"]]
        b = [f for f in fs if not f["correct"]]
        if not a or not b:
            continue
        row = " ".join(f"{k}={rank_biserial([f[k] for f in a], [f[k] for f in b]):+.2f}"
                       for k in ["opens", "waits", "verify_dup", "prune_durable",
                                 "closure_att"])
        print(f"  p{pid:02d} {fs[0]['domain']:10s} {len(a)}c/{len(b)}w  {row}")

    print("\n== per-family means (correct | wrong) ==")
    byf = defaultdict(list)
    for f in L:
        byf[f["domain"]].append(f)
    for fam, fs in sorted(byf.items()):
        a = [f for f in fs if f["correct"]]
        b = [f for f in fs if not f["correct"]]
        def m(g, k):
            return statistics.mean(f[k] for f in g) if g else float("nan")
        print(f"  {fam:10s} vd={m(a,'verify_dup'):.2f}|{m(b,'verify_dup'):.2f} "
              f"pd={m(a,'prune_durable'):.2f}|{m(b,'prune_durable'):.2f} "
              f"waits={m(a,'waits'):.1f}|{m(b,'waits'):.1f} "
              f"verifies={m(a,'verifies'):.1f}|{m(b,'verifies'):.1f} "
              f"opens={m(a,'opens'):.1f}|{m(b,'opens'):.1f}")


if __name__ == "__main__":
    main()
