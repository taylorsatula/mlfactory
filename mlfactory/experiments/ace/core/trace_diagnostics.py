#!/usr/bin/env python3
"""Text-level trace diagnostics for ACE probe outputs (no GPU required).

Computed per trace, from the completion text alone:
  - n_new_tokens, truncated, soft `correct` (advisory; calibrate.py is authoritative)
  - verbatim terminal-loop detection: period, onset char offset, post-onset fraction
  - emission-paralysis flag: loop unit contains closure-intent language
  - word-level recurrence proxy: fraction of 8-grams seen earlier in the trace

Aggregates by family, outcome, and truncation. This is the Phase-1
pre-screen: cheap text observables that decide which model-based
measurements (teacher-forced entropy/recurrence/tortuosity) are worth GPU
time after the probe completes.

Usage:
  python3 -m mlfactory.experiments.ace.core.trace_diagnostics data/acegen_probe_b1_gpu0.jsonl data/acegen_probe_b1_gpu1.jsonl
  python3 -m mlfactory.experiments.ace.core.trace_diagnostics data/frontier_rollouts_pass1.jsonl   # frozen-30
"""
from __future__ import annotations

import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

CLOSURE_PAT = re.compile(
    r"(final answer|i'll write|i will (now )?(write|output|present|give)|"
    r"let me (now )?(write|output|present)|putting it all together|"
    r"time to (write|output)|now i (can|will) (write|present|output))",
    re.IGNORECASE,
)


def find_terminal_loop(text: str, min_repeats: int = 3, max_tail: int = 200_000) -> dict | None:
    """Detect a verbatim periodic suffix. Returns period/onset or None.

    onset = char offset where the periodic regime begins (start of the first
    full repeat of the unit). Everything from onset onward is post-novelty.
    """
    tail = text[-max_tail:] if len(text) > max_tail else text
    base = len(text) - len(tail)
    n = len(tail)
    # try periods from small to n//min_repeats; prefer the LONGEST periodic
    # suffix (check largest k*p coverage). Simple O(n^2)-ish but bounded.
    best = None
    for p in range(40, n // min_repeats + 1):
        # how far back does period p hold?
        k = 1
        while k * p < n and tail[n - (k + 1) * p : n - k * p] == tail[n - p :]:
            k += 1
        if k >= min_repeats:
            cover = k * p
            if best is None or cover > best["cover"]:
                best = {"period": p, "cover": cover, "repeats": k}
    if best is None:
        return None
    onset = base + n - best["cover"]
    unit = text[onset : onset + best["period"]]
    return {
        "period_chars": best["period"],
        "repeats": best["repeats"],
        "onset_char": onset,
        "post_onset_frac": (len(text) - onset) / max(1, len(text)),
        "closure_intent": bool(CLOSURE_PAT.search(unit)),
        "unit_preview": " ".join(unit.split())[:120],
    }


def recurrence_8gram(text: str) -> float:
    """Fraction of word 8-grams (after position 200 words) that occurred earlier."""
    words = text.split()
    if len(words) < 220:
        return 0.0
    seen: set[tuple] = set()
    for i in range(0, min(200, len(words) - 8)):
        seen.add(tuple(words[i : i + 8]))
    total = 0
    dup = 0
    for i in range(200, len(words) - 8):
        total += 1
        g = tuple(words[i : i + 8])
        if g in seen:
            dup += 1
        seen.add(g)
    return dup / max(1, total)


def analyze(rows: list[dict]) -> list[dict]:
    out = []
    for r in rows:
        text = r.get("completion", "")
        d = {
            "proposal_id": r.get("proposal_id"),
            "sample_i": r.get("sample_i"),
            "domain": r.get("domain"),
            "tokens": r.get("n_new_tokens", 0),
            "truncated": bool(r.get("truncated")),
            "correct": bool(r.get("correct")),
            "recurrence": recurrence_8gram(text),
            "loop": find_terminal_loop(text),
        }
        out.append(d)
    return out


def pct(a: int, b: int) -> str:
    return f"{a}/{b} ({100*a/b:.0f}%)" if b else "0/0"


def report(diags: list[dict]) -> None:
    n = len(diags)
    trunc = [d for d in diags if d["truncated"]]
    loops = [d for d in diags if d["loop"]]
    paral = [d for d in diags if d["loop"] and d["loop"]["closure_intent"]]

    print(f"traces: {n}   truncated: {pct(len(trunc), n)}   "
          f"terminal loops: {pct(len(loops), n)}   closure-intent loops: {pct(len(paral), n)}")

    print("\n== outcome x length ==")
    for key, grp in [("correct", [d for d in diags if d["correct"]]),
                     ("wrong  ", [d for d in diags if not d["correct"]])]:
        if grp:
            toks = [d["tokens"] for d in grp]
            print(f"  {key} n={len(grp):3d}  mean_tok={statistics.mean(toks):7.0f}  "
                  f"median={statistics.median(toks):7.0f}  trunc={pct(sum(d['truncated'] for d in grp), len(grp))}  "
                  f"loops={pct(sum(bool(d['loop']) for d in grp), len(grp))}  "
                  f"mean_recur={statistics.mean(d['recurrence'] for d in grp):.3f}")

    print("\n== loops by truncation ==")
    for key, grp in [("truncated", trunc), ("finished ", [d for d in diags if not d["truncated"]])]:
        if grp:
            gl = [d for d in grp if d["loop"]]
            print(f"  {key} n={len(grp):3d}  loops={pct(len(gl), len(grp))}")
            if gl:
                print(f"    post-onset token frac: mean={statistics.mean(d['loop']['post_onset_frac'] for d in gl):.2f}"
                      f"  median={statistics.median(d['loop']['post_onset_frac'] for d in gl):.2f}"
                      f"  closure-intent={pct(sum(d['loop']['closure_intent'] for d in gl), len(gl))}")

    print("\n== per-family ==")
    fams = defaultdict(list)
    for d in diags:
        fams[d["domain"]].append(d)
    for fam in sorted(fams, key=str):
        g = fams[fam]
        fl = [d for d in g if d["loop"]]
        print(f"  {str(fam):12s} n={len(g):3d} soft={sum(d['correct'] for d in g):3d}/{len(g):<3d} "
              f"trunc={pct(sum(d['truncated'] for d in g), len(g)):>10s} loops={pct(len(fl), len(g)):>10s} "
              f"mean_recur={statistics.mean(d['recurrence'] for d in g):.3f}")

    print("\n== closure-intent loop cases (emission-paralysis candidates) ==")
    for d in paral:
        lp = d["loop"]
        print(f"  p{d['proposal_id']:02d} s{d['sample_i']} {str(d['domain']):10s} "
              f"tok={d['tokens']:6d} correct={d['correct']} post-onset={lp['post_onset_frac']:.2f} "
              f"x{lp['repeats']} | {lp['unit_preview']}")


def main() -> None:
    rows = []
    for arg in sys.argv[1:]:
        for line in Path(arg).read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    # dedup by (proposal_id, sample_i) — stale pass2 files may overlap
    seen = set()
    uniq = []
    for r in rows:
        k = (r.get("proposal_id"), r.get("sample_i"))
        if k not in seen:
            seen.add(k)
            uniq.append(r)
    if len(uniq) != len(rows):
        print(f"[dedup] {len(rows)} -> {len(uniq)} rows", file=sys.stderr)
    report(analyze(uniq))


if __name__ == "__main__":
    main()
