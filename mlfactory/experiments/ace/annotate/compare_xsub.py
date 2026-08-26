"""Cross-substrate band comparison: q8 vs bf16 on the same 6 prompts.

Reads the three xsub collection files (q8 local, bf16 gpu0/gpu1), reports
per-prompt success/cap/length tables for each substrate, seed-paired
outcome agreement across substrates, and failure-species counts
(committed-wrong vs budget-exhaustion). Pure CPU analysis over collected
evidence; run after retrieval, results feed the lab note.

Usage (repo root or anywhere — paths explicit):
    python -m mlfactory.experiments.ace.annotate.compare_xsub \
        data/xsub_q8.jsonl data/xsub_bf16_gpu0.jsonl data/xsub_bf16_gpu1.jsonl
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def _b(v) -> bool:
    return str(v).strip().lower() in ("true", "1", "yes")


def _i(v) -> int:
    return int(str(v).strip())


def load_rows(path: Path) -> list[dict]:
    rows = []
    for line in path.open():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        r["proposal_id"] = _i(r["proposal_id"])
        r["sample_i"] = _i(r["sample_i"])
        r["n_new_tokens"] = _i(r["n_new_tokens"])
        r["correct"] = _b(r["correct"])
        r["truncated"] = _b(r["truncated"])
        r["elapsed_s"] = float(r.get("elapsed_s", 0) or 0)
        rows.append(r)
    return rows


def species(r: dict) -> str:
    """Failure species (correct rows are 'correct')."""
    if r["correct"]:
        return "correct"
    # truncated/cap-hit = budget exhaustion; else committed wrong answer
    return "budget" if r["truncated"] else "committed_wrong"


def fmt_n(xs: list[int]) -> str:
    if not xs:
        return "-"
    return f"med={statistics.median(xs):.0f} mean={statistics.mean(xs):.0f}"


def per_prompt_table(rows: list[dict], substrate: str) -> None:
    print(f"\n== {substrate}: per-prompt (n=8 each) ==")
    print(f"{'domain':<10} {'pid':>4} {'succ':>5} {'cap':>4} {'sp/wrong':>9}  n_new tokens")
    by_prompt = defaultdict(list)
    for r in rows:
        by_prompt[(r["domain"], r["proposal_id"])].append(r)
    tot_c = tot_cap = 0
    all_lens: list[int] = []
    for (dom, pid), rs in sorted(by_prompt.items()):
        c = sum(r["correct"] for r in rs)
        cap = sum(r["truncated"] for r in rs)
        sp = [r["n_new_tokens"] for r in rs]
        tot_c += c
        tot_cap += cap
        all_lens += sp
        print(f"{dom:<10} {pid:>4} {c:>4}/8 {cap:>3}/8 {fmt_n(sp):>24}")
    n = len(rows)
    print(f"{'TOTAL':<15} {tot_c:>3}/{n} {tot_cap:>3}/{n} {fmt_n(all_lens):>24}")


def paired_agreement(q8: list[dict], bf16: list[dict]) -> None:
    print("\n== seed-paired outcome agreement (same prompt, same sample_i) ==")
    q = {(r["proposal_id"], r["sample_i"]): r for r in q8}
    b = {(r["proposal_id"], r["sample_i"]): r for r in bf16}
    both = sorted(set(q) & set(b))
    if not both:
        print("no paired keys yet")
        return
    agree = sum(q[k]["correct"] == b[k]["correct"] for k in both)
    flips = [(k, q[k]["correct"], b[k]["correct"]) for k in both if q[k]["correct"] != b[k]["correct"]]
    print(f"paired: {len(both)}/{len(q)} q8 rows have bf16 counterparts")
    print(f"outcome agreement: {agree}/{len(both)} = {agree / len(both):.2f}")
    if flips:
        print("flips (q8 -> bf16):")
        for (pid, si), qc, bc in flips:
            qs, bs = species(q[(pid, si)]), species(b[(pid, si)])
            print(f"  p{pid} s{si}: {qc} ({qs}) -> {bc} ({bs})")


def failure_species(q8: list[dict], bf16: list[dict]) -> None:
    print("\n== failure species ==")
    for name, rows in (("q8", q8), ("bf16", bf16)):
        counts = defaultdict(int)
        for r in rows:
            counts[species(r)] += 1
        print(f"  {name:<5} " + "  ".join(f"{k}={v}" for k, v in sorted(counts.items())))


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    paths = [Path(p) for p in sys.argv[1:]]
    for p in paths:
        if not p.exists():
            print(f"missing: {p}")
            sys.exit(1)
    q8 = load_rows(paths[0])
    bf16 = []
    for p in paths[1:]:
        bf16 += load_rows(p)
    print(f"loaded: q8={len(q8)} rows, bf16={len(bf16)} rows")
    per_prompt_table(q8, "Q8_0-MTP")
    per_prompt_table(bf16, "BF16-MTP")
    paired_agreement(q8, bf16)
    failure_species(q8, bf16)


if __name__ == "__main__":
    main()
