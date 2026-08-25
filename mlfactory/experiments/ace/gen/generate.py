"""Generate ACE probe pools: solver-built problems with knob metadata.

Usage (from mlfactory/experiments/ace):
    .venv/bin/python -m gen.generate --out data/acegen_probe_b1.jsonl
    .venv/bin/python -m gen.generate --family grid --n-per 20 --seed 9000
    .venv/bin/python -m gen.generate --self-test

Every generated row is self-checked against its family verifier before
being written, so the pool is correct by construction. Probe the pool with
the existing collector:

    CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u collect_qwen_frontier_30.py \
        --candidates data/acegen_probe_b1.jsonl \
        --out data/acegen_probe_b1_rollouts.jsonl --n-samples 8

then score with gen/calibrate.py.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from . import adversary, assign, certify, construct, grid, hypothesis, \
    machine, revise

FAMILIES = {
    "assign": assign,
    "machine": machine,
    "adversary": adversary,
    "certify": certify,
    "grid": grid,
    "hypothesis": hypothesis,
    "construct": construct,
    "revise": revise,
}

DEFAULT_KNOBS = {
    "assign":     {"n_items": 6, "n_bins": 4, "delayed": True},
    "machine":    {"n_states": 5, "n_events": 6, "log_len": 13},
    "adversary":  {"n_modes": 4, "target_depth": 4, "max_witness": 3},
    "certify":    {"n_nodes": 7, "k": 3, "trap": True, "none_prob": 0.25},
    "grid":       {"n_pos": 5},
    "hypothesis": {"n_sales": 5, "n_payouts": 2, "spread": 15},
    "construct":  {"n_items": 6, "budget": False, "none_prob": 0.2},
    "revise":     {"n_records": 4, "spread": 15, "decoy": True},
}

# Harder preset: nudge knobs toward the top of each range. The calibration
# loop should move along these axes when prompts land DEAD-EASY.
HARD_KNOBS = {
    "assign":     {"n_items": 8, "n_bins": 5, "delayed": True},
    "machine":    {"n_states": 6, "n_events": 7, "log_len": 17},
    "adversary":  {"n_modes": 4, "target_depth": 5, "max_witness": 2},
    "certify":    {"n_nodes": 9, "k": 3, "trap": True, "none_prob": 0.3},
    "grid":       {"n_pos": 6, "max_at": 1},
    "hypothesis": {"n_sales": 6, "n_payouts": 3, "spread": 12,
                   "n_voids": 3},
    "construct":  {"n_items": 7, "budget": True, "none_prob": 0.25},
    "revise":     {"n_records": 5, "spread": 12, "decoy": True},
}

# Easier preset: bottom of each range, for families landing DEAD-HARD.
EASY_KNOBS = {
    "assign":     {"n_items": 5, "n_bins": 3, "delayed": False},
    "machine":    {"n_states": 4, "n_events": 5, "log_len": 10},
    "adversary":  {"n_modes": 3, "target_depth": 4, "max_witness": 3},
    "certify":    {"n_nodes": 6, "k": 3, "trap": False, "none_prob": 0.15},
    "grid":       {"n_pos": 4},
    "hypothesis": {"n_sales": 4, "n_payouts": 1, "spread": 20},
    "construct":  {"n_items": 5, "budget": False, "none_prob": 0.15},
    "revise":     {"n_records": 3, "spread": 20, "decoy": False},
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--family", choices=list(FAMILIES) + ["all"],
                   default="all")
    p.add_argument("--n-per", type=int, default=10)
    p.add_argument("--seed", type=int, default=7000)
    p.add_argument("--start-id", type=int, default=1)
    p.add_argument("--preset", choices=["default", "hard", "easy"],
                   default="default")
    p.add_argument("--out", type=Path,
                   default=Path("data/acegen_probe.jsonl"))
    p.add_argument("--self-test", action="store_true",
                   help="Generate 2 per family, verify round-trip, print.")
    return p.parse_args()


def main() -> None:
    cfg = parse_args()
    presets = {"default": DEFAULT_KNOBS, "hard": HARD_KNOBS,
               "easy": EASY_KNOBS}
    knobs_table = presets[cfg.preset]
    fams = list(FAMILIES) if cfg.family == "all" else [cfg.family]
    n_per = 2 if cfg.self_test else cfg.n_per

    rows = []
    pid = cfg.start_id
    for fi, fam in enumerate(fams):
        mod = FAMILIES[fam]
        for i in range(n_per):
            rng = random.Random(cfg.seed + fi * 100003 + i * 7919)
            prob = mod.make(rng, dict(knobs_table[fam]))
            # self-check: the family's own strict verifier must accept
            # the reference answer it just produced.
            ok = mod.check("Answer: " + prob.answer, prob.answer,
                           prob.knobs)
            if not ok:
                raise RuntimeError(
                    f"{fam} #{i}: self-check failed for {prob.answer!r}")
            rows.append(prob.to_row(pid))
            pid += 1

    if cfg.self_test:
        for r in rows:
            print(f"--- p{r['provenance']['proposal_id']} {r['domain']} "
                  f"knobs={r['knobs']} ---")
            print(r["prose"][:600])
            print(f"Q: {r['surface_question']}")
            print(f"A: {r['problem']['reference_answer']}")
            print()
        print(f"self-test: {len(rows)}/{len(rows)} round-trip OK")
        return

    cfg.out.parent.mkdir(parents=True, exist_ok=True)
    with cfg.out.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} problems -> {cfg.out}")


if __name__ == "__main__":
    main()
