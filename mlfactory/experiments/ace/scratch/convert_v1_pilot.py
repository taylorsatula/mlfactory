#!/usr/bin/env python3
"""One-shot migration (2026-08-28): rebuild R4v2-schema rows for the
judge-hillclimb pilot from the halted R4 v1 partial results.

Why reuse works: v2 seed derivation is bit-equivalent to v1's
fresh-run sub-batch-1 seeds (verified 2026-08-28), same prefix/hook/
params/FLASH backend — a v1 continuation truncated to the first 2048
post-fork tokens IS the v2 window. Rows are derived, never rewritten:
v1 files stay untouched; this writes a new v2-schema file.

Guard: the WINDOW slice itself must round-trip exactly (decode ->
retokenize -> same ids). Off-by-one full-length mismatches are a
known stop-token boundary artifact (verified 2026-08-28: all prefixes
<= 7000 round-trip clean on the first dropped row); a clean window
slice is what the judge reads, so that is what gets verified.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "/home/admin/mlfactory")
from transformers import AutoTokenizer
from mlfactory.experiments.ace.core.steering_controller import (
    MODEL_PATH, build_prompt_ids)

ACE = Path("/home/admin/mlfactory/mlfactory/experiments/ace")
PLAN = Path("/home/admin/mlfactory/artifacts/fork_plan_r4.jsonl")
V1_FILES = sorted(ACE.glob("data/r4fork-*/fork_r4_results_*.jsonl"))
OUT = ACE / "data" / "fork_r4v2_pilot.jsonl"
STATES = {"r4_cycle_00", "r4_cycle_02", "r4_cycle_03"}
SEEDS = set(range(8))
ARMS = {"noop", "toward_healthy"}
WINDOW, TAIL, CAP_ABS = 2048, 512, 26000

plan = {r["state_id"]: r for r in map(json.loads, PLAN.open())}
tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

# prefix_tail per state (same derivation as fork_r4v2.run)
tails = {}
for sid, st in plan.items():
    if sid not in STATES:
        continue
    prefix = build_prompt_ids(tok, st["prompt_text"], enable_thinking=True)
    c_ids = tok(st["completion"][:CAP_ABS * 4],
                add_special_tokens=False)["input_ids"][:CAP_ABS]
    prefix_ids = (prefix + c_ids)[: st["fork_abs"]]
    assert len(prefix_ids) == st["fork_abs"], sid
    tails[sid] = tok.decode(prefix_ids[max(0, len(prefix_ids) - TAIL):],
                            skip_special_tokens=False)

kept = dropped_rt = 0
out_rows = []
for f in V1_FILES:
    for l in f.open():
        r = json.loads(l)
        if (r["state_id"] not in STATES or r["seed_i"] not in SEEDS
                or r["arm"] not in ARMS):
            continue
        ids = tok(r["completion"], add_special_tokens=False)["input_ids"]
        if abs(len(ids) - r["n_new"]) > 1:
            print(f"BIG round-trip mismatch, dropping: {r['state_id']} "
                  f"{r['arm']} seed {r['seed_i']} "
                  f"({len(ids)} vs {r['n_new']})")
            dropped_rt += 1
            continue
        window_ids = ids[:WINDOW]
        # alignment must be clean up to the window boundary (a tail
        # artifact shifting tokens inside the window would leave the
        # slice self-consistent but wrong)
        probe = min(r["n_new"], WINDOW)
        if len(tok(tok.decode(ids[:probe], skip_special_tokens=False),
                   add_special_tokens=False)["input_ids"]) != probe:
            print(f"prefix alignment mismatch within window, dropping: "
                  f"{r['state_id']} {r['arm']} seed {r['seed_i']}")
            dropped_rt += 1
            continue
        # the judge reads the slice: verify the slice round-trips
        rt = tok(tok.decode(window_ids, skip_special_tokens=False),
                 add_special_tokens=False)["input_ids"]
        if rt != window_ids:
            print(f"window-slice round-trip mismatch, dropping: "
                  f"{r['state_id']} {r['arm']} seed {r['seed_i']}")
            dropped_rt += 1
            continue
        assert r["fork_token"] == plan[r["state_id"]]["fork_abs"]
        out_rows.append({
            "state_id": r["state_id"], "class": r["class"],
            "substrate": r["substrate"], "pid": r["pid"],
            "sample_i": r["sample_i"], "arm": r["arm"],
            "lam": r["lam"], "layer": r["layer"],
            "fork_token": r["fork_token"],
            "seed_i": r["seed_i"], "seed_batch": r["seed_batch"],
            "n_new": len(window_ids),
            "window_capped": r["n_new"] >= WINDOW,
            "elapsed_s": r["elapsed_s"],
            "prefix_tail": tails[r["state_id"]],
            "window": tok.decode(window_ids, skip_special_tokens=False),
            "derived_from": f"r4 v1 {f.name} (truncated to {WINDOW})",
        })
        kept += 1

with OUT.open("w") as fh:
    for r in sorted(out_rows,
                    key=lambda r: (r["state_id"], r["arm"], r["seed_i"])):
        fh.write(json.dumps(r) + "\n")
print(f"wrote {OUT.name}: {kept} rows "
      f"({dropped_rt} dropped on round-trip)")
from collections import Counter
print(Counter((r["state_id"], r["arm"]) for r in out_rows))
