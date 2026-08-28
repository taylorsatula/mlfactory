"""Step-1 gate checks for a dry-run out dir (run remotely).

Usage: python3 step1_gates.py /workspace/s1_dry
"""
import json, sys
from pathlib import Path

out = Path(sys.argv[1])
rows = [json.loads(l) for l in (out / "rollout_rows.jsonl").read_text().splitlines() if l.strip()]
summary = json.loads((out / "summary.json").read_text())
fails, notes = [], []

# (b) row count + key uniqueness
keys = [(r["id"], r["arm"], r["sample_i"], r["iter"]) for r in rows]
dups = len(keys) - len(set(keys))
if dups: fails.append(f"DUPLICATE row keys: {dups}")
if len(rows) != 48: fails.append(f"row count {len(rows)} != 48")

# (a) iter-0 steered==base identity (zero-init controller = exact no-op)
b = {(r["id"], r["sample_i"]): r for r in rows if r["iter"] == 0 and r["arm"] == "base"}
s = {(r["id"], r["sample_i"]): r for r in rows if r["iter"] == 0 and r["arm"] == "steered"}
if set(b) != set(s): fails.append(f"iter0 key mismatch: base {len(b)} vs steered {len(s)}")
mm = []
for k in set(b) & set(s):
    for f in ("reward", "n_new", "seed", "truncated", "eos"):
        if b[k][f] != s[k][f]: mm.append((k, f, b[k][f], s[k][f]))
if mm: fails.append(f"iter0 steered!=base on {len(mm)} fields: {mm[:4]}")
else: notes.append(f"(a) iter0 identity OK on {len(b)} pairs")

# (c) fingerprint
fp = summary.get("base_fingerprints_unchanged")
if fp is not True: fails.append(f"base_fingerprints_unchanged={fp}")
else: notes.append("(c) fingerprint flag true")

# (d) cap counts + peak mem (from iter lines in train.jsonl)
tl = [json.loads(l) for l in (out / "train.jsonl").read_text().splitlines() if l.strip()]
for t in tl:
    pk = t.get("peak_mem_gb", t.get("replay_peak_mem_gb"))
    if pk is not None and pk > 130: fails.append(f"iter{t.get('iter')} peak_mem {pk} > 130")
    notes.append(f"(d) iter{t.get('iter')} cap={t.get('cap_hits')}/{t.get('n_rows', t.get('rows'))} peak={pk}")

# (e) gate stats non-degenerate
import statistics as st
for arm in ("steered",):
    gm = [r["gate_mean"] for r in rows if r["arm"] == arm and r["gate_mean"] is not None]
    gs = [r["gate_std"] for r in rows if r["arm"] == arm and r["gate_std"] is not None]
    if not gm: fails.append(f"no gate_mean for {arm}"); continue
    notes.append(f"(e) {arm} gate_mean {min(gm):.4f}-{max(gm):.4f} gate_std {min(gs):.4f}-{max(gs):.4f}")
    if max(gs) == 0: fails.append(f"{arm} gate_std identically 0 (saturated)")

rew = [r["reward"] for r in rows]
notes.append(f"rewards: {sum(rew)}/{len(rew)} nonzero across all rows")

print("PASS" if not fails else "FAIL")
for n in notes: print(" ", n)
for f in fails: print("  !!", f)
sys.exit(0 if not fails else 1)
