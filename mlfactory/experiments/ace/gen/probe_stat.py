#!/usr/bin/env python3
"""Scalar stats for the ACE probe dashboard (called by shell_command probes).

Usage: probe_stat.py <what>
    done    total samples across both GPU outputs
    rate0   mean tok/s over last 8 samples on GPU 0
    rate1   mean tok/s over last 8 samples on GPU 1
    eta     hours remaining (remaining/2 * avg sample time, both GPUs)
    alive0  1 if the GPU 0 collector process is running
    alive1  1 if the GPU 1 collector process is running
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"

# Per-batch configuration; selected by optional second CLI arg (default b1).
BATCHES = {
    "b1": {
        "logs": {"0": DATA / "acegen_probe_b1_gpu0.log",
                 "1": DATA / "acegen_probe_b1_gpu1.log"},
        "outs": {"0": DATA / "acegen_probe_b1_gpu0.jsonl",
                 "1": DATA / "acegen_probe_b1_gpu1.jsonl"},
        "total": 384,
        "pgrep": {"0": "[c]ollect_rollouts .*candidate-range 24:48",
                  "1": "[c]ollect_rollouts .*candidate-range 0:24"},
    },
    "b2": {
        "logs": {"0": DATA / "acegen_probe_b2_gpu0.log",
                 "1": DATA / "acegen_probe_b2_gpu1.log"},
        "outs": {"0": DATA / "acegen_probe_b2_gpu0.jsonl",
                 "1": DATA / "acegen_probe_b2_gpu1.jsonl"},
        "total": 384,
        "pgrep": {"0": "[c]ollect_rollouts_api.*candidate-range 0:24",
                  "1": "[c]ollect_rollouts_api.*candidate-range 24:48"},
    },
}
BATCH = BATCHES[sys.argv[2] if len(sys.argv) > 2 else "b1"]
LOGS = BATCH["logs"]
OUTS = BATCH["outs"]
TOTAL = BATCH["total"]
PGREP_PAT = BATCH["pgrep"]


def samples(which: str, n: int = 8) -> list[dict]:
    path = LOGS[which]
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(errors="ignore").splitlines():
        line = line.strip()
        if not line.startswith('{"proposal_id"'):
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows[-n:]


def done() -> int:
    total = 0
    for p in OUTS.values():
        if p.exists():
            total += sum(1 for _ in p.open("rb"))
    return total


def alive(which: str) -> int:
    out = subprocess.run(["pgrep", "-f", PGREP_PAT[which]],
                         capture_output=True, text=True)
    return 1 if out.stdout.strip() else 0


def main() -> None:
    what = sys.argv[1]
    if what == "done":
        print(done())
    elif what in ("rate0", "rate1"):
        rows = samples(what[-1])
        rates = [r["n_new"] / r["dt_s"] for r in rows if r.get("dt_s")]
        print(f"{sum(rates) / len(rates):.1f}" if rates else "n/a")
    elif what == "eta":
        dts = [r["dt_s"] for w in ("0", "1") for r in samples(w)
               if r.get("dt_s")]
        if not dts:
            print("n/a")
            return
        avg = sum(dts) / len(dts)
        remaining = max(0, TOTAL - done())
        print(f"{remaining / 2 * avg / 3600:.1f}")
    elif what in ("alive0", "alive1"):
        print(alive(what[-1]))
    else:
        print("unknown", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
