#!/usr/bin/env python3
"""Generate ACE R4v2 fork run status JSON snapshot for the web dashboard.

Reads the r4v2 checkpoint JSONL files and judge files, writes a compact JSON
snapshot to /opt/crm_mira/web/assets/ace-status.json.
"""
import json
import glob
import os
import re
import subprocess
from collections import defaultdict, Counter
from datetime import datetime, timezone

DATA_DIR = "/home/admin/mlfactory/mlfactory/experiments/ace/data"
OUT_PATH = "/opt/crm_mira/web/assets/ace-status.json"

# 27 target states × 3 arms × 24 seeds = 1944
TARGET_STATES = [
    "r4_cycle_00", "r4_cycle_01", "r4_cycle_02", "r4_cycle_03", "r4_cycle_04",
    "r4_cycle_05", "r4_cycle_06", "r4_cycle_07", "r4_cycle_08", "r4_cycle_09",
    "r4_loop_00", "r4_loop_01", "r4_loop_02", "r4_loop_03", "r4_loop_04",
    "r4_loop_05", "r4_loop_06", "r4_loop_07", "r4_loop_08", "r4_loop_09",
    "r4_muse_00", "r4_muse_01", "r4_muse_02", "r4_muse_03", "r4_muse_04",
    "r4_muse_05", "r4_muse_06",
]
ARMS = ("noop", "toward_healthy", "toward_diverge")
SEEDS_PER_STATE = 24
ROWS_PER_ARM = len(TARGET_STATES) * SEEDS_PER_STATE  # 648
TARGET_ROWS = len(TARGET_STATES) * len(ARMS) * SEEDS_PER_STATE  # 1944

# Only count the main run files, not pilots/smoke/equivalence
RUN_GLOBS = [
    f"{DATA_DIR}/fork_r4v2_run_gpu0.jsonl",
    f"{DATA_DIR}/fork_r4v2_run_gpu1.jsonl",
]

# Judge files (pilot judge runs done so far)
JUDGE_GLOBS = [
    f"{DATA_DIR}/judge_r4v2_pilot_v2.jsonl",
    f"{DATA_DIR}/judge_r4v2_pilot2.jsonl",
    f"{DATA_DIR}/judge_r4v2_pilot2_v3.jsonl",
]


def state_class(state_id):
    if "cycle" in state_id:
        return "CYCLE"
    if "loop" in state_id:
        return "LOOP"
    if "muse" in state_id:
        return "MUSE"
    return "OTHER"


def load_rows():
    rows = []
    for pattern in RUN_GLOBS:
        for f in sorted(glob.glob(pattern)):
            with open(f) as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
    return rows


def check_processes():
    """Check if the fork_r4v2 run processes are active."""
    try:
        result = subprocess.run(
            ["bash", "-c", "ps aux | grep fork_r4v2 | grep -v grep"],
            capture_output=True, text=True, timeout=5,
        )
        procs = result.stdout.strip().split("\n") if result.stdout.strip() else []
        gpu_active = {"gpu0": False, "gpu1": False}
        for line in procs:
            if "gpu0" in line:
                gpu_active["gpu0"] = True
            if "gpu1" in line:
                gpu_active["gpu1"] = True
        return {
            "active": len(procs) > 0,
            "gpu0": gpu_active["gpu0"],
            "gpu1": gpu_active["gpu1"],
            "process_count": len(procs),
        }
    except Exception:
        return {"active": False, "gpu0": False, "gpu1": False, "process_count": 0}


def load_judge_verdicts():
    """Parse all judge files and return per-(state, seed) verdicts."""
    all_verdicts = []
    for pattern in JUDGE_GLOBS:
        for f in sorted(glob.glob(pattern)):
            with open(f) as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    for p in row.get("passes", []):
                        label_to_arm = p.get("label_to_arm", {})
                        raw = p.get("raw", "")
                        json_match = re.search(r"\{.*\}", raw, re.DOTALL)
                        if not json_match:
                            continue
                        try:
                            parsed = json.loads(json_match.group())
                        except json.JSONDecodeError:
                            continue
                        ranking = parsed.get("ranking", [])
                        for rank_idx, entry in enumerate(ranking):
                            label = entry[0] if isinstance(entry, list) else str(entry)
                            arm = label_to_arm.get(label, label)
                            char = ""
                            mode = ""
                            if label in parsed and isinstance(parsed[label], dict):
                                char = parsed[label].get("characterization", "")
                                mode = parsed[label].get("mode", "")
                            all_verdicts.append({
                                "state_id": row["state_id"],
                                "seed_i": row["seed_i"],
                                "arm": arm,
                                "rank": rank_idx,
                                "mode": mode,
                                "characterization": char,
                                "source": os.path.basename(f),
                            })
    return all_verdicts


def compute_judge_summary(verdicts):
    """Aggregate judge verdicts into summary stats."""
    if not verdicts:
        return None

    # Majority vote per (state, seed): which arm wins most often
    pair_votes = defaultdict(Counter)
    for v in verdicts:
        if v["rank"] == 0:
            pair_votes[(v["state_id"], v["seed_i"])][v["arm"]] += 1

    winners = Counter()
    for pair, votes in pair_votes.items():
        winner = votes.most_common(1)[0][0]
        winners[winner] += 1

    # Average rank by arm
    arm_ranks = defaultdict(list)
    arm_modes = defaultdict(Counter)
    for v in verdicts:
        arm_ranks[v["arm"]].append(v["rank"])
        if v["mode"]:
            arm_modes[v["arm"]][v["mode"]] += 1

    arm_stats = {}
    for arm in ARMS:
        ranks = arm_ranks.get(arm, [])
        arm_stats[arm] = {
            "avg_rank": round(sum(ranks) / len(ranks), 2) if ranks else None,
            "wins": winners.get(arm, 0),
            "modes": dict(arm_modes.get(arm, {})),
        }

    return {
        "pairs_judged": len(pair_votes),
        "total_verdicts": len(verdicts),
        "winners": dict(winners),
        "arm_stats": arm_stats,
    }


def compute_stats(rows, judge_verdicts):
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    total = len(rows)

    # By arm
    by_arm = {}
    for arm in ARMS:
        arm_rows = [r for r in rows if r["arm"] == arm]
        n = len(arm_rows)
        el = [r.get("elapsed_s", 0) for r in arm_rows]
        by_arm[arm] = {
            "rows": n,
            "target": ROWS_PER_ARM,
            "avg_elapsed_s": round(sum(el) / len(el)) if el else 0,
        }

    # By state
    by_state = {}
    for r in rows:
        sid = r["state_id"]
        if sid not in by_state:
            by_state[sid] = {"rows": 0, "arms": set(), "seeds": set()}
        by_state[sid]["rows"] += 1
        by_state[sid]["arms"].add(r["arm"])
        by_state[sid]["seeds"].add(r["seed_i"])

    # Include target states with zero rows
    for sid in TARGET_STATES:
        if sid not in by_state:
            by_state[sid] = {"rows": 0, "arms": set(), "seeds": set()}

    state_out = {}
    for sid in sorted(by_state):
        d = by_state[sid]
        triples = len(d["seeds"])  # seeds with at least 1 arm = potential triples
        complete = sum(
            1 for s in d["seeds"]
            if all(
                any(r["state_id"] == sid and r["seed_i"] == s and r["arm"] == a for r in rows)
                for a in ARMS
            )
        )
        state_out[sid] = {
            "rows": d["rows"],
            "target": len(ARMS) * SEEDS_PER_STATE,
            "class": state_class(sid),
            "seeds_started": triples,
            "complete_triples": complete,
        }

    # By class
    by_class = {}
    for cls in ["CYCLE", "LOOP", "MUSE"]:
        cls_states = [s for s in TARGET_STATES if state_class(s) == cls]
        cls_rows = sum(by_state.get(s, {}).get("rows", 0) for s in cls_states)
        cls_target = len(cls_states) * len(ARMS) * SEEDS_PER_STATE
        by_class[cls] = {
            "rows": cls_rows,
            "target": cls_target,
            "states": len(cls_states),
            "states_started": sum(1 for s in cls_states if by_state.get(s, {}).get("rows", 0) > 0),
        }

    # Token / time stats (n_new is always 2048 now, but track elapsed)
    all_el = [r.get("elapsed_s", 0) for r in rows]
    all_toks = [r.get("n_new", 0) for r in rows]

    pct = round(total / TARGET_ROWS * 100, 1) if TARGET_ROWS else 0

    # Process status
    proc_status = check_processes()

    # Judge summary
    judge_summary = compute_judge_summary(judge_verdicts)

    return {
        "generated_at": now_utc,
        "version": "r4v2",
        "total_rows": total,
        "target_rows": TARGET_ROWS,
        "pct_complete": pct,
        "total_tokens": sum(all_toks) if all_toks else 0,
        "tokens": {
            "total": sum(all_toks) if all_toks else 0,
            "mean": round(sum(all_toks) / len(all_toks)) if all_toks else 0,
            "fixed_window": 2048,
        },
        "elapsed": {
            "mean_s": round(sum(all_el) / len(all_el)) if all_el else 0,
            "min_s": round(min(all_el)) if all_el else 0,
            "max_s": round(max(all_el)) if all_el else 0,
        },
        "by_arm": by_arm,
        "by_state": state_out,
        "by_class": by_class,
        "processes": proc_status,
        "judge": judge_summary,
    }


def main():
    rows = load_rows()
    judge_verdicts = load_judge_verdicts()
    stats = compute_stats(rows, judge_verdicts)
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Wrote {OUT_PATH}: {stats['total_rows']}/{stats['target_rows']} rows ({stats['pct_complete']}%)")


if __name__ == "__main__":
    main()
