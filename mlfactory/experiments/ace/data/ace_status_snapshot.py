#!/usr/bin/env python3
"""Generate ACE R4 fork run status JSON snapshot for the web dashboard.

Reads the checkpoint JSONL files and attendance log, writes a compact JSON
snapshot to /opt/crm_mira/web/assets/ace-status.json.
"""
import json
import glob
import os
from collections import defaultdict
from datetime import datetime, timezone

DATA_DIR = "/home/admin/mlfactory/mlfactory/experiments/ace/data"
ATTENDANCE_LOG = "/home/admin/mlfactory/mlfactory/experiments/ace/annotate/out/r4_attendance.log"
OUT_PATH = "/opt/crm_mira/web/assets/ace-status.json"

TARGET_ROWS = 1944  # 27 states × 3 arms × 24 seeds
ARMS = ("noop", "toward_healthy", "toward_diverge")


def load_rows():
    rows = []
    for f in sorted(
        glob.glob(f"{DATA_DIR}/r4fork-a/fork_r4_results_*.jsonl")
        + glob.glob(f"{DATA_DIR}/r4fork-b/fork_r4_results_*.jsonl")
    ):
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def parse_attendance():
    """Parse the last few attendance log entries."""
    entries = []
    if not os.path.exists(ATTENDANCE_LOG):
        return entries
    with open(ATTENDANCE_LOG) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entries.append(line)
    return entries[-5:]


def compute_promise(rows, by_arm, overall_accuracy, total_hit_cap, total):
    """Composite 'is this run promising?' signal from multiple indicators.

    Combines: overall accuracy, hit-cap (thrashing) rate, arm separation
    (toward_healthy vs noop — the key experimental signal), and a rolling
    accuracy trend over the last two windows.
    """
    signals = []
    score = 50  # neutral baseline

    # 1. Accuracy signal
    if overall_accuracy >= 75:
        score += 15; signals.append("Strong solve rate")
    elif overall_accuracy >= 50:
        score += 5; signals.append("Moderate solve rate")
    elif overall_accuracy < 30:
        score -= 15; signals.append("Low solve rate")

    # 2. Hit-cap (thrashing) signal
    hit_cap_pct = round(total_hit_cap / total * 100, 1) if total else 0
    if hit_cap_pct <= 20:
        score += 10; signals.append("Traces terminating cleanly")
    elif hit_cap_pct >= 50:
        score -= 10; signals.append("High cap-hit — possible thrashing")

    # 3. Arm separation — the key experimental signal
    th = by_arm.get("toward_healthy", {})
    noop = by_arm.get("noop", {})
    if th.get("rows", 0) >= 5 and noop.get("rows", 0) >= 5:
        delta = th["accuracy"] - noop["accuracy"]
        if delta > 5:
            score += 25; signals.append(f"toward_healthy +{delta:.0f}% vs noop")
        elif delta > 0:
            score += 10; signals.append("toward_healthy edges out noop")
        elif delta < -10:
            score -= 15; signals.append("toward_healthy underperforming")
        else:
            signals.append("Arm separation inconclusive")
    else:
        signals.append("Arm comparison pending")

    # 4. Rolling accuracy trend (last 15 vs previous 15)
    WINDOW = 15
    if total >= WINDOW * 2:
        recent = rows[-WINDOW:]
        earlier = rows[-2 * WINDOW:-WINDOW]
        recent_acc = round(sum(1 for r in recent if r.get("correct")) / len(recent) * 100, 1)
        earlier_acc = round(sum(1 for r in earlier if r.get("correct")) / len(earlier) * 100, 1)
        if recent_acc > earlier_acc + 5:
            score += 10; signals.append(f"Accuracy trending up ({earlier_acc}%→{recent_acc}%)")
        elif recent_acc < earlier_acc - 10:
            score -= 10; signals.append(f"Accuracy trending down ({earlier_acc}%→{recent_acc}%)")
        trend_detail = {"recent": recent_acc, "earlier": earlier_acc}
    else:
        trend_detail = {"recent": overall_accuracy, "earlier": overall_accuracy}

    score = max(0, min(100, score))

    if score >= 70:
        label, arrow = "Promising", "up"
    elif score >= 40:
        label, arrow = "Steady", "flat"
    else:
        label, arrow = "Watch", "down"

    return {
        "score": score,
        "label": label,
        "arrow": arrow,
        "signal": signals[0] if signals else "Evaluating",
        "signals": signals,
        "trend": trend_detail,
    }


def compute_stats(rows):
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    total = len(rows)
    total_correct = sum(1 for r in rows if r.get("correct"))
    total_hit_cap = sum(1 for r in rows if r.get("hit_cap"))

    # By arm
    by_arm = {}
    for arm in ARMS:
        arm_rows = [r for r in rows if r["arm"] == arm]
        if not arm_rows:
            by_arm[arm] = {
                "rows": 0, "correct": 0, "accuracy": 0,
                "hit_cap": 0, "hit_cap_pct": 0,
                "avg_tokens": 0, "avg_elapsed_s": 0,
            }
            continue
        n = len(arm_rows)
        c = sum(1 for r in arm_rows if r.get("correct"))
        hc = sum(1 for r in arm_rows if r.get("hit_cap"))
        toks = [r.get("n_new", 0) for r in arm_rows]
        el = [r.get("elapsed_s", 0) for r in arm_rows]
        by_arm[arm] = {
            "rows": n,
            "correct": c,
            "accuracy": round(c / n * 100, 1) if n else 0,
            "hit_cap": hc,
            "hit_cap_pct": round(hc / n * 100, 1) if n else 0,
            "avg_tokens": round(sum(toks) / len(toks)) if toks else 0,
            "avg_elapsed_s": round(sum(el) / len(el)) if el else 0,
        }

    # By state
    by_state = {}
    for r in rows:
        sid = r["state_id"]
        if sid not in by_state:
            by_state[sid] = {"rows": 0, "correct": 0, "hit_cap": 0, "arms": set()}
        by_state[sid]["rows"] += 1
        if r.get("correct"):
            by_state[sid]["correct"] += 1
        if r.get("hit_cap"):
            by_state[sid]["hit_cap"] += 1
        by_state[sid]["arms"].add(r["arm"])
    # Convert sets to sorted lists for JSON
    for sid in by_state:
        d = by_state[sid]
        d["arms"] = sorted(d["arms"])
        d["accuracy"] = round(d["correct"] / d["rows"] * 100, 1) if d["rows"] else 0
        d["target"] = 72  # 3 arms × 24 seeds per state

    # Token stats
    all_toks = [r.get("n_new", 0) for r in rows]
    all_el = [r.get("elapsed_s", 0) for r in rows]

    # Parse spend from attendance log
    spend = "~$0"
    attendance = parse_attendance()
    for line in reversed(attendance):
        if "spend=" in line:
            spend = line.split("spend=")[1].split("|")[0].strip()
            break

    # Box info from attendance
    box_a_status = "UNKNOWN"
    box_b_status = "UNKNOWN"
    for line in reversed(attendance):
        if "A:" in line and "B:" in line:
            parts = line.split("|")
            for p in parts:
                p = p.strip()
                if p.startswith("A:"):
                    box_a_status = p
                elif p.startswith("B:"):
                    box_b_status = p
            break

    pct = round(total / TARGET_ROWS * 100, 1) if TARGET_ROWS else 0

    return {
        "generated_at": now_utc,
        "total_rows": total,
        "target_rows": TARGET_ROWS,
        "pct_complete": pct,
        "overall_correct": total_correct,
        "overall_accuracy": round(total_correct / total * 100, 1) if total else 0,
        "total_hit_cap": total_hit_cap,
        "hit_cap_pct": round(total_hit_cap / total * 100, 1) if total else 0,
        "spend": spend,
        "box_a": box_a_status,
        "box_b": box_b_status,
        "by_arm": by_arm,
        "by_state": {k: by_state[k] for k in sorted(by_state)},
        "tokens": {
            "total": sum(all_toks) if all_toks else 0,
            "min": min(all_toks) if all_toks else 0,
            "max": max(all_toks) if all_toks else 0,
            "mean": round(sum(all_toks) / len(all_toks)) if all_toks else 0,
            "median": sorted(all_toks)[len(all_toks) // 2] if all_toks else 0,
        },
        "elapsed": {
            "min_s": round(min(all_el)) if all_el else 0,
            "max_s": round(max(all_el)) if all_el else 0,
            "mean_s": round(sum(all_el) / len(all_el)) if all_el else 0,
        },
        "attendance_tail": attendance,
        "promise": compute_promise(rows, by_arm, round(total_correct / total * 100, 1) if total else 0, total_hit_cap, total),
    }


def main():
    rows = load_rows()
    stats = compute_stats(rows)
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Wrote {OUT_PATH}: {stats['total_rows']}/{stats['target_rows']} rows ({stats['pct_complete']}%)")


if __name__ == "__main__":
    main()
