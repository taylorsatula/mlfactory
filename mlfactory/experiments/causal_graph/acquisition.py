"""Candidate generation and controlled acquisition branches."""
from __future__ import annotations

import random
from collections import Counter, defaultdict
from typing import Any

from .analysis import score_candidates, select_frontier_batch
from .generator import TRAIN_WORLDS, canonical_trace, generate_task, verify_task


def generate_candidate_pool(seed: int, count: int, *, d50: float | None = None, max_depth: int = 48) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    center = int(round(d50)) if d50 is not None else 12
    low = max(2, center - 8)
    high = min(max_depth, center + 14)
    pool: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index in range(count * 3):
        depth = rng.randint(low, high)
        gates = rng.randint(0, min(depth, 8))
        extra = rng.randint(0, min(gates, 4))
        task = generate_task(
            seed + index, depth=depth, relevant_nodes=depth + 1 + extra,
            distractor_nodes=rng.randint(0, 6), binary_gate_count=gates,
            negation_count=rng.randint(0, min(depth - gates, 8)), source_update_count=rng.randint(0, 3),
            world_id=rng.choice(TRAIN_WORLDS), render_template_id=rng.randrange(3),
        )
        if task["id"] in seen:
            continue
        verify_task(task)
        task["canonical_trace"] = canonical_trace(task)
        pool.append(task); seen.add(task["id"])
        if len(pool) >= count:
            break
    if len(pool) < count:
        raise RuntimeError(f"candidate generator produced only {len(pool)}/{count} unique tasks")
    return pool


def select_random_batch(pool: list[dict[str, Any]], batch_size: int, seed: int = 20260811, cell_fraction: float = 0.10) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    rows = list(pool); rng.shuffle(rows)
    cap = max(1, int(batch_size * cell_fraction)); counts: Counter[tuple[Any, ...]] = Counter(); worlds: Counter[str] = Counter()
    selected: list[dict[str, Any]] = []
    for row in rows:
        cell = (row["depth"], row["relevant_nodes"] // 2, row["distractor_nodes"] // 2, row["binary_gate_count"] // 2, row["source_update_count"])
        if counts[cell] >= cap or worlds[row["world_id"]] >= int(batch_size * 0.35):
            continue
        counts[cell] += 1; worlds[row["world_id"]] += 1; selected.append(row)
        if len(selected) >= batch_size: break
    if len(selected) < batch_size:
        for row in rows:
            if row in selected: continue
            cell = (row["depth"], row["relevant_nodes"] // 2, row["distractor_nodes"] // 2, row["binary_gate_count"] // 2, row["source_update_count"])
            if counts[cell] >= cap: continue
            counts[cell] += 1; selected.append(row)
            if len(selected) >= batch_size: break
    for row in selected: row["selected_for_training"] = True
    return selected


def select_depth_matched(pool: list[dict[str, Any]], targeted: list[dict[str, Any]], seed: int = 20260811) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    by_depth: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in pool: by_depth[int(row["depth"])].append(row)
    selected: list[dict[str, Any]] = []
    used: set[str] = set()
    for depth, count in Counter(int(row["depth"]) for row in targeted).items():
        choices = list(by_depth.get(depth, [])); rng.shuffle(choices)
        for row in choices:
            if row["id"] in used: continue
            selected.append(row); used.add(row["id"]); row["selected_for_training"] = True
            if sum(1 for x in selected if x["depth"] == depth) >= count: break
    return selected


def coordinate_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows: return {"count": 0}
    fields = ["depth", "relevant_nodes", "distractor_nodes", "binary_gate_count", "negation_count", "source_update_count"]
    return {"count": len(rows), **{f"{name}_mean": sum(float(row[name]) for row in rows) / len(rows) for name in fields}, **{f"{name}_median": sorted(float(row[name]) for row in rows)[len(rows)//2] for name in fields}, "world_counts": dict(Counter(row["world_id"] for row in rows))}
