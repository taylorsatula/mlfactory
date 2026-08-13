"""The three controlled Round 2 acquisition policies."""
from __future__ import annotations

import random
from collections import Counter, defaultdict
from typing import Callable

from .analysis import score_candidates


def targeted(pool: list[dict], surface: dict, batch_size: int, seed: int) -> tuple[list[dict], list[dict]]:
    if batch_size % 2:
        raise ValueError("batch_size must be even")
    scored = score_candidates(pool, surface)
    ordered = sorted(scored, key=lambda row: (-float(row["boundary_score"]), str(row["id"])))
    answer_cap = batch_size // 2
    world_cap = int(batch_size * 0.35)
    cell_cap = max(1, int(batch_size * 0.10))
    answers: Counter[str] = Counter()
    worlds: Counter[str] = Counter()
    cells: Counter[tuple] = Counter()
    selected: list[dict] = []
    for row in ordered:
        answer = row["canonical_answer"]
        world = row["world_id"]
        cell = (row["depth"], row["relevant_nodes"] // 2, row["distractor_nodes"] // 2, row["binary_gate_count"] // 2, row["source_update_count"])
        if answers[answer] >= answer_cap or worlds[world] >= world_cap or cells[cell] >= cell_cap:
            continue
        selected.append(row)
        answers[answer] += 1
        worlds[world] += 1
        cells[cell] += 1
        if len(selected) == batch_size:
            break
    if len(selected) != batch_size:
        raise RuntimeError(f"targeted selection filled only {len(selected)}/{batch_size}")
    return selected, scored


def token_matched_batch(pool: list[dict], reference: list[dict], seed: int) -> list[dict]:
    """Randomly sample a batch while matching cumulative prompt/target tokens."""
    if len(reference) % 2:
        raise ValueError("reference batch must be even")
    rng = random.Random(seed)
    target_prompt = sum(int(row["prompt_token_count"]) for row in reference)
    target_target = sum(int(row["target_token_count"]) for row in reference)
    answer_cap = len(reference) // 2
    world_cap = max(1, int(len(reference) * 0.35))
    available = list(pool)
    selected: list[dict] = []
    answers: Counter[str] = Counter()
    worlds: Counter[str] = Counter()
    prompt_total = target_total = 0
    for index in range(len(reference)):
        remaining = len(reference) - index - 1
        feasible = [
            row for row in available
            if answers[row["canonical_answer"]] < answer_cap
            and worlds[row["world_id"]] < world_cap
        ]
        if len(feasible) < remaining + 1:
            raise RuntimeError("token-matched sampler ran out of feasible candidates")
        desired_prompt = target_prompt * (index + 1) / len(reference)
        desired_target = target_target * (index + 1) / len(reference)
        rng.shuffle(feasible)
        feasible.sort(key=lambda row: (
            abs(prompt_total + int(row["prompt_token_count"]) - desired_prompt) / max(1, target_prompt),
            abs(target_total + int(row["target_token_count"]) - desired_target) / max(1, target_target),
        ))
        row = feasible[0]
        selected.append(row)
        available.remove(row)
        answers[row["canonical_answer"]] += 1
        worlds[row["world_id"]] += 1
        prompt_total += int(row["prompt_token_count"])
        target_total += int(row["target_token_count"])
    return selected




def depth_matched(pool: list[dict], reference: list[dict], seed: int) -> list[dict]:
    wanted = Counter(int(row["depth"]) for row in reference)
    by_depth: dict[int, list[dict]] = defaultdict(list)
    for row in pool:
        by_depth[int(row["depth"])].append(row)
    rng = random.Random(seed)
    selected: list[dict] = []
    for depth, count in sorted(wanted.items()):
        choices = list(by_depth[depth])
        reference_rows = [row for row in reference if int(row["depth"]) == depth]
        if len(choices) < count:
            raise RuntimeError(f"depth-matched pool lacks depth {depth}: {len(choices)}<{count}")
        target_prompt = sum(int(row["prompt_token_count"]) for row in reference_rows)
        target_target = sum(int(row["target_token_count"]) for row in reference_rows)
        prompt_total = target_total = 0
        for index in range(count):
            desired_prompt = target_prompt * (index + 1) / count
            desired_target = target_target * (index + 1) / count
            rng.shuffle(choices)
            choices.sort(key=lambda row: (
                abs(prompt_total + int(row["prompt_token_count"]) - desired_prompt) / max(1, target_prompt),
                abs(target_total + int(row["target_token_count"]) - desired_target) / max(1, target_target),
            ))
            row = choices.pop(0)
            selected.append(row)
            prompt_total += int(row["prompt_token_count"])
            target_total += int(row["target_token_count"])
    rng.shuffle(selected)
    return selected


def token_summary(rows: list[dict]) -> dict[str, int]:
    return {
        "examples": len(rows),
        "prompt_tokens": sum(int(row["prompt_token_count"]) for row in rows),
        "target_tokens": sum(int(row["target_token_count"]) for row in rows),
    }


def closest_token_match(selector: Callable[[int], list[dict]], reference: list[dict], attempts: int = 32) -> tuple[list[dict], dict]:
    """Choose the closest of repeated policy-valid random draws."""
    target = token_summary(reference)
    best_rows: list[dict] | None = None
    best_report: dict | None = None
    best_error = float("inf")
    for attempt in range(attempts):
        rows = selector(attempt)
        summary = token_summary(rows)
        deltas = {
            key: (summary[key] - target[key]) / max(1, target[key])
            for key in ("prompt_tokens", "target_tokens")
        }
        error = max(abs(value) for value in deltas.values())
        if error < best_error:
            best_rows = rows
            best_report = {"reference": target, "selected": summary, "relative_deltas": deltas, "attempt": attempt}
            best_error = error
    assert best_rows is not None and best_report is not None
    return best_rows, best_report
