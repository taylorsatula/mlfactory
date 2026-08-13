"""Round 2 task rendering built on the validated Round 1 symbolic generator."""
from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from typing import Any

from mlfactory.experiments.causal_graph.generator import (
    HELDOUT_WORLDS,
    TRAIN_WORLDS,
    WORLD_IDS,
    generate_task as _generate_symbolic_task,
    verify_task,
)

from .contract import chat_prompt, instruction, target


def symbolic_hash(task: dict[str, Any]) -> str:
    nodes = {
        node_id: {key: node[key] for key in ("kind", "op", "parents") if key in node}
        for node_id, node in task["graph"]["nodes"].items()
    }
    payload = {
        "nodes": nodes,
        "sources": task["source_values_initial"],
        "updates": task["source_updates"],
        "query": task["query_node"],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def generate_task(seed: int, **kwargs: Any) -> dict[str, Any]:
    """Generate a symbolic task and apply the single Round 2 renderer contract."""
    task = _generate_symbolic_task(seed, **kwargs)
    verify_task(task)
    state_count = int(task["relevant_nodes"])
    lines = task["rendered_prompt"].splitlines()
    lines[-1] = instruction(state_count)
    task["rendered_prompt"] = "\n".join(lines)
    task["response_mode"] = "bittrace"
    task["canonical_trace"] = target(task)
    task["graph_hash"] = symbolic_hash(task)
    return task


def regenerate_matches(task: dict[str, Any]) -> bool:
    replay = generate_task(
        task["seed"],
        depth=task["depth"],
        relevant_nodes=task["relevant_nodes"],
        distractor_nodes=task["distractor_nodes"],
        binary_gate_count=task["binary_gate_count"],
        negation_count=task["negation_count"],
        source_update_count=task["source_update_count"],
        world_id=task["world_id"],
        render_template_id=task["render_template_id"],
    )
    return (
        replay["rendered_prompt"] == task["rendered_prompt"]
        and replay["canonical_trace"] == task["canonical_trace"]
        and replay["canonical_answer"] == task["canonical_answer"]
    )


def audit_task(task: dict[str, Any], tokenizer: Any, *, max_target_tokens: int = 128, max_length: int = 1024) -> tuple[dict[str, Any] | None, str | None]:
    """Validate symbolic semantics and the complete tokenized training contract."""
    try:
        verify_task(task)
        prompt_ids = tokenizer(chat_prompt(tokenizer, task["rendered_prompt"]), add_special_tokens=False)["input_ids"]
        target_ids = tokenizer(task["canonical_trace"] + (tokenizer.eos_token or ""), add_special_tokens=False)["input_ids"]
        if len(target_ids) > max_target_tokens:
            return None, f"target budget {len(target_ids)}>{max_target_tokens}"
        if len(prompt_ids) + len(target_ids) > max_length:
            return None, f"context budget {len(prompt_ids) + len(target_ids)}>{max_length}"
        decoded = tokenizer.decode(target_ids, skip_special_tokens=True).strip()
        if not decoded.endswith(f"FINAL: {task['canonical_answer']}"):
            return None, "decoded target lost terminal answer"
        row = dict(task)
        row["prompt_token_count"] = len(prompt_ids)
        row["target_token_count"] = len(target_ids)
        return row, None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def probe_tasks(seed: int, depths: list[int], per_depth: int, worlds: tuple[str, ...]) -> list[dict[str, Any]]:
    """Generate balanced fixed-control probes with an easy-to-hard transition."""
    if per_depth % 2:
        raise ValueError("per_depth must be even for exact label balance")
    tasks: list[dict[str, Any]] = []
    for depth in depths:
        gates = 0 if depth <= 6 else 1 if depth <= 12 else 2
        negations = 0 if depth <= 6 else 1
        relevant = depth + 1 + min(gates, 2)
        wanted = per_depth // 2
        labels = Counter()
        index = 0
        while len([task for task in tasks if task["depth"] == depth]) < per_depth:
            task = generate_task(
                seed + depth * 100_000 + index,
                depth=depth,
                relevant_nodes=relevant,
                distractor_nodes=index % (2 if depth <= 4 else 3),
                binary_gate_count=gates,
                negation_count=negations,
                source_update_count=0 if depth <= 8 else index % 2,
                world_id=worlds[(index + depth) % len(worlds)],
                render_template_id=index % 3,
            )
            index += 1
            answer = task["canonical_answer"]
            if labels[answer] >= wanted:
                continue
            labels[answer] += 1
            tasks.append(task)
    return tasks


def contract_anchors(seed: int, count: int) -> list[dict[str, Any]]:
    """Shared long-trace examples that teach every branch the output grammar."""
    depths = [1, 2, 4, 6, 8, 10, 12, 16, 20, 24]
    rows: list[dict[str, Any]] = []
    labels = Counter()
    seen: set[str] = set()
    index = 0
    while len(rows) < count:
        depth = depths[index % len(depths)]
        gates = 0 if depth <= 6 else 1 if depth <= 12 else 2
        task = generate_task(
            seed + index,
            depth=depth,
            relevant_nodes=depth + 1 + (index % 2 if depth >= 8 else 0),
            distractor_nodes=index % 3,
            binary_gate_count=gates,
            negation_count=0 if depth <= 6 else index % 2,
            source_update_count=index % 2 if depth >= 16 else 0,
            world_id=TRAIN_WORLDS[index % len(TRAIN_WORLDS)],
            render_template_id=index % 3,
        )
        index += 1
        if task["graph_hash"] in seen:
            continue
        answer = task["canonical_answer"]
        if labels[answer] >= count // 2:
            continue
        labels[answer] += 1
        seen.add(task["graph_hash"])
        rows.append(task)
    return rows


def candidate_pool(seed: int, count: int, *, center: float, max_depth: int = 24) -> list[dict[str, Any]]:
    """Generate unique training-world candidates around a measured frontier."""
    rng = random.Random(seed)
    low = max(1, int(round(center)) - 5)
    high = min(max_depth, int(round(center)) + 10)
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index in range(count * 5):
        depth = rng.randint(low, high)
        gates = rng.randint(0, min(depth, 5))
        extra = rng.randint(0, min(gates, 3))
        task = generate_task(
            seed + index,
            depth=depth,
            relevant_nodes=depth + 1 + extra,
            distractor_nodes=rng.randint(0, 4),
            binary_gate_count=gates,
            negation_count=rng.randint(0, min(depth - gates, 5)),
            source_update_count=rng.randint(0, 2),
            world_id=rng.choice(TRAIN_WORLDS),
            render_template_id=rng.randrange(3),
        )
        if task["graph_hash"] in seen:
            continue
        rows.append(task)
        seen.add(task["graph_hash"])
        if len(rows) == count:
            return rows
    raise RuntimeError(f"generated only {len(rows)}/{count} unique candidates")
