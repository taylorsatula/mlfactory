"""Model probing for CausalGraph, with no judge or self-consistency."""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Iterable

import requests

from .generator import TRAIN_WORLDS, WORLD_IDS, generate_task, verify_task

FINAL_RE = re.compile(r"FINAL\s*:\s*(YES|NO)\b", re.IGNORECASE)


def local_api_key() -> str:
    if os.environ.get("CAUSAL_GRAPH_API_KEY"):
        return os.environ["CAUSAL_GRAPH_API_KEY"]
    path = Path("/etc/llama-server/api-keys")
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                return line
    return "none"


class CausalGraphClient:
    def __init__(self, base_url: str = "http://127.0.0.1:3090/v1", model: str = "f16-jackrongds4qwen", timeout: float = 180.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {local_api_key()}", "Content-Type": "application/json"})

    def complete(self, prompt: str, max_tokens: int = 256) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "Solve the symbolic state problem. Do not restate the prompt. Use at most four short derivation lines, then put exactly FINAL: YES or FINAL: NO on the last line."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0,
            "top_p": 1,
            "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        last: Exception | None = None
        for attempt in range(3):
            try:
                response = self.session.post(f"{self.base_url}/chat/completions", json=payload, timeout=self.timeout)
                response.raise_for_status()
                data = response.json()
                message = data.get("choices", [{}])[0].get("message", {})
                content = message.get("content")
                if content is None:
                    content = message.get("reasoning_content", "")
                return str(content)
            except Exception as exc:
                last = exc
                if attempt < 2:
                    time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"model request failed: {last}")


def parse_final(text: str) -> tuple[str | None, bool]:
    matches = FINAL_RE.findall(text or "")
    if not matches:
        return None, True
    return matches[-1].upper(), False


def make_coarse_tasks(seed: int = 20260811, examples_per_depth: int = 64, worlds: Iterable[str] = TRAIN_WORLDS) -> list[dict[str, Any]]:
    depths = [2, 4, 6, 8, 10, 12, 16, 20, 24, 32]
    worlds = tuple(worlds)
    tasks: list[dict[str, Any]] = []
    for depth in depths:
        gates = min(max(depth // 5, 1), 6)
        negations = min(max(depth // 9, 0), depth - gates)
        relevant = depth + 1 + min(gates, 3)
        # Construct an exactly balanced YES/NO slice at each depth.  This
        # prevents a depth-dependent Boolean prior from masquerading as a
        # competence curve, especially for long chains of negations.
        target_each = examples_per_depth // 2
        counts = {"YES": 0, "NO": 0}
        index = 0
        depth_start = sum(examples_per_depth for _ in depths[:depths.index(depth)])
        while len(tasks) < depth_start + target_each * 2:
            task_seed = seed + depth * 100_000 + index
            task = generate_task(
                task_seed, depth=depth, relevant_nodes=relevant,
                distractor_nodes=index % 3, binary_gate_count=gates,
                negation_count=negations, source_update_count=index % 2,
                world_id=worlds[(index + depth) % len(worlds)], render_template_id=index % 3,
            )
            index += 1
            if counts[task["canonical_answer"]] >= target_each:
                continue
            verify_task(task)
            counts[task["canonical_answer"]] += 1
            tasks.append(task)
        # For odd budgets, add one deterministic example after the balanced
        # pair construction.
        while counts["YES"] + counts["NO"] < examples_per_depth:
            task = generate_task(seed + depth * 100_000 + index, depth=depth, relevant_nodes=relevant, distractor_nodes=index % 3, binary_gate_count=gates, negation_count=negations, source_update_count=index % 2, world_id=worlds[(index + depth) % len(worlds)], render_template_id=index % 3)
            index += 1
            if counts[task["canonical_answer"]] <= target_each:
                verify_task(task)
                counts[task["canonical_answer"]] += 1
                tasks.append(task)
    return tasks


def evaluate_tasks(
    tasks: list[dict[str, Any]], client: CausalGraphClient, output_path: str | Path,
    checkpoint: str = "BASELINE", pause_seconds: float = 0.0,
    max_tokens: int = 256,
) -> list[dict[str, Any]]:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    with output_path.open("w", encoding="utf-8") as handle:
        for index, task in enumerate(tasks, 1):
            started = time.time()
            raw = client.complete(task["rendered_prompt"], max_tokens=max_tokens)
            parsed, parse_failure = parse_final(raw)
            record = {
                "example_id": task["id"], "checkpoint": checkpoint,
                "world": task["world_id"], "depth": task["depth"],
                "relevant_nodes": task["relevant_nodes"], "distractor_nodes": task["distractor_nodes"],
                "binary_gate_count": task["binary_gate_count"], "negation_count": task["negation_count"],
                "source_update_count": task["source_update_count"],
                "raw_model_output": raw, "parsed_answer": parsed,
                "gold_answer": task["canonical_answer"],
                "correct": bool(parsed == task["canonical_answer"]),
                "parse_failure": bool(parse_failure),
                "prompt_chars": len(task["rendered_prompt"]),
                "elapsed_seconds": round(time.time() - started, 4),
            }
            records.append(record)
            handle.write(json.dumps({**task, "model_output": record}, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            if pause_seconds:
                time.sleep(pause_seconds)
            if index % 25 == 0:
                print(f"evaluated {index}/{len(tasks)}", flush=True)
    return records


def write_tasks(path: str | Path, tasks: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for task in tasks:
            handle.write(json.dumps(task, ensure_ascii=False, sort_keys=True) + "\n")
