"""The one fixed response contract used throughout CausalGraph Round 2."""
from __future__ import annotations

import re
from typing import Any

SYSTEM_PROMPT = "You are a precise symbolic state evaluator. /nothink"
FINAL_RE = re.compile(r"FINAL:\s*(YES|NO)\s*$")
TRACE_RE = re.compile(r"^TRACE:\s*([TF](?:\s*,\s*[TF])*)\s*\nFINAL:\s*(YES|NO)\s*$")
THINK_MARKERS = ("<think>", "</think>", "reasoning_content")


def instruction(state_count: int) -> str:
    return (
        "Do not explain or use prose. Output exactly two lines. First output "
        f"TRACE: followed by exactly {state_count} comma-separated T/F values: "
        "all query-relevant source states in their listed order, then every "
        "derived state in causal order through the query. Then output exactly "
        "FINAL: YES or FINAL: NO. Output nothing else."
    )


def chat_prompt(tokenizer: Any, prompt: str) -> str:
    """Render Qwen3.5 with both `/nothink` and thinking disabled."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    try:
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError as exc:
        raise RuntimeError("tokenizer does not support enable_thinking=False") from exc
    # Qwen's supported non-thinking template ends with a closed empty scaffold.
    # Removing it causes Qwen3.5 to open a new thinking trace during generation.
    if not rendered.endswith("<think>\n\n</think>\n\n"):
        raise RuntimeError("unexpected Qwen non-thinking chat template")
    return rendered


def target(task: dict[str, Any]) -> str:
    graph = task["graph"]["nodes"]
    relevant = [
        node_id for node_id in task["topological_order"]
        if graph[node_id]["kind"] != "distractor"
    ]
    bits = ",".join("T" if task["canonical_node_states"][node_id] else "F" for node_id in relevant)
    return f"TRACE: {bits}\nFINAL: {task['canonical_answer']}"


def parse(text: str, task: dict[str, Any]) -> dict[str, Any]:
    """Parse and score the complete trace and terminal answer."""
    raw = text or ""
    thinking = any(marker.lower() in raw.lower() for marker in THINK_MARKERS)
    match = TRACE_RE.fullmatch(raw.strip()) if not thinking else None
    graph = task["graph"]["nodes"]
    relevant = [
        node_id for node_id in task["topological_order"]
        if graph[node_id]["kind"] != "distractor"
    ]
    gold_bits = ["T" if task["canonical_node_states"][node_id] else "F" for node_id in relevant]
    if match is None:
        return {
            "parsed_answer": None,
            "trace_bits": None,
            "gold_trace_bits": gold_bits,
            "terminal_correct": False,
            "trace_exact": False,
            "trace_bit_accuracy": 0.0,
            "correct": False,
            "parse_failure": True,
            "thinking_marker": thinking,
        }
    predicted = [value.strip() for value in match.group(1).split(",")]
    answer = match.group(2)
    bit_correct = sum(left == right for left, right in zip(predicted, gold_bits))
    bit_accuracy = bit_correct / max(1, max(len(predicted), len(gold_bits)))
    trace_exact = predicted == gold_bits
    terminal_correct = answer == task["canonical_answer"]
    return {
        "parsed_answer": answer,
        "trace_bits": predicted,
        "gold_trace_bits": gold_bits,
        "terminal_correct": terminal_correct,
        "trace_exact": trace_exact,
        "trace_bit_accuracy": bit_accuracy,
        "correct": trace_exact and terminal_correct,
        "parse_failure": False,
        "thinking_marker": False,
    }
