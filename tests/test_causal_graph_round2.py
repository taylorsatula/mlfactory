"""Tests for the single fixed Round 2 contract."""
from __future__ import annotations

from collections import Counter

from mlfactory.experiments.causal_graph_round2.contract import chat_prompt, parse, target
from mlfactory.experiments.causal_graph_round2.generator import (
    TRAIN_WORLDS,
    audit_task,
    generate_task,
    probe_tasks,
    regenerate_matches,
    verify_task,
)


class FakeTokenizer:
    eos_token = "<eos>"

    def apply_chat_template(self, messages, **kwargs):
        assert messages[0]["role"] == "system"
        assert "/nothink" in messages[0]["content"]
        assert kwargs["enable_thinking"] is False
        return "<system>/nothink</system><user>prompt</user><assistant><think>\n\n</think>\n\n"

    def __call__(self, text, **kwargs):
        return {"input_ids": list(range(len(text.split()) + 1))}

    def decode(self, ids, **kwargs):
        return self._decoded


def test_chat_prompt_explicitly_disables_thinking() -> None:
    rendered = chat_prompt(FakeTokenizer(), "prompt")
    assert rendered.endswith("<think>\n\n</think>\n\n")


def test_fixed_target_and_parser() -> None:
    task = generate_task(42, depth=4, relevant_nodes=5, binary_gate_count=0, world_id="greenhouse")
    text = target(task)
    result = parse(text, task)
    assert result["correct"]
    assert len(result["trace_bits"]) == task["relevant_nodes"]
    assert text == task["canonical_trace"]


def test_parser_rejects_extra_text_and_thinking() -> None:
    task = generate_task(43, depth=2, relevant_nodes=3, binary_gate_count=0, world_id="security")
    assert parse(target(task) + "\nextra", task)["parse_failure"]
    assert parse("<think>hidden</think>\n" + target(task), task)["thinking_marker"]


def test_audit_rejects_no_complete_target() -> None:
    task = generate_task(44, depth=2, relevant_nodes=3, binary_gate_count=0, world_id="factory")
    tokenizer = FakeTokenizer()
    tokenizer._decoded = task["canonical_trace"]
    accepted, reason = audit_task(task, tokenizer, max_target_tokens=128, max_length=1024)
    assert reason is None
    assert accepted is not None
    broken = dict(task)
    broken["canonical_trace"] = "TRACE: T"
    tokenizer._decoded = broken["canonical_trace"]
    accepted, reason = audit_task(broken, tokenizer)
    assert accepted is None
    assert reason


def test_generator_replay_and_verification() -> None:
    task = generate_task(45, depth=6, relevant_nodes=8, binary_gate_count=1, distractor_nodes=2, world_id="household")
    verify_task(task)
    assert regenerate_matches(task)


def test_all_training_worlds() -> None:
    for index, world in enumerate(TRAIN_WORLDS):
        task = generate_task(100 + index, depth=2, relevant_nodes=3, binary_gate_count=0, world_id=world)
        assert task["world_id"] == world
        assert "TRACE:" in task["canonical_trace"]


def test_probe_balance() -> None:
    rows = probe_tasks(500, [1, 2, 4], 8, tuple(TRAIN_WORLDS))
    for depth in (1, 2, 4):
        answers = Counter(row["canonical_answer"] for row in rows if row["depth"] == depth)
        assert answers == {"YES": 4, "NO": 4}
