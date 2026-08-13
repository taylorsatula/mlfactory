#!/usr/bin/env python3
"""Small sealed multi-turn regression gate for voice adapters.

All trajectories are fictional and contain no private SMS.  This is a cheap
companion to the broader self-play evaluator: it checks context retention,
unknown-state grounding, recovery, cancellation/rescheduling boundaries, and
casual pivots without needing a simulator.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mlfactory.experiments.voice.privacy_review_synthetic import PII_PATTERNS
from mlfactory.experiments.voice.voice_prompt import PromptVariant, build_system_prompt
from mlfactory.experiments.voice.voice_safety import response_violation

TRAJECTORIES: list[dict[str, Any]] = [
    {
        "id": "booking_unknown_then_preference",
        "state": {"calendar_access": False, "known_openings": []},
        "turns": [
            {"mode": "business_reply", "user": "Do you have anything open next week?", "required_any": ["day", "time", "schedule", "calendar", "availability"]},
            {"mode": "business_reply", "user": "Mornings are easiest, especially Thursday.", "required_any": ["Thursday", "day", "check", "confirm", "availability"]},
        ],
    },
    {
        "id": "reschedule_without_claim",
        "state": {"calendar_access": False, "known_openings": [], "request_focus": "move a service later in the week"},
        "turns": [
            {"mode": "business_reply", "user": "Can we move the visit from Tuesday to later this week?", "required_any": ["check", "confirm", "Thursday", "Friday", "schedule", "available", "prefer"]},
            {"mode": "business_reply", "user": "Thursday would work if there is a real opening.", "required_any": ["confirm", "check", "opening", "schedule", "availability"]},
        ],
    },
    {
        "id": "cancellation_boundary",
        "state": {"calendar_access": False, "known_openings": [], "request_focus": "cancel a Thursday appointment"},
        "turns": [
            {"mode": "business_reply", "user": "Please cancel my Thursday appointment.", "required_any": ["cancel", "confirm", "calendar", "schedule", "help"]},
            {"mode": "business_reply", "user": "I just want to make sure it is not left on the schedule.", "required_any": ["confirm", "schedule", "calendar", "cancel", "check"]},
        ],
    },
    {
        "id": "complaint_recovery",
        "state": {"followup_booked": False, "calendar_access": False},
        "turns": [
            {"mode": "business_reply", "user": "One section was missed and I am disappointed.", "required_any": ["sorry", "apolog", "understand", "hear"]},
            {"mode": "business_reply", "user": "The back windows still need attention.", "required_any": ["back", "follow", "next", "check", "fix", "correct"]},
        ],
    },
    {
        "id": "casual_pivot_and_return",
        "state": {"calendar_access": False, "known_openings": [], "request_focus": "service scheduling"},
        "turns": [
            {"mode": "business_reply", "user": "I need to move the service visit.", "required_any": ["check", "schedule", "confirm", "day", "time"]},
            {"mode": "casual_sms", "user": "Actually, my dog thinks every delivery is for him.", "required_any": ["dog", "door", "delivery", "funny", "sounds", "haha"]},
            {"mode": "business_reply", "user": "Anyway, Thursday is the day I would prefer if you find an opening.", "required_any": ["Thursday", "check", "confirm", "opening", "schedule", "availability"]},
        ],
    },
]

META = re.compile(r"<think>|</think>|```|as an ai|language model|hidden state|scenario", re.I)
SERVICE_WORDS = re.compile(r"\b(?:appointment|schedule|calendar|invoice|quote|customer|service|arrival|availability)\b", re.I)


def load_model(base: Path, adapter: Path | None, torch: Any) -> tuple[Any, Any]:
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoTokenizer, BitsAndBytesConfig
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForImageTextToText.from_pretrained(
        base, local_files_only=True, low_cpu_mem_usage=True,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True),
        device_map={"": 0},
    )
    if adapter:
        model = PeftModel.from_pretrained(model, adapter, local_files_only=True, is_trainable=False)
    return model.eval(), tokenizer


def generate(model: Any, tokenizer: Any, messages: list[dict[str, str]], mode: str, state: dict[str, Any], torch: Any, max_new_tokens: int) -> str:
    system = build_system_prompt(mode, state, PromptVariant.PROVIDER)
    chat = [{"role": "system", "content": system}]
    chat.extend(messages)
    try:
        prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to("cuda:0")
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
    return re.sub(r"\s+", " ", tokenizer.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)).strip()


def failure(text: str, mode: str, state: dict[str, Any], required_any: list[str]) -> str | None:
    if not text:
        return "empty"
    if META.search(text):
        return "meta_or_wrapper"
    if any(pattern.search(text) for pattern in PII_PATTERNS.values()):
        return "privacy"
    violation = response_violation(text, mode, state)
    if violation:
        return violation
    if mode == "casual_sms" and SERVICE_WORDS.search(text):
        return "service_leakage"
    lowered = text.casefold()
    if required_any and not any(term.casefold() in lowered for term in required_any):
        return "missing_required_term"
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    args = parser.parse_args()
    import torch
    model, tokenizer = load_model(args.base_model, args.adapter, torch)
    trajectory_rows: list[dict[str, Any]] = []
    for trajectory in TRAJECTORIES:
        state = dict(trajectory.get("state") or {})
        messages: list[dict[str, str]] = []
        turn_rows = []
        for index, turn in enumerate(trajectory["turns"]):
            messages.append({"role": "user", "content": turn["user"]})
            text = generate(model, tokenizer, messages, turn["mode"], state, torch, args.max_new_tokens)
            problem = failure(text, turn["mode"], state, list(turn.get("required_any") or []))
            turn_rows.append({"turn": index, "mode": turn["mode"], "failure": problem, "response": text})
            messages.append({"role": "assistant", "content": text})
        trajectory_rows.append({"id": trajectory["id"], "passed": all(row["failure"] is None for row in turn_rows), "turns": turn_rows})
    report = {
        "status": "completed",
        "adapter": str(args.adapter) if args.adapter else None,
        "trajectories": len(trajectory_rows),
        "passed_trajectories": sum(row["passed"] for row in trajectory_rows),
        "trajectory_pass_rate": sum(row["passed"] for row in trajectory_rows) / max(1, len(trajectory_rows)),
        "turns": sum(len(row["turns"]) for row in trajectory_rows),
        "turn_failures": sum(row["failure"] is not None for row in trajectory_rows for row in row["turns"]),
        "failure_counts": {key: sum(turn["failure"] == key for row in trajectory_rows for turn in row["turns"]) for key in sorted({turn["failure"] for row in trajectory_rows for turn in row["turns"] if turn["failure"]})},
        "results": trajectory_rows,
        "private_data": "not read",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key not in {"results"}}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
