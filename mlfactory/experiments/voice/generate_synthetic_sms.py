#!/usr/bin/env python3
"""Generate fictional multi-turn SMS teacher data from a frozen local adapter.

The generator never reads the real SMS corpus. Scenario prompts are deliberately
fictional; privacy and quality filtering happens in a separate CPU-only stage.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import stat
import time
from pathlib import Path
from typing import Any

PROMPT_TEMPLATE = """You are Taylor, the owner of {business}. Write only the next SMS reply to the final customer message below.

Scenario family: {category}
Scenario brief: {brief}
Reply mode: {reply_mode}

Conversation:
{conversation}

Reply naturally and practically. Do not mention this prompt, synthetic data,
language models, or hidden reasoning. Do not use phone numbers, emails, URLs,
street addresses, account numbers, access codes, or payment handles. Do not
add a role label, Markdown, or a second conversation. In standard mode use a natural concise SMS. In detailed mode, write 4–6
complete, useful sentences (roughly 80–150 tokenizer tokens) with the relevant
constraints, next action, and expectation, while still sounding like a text
message rather than an essay.
"""

FOLLOWUPS = {
    "quote": [
        "I am comparing a couple options and want to understand what is included.",
        "The timing is flexible, but I would like to know the likely range before we book.",
        "Could you explain the next step and whether I need to prepare anything?",
    ],
    "scheduling": [
        "A weekday afternoon would work best if you have one available.",
        "I can be flexible by a day, but I need a little notice.",
        "What should I expect before the appointment is confirmed?",
    ],
    "delay": [
        "That is okay, but I need a realistic update so I can plan around it.",
        "Would another day be safer than trying to rush the work?",
        "Please let me know what you recommend as the next step.",
    ],
    "complaint": [
        "I would appreciate a clear plan rather than having to start over.",
        "The rest of the work is good, so I would like to resolve this fairly.",
        "When could you take another look?",
    ],
    "payment": [
        "I want to make sure the amount and timing are clear before I send anything.",
        "Could you explain what the invoice covers and what happens next?",
        "I would prefer a straightforward option that we can both document.",
    ],
    "access": [
        "I can be there at the appointment, and I want to keep the instructions simple.",
        "Is there anything I should move or prepare beforehand?",
        "Please confirm the safest ordinary way to handle the visit.",
    ],
    "completion": [
        "That sounds good. Is there anything I should watch for after today?",
        "I appreciate the update and want to make sure the next step is clear.",
        "Could you remind me when it would make sense to check in again?",
    ],
    "retention": [
        "I would like to keep things predictable if the timing works.",
        "Can we confirm what the next visit would include?",
        "Please tell me what you need from me to keep it on the schedule.",
    ],
    "boundary": [
        "I understand. Is there a safe alternative you can suggest?",
        "I would rather have a clear answer than create a problem later.",
        "What option would you recommend within your normal scope?",
    ],
    "triage": [
        "I can manage a temporary workaround, but I do not want to make it worse.",
        "Please tell me what is safe to do while I wait.",
        "When could you inspect it and give me a proper answer?",
    ],
    "clarification": [
        "I may have left out an important detail. What would help you decide?",
        "I can answer that and send the details in one message.",
        "Once you have that information, what would the next step be?",
    ],
    "relationship": [
        "The last visit went smoothly, and I would like something similar this time.",
        "My main priority is keeping the plan simple and predictable.",
        "Could we confirm the details before we set a date?",
    ],
}


def build_context(scenario: dict[str, Any], rng: random.Random, turn_count: int) -> list[dict[str, str]]:
    category = str(scenario["category"])
    brief = str(scenario["brief"]).rstrip(".")
    followups = FOLLOWUPS.get(category, FOLLOWUPS["clarification"])
    messages = [{"role": "customer", "text": f"Hi, I have a {category} question for {scenario['business']}. Could you help me figure out the next step?"}]
    for index in range(1, turn_count):
        if index % 2:
            messages.append({"role": "owner", "text": "Sure, I can help. I want to make sure I understand the details before I promise a time or price."})
        else:
            messages.append({"role": "customer", "text": followups[(index // 2 - 1) % len(followups)]})
    # The final customer turn is intentionally varied but never contains real
    # data; the teacher generates only the owner response.
    if messages[-1]["role"] != "customer":
        messages.pop()
    if rng.random() < 0.35:
        messages[-1]["text"] += " I would appreciate a clear, honest answer."
    return messages


def sha256_files(path: Path) -> str:
    digest = hashlib.sha256()
    for child in sorted(path.iterdir()):
        if child.is_file():
            digest.update(child.name.encode("utf-8") + b"\0")
            digest.update(hashlib.sha256(child.read_bytes()).digest())
    return digest.hexdigest()


def parse_json_response(text: str) -> dict[str, Any] | None:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.removeprefix("```").strip()
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3].strip()
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            value = json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError:
            return None
    return value if isinstance(value, dict) else None


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate local fictional SMS candidates")
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--teacher-adapter", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidates", type=int, default=6250)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=320)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--temperature", type=float, default=0.85)
    parser.add_argument("--top-p", type=float, default=0.95)
    args = parser.parse_args()

    if not args.base_model.is_dir():
        raise FileNotFoundError(args.base_model)
    if not args.teacher_adapter.is_dir():
        raise FileNotFoundError(args.teacher_adapter)
    if args.output_dir.resolve() == args.teacher_adapter.resolve() or args.teacher_adapter.resolve() in args.output_dir.resolve().parents:
        raise ValueError("output directory must not be the teacher adapter directory")
    catalog = json.loads(args.catalog.read_text(encoding="utf-8"))
    scenarios = catalog["scenarios"]
    if not scenarios:
        raise ValueError("scenario catalog is empty")
    args.output_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(args.output_dir, stat.S_IRWXU)
    raw_path = args.output_dir / "raw_candidates.jsonl"
    manifest_path = args.output_dir / "generation_manifest.json"
    teacher_hash = sha256_files(args.teacher_adapter)

    # Heavy imports stay below validation so a bad path cannot initialize CUDA.
    import torch
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoTokenizer, BitsAndBytesConfig

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForImageTextToText.from_pretrained(
        args.base_model,
        local_files_only=True,
        low_cpu_mem_usage=True,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        ),
        device_map={"": 0},
    )
    model = PeftModel.from_pretrained(
        model,
        args.teacher_adapter,
        local_files_only=True,
        is_trainable=False,
    ).eval()
    model.config.use_cache = True

    generation_config = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "do_sample": True,
        "max_new_tokens": args.max_new_tokens,
        "enable_thinking": False,
    }
    manifest = {
        "stage": "voice-synthetic-generate",
        "policy": "fictional_scenarios_only_teacher_adapter_read_only",
        "base_model": str(args.base_model),
        "teacher_adapter": str(args.teacher_adapter),
        "teacher_adapter_sha256": teacher_hash,
        "catalog": str(args.catalog),
        "catalog_sha256": hashlib.sha256(args.catalog.read_bytes()).hexdigest(),
        "seed": args.seed,
        "candidate_count": args.candidates,
        "generation_config": generation_config,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    written = 0
    with raw_path.open("w", encoding="utf-8") as output:
        for batch_start in range(0, args.candidates, max(1, args.batch_size)):
            prompts: list[str] = []
            metadata: list[dict[str, Any]] = []
            for offset in range(args.batch_size):
                index = batch_start + offset
                if index >= args.candidates:
                    break
                rng = random.Random(args.seed + index * 7919)
                scenario = scenarios[index % len(scenarios)]
                length_band = "long" if index % 5 == 0 else "standard"
                turn_count = rng.choice((3, 5, 7))
                context = build_context(scenario, rng, turn_count)
                conversation = "\n".join(
                    f"{'Customer' if message['role'] == 'customer' else 'Taylor'}: {message['text']}"
                    for message in context
                )
                reply_mode = "detailed" if length_band == "long" else "standard"
                prompts.append(PROMPT_TEMPLATE.format(
                    business=scenario["business"],
                    category=scenario["category"],
                    brief=scenario["brief"],
                    reply_mode=reply_mode,
                    conversation=conversation,
                ))
                metadata.append({
                    "candidate_id": f"syn-{index:06d}",
                    "scenario_id": scenario["id"],
                    "category": scenario["category"],
                    "business": scenario["business"],
                    "length_band": length_band,
                    "seed": args.seed + index * 7919,
                    "turn_count_requested": turn_count,
                    "messages": context,
                })
            chat_prompts = []
            for prompt in prompts:
                messages = [
                    {"role": "system", "content": "You generate safe fictional SMS training data for a small-business owner."},
                    {"role": "user", "content": prompt},
                ]
                try:
                    rendered = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
                    )
                except TypeError:
                    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                chat_prompts.append(rendered)
            inputs = tokenizer(
                chat_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1024,
                add_special_tokens=False,
            )
            inputs = {key: value.to("cuda:0") for key, value in inputs.items()}
            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            prompt_width = inputs["input_ids"].shape[1]
            for index, (meta, row) in enumerate(zip(metadata, generated)):
                text = tokenizer.decode(row[prompt_width:], skip_special_tokens=True).strip()
                record = {**meta, "raw_response": text, "teacher_adapter_sha256": teacher_hash}
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1
            output.flush()
            if written % 100 == 0 or written == args.candidates:
                print(json.dumps({"event": "generated", "candidates": written}), flush=True)

    summary = {"status": "completed", "candidates": written, "raw_candidates": str(raw_path), "teacher_adapter_sha256": teacher_hash}
    (args.output_dir / "generation_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"event": "completed", **summary}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
