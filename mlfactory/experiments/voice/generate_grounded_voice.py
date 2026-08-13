#!/usr/bin/env python3
"""Generate fictional, state-grounded SMS candidates from a frozen local teacher.

This generator deliberately does not read the private SMS corpus.  Its scenario
plans are fictional and every fact required by a target is visible in the
conversation or verified_state.  Partitions can run concurrently on separate
GPUs; only aggregate generation metadata is written beside the synthetic data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import stat
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mlfactory.experiments.voice.voice_prompt import PromptVariant, build_system_prompt

STYLE_SUFFIXES = {
    "terse": ("Just need a clear answer.", "What is the next step?"),
    "warm": ("Thanks for helping me sort this out.", "I appreciate the help."),
    "fragmented": ("Sorry, short version: what should I do?", "Just checking what happens next."),
    "checking_in": ("Just checking in so I can plan.", "A quick update would help."),
    "change_of_mind": ("Actually, I may need a different option.", "I may have changed what works for me."),
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_catalog(path: Path) -> dict[str, Any]:
    catalog = json.loads(path.read_text(encoding="utf-8"))
    scenarios = catalog.get("scenarios")
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError("scenario catalog has no scenarios")
    return catalog


def make_case(scenario: dict[str, Any], index: int) -> tuple[list[dict[str, str]], str, str, dict[str, Any]]:
    rng = random.Random(index * 7919 + 20260807)
    history = [{"role": role, "text": text} for role, text in scenario["history"]]
    # Scenario authors may stop on a customer clarification. Insert a short
    # visible business acknowledgement before the generated final customer
    # turn so every training context has strict customer/owner alternation.
    if history and history[-1]["role"] == "customer":
        history.append({"role": "owner", "text": "Thanks for clarifying. I want to make sure the next step is clear."})
    final_options = list(scenario.get("final_customers") or [])
    final = final_options[index % len(final_options)]
    style_names = list(STYLE_SUFFIXES)
    style = style_names[(index // max(1, len(final_options))) % len(style_names)]
    suffix = STYLE_SUFFIXES[style][(index // 3) % len(STYLE_SUFFIXES[style])]
    # Keep some candidates clean and make the rest reflect the short,
    # checking-in, change-heavy customer turns found in the target channel.
    if index % 4 != 0 and suffix.casefold() not in final.casefold():
        final = f"{final} {suffix}"
    if index % 11 == 0 and history:
        history[-1]["text"] = f"{history[-1]['text']} {STYLE_SUFFIXES['checking_in'][0]}"
    mode = "casual_sms" if scenario["family"] == "casual_pivot" else "business_reply"
    state = dict(scenario.get("state") or {})
    metadata = {
        "scenario_id": scenario["id"],
        "family": scenario["family"],
        "domain": scenario["domain"],
        "split": scenario["split"],
        "length_band": scenario.get("length_band", "standard"),
        "style": style,
        "topic_terms": list(scenario.get("topic_terms") or []),
        "scenario_variant": index,
    }
    return history + [{"role": "customer", "text": final}], mode, json.dumps(state, ensure_ascii=False, sort_keys=True), metadata


def build_prompt(tokenizer: Any, messages: list[dict[str, str]], mode: str, state: dict[str, Any]) -> str:
    system = build_system_prompt(mode, state, PromptVariant.PROVIDER)
    system += (
        "\n\nYou are authoring one fictional training target. Reply only with the next "
        "SMS from the business representative. Do not explain your reasoning, repeat "
        "the conversation, mention the scenario, or use labels. Keep it natural and "
        "appropriately brief for SMS. A customer-supplied date, count, or preference "
        "may be acknowledged, but never turn an unverified request into a completed "
        "booking, quote, arrival, payment, message, or other action."
    )
    chat = [{"role": "system", "content": system}]
    chat.extend({"role": "user" if row["role"] == "customer" else "assistant", "content": row["text"]} for row in messages)
    try:
        return tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)


def clean_target(text: str) -> str:
    value = re.sub(r"<think>.*?</think>", "", str(text or ""), flags=re.I | re.S).strip()
    value = re.sub(r"^\s*(?:assistant|business representative|owner|taylor)\s*:\s*", "", value, flags=re.I)
    value = value.replace("```", "").strip()
    return re.sub(r"\s+", " ", value)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, default=None)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-count", type=int, default=800)
    parser.add_argument("--part-index", type=int, default=0)
    parser.add_argument("--part-count", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--temperature", type=float, default=0.78)
    parser.add_argument("--top-p", type=float, default=0.92)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.part_index < 0 or args.part_index >= args.part_count:
        raise ValueError("part-index must be within part-count")
    if not args.base_model.is_dir() or not (args.base_model / "config.json").is_file():
        raise FileNotFoundError(args.base_model)
    if args.adapter and not args.adapter.is_dir():
        raise FileNotFoundError(args.adapter)
    catalog = load_catalog(args.catalog)
    scenarios = catalog["scenarios"]
    args.output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(args.output.parent, stat.S_IRWXU)

    import torch
    from transformers import AutoModelForImageTextToText, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

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
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter, local_files_only=True, is_trainable=False)
    model.eval()
    model.config.use_cache = True

    selected = list(range(args.part_index, args.candidate_count, args.part_count))
    manifest = {
        "status": "started",
        "policy": "fictional_visible_state_only",
        "base_model": str(args.base_model),
        "base_model_config_sha256": sha256_file(args.base_model / "config.json"),
        "adapter": str(args.adapter) if args.adapter else None,
        "catalog": str(args.catalog),
        "catalog_sha256": sha256_file(args.catalog),
        "candidate_count": args.candidate_count,
        "part_index": args.part_index,
        "part_count": args.part_count,
        "selected_count": len(selected),
        "seed": args.seed,
        "device": args.device,
        "generation": {"temperature": args.temperature, "top_p": args.top_p, "max_new_tokens": args.max_new_tokens},
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    manifest_path = args.output.with_suffix(args.output.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    written = 0
    with args.output.open("w", encoding="utf-8") as stream:
        for start in range(0, len(selected), max(1, args.batch_size)):
            batch_indices = selected[start : start + max(1, args.batch_size)]
            prompts, metadata = [], []
            for global_index in batch_indices:
                scenario = scenarios[global_index % len(scenarios)]
                messages, mode, state_text, meta = make_case(scenario, global_index)
                state = json.loads(state_text)
                prompts.append(build_prompt(tokenizer, messages, mode, state))
                metadata.append({
                    "candidate_id": f"grounded-{global_index:06d}",
                    "messages": messages,
                    "mode": mode,
                    "verified_state": state,
                    "source": "fictional_base_teacher",
                    "base_model_config_sha256": manifest["base_model_config_sha256"],
                    **meta,
                })
            inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=1536, add_special_tokens=False)
            inputs = {key: value.to(args.device) for key, value in inputs.items()}
            torch.manual_seed(args.seed + batch_indices[0] * 7919)
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
            width = inputs["input_ids"].shape[1]
            for meta, row in zip(metadata, generated):
                meta["raw_response"] = clean_target(tokenizer.decode(row[width:], skip_special_tokens=True))
                meta["generation_seed"] = args.seed + int(meta["scenario_variant"]) * 7919
                stream.write(json.dumps(meta, ensure_ascii=False) + "\n")
                written += 1
            stream.flush()
            if written % 100 < len(batch_indices) or written == len(selected):
                print(json.dumps({"event": "generated", "part": args.part_index, "count": written}), flush=True)
    manifest["status"] = "completed"
    manifest["written"] = written
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"event": "completed", "output": str(args.output), "written": written, "part": args.part_index}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
