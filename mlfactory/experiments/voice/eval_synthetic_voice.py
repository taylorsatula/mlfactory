#!/usr/bin/env python3
"""Paired local evaluation of base, frozen teacher, and second SMS adapters."""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mlfactory.experiments.voice.privacy_review_synthetic import PII_PATTERNS
from mlfactory.experiments.voice.train_synthetic_voice import collate, load_jsonl, render_prompt, tokenize_rows


def load_policy(base: Path, adapter: Path | None, torch: Any) -> tuple[Any, Any]:
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForImageTextToText.from_pretrained(
        base,
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
    if adapter is not None:
        model = PeftModel.from_pretrained(model, adapter, local_files_only=True, is_trainable=False)
    model.eval()
    return model, tokenizer


def loss_on_rows(model: Any, rows: list[dict[str, Any]], tokenizer: Any, device: Any, torch: Any, limit: int) -> float | None:
    values: list[float] = []
    args = type("Args", (), {"pad_token_id": tokenizer.pad_token_id})()
    with torch.no_grad():
        for row in rows[:limit]:
            batch = collate([row], tokenizer.pad_token_id, torch)
            batch = {key: value.to(device) for key, value in batch.items()}
            value = float(model(**batch).loss.detach().cpu())
            if math.isfinite(value):
                values.append(value)
    return sum(values) / len(values) if values else None


def generate(model: Any, tokenizer: Any, record: dict[str, Any], device: Any, torch: Any, max_new_tokens: int) -> str:
    prompt = render_prompt(tokenizer, record["messages"])
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def privacy_failure(text: str) -> str | None:
    if re.search(r"<think>|</think>|```|as an ai|language model", text, re.I):
        return "analysis_or_wrapper"
    for name, pattern in PII_PATTERNS.items():
        if pattern.search(text):
            return name
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--teacher-adapter", type=Path, required=True)
    parser.add_argument("--second-adapter", type=Path, required=True)
    parser.add_argument("--eval-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--loss-examples", type=int, default=256)
    parser.add_argument("--generation-examples", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True, mode=0o700)

    import torch
    records = load_jsonl(args.eval_file)
    tokenizer = __import__("transformers").AutoTokenizer.from_pretrained(args.base_model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    rows = tokenize_rows(records, tokenizer, 512, 192)
    device = torch.device("cuda:0")
    results: dict[str, Any] = {}
    paired: list[dict[str, Any]] = []
    for label, adapter in (("base", None), ("teacher", args.teacher_adapter), ("second", args.second_adapter)):
        model, loaded_tokenizer = load_policy(args.base_model, adapter, torch)
        loss = loss_on_rows(model, rows, loaded_tokenizer, device, torch, args.loss_examples)
        outputs: list[str] = []
        failures = 0
        for record in records[: args.generation_examples]:
            text = generate(model, loaded_tokenizer, record, device, torch, args.max_new_tokens)
            outputs.append(text)
            failure = privacy_failure(text)
            failures += int(failure is not None)
            if label != "base":
                paired.append({"example_id": record.get("example_id"), "model": label, "response": text, "privacy_failure": failure})
        lengths = [len(loaded_tokenizer(text, add_special_tokens=False)["input_ids"]) for text in outputs]
        results[label] = {
            "heldout_loss": loss,
            "generated_examples": len(outputs),
            "mean_response_tokens": sum(lengths) / max(1, len(lengths)),
            "p95_response_tokens": sorted(lengths)[min(len(lengths) - 1, int(len(lengths) * 0.95))] if lengths else 0,
            "privacy_failures": failures,
        }
        del model
        torch.cuda.empty_cache()
    (args.output_dir / "paired_samples.json").write_text(json.dumps(paired, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "comparison.json").write_text(json.dumps({"status": "completed", "models": results}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"event": "completed", "models": results}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
