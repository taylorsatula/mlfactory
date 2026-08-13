"""Deterministic fixed-contract evaluation for base and QLoRA checkpoints."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .contract import chat_prompt, parse
from .progress import emit


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    base = Path(args.base_model).resolve()
    tokenizer = AutoTokenizer.from_pretrained(str(base), trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(base),
        trust_remote_code=True,
        quantization_config=quant,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    if args.adapter:
        model = PeftModel.from_pretrained(model, str(Path(args.adapter).resolve()))
    model.eval()
    device = next(parameter for parameter in model.parameters() if parameter.device.type != "meta").device
    tasks = [json.loads(line) for line in Path(args.input).open(encoding="utf-8") if line.strip()]
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    emit(args.dashboard_file, "stage_start", stage=args.stage, current=0, total=len(tasks))

    with output_path.open("w", encoding="utf-8") as output:
        for start in range(0, len(tasks), args.batch_size):
            chunk = tasks[start:start + args.batch_size]
            prompts = [chat_prompt(tokenizer, task["rendered_prompt"]) for task in chunk]
            encoded = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_length,
            ).to(device)
            prompt_width = encoded["input_ids"].shape[1]
            with torch.inference_mode():
                generated = model.generate(
                    **encoded,
                    do_sample=False,
                    max_new_tokens=args.max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                )
            texts = [tokenizer.decode(generated[index][prompt_width:], skip_special_tokens=True) for index in range(len(chunk))]
            prompt_lengths = encoded["attention_mask"].sum(dim=1).detach().cpu().tolist()
            for task, text, prompt_tokens in zip(chunk, texts, prompt_lengths):
                scores = parse(text, task)
                output_tokens = len(tokenizer(text, add_special_tokens=False)["input_ids"])
                record = {
                    "example_id": task["id"],
                    "checkpoint": args.checkpoint,
                    "world": task["world_id"],
                    "depth": task["depth"],
                    "relevant_nodes": task["relevant_nodes"],
                    "distractor_nodes": task["distractor_nodes"],
                    "binary_gate_count": task["binary_gate_count"],
                    "negation_count": task["negation_count"],
                    "source_update_count": task["source_update_count"],
                    "gold_answer": task["canonical_answer"],
                    "raw_model_output": text,
                    "prompt_tokens": int(prompt_tokens),
                    "output_tokens": output_tokens,
                    "budget_exhausted": output_tokens >= args.max_new_tokens,
                    **scores,
                }
                records.append(record)
                output.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            output.flush()
            emit(
                args.dashboard_file,
                "evaluation_progress",
                stage=args.stage,
                current=len(records),
                total=len(tasks),
                parse_failures=sum(row["parse_failure"] for row in records),
            )

    parsed = [row for row in records if not row["parse_failure"]]
    summary = {
        "checkpoint": args.checkpoint,
        "n": len(records),
        "accuracy": sum(row["correct"] for row in records) / max(1, len(records)),
        "terminal_accuracy": sum(row["terminal_correct"] for row in records) / max(1, len(records)),
        "trace_bit_accuracy": sum(row["trace_bit_accuracy"] for row in records) / max(1, len(records)),
        "conditional_accuracy": sum(row["correct"] for row in parsed) / max(1, len(parsed)),
        "parse_failures": sum(row["parse_failure"] for row in records),
        "budget_exhausted": sum(row["budget_exhausted"] for row in records),
        "thinking_markers": sum(row["thinking_marker"] for row in records),
    }
    emit(args.dashboard_file, "stage_complete", stage=args.stage, current=len(records), total=len(tasks), **summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--checkpoint", default="HF")
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--dashboard-file")
    parser.add_argument("--stage", default="evaluation")
    print(json.dumps(evaluate(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
