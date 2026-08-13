#!/usr/bin/env python3
"""Train a second adapter on selected synthetic SMS while preserving its teacher.

The teacher adapter is loaded read-only from a separate directory and saved only
as a hash in this run. The output adapter is always written under this run dir.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mlfactory.core.metrics import MetricsLogger
from mlfactory.experiments.voice.voice_prompt import build_system_prompt

SYSTEM_PROMPT = build_system_prompt()


def hash_directory(path: Path) -> str:
    digest = hashlib.sha256()
    for child in sorted(path.iterdir()):
        if child.is_file():
            digest.update(child.name.encode("utf-8") + b"\0")
            digest.update(hashlib.sha256(child.read_bytes()).digest())
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def render_prompt(tokenizer: Any, messages: list[dict[str, str]], mode: str = "business_reply", state: dict[str, Any] | None = None, variant: str | None = None) -> str:
    converted = [{"role": "system", "content": build_system_prompt(mode, state, variant)}]
    for message in messages:
        role = str(message.get("role"))
        converted.append({
            "role": "user" if role == "customer" else "assistant",
            "content": str(message.get("text") or ""),
        })
    try:
        return tokenizer.apply_chat_template(
            converted, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        return tokenizer.apply_chat_template(converted, tokenize=False, add_generation_prompt=True)


def tokenize_rows(records: list[dict[str, Any]], tokenizer: Any, max_length: int, max_target_tokens: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        messages = record.get("messages")
        target = str(record.get("target") or "").strip()
        if not isinstance(messages, list) or not target:
            continue
        target_ids = tokenizer(target + (tokenizer.eos_token or ""), add_special_tokens=False)["input_ids"][:max_target_tokens]
        prompt_ids = tokenizer(render_prompt(tokenizer, messages, str(record.get("mode") or "business_reply"), record.get("verified_state"), record.get("prompt_variant")), add_special_tokens=False)["input_ids"]
        prompt_budget = max_length - len(target_ids)
        if prompt_budget < 1 or not target_ids:
            continue
        prompt_ids = prompt_ids[-prompt_budget:]
        rows.append({
            "input_ids": prompt_ids + target_ids,
            "labels": [-100] * len(prompt_ids) + target_ids,
            "example_id": str(record.get("example_id") or "unknown"),
            "length_band": str(record.get("length_band") or "standard"),
        })
    if not rows:
        raise RuntimeError("no synthetic examples survived tokenization")
    return rows


def collate(batch: list[dict[str, Any]], pad_id: int, torch: Any) -> dict[str, Any]:
    width = max(len(row["input_ids"]) for row in batch)
    input_ids, labels, masks = [], [], []
    for row in batch:
        pad = width - len(row["input_ids"])
        input_ids.append(row["input_ids"] + [pad_id] * pad)
        labels.append(row["labels"] + [-100] * pad)
        masks.append([1] * len(row["input_ids"]) + [0] * pad)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(masks, dtype=torch.long),
    }


def evaluate_loss(model: Any, rows: list[dict[str, Any]], args: argparse.Namespace, torch: Any) -> float | None:
    values: list[float] = []
    model.eval()
    with torch.no_grad():
        for row in rows[: args.eval_examples]:
            batch = collate([row], args.pad_token_id, torch)
            batch = {key: value.to(args.device) for key, value in batch.items()}
            output = model(**batch)
            value = float(output.loss.detach().cpu())
            if math.isfinite(value):
                values.append(value)
    return sum(values) / len(values) if values else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--init-adapter", type=Path, required=True)
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--eval-file", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-target-tokens", type=int, default=192)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=900)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--eval-examples", type=int, default=256)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--load-in-4bit", action="store_true")
    args = parser.parse_args()

    for path in (args.base_model, args.init_adapter, args.train_file, args.eval_file):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.run_dir.resolve() == args.init_adapter.resolve() or args.init_adapter.resolve() in args.run_dir.resolve().parents:
        raise ValueError("run directory must not be the teacher adapter directory")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    (args.run_dir / "artifacts").mkdir(parents=True, exist_ok=True)
    (args.run_dir / "logs").mkdir(parents=True, exist_ok=True)
    teacher_hash_before = hash_directory(args.init_adapter)

    import torch
    from peft import PeftModel, prepare_model_for_kbit_training
    from transformers import AutoModelForImageTextToText, AutoTokenizer, BitsAndBytesConfig
    from torch.utils.data import DataLoader, Dataset

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    args.device = str(torch.device(args.device))
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    args.pad_token_id = tokenizer.pad_token_id
    train_records = load_jsonl(args.train_file)
    eval_records = load_jsonl(args.eval_file)
    train_rows = tokenize_rows(train_records, tokenizer, args.max_length, args.max_target_tokens)
    eval_rows = tokenize_rows(eval_records, tokenizer, args.max_length, args.max_target_tokens)
    started = time.time()
    (args.run_dir / "artifacts" / "training_config.json").write_text(json.dumps({
        "base_model": str(args.base_model),
        "init_adapter": str(args.init_adapter),
        "init_adapter_sha256": teacher_hash_before,
        "train_file": str(args.train_file),
        "eval_file": str(args.eval_file),
        "train_records": len(train_records),
        "eval_records": len(eval_records),
        "tokenized_train": len(train_rows),
        "tokenized_eval": len(eval_rows),
        "training": {key: getattr(args, key) for key in ("max_length", "max_target_tokens", "batch_size", "gradient_accumulation_steps", "max_steps", "eval_every", "save_every", "learning_rate", "weight_decay", "max_grad_norm", "warmup_steps")},
        "privacy": "Synthetic selected records only; teacher adapter is read-only.",
    }, indent=2) + "\n", encoding="utf-8")

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model_kwargs: dict[str, Any] = {"local_files_only": True, "low_cpu_mem_usage": True}
    if args.load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["device_map"] = {"": 0}
    else:
        model_kwargs["torch_dtype"] = dtype
    model = AutoModelForImageTextToText.from_pretrained(args.base_model, **model_kwargs)
    if not args.load_in_4bit:
        model.to(device)
    model.config.pad_token_id = tokenizer.pad_token_id
    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    model = PeftModel.from_pretrained(model, args.init_adapter, local_files_only=True, is_trainable=True)
    model.config.use_cache = False
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.print_trainable_parameters()

    class Rows(Dataset):
        def __init__(self, values: list[dict[str, Any]]):
            self.values = values
        def __len__(self) -> int:
            return len(self.values)
        def __getitem__(self, index: int) -> dict[str, Any]:
            return self.values[index]

    loader = DataLoader(
        Rows(train_rows), batch_size=args.batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
        collate_fn=lambda batch: collate(batch, tokenizer.pad_token_id, torch),
    )
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable or any("lora_" not in name.lower() for name, parameter in model.named_parameters() if parameter.requires_grad):
        raise RuntimeError("teacher/base parameters unexpectedly trainable")
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=args.weight_decay)
    total_schedule_steps = max(1, args.max_steps)
    def lr_lambda(current: int) -> float:
        if args.warmup_steps > 0 and current < args.warmup_steps:
            return max(1e-3, current / args.warmup_steps)
        progress = (current - args.warmup_steps) / max(1, total_schedule_steps - args.warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, progress))))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    accumulation = max(1, args.gradient_accumulation_steps)
    metrics = MetricsLogger(args.run_dir)
    optimizer.zero_grad(set_to_none=True)
    model.train()
    step = 0
    micro = 0
    losses: list[float] = []
    stop = False
    while not stop:
        for raw_batch in loader:
            micro += 1
            batch = {key: value.to(device) for key, value in raw_batch.items()}
            output = model(**batch)
            loss = output.loss
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at step {step}")
            (loss / accumulation).backward()
            if micro % accumulation == 0 or micro == len(loader):
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                loss_value = float(loss.detach().cpu())
                losses.append(loss_value)
                metrics.step(step, loss=loss_value, grad_norm=float(grad_norm.detach().cpu()), lr=float(optimizer.param_groups[0]["lr"]), memory_allocated_gib=torch.cuda.memory_allocated(device) / 2**30, memory_reserved_gib=torch.cuda.memory_reserved(device) / 2**30)
                if args.eval_every > 0 and step % args.eval_every == 0:
                    value = evaluate_loss(model, eval_rows, args, torch)
                    metrics.log("validation_loss", value, step=step)
                    model.train()
                if args.save_every > 0 and step % args.save_every == 0:
                    checkpoint_dir = args.run_dir / "artifacts" / f"checkpoint-{step:06d}"
                    model.save_pretrained(checkpoint_dir, safe_serialization=True)
                    (checkpoint_dir / "step.json").write_text(json.dumps({"step": step, "loss": loss_value}, indent=2) + "\n", encoding="utf-8")
                torch.cuda.reset_peak_memory_stats(device)
                if args.max_steps > 0 and step >= args.max_steps:
                    stop = True
                    break
        if args.max_steps <= 0 or step >= args.max_steps:
            break

    model.eval()
    validation_loss = evaluate_loss(model, eval_rows, args, torch)
    adapter_dir = args.run_dir / "artifacts" / "adapter"
    model.save_pretrained(adapter_dir, safe_serialization=True)
    tokenizer.save_pretrained(adapter_dir)
    teacher_hash_after = hash_directory(args.init_adapter)
    if teacher_hash_after != teacher_hash_before:
        raise RuntimeError("teacher adapter changed during second training")
    summary = {
        "status": "completed",
        "backend": "transformers_lora_synthetic_sft",
        "base_model": str(args.base_model),
        "teacher_adapter": str(args.init_adapter),
        "teacher_adapter_sha256": teacher_hash_before,
        "teacher_adapter_unchanged": teacher_hash_after == teacher_hash_before,
        "adapter": str(adapter_dir),
        "optimizer_steps": step,
        "train_examples": len(train_rows),
        "eval_examples": len(eval_rows),
        "initial_loss": losses[0] if losses else None,
        "final_loss": losses[-1] if losses else None,
        "validation_loss": validation_loss,
        "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
        "elapsed_seconds": round(time.time() - started, 2),
        "privacy": "Only selected synthetic records were used; teacher adapter remained read-only.",
    }
    (args.run_dir / "artifacts" / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"event": "completed", **summary}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
