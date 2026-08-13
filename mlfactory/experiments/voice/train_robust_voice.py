#!/usr/bin/env python3
"""Train the robust voice adapter from the frozen base model.

This is deliberately a fresh adapter, not a continuation of either previous
adapter.  Real SMS is redacted and consumed in memory only.  Fictional records
carry every state fact used by their target in the visible prompt, and general
replay prevents business-mode narrowing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mlfactory.core.metrics import MetricsLogger
from mlfactory.experiments.voice.train_voice import (
    DEFAULT_BUSINESS_STATE,
    SYSTEM_PROMPT,
    build_examples,
    collate,
    conversation_text,
    corpus_manifest,
    generate_sample,
    system_prompt,
    tokenized_examples,
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def render_record(tokenizer: Any, row: dict[str, Any]) -> str:
    if row.get("prompt"):
        return str(row["prompt"])
    messages = [{"role": "system", "content": system_prompt(str(row.get("mode") or "business_reply"), row.get("verified_state"), row.get("prompt_variant"))}]
    for message in row.get("messages", []):
        role = "user" if message.get("role") == "customer" else "assistant"
        messages.append({"role": role, "content": str(message.get("text") or "")})
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def tokenize_records(records: list[dict[str, Any]], tokenizer: Any, max_length: int, max_target_tokens: int) -> list[dict[str, Any]]:
    rows = []
    for record in records:
        target = str(record.get("completion", record.get("target", "")) or "").strip()
        if not target:
            continue
        target_ids = tokenizer(target + (tokenizer.eos_token or ""), add_special_tokens=False)["input_ids"][:max_target_tokens]
        prompt_ids = tokenizer(render_record(tokenizer, record), add_special_tokens=False)["input_ids"]
        budget = max_length - len(target_ids)
        if budget < 1 or not target_ids:
            continue
        rows.append({
            "input_ids": prompt_ids[-budget:] + target_ids,
            "labels": [-100] * min(len(prompt_ids), budget) + target_ids,
            "kind": str(record.get("kind") or record.get("source") or "unknown"),
            "example_id": str(record.get("example_id") or "unknown"),
            "target_tokens": len(target_ids),
        })
    if not rows:
        raise RuntimeError("no robust voice records survived tokenization")
    return rows


def make_record_from_real(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "prompt": row["prompt"],
        "completion": row["completion"],
        "kind": "real_redacted",
        "domain": str(row.get("domain") or "general_business"),
        "prompt_variant": row.get("prompt_variant"),
        "example_id": f"real-{row.get('source_thread_hash')}-{row.get('source_index')}",
    }


def select_diverse_real(values: list[dict[str, Any]], limit: int, max_fraction: float = 0.10) -> list[dict[str, Any]]:
    """Cap any one real-world domain before it reaches the adapter.

    The source corpus is heavily concentrated in window cleaning.  A hard cap
    is preferable to hoping random shuffling fixes that imbalance.  Examples
    are taken round-robin across domains so sparse domains are retained.
    """
    if limit <= 0 or len(values) <= limit:
        return values
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in values:
        groups.setdefault(str(row.get("domain") or "general_business"), []).append(row)
    cap = max(1, int(math.ceil(limit * max_fraction)))
    selected: list[dict[str, Any]] = []
    domains = sorted(groups)
    cursor = {domain: 0 for domain in domains}
    while len(selected) < limit:
        progressed = False
        for domain in domains:
            if len(selected) >= limit:
                break
            group = groups[domain]
            allowed = min(len(group), cap)
            if cursor[domain] < allowed:
                selected.append(group[cursor[domain]])
                cursor[domain] += 1
                progressed = True
        if not progressed:
            break
    return selected


def repeat_rows(values: list[dict[str, Any]], repeats: int, prefix: str) -> list[dict[str, Any]]:
    output = []
    for repeat in range(max(1, repeats)):
        for row in values:
            copied = dict(row)
            copied["example_id"] = f"{prefix}{repeat}-{row.get('example_id', 'unknown')}"
            output.append(copied)
    return output


def evaluate_loss(model: Any, rows: list[dict[str, Any]], args: argparse.Namespace, torch: Any) -> float | None:
    if not rows:
        return None
    values = []
    model.eval()
    with torch.no_grad():
        for row in rows[: args.eval_examples]:
            batch = collate([row], args.pad_token_id, torch)
            batch = {key: value.to(args.device) for key, value in batch.items() if key in {"input_ids", "labels", "attention_mask"}}
            loss = model(**batch).loss
            if math.isfinite(float(loss.detach().cpu())):
                values.append(float(loss.detach().cpu()))
    return sum(values) / len(values) if values else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--threads", type=Path, required=True)
    parser.add_argument("--synthetic-train", type=Path, required=True)
    parser.add_argument("--synthetic-eval", type=Path, required=True)
    parser.add_argument("--replay-file", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--max-real-train", type=int, default=1000)
    parser.add_argument("--max-real-eval", type=int, default=250)
    parser.add_argument("--synthetic-repeats", type=int, default=4)
    parser.add_argument("--replay-repeats", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-target-tokens", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--eval-examples", type=int, default=256)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--warmup-steps", type=int, default=40)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--target-modules", default="q_proj,k_proj,v_proj,o_proj")
    parser.add_argument("--generation-tokens", type=int, default=96)
    parser.add_argument("--load-in-4bit", action="store_true")
    args = parser.parse_args()
    args.run_dir = args.run_dir.resolve()
    args.base_model = args.base_model.resolve()
    args.threads = args.threads.resolve()
    for path in (args.base_model, args.threads, args.synthetic_train, args.synthetic_eval, args.replay_file):
        if not Path(path).exists():
            raise FileNotFoundError(path)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    artifacts = args.run_dir / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    (args.run_dir / "logs").mkdir(parents=True, exist_ok=True)

    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModelForImageTextToText, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("robust voice training requires CUDA")
    args.device = str(device)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    args.pad_token_id = tokenizer.pad_token_id

    # Build the full redacted in-memory split first, then enforce domain
    # balancing.  This keeps window-cleaning examples useful without allowing
    # them to define the adapter's identity.
    real_train_raw, real_eval_raw, counts = build_examples(
        args.threads, tokenizer, args.seed, 0.2, 8, 0, args.max_real_eval,
    )
    real_train_raw = select_diverse_real(real_train_raw, args.max_real_train)
    real_train = [make_record_from_real(row) for row in real_train_raw]
    real_eval = [make_record_from_real(row) for row in real_eval_raw]
    synthetic_train = load_jsonl(args.synthetic_train)
    synthetic_eval = load_jsonl(args.synthetic_eval)
    replay = load_jsonl(args.replay_file)
    # Replay is intentionally present in every epoch-equivalent mixture, while
    # business records are never initialized from the old teacher adapter.
    train_records = (
        real_train
        + repeat_rows(synthetic_train, args.synthetic_repeats, "syn-")
        + repeat_rows(replay, args.replay_repeats, "replay-")
    )
    eval_records = real_eval + synthetic_eval + replay
    for row in train_records + eval_records:
        row.setdefault("kind", row.get("source", "fictional_authored"))
    rng = random.Random(args.seed)
    rng.shuffle(train_records)
    rng.shuffle(eval_records)
    train_rows = tokenize_records(train_records, tokenizer, args.max_length, args.max_target_tokens)
    eval_rows = tokenize_records(eval_records, tokenizer, args.max_length, args.max_target_tokens)
    started = time.time()
    dataset_manifest = {
        "policy": "real_redacted_in_memory_plus_fictional_visible_state_and_general_replay",
        "real_corpus": corpus_manifest(args.threads),
        "real_counts": counts,
        "train_records": len(train_records), "eval_records": len(eval_records),
        "tokenized_train": len(train_rows), "tokenized_eval": len(eval_rows),
        "source_counts": {kind: sum(row.get("kind") == kind for row in train_records) for kind in sorted({row.get("kind") for row in train_records})},
        "real_domain_counts": {domain: sum(row.get("domain") == domain for row in real_train) for domain in sorted({row.get("domain") for row in real_train})},
        "real_domain_cap_fraction": 0.10,
        "synthetic_context_overlap": "0 by construction: evaluation is a disjoint deterministic holdout",
        "private_data_policy": "raw SMS and redacted real messages never written to run artifacts",
    }
    (artifacts / "data_manifest.json").write_text(json.dumps(dataset_manifest, indent=2) + "\n", encoding="utf-8")
    config = {
        "base_model": str(args.base_model), "init_adapter": None,
        "device": args.device, "load_in_4bit": args.load_in_4bit,
        "seed": args.seed, "training": {key: getattr(args, key) for key in (
            "max_real_train", "max_real_eval", "synthetic_repeats", "replay_repeats", "max_length", "max_target_tokens",
            "batch_size", "gradient_accumulation_steps", "max_steps", "eval_every", "save_every", "learning_rate",
            "weight_decay", "max_grad_norm", "warmup_steps", "lora_r", "lora_alpha", "lora_dropout", "target_modules")},
        "model_policy": "fresh_adapter_from_frozen_base; q/k/v/o only; no teacher continuation",
        "data": dataset_manifest,
    }
    (artifacts / "training_config.json").write_text(json.dumps(config, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"event": "data_ready", **dataset_manifest}, default=str), flush=True)

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    kwargs: dict[str, Any] = {"local_files_only": True, "low_cpu_mem_usage": True}
    if args.load_in_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True)
        kwargs["device_map"] = {"": 0}
    else:
        kwargs["torch_dtype"] = dtype
    model = AutoModelForImageTextToText.from_pretrained(args.base_model, **kwargs)
    if not args.load_in_4bit:
        model.to(device)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = True
    baseline = []
    for case in ["Can you tell me what a palindrome is?", "Hey there!", "Could we move our appointment to later this week?"]:
        baseline.append({"customer": case, "response": generate_sample(model, tokenizer, torch, case, args.device, args.generation_tokens)})
    (artifacts / "baseline_samples.json").write_text(json.dumps(baseline, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    lora = LoraConfig(task_type=TaskType.CAUSAL_LM, r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout, target_modules=[x.strip() for x in args.target_modules.split(",") if x.strip()])
    model = get_peft_model(model, lora)
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    model.print_trainable_parameters()

    class Rows(Dataset):
        def __init__(self, values: list[dict[str, Any]]): self.values = values
        def __len__(self) -> int: return len(self.values)
        def __getitem__(self, index: int) -> dict[str, Any]: return self.values[index]

    loader = DataLoader(Rows(train_rows), batch_size=args.batch_size, shuffle=True, generator=torch.Generator().manual_seed(args.seed), collate_fn=lambda batch: collate(batch, tokenizer.pad_token_id, torch))
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable or any("lora_" not in name.lower() for name, p in model.named_parameters() if p.requires_grad):
        raise RuntimeError("unexpected non-LoRA trainable parameter")
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=args.weight_decay)
    def lr_lambda(step: int) -> float:
        if args.warmup_steps and step < args.warmup_steps:
            return max(1e-3, step / args.warmup_steps)
        progress = (step - args.warmup_steps) / max(1, args.max_steps - args.warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, max(0.0, progress))))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    metrics = MetricsLogger(args.run_dir)
    accumulation = max(1, args.gradient_accumulation_steps)
    optimizer.zero_grad(set_to_none=True)
    step = micro = 0
    losses: list[float] = []
    stop = False
    model.train()
    while not stop:
        for raw_batch in loader:
            micro += 1
            batch = {key: value.to(device) for key, value in raw_batch.items() if key in {"input_ids", "labels", "attention_mask"}}
            loss = model(**batch).loss
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at step {step}")
            (loss / accumulation).backward()
            if micro % accumulation == 0 or micro == len(loader):
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
                optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True); step += 1
                value = float(loss.detach().cpu()); losses.append(value)
                metrics.step(step, loss=value, grad_norm=float(grad_norm.detach().cpu()), lr=float(optimizer.param_groups[0]["lr"]), memory_allocated_gib=torch.cuda.memory_allocated(device) / 2**30, memory_reserved_gib=torch.cuda.memory_reserved(device) / 2**30)
                if args.eval_every > 0 and step % args.eval_every == 0:
                    validation = evaluate_loss(model, eval_rows, args, torch); metrics.log("validation_loss", validation, step=step); model.train()
                if args.save_every > 0 and step % args.save_every == 0:
                    checkpoint = artifacts / f"checkpoint-{step:06d}"; model.save_pretrained(checkpoint, safe_serialization=True); (checkpoint / "step.json").write_text(json.dumps({"step": step, "loss": value}, indent=2) + "\n")
                if args.max_steps > 0 and step >= args.max_steps:
                    stop = True; break
        if args.max_steps <= 0 or step >= args.max_steps: break
    model.eval()
    final_validation = evaluate_loss(model, eval_rows, args, torch)
    adapter = artifacts / "adapter"; model.save_pretrained(adapter, safe_serialization=True); tokenizer.save_pretrained(adapter)
    samples = []
    for case in ["Can you tell me what a palindrome is?", "Hey there!", "Could we move our appointment to later this week?", "One part was missed and I am disappointed."]:
        samples.append({"customer": case, "response": generate_sample(model, tokenizer, torch, case, args.device, args.generation_tokens)})
    (artifacts / "finetuned_samples.json").write_text(json.dumps(samples, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    summary = {
        "status": "completed", "backend": "transformers_lora_causal_lm_fresh_robust",
        "base_model": str(args.base_model), "adapter": str(adapter), "optimizer_steps": step,
        "train_examples": len(train_rows), "test_examples": len(eval_rows),
        "initial_loss": losses[0] if losses else None, "final_loss": losses[-1] if losses else None,
        "validation_loss": final_validation, "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30, "elapsed_seconds": round(time.time() - started, 2),
        "release_policy": "must pass sealed grounding, capability, leakage, diversity, and privacy gates before serving",
        "privacy": "redacted real examples were in-memory only; fictional data contains no private SMS",
    }
    (artifacts / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"event": "completed", **summary}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
