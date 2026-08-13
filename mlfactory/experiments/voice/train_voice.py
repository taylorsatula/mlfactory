#!/usr/bin/env python3
"""Small local Qwen voice fine-tuning proof of concept.

This script deliberately keeps the private SMS corpus on the local machine.  It
builds redacted prompt/completion examples in memory from the raw thread files,
trains a LoRA adapter, and writes only aggregate provenance plus generic
before/after samples to the mlfactory run artifacts.
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

# Allow direct execution from any working directory.
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mlfactory.core.metrics import MetricsLogger
from mlfactory.core.manifest import sha256_file
from mlfactory.experiments.voice.voice_prompt import (
    DEFAULT_BUSINESS_STATE,
    PromptVariant,
    build_system_prompt,
    format_business_state,
    variant_for_key,
)

EXCLUDED_CUSTOMERS = {
    "taylor satula",
    "josiah quality care exteriors",
    "annika rettstadt",
    "charlie rettstadt",
}
# Kept as a compatibility export for older experiment scripts. New code uses
# build_system_prompt() with a rotated PromptVariant.
SYSTEM_PROMPT = build_system_prompt()
REDACTION_TOKENS = ("<URL>", "<EMAIL>", "<PHONE>", "<ADDRESS>", "<ZIP>", "<NAME>")
EVAL_CASES = [
    {
        "case_id": "availability",
        "customer": "Hey, can you let me know when you might be able to come by tomorrow?",
    },
    {
        "case_id": "price_question",
        "customer": "I'm sorry, the invoice is higher than I expected. Can you explain it?",
    },
    {
        "case_id": "thanks",
        "customer": "Thanks, that works great!",
    },
    {
        "case_id": "reschedule",
        "customer": "Could we move our appointment from Tuesday to Thursday instead?",
    },
    {
        "case_id": "frustrated",
        "customer": "The issue is still not fixed and I'm getting frustrated.",
    },
]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def redact_text(text: str, customer: dict[str, Any]) -> str:
    """Conservatively remove common direct identifiers while preserving voice."""
    value = str(text or "").strip()
    # Apply structured redactions first so a customer's name inside an email
    # address is not left attached to the replacement token.
    value = re.sub(r"https?://\S+|www\.\S+", "<URL>", value, flags=re.IGNORECASE)
    value = re.sub(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", "<EMAIL>", value, flags=re.IGNORECASE)
    value = re.sub(
        r"(?<!\d)(?:\+?1[ .-]?)?(?:\(\d{3}\)|\d{3})[ .-]\d{3}[ .-]\d{4}(?!\d)",
        "<PHONE>",
        value,
    )
    # Common US street-address shapes.  This is intentionally conservative;
    # the POC is local and the transformation is recorded in the run config.
    value = re.sub(
        r"\b\d{1,6}\s+[A-Za-z0-9.'-]+(?:\s+[A-Za-z0-9.'-]+){0,5}\s+"
        r"(?:street|st|road|rd|avenue|ave|drive|dr|lane|ln|court|ct|circle|cir|"
        r"highway|hwy|boulevard|blvd|way|parkway|pkwy)\b[^,;.!?]*",
        "<ADDRESS>",
        value,
        flags=re.IGNORECASE,
    )
    value = re.sub(r"\b\d{5}(?:-\d{4})?\b", "<ZIP>", value)
    names = [str(customer.get("name") or "").strip()]
    names.extend(
        part.strip()
        for part in str(customer.get("name") or "").split()
        if len(part.strip()) >= 3
    )
    for name in sorted({n for n in names if n}, key=len, reverse=True):
        value = re.sub(re.escape(name), "<NAME>", value, flags=re.IGNORECASE)
    # Do not let a sender's personal name become the adapter's persona token.
    # Speaker identity is already represented by the chat role and the
    # business-representative prompt variants.
    value = re.sub(r"Taylor(?:'s)?", "I", value, flags=re.IGNORECASE)
    return value


def customer_key(thread: dict[str, Any]) -> str:
    return str(thread.get("customer", {}).get("name") or "").strip().lower()


DOMAIN_TERMS: dict[str, tuple[str, ...]] = {
    "window_cleaning": ("window cleaning", "window wash", "soft wash", "squeegee", "pressure washing"),
    "landscaping": ("lawn", "landscap", "yard", "mow", "mulch", "tree service"),
    "painting": ("paint", "painting", "stain", "color consultation"),
    "roofing": ("roof", "gutter", "shingle", "leak"),
    "repair_trades": ("repair", "plumb", "electric", "hvac", "appliance", "install"),
    "auto_or_moving": ("car", "truck", "vehicle", "auto", "u-haul", "move", "moving"),
    "beauty_or_wellness": ("salon", "hair", "spa", "massage", "fitness", "trainer"),
    "medical_or_dental": ("dentist", "dental", "doctor", "clinic", "medical", "therapy"),
    "property_or_real_estate": ("property", "tenant", "rental", "lease", "landlord", "real estate"),
    "events_or_creative": ("wedding", "event", "photograph", "photo", "venue", "catering"),
    "professional_services": ("invoice", "accounting", "bookkeeping", "consult", "contract", "business"),
}


def classify_domain(thread: dict[str, Any]) -> str:
    text = " ".join(str(message.get("text") or "") for message in thread.get("messages", [])).casefold()
    scores = {domain: sum(text.count(term) for term in terms) for domain, terms in DOMAIN_TERMS.items()}
    best, score = max(scores.items(), key=lambda item: item[1])
    return best if score else "general_business"


def corpus_manifest(threads_dir: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    count = total_messages = 0
    for path in sorted(threads_dir.glob("*.json")):
        file_digest = sha256_file(path)
        digest.update(path.name.encode("utf-8") + b"\0" + file_digest.encode("ascii") + b"\n")
        count += 1
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
            total_messages += len(row.get("messages", []))
        except Exception:
            pass
    return {
        "policy": "raw_local_threads_redacted_in_memory",
        "thread_count": count,
        "message_count": total_messages,
        "aggregate_sha256": digest.hexdigest(),
        "excluded_customer_keys": sorted(EXCLUDED_CUSTOMERS),
    }


def system_prompt(mode: str = "business_reply", state: dict[str, Any] | None = None, variant: str | PromptVariant | None = None) -> str:
    return build_system_prompt(mode, state, variant)


def chat_prompt(tokenizer: Any, user_text: str, mode: str = "business_reply", state: dict[str, Any] | None = None, variant: str | PromptVariant | None = None) -> str:
    messages = [
        {"role": "system", "content": system_prompt(mode, state, variant)},
        {"role": "user", "content": user_text},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def conversation_text(history: list[dict[str, str]]) -> str:
    lines = []
    for message in history:
        speaker = "Customer" if message["role"] == "customer" else "Business representative"
        lines.append(f"{speaker}: {message['text']}")
    lines.append("Business representative:")
    return "\n".join(lines)


def build_examples(
    threads_dir: Path,
    tokenizer: Any,
    seed: int,
    test_fraction: float,
    history_messages: int,
    max_train_examples: int,
    max_test_examples: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    train: list[dict[str, Any]] = []
    test: list[dict[str, Any]] = []
    thread_counts = {"train": 0, "test": 0, "excluded": 0, "redacted_targets_dropped": 0}
    for path in sorted(threads_dir.glob("*.json")):
        thread = json.loads(path.read_text(encoding="utf-8"))
        if customer_key(thread) in EXCLUDED_CUSTOMERS:
            thread_counts["excluded"] += 1
            continue
        digest = hashlib.sha256(f"{seed}:{path.name}".encode("utf-8")).digest()
        split = "test" if int.from_bytes(digest[:8], "big") / 2**64 < test_fraction else "train"
        thread_counts[split] += 1
        source_messages = thread.get("messages", [])
        domain = classify_domain(thread)
        messages = [
            {"role": str(row.get("from")), "text": redact_text(str(row.get("text") or ""), thread.get("customer", {}))}
            for row in source_messages
            if str(row.get("from")) in {"customer", "taylor"} and str(row.get("text") or "").strip()
        ]
        for index, message in enumerate(messages):
            if message["role"] != "taylor" or index == 0:
                continue
            history = messages[max(0, index - history_messages):index]
            # A business-representative target is a response to the immediately
            # preceding customer turn. The old "any customer in the window" rule
            # mislabeled proactive ETA/arrival updates as generic replies and
            # taught the adapter to answer greetings with an invented trip.
            if messages[index - 1]["role"] != "customer":
                continue
            target = message["text"]
            # Never teach the adapter to emit a privacy marker. Context may
            # contain markers, but a target containing one is omitted from this
            # small POC rather than turning redaction into a learned phrase.
            if any(token in target for token in REDACTION_TOKENS):
                thread_counts["redacted_targets_dropped"] += 1
                continue
            user_text = conversation_text(history)
            prompt_variant = variant_for_key(f"{path.name}:{index}")
            prompt = chat_prompt(tokenizer, user_text, variant=prompt_variant)
            row = {
                "prompt": prompt,
                "completion": target,
                "mode": "business_reply",
                "verified_state": {},
                "target_role": "business_representative",
                "prompt_variant": prompt_variant.value,
                "source_thread_hash": sha256_bytes(path.name.encode("utf-8"))[:16],
                "source_index": index,
                "domain": domain,
            }
            (test if split == "test" else train).append(row)
    rng = random.Random(seed)
    rng.shuffle(train)
    rng.shuffle(test)
    if max_train_examples > 0:
        train = train[:max_train_examples]
    if max_test_examples > 0:
        test = test[:max_test_examples]
    return train, test, {"thread_counts": thread_counts, "train_examples": len(train), "test_examples": len(test)}


def tokenized_examples(
    examples: list[dict[str, Any]], tokenizer: Any, max_length: int, max_target_tokens: int
) -> list[dict[str, Any]]:
    result = []
    for index, row in enumerate(examples):
        target_ids = tokenizer(row["completion"] + (tokenizer.eos_token or ""), add_special_tokens=False)["input_ids"]
        target_ids = target_ids[:max_target_tokens]
        if not target_ids:
            continue
        prompt_ids = tokenizer(row["prompt"], add_special_tokens=False)["input_ids"]
        prompt_budget = max_length - len(target_ids)
        if prompt_budget < 1:
            continue
        # Keep the most recent conversation tokens when a long thread exceeds
        # the POC window; the system instruction remains in the original text,
        # but the target still receives a valid causal context.
        prompt_ids = prompt_ids[-prompt_budget:]
        input_ids = prompt_ids + target_ids
        result.append({
            "input_ids": input_ids,
            "labels": [-100] * len(prompt_ids) + target_ids,
            "prompt_tokens": len(prompt_ids),
            "target_tokens": len(target_ids),
            "source_thread_hash": row["source_thread_hash"],
            "source_index": row["source_index"],
        })
    if not result:
        raise RuntimeError("no examples survived tokenization")
    return result


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


def generate_sample(model: Any, tokenizer: Any, torch: Any, customer_text: str, device: str, max_new_tokens: int) -> str:
    prompt = chat_prompt(tokenizer, f"Customer: {customer_text}\nBusiness representative:")
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    previous_cache = getattr(model.config, "use_cache", None)
    model.config.use_cache = True
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    model.config.use_cache = previous_cache
    generated = output[0, inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(generated, skip_special_tokens=True).strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    return text


def evaluate_loss(model: Any, examples: list[dict[str, Any]], args: argparse.Namespace, torch: Any) -> float | None:
    if not examples:
        return None
    model.eval()
    values = []
    with torch.no_grad():
        for row in examples[: int(args.eval_examples)]:
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
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--threads", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--history-messages", type=int, default=8)
    parser.add_argument("--max-train-examples", type=int, default=300)
    parser.add_argument("--max-test-examples", type=int, default=60)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--max-target-tokens", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--eval-examples", type=int, default=24)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--target-modules", default="q_proj,k_proj,v_proj,o_proj")
    parser.add_argument("--generation-tokens", type=int, default=96)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--model-class", choices=("causal_lm", "image_text_to_text"), default="causal_lm")
    args = parser.parse_args()
    args.run_dir = args.run_dir.resolve()
    args.threads = args.threads.resolve()
    args.device = str(args.device)
    args.pad_token_id = None
    if not args.threads.is_dir():
        raise FileNotFoundError(args.threads)
    if not Path(args.base_model).exists():
        raise FileNotFoundError(f"base model must already be downloaded locally: {args.base_model}")

    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the local voice POC")
    torch.set_float32_matmul_precision("high")
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("voice POC must run on CUDA")
    args.device = str(device)
    run_dir = args.run_dir
    artifacts = run_dir / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    metrics = MetricsLogger(run_dir)
    started = time.time()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    args.pad_token_id = tokenizer.pad_token_id
    train_raw, test_raw, counts = build_examples(
        args.threads, tokenizer, args.seed, args.test_fraction,
        args.history_messages, args.max_train_examples, args.max_test_examples,
    )
    train_rows = tokenized_examples(train_raw, tokenizer, args.max_length, args.max_target_tokens)
    test_rows = tokenized_examples(test_raw, tokenizer, args.max_length, args.max_target_tokens)
    data_info = {
        **corpus_manifest(args.threads),
        **counts,
        "tokenized_train_examples": len(train_rows),
        "tokenized_test_examples": len(test_rows),
        "redaction": "customer metadata, URL, email, phone, common street address, ZIP",
        "max_length": args.max_length,
        "history_messages": args.history_messages,
    }
    (artifacts / "data_manifest.json").write_text(json.dumps(data_info, indent=2) + "\n", encoding="utf-8")
    config = {
        "base_model": str(args.base_model),
        "base_model_revision": None,
        "device": args.device,
        "load_in_4bit": args.load_in_4bit,
        "model_class": args.model_class,
        "seed": args.seed,
        "training": {key: getattr(args, key) for key in (
            "max_train_examples", "max_test_examples", "max_length", "max_target_tokens",
            "batch_size", "gradient_accumulation_steps", "max_steps", "eval_every", "save_every",
            "learning_rate", "weight_decay", "max_grad_norm", "lora_r", "lora_alpha",
            "lora_dropout", "target_modules",
        )},
        "data": data_info,
        "gpu": [
            {"index": i, "name": torch.cuda.get_device_name(i), "capability": torch.cuda.get_device_capability(i)}
            for i in range(torch.cuda.device_count())
        ],
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
    }
    (artifacts / "training_config.json").write_text(json.dumps(config, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"event": "data_ready", **data_info}, default=str), flush=True)

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model_cls = AutoModelForImageTextToText if args.model_class == "image_text_to_text" else AutoModelForCausalLM
    model_kwargs = {
        "local_files_only": True,
        "low_cpu_mem_usage": True,
    }
    if args.load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )
        # CUDA_VISIBLE_DEVICES is set by the launch spec, so logical device 0
        # is the intended physical training GPU.
        model_kwargs["device_map"] = {"": 0}
    else:
        model_kwargs["torch_dtype"] = dtype
    print(json.dumps({"event": "loading_model", "dtype": str(dtype), "device": args.device, "quantized": args.load_in_4bit, "model_class": args.model_class}), flush=True)
    model = model_cls.from_pretrained(args.base_model, **model_kwargs)
    if not args.load_in_4bit:
        model.to(device)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = True
    model.eval()

    baseline_samples = []
    for case in EVAL_CASES:
        text = generate_sample(model, tokenizer, torch, case["customer"], args.device, args.generation_tokens)
        baseline_samples.append({"case_id": case["case_id"], "customer": case["customer"], "response": text})
    (artifacts / "baseline_samples.json").write_text(json.dumps(baseline_samples, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=[value.strip() for value in args.target_modules.split(",") if value.strip()],
    )
    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    model = get_peft_model(model, lora_config)
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
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
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=args.weight_decay)
    accumulation = max(1, args.gradient_accumulation_steps)
    optimizer.zero_grad(set_to_none=True)
    model.train()
    step = 0
    micro = 0
    losses = []
    stop = False
    while not stop:
        for raw_batch in loader:
            micro += 1
            batch = {key: value.to(device) for key, value in raw_batch.items()}
            output = model(**batch)
            loss = output.loss
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at step {step}: {loss.detach().item()}")
            (loss / accumulation).backward()
            if micro % accumulation == 0 or micro == len(loader):
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                loss_value = float(loss.detach().cpu())
                losses.append(loss_value)
                gpu_metrics = {
                    "loss": loss_value,
                    "grad_norm": float(grad_norm.detach().cpu()),
                    "lr": float(optimizer.param_groups[0]["lr"]),
                    "memory_allocated_gib": torch.cuda.memory_allocated(device) / 2**30,
                    "memory_reserved_gib": torch.cuda.memory_reserved(device) / 2**30,
                    "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
                    "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
                }
                metrics.step(step, **gpu_metrics)
                if args.eval_every > 0 and step % args.eval_every == 0:
                    validation_loss = evaluate_loss(model, test_rows, args, torch)
                    metrics.log("validation_loss", validation_loss, step=step)
                    model.train()
                if args.save_every > 0 and step % args.save_every == 0:
                    checkpoint_dir = artifacts / f"checkpoint-{step:06d}"
                    model.save_pretrained(checkpoint_dir, safe_serialization=True)
                    (checkpoint_dir / "step.json").write_text(
                        json.dumps({"step": step, "loss": float(loss.detach().cpu())}, indent=2) + "\n",
                        encoding="utf-8",
                    )
                    metrics.event("checkpoint_saved", {"step": step, "path": str(checkpoint_dir)})
                torch.cuda.reset_peak_memory_stats(device)
                if args.max_steps > 0 and step >= args.max_steps:
                    stop = True
                    break
        if step == 0 and not loader:
            raise RuntimeError("training loader is empty")
        if args.max_steps <= 0 or step >= args.max_steps:
            break

    model.eval()
    final_validation_loss = evaluate_loss(model, test_rows, args, torch)
    adapter_dir = artifacts / "adapter"
    model.save_pretrained(adapter_dir, safe_serialization=True)
    tokenizer.save_pretrained(adapter_dir)
    finetuned_samples = []
    for case in EVAL_CASES:
        text = generate_sample(model, tokenizer, torch, case["customer"], args.device, args.generation_tokens)
        finetuned_samples.append({"case_id": case["case_id"], "customer": case["customer"], "response": text})
    (artifacts / "finetuned_samples.json").write_text(json.dumps(finetuned_samples, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    summary = {
        "status": "completed",
        "backend": "transformers_lora_causal_lm",
        "base_model": str(args.base_model),
        "adapter": str(adapter_dir),
        "optimizer_steps": step,
        "train_examples": len(train_rows),
        "test_examples": len(test_rows),
        "initial_loss": losses[0] if losses else None,
        "final_loss": losses[-1] if losses else None,
        "validation_loss": final_validation_loss,
        "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
        "elapsed_seconds": round(time.time() - started, 2),
        "privacy": "Only redacted in-memory examples and aggregate corpus provenance were used; raw SMS files were not copied into the run.",
    }
    (artifacts / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"event": "completed", **summary}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
