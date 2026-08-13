#!/usr/bin/env python3
"""Conservative local DPO style pass from human SMS preferences.

The private corpus is read, redacted, and consumed in memory. Preference pairs
are never written: chosen is a real redacted human reply and rejected is a
response sampled from the robust adapter for the identical visible prompt.
The reference adapter and trainable policy adapter share one quantized base
model, keeping this within a 24-GB GPU.
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
from mlfactory.experiments.voice.train_voice import build_examples
from mlfactory.experiments.voice.train_robust_voice import select_diverse_real


def normalized(text: str) -> str:
    return re.sub(r"\s+", " ", text.casefold()).strip()


def load_policy(base: Path, adapter: Path, torch: Any, tokenizer: Any) -> Any:
    from peft import PeftModel, prepare_model_for_kbit_training
    from transformers import AutoModelForImageTextToText, BitsAndBytesConfig
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForImageTextToText.from_pretrained(
        base, local_files_only=True, low_cpu_mem_usage=True,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True),
        device_map={"": 0},
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    # Both adapters are copies of the frozen robust adapter. Only `policy` is
    # trainable; `reference` is the fixed DPO baseline.
    model = PeftModel.from_pretrained(model, adapter, adapter_name="policy", local_files_only=True, is_trainable=True)
    model.load_adapter(adapter, adapter_name="reference", local_files_only=True, is_trainable=False)
    model.set_adapter("reference")
    model.config.use_cache = True
    return model


def generate_rejected(model: Any, tokenizer: Any, prompts: list[str], torch: Any, batch_size: int, max_new_tokens: int, seed: int) -> list[str]:
    tokenizer.padding_side = "left"
    outputs: list[str] = []
    model.set_adapter("reference")
    for start in range(0, len(prompts), max(1, batch_size)):
        chunk = prompts[start : start + max(1, batch_size)]
        inputs = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True, max_length=512, add_special_tokens=False)
        inputs = {key: value.to("cuda:0") for key, value in inputs.items()}
        torch.manual_seed(seed + start)
        with torch.inference_mode():
            generated = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.7, top_p=0.9,
                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
            )
        width = inputs["input_ids"].shape[1]
        outputs.extend(tokenizer.batch_decode(generated[:, width:], skip_special_tokens=True))
    tokenizer.padding_side = "right"
    return [re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip() for text in outputs]


def make_pairs(rows: list[dict[str, Any]], rejected: list[str]) -> list[dict[str, Any]]:
    pairs = []
    for row, text in zip(rows, rejected):
        chosen = str(row["completion"]).strip()
        rejected_text = str(text).strip()
        if not chosen or not rejected_text or normalized(chosen) == normalized(rejected_text):
            continue
        pairs.append({"prompt": row["prompt"], "chosen": chosen, "rejected": rejected_text, "example_id": f"{row.get('source_thread_hash')}-{row.get('source_index')}"})
    return pairs


def encode_pair(pair: dict[str, str], tokenizer: Any, max_length: int, max_target_tokens: int) -> dict[str, Any] | None:
    prompt = tokenizer(pair["prompt"], add_special_tokens=False)["input_ids"]
    chosen = tokenizer(pair["chosen"] + (tokenizer.eos_token or ""), add_special_tokens=False)["input_ids"][:max_target_tokens]
    rejected = tokenizer(pair["rejected"] + (tokenizer.eos_token or ""), add_special_tokens=False)["input_ids"][:max_target_tokens]
    budget = max_length - max(len(chosen), len(rejected))
    if budget < 1 or not chosen or not rejected:
        return None
    prompt = prompt[-budget:]
    return {
        "chosen_ids": prompt + chosen, "chosen_labels": [-100] * len(prompt) + chosen,
        "rejected_ids": prompt + rejected, "rejected_labels": [-100] * len(prompt) + rejected,
    }


def collate(encoded: list[dict[str, Any]], pad_id: int, torch: Any) -> dict[str, Any]:
    chosen_width = max(len(row["chosen_ids"]) for row in encoded)
    rejected_width = max(len(row["rejected_ids"]) for row in encoded)
    def pad(row: dict[str, Any], key: str, label_key: str, width: int) -> tuple[list[int], list[int], list[int]]:
        ids = row[key]; labels = row[label_key]; n = width - len(ids)
        return ids + [pad_id] * n, labels + [-100] * n, [1] * len(ids) + [0] * n
    ci, cl, cm, ri, rl, rm = [], [], [], [], [], []
    for row in encoded:
        x, y, z = pad(row, "chosen_ids", "chosen_labels", chosen_width); ci.append(x); cl.append(y); cm.append(z)
        x, y, z = pad(row, "rejected_ids", "rejected_labels", rejected_width); ri.append(x); rl.append(y); rm.append(z)
    return {"chosen_ids": torch.tensor(ci, dtype=torch.long), "chosen_labels": torch.tensor(cl, dtype=torch.long), "chosen_mask": torch.tensor(cm, dtype=torch.long), "rejected_ids": torch.tensor(ri, dtype=torch.long), "rejected_labels": torch.tensor(rl, dtype=torch.long), "rejected_mask": torch.tensor(rm, dtype=torch.long)}


def sequence_logprob(model: Any, ids: Any, labels: Any, mask: Any, torch: Any, grad: bool) -> Any:
    context = torch.enable_grad() if grad else torch.no_grad()
    with context:
        logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits[:, :-1, :]
        next_labels = labels[:, 1:]
        valid = next_labels.ne(-100)
        safe_labels = next_labels.masked_fill(~valid, 0)
        log_probs = torch.log_softmax(logits, dim=-1)
        token_values = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
        return (token_values * valid).sum(dim=-1)


def dpo_batch(model: Any, batch: dict[str, Any], tokenizer: Any, args: argparse.Namespace, torch: Any, train: bool = True) -> tuple[Any, Any, Any]:
    device = args.device
    chosen_ids = batch["chosen_ids"].to(device); chosen_labels = batch["chosen_labels"].to(device); chosen_mask = batch["chosen_mask"].to(device)
    rejected_ids = batch["rejected_ids"].to(device); rejected_labels = batch["rejected_labels"].to(device); rejected_mask = batch["rejected_mask"].to(device)
    model.set_adapter("reference")
    with torch.no_grad():
        ref_c = sequence_logprob(model, chosen_ids, chosen_labels, chosen_mask, torch, False)
        ref_r = sequence_logprob(model, rejected_ids, rejected_labels, rejected_mask, torch, False)
    model.set_adapter("policy")
    pi_c = sequence_logprob(model, chosen_ids, chosen_labels, chosen_mask, torch, train)
    pi_r = sequence_logprob(model, rejected_ids, rejected_labels, rejected_mask, torch, train)
    advantage = (pi_c - pi_r) - (ref_c - ref_r)
    loss = -torch.nn.functional.logsigmoid(args.beta * advantage).mean()
    accuracy = (advantage > 0).float().mean()
    margin = advantage.mean().detach()
    return loss, accuracy.detach(), margin


def evaluate(model: Any, encoded: list[dict[str, Any]], tokenizer: Any, args: argparse.Namespace, torch: Any) -> tuple[float, float, float]:
    values, accuracies, margins = [], [], []
    model.eval()
    for start in range(0, min(len(encoded), args.eval_examples), args.batch_size):
        batch = collate(encoded[start : start + args.batch_size], tokenizer.pad_token_id, torch)
        with torch.no_grad():
            loss, accuracy, margin = dpo_batch(model, batch, tokenizer, args, torch, train=False)
        values.append(float(loss.cpu())); accuracies.append(float(accuracy.cpu())); margins.append(float(margin.cpu()))
    return (sum(values) / max(1, len(values)), sum(accuracies) / max(1, len(accuracies)), sum(margins) / max(1, len(margins)))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--init-adapter", type=Path, required=True)
    parser.add_argument("--threads", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--max-real-train", type=int, default=1000)
    parser.add_argument("--max-real-eval", type=int, default=250)
    parser.add_argument("--generation-batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-target-tokens", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=180)
    parser.add_argument("--eval-every", type=int, default=30)
    parser.add_argument("--eval-examples", type=int, default=128)
    parser.add_argument("--save-every", type=int, default=90)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--beta", type=float, default=0.1)
    args = parser.parse_args()
    args.run_dir = args.run_dir.resolve(); args.base_model = args.base_model.resolve(); args.init_adapter = args.init_adapter.resolve(); args.threads = args.threads.resolve(); args.device = str(args.device)
    for path in (args.base_model, args.init_adapter, args.threads):
        if not path.exists(): raise FileNotFoundError(path)
    if args.run_dir == args.init_adapter or args.init_adapter in args.run_dir.parents:
        raise ValueError("DPO run must not write into the robust adapter")
    args.run_dir.mkdir(parents=True, exist_ok=True); artifacts = args.run_dir / "artifacts"; artifacts.mkdir(parents=True, exist_ok=True); (args.run_dir / "logs").mkdir(parents=True, exist_ok=True)

    import torch
    from transformers import AutoTokenizer
    random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    if not torch.cuda.is_available(): raise RuntimeError("CUDA is required")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, local_files_only=True)
    if tokenizer.pad_token_id is None: tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    # This uses the corrected immediate-customer attribution and domain cap;
    # raw/private examples never leave this process.
    real_train, real_eval, counts = build_examples(args.threads, tokenizer, args.seed, 0.2, 8, 0, args.max_real_eval)
    real_train = select_diverse_real(real_train, args.max_real_train, max_fraction=0.10)
    model = load_policy(args.base_model, args.init_adapter, torch, tokenizer)
    prompts = [row["prompt"] for row in real_train + real_eval]
    rejected = generate_rejected(model, tokenizer, prompts, torch, args.generation_batch_size, args.max_new_tokens, args.seed)
    train_rejected = rejected[:len(real_train)]; eval_rejected = rejected[len(real_train):]
    train_pairs = make_pairs(real_train, train_rejected); eval_pairs = make_pairs(real_eval, eval_rejected)
    encoded_train = [x for pair in train_pairs if (x := encode_pair(pair, tokenizer, args.max_length, args.max_target_tokens))]
    encoded_eval = [x for pair in eval_pairs if (x := encode_pair(pair, tokenizer, args.max_length, args.max_target_tokens))]
    if not encoded_train or not encoded_eval: raise RuntimeError("DPO pair construction produced no usable examples")
    dataset = {"train_candidates": len(real_train), "eval_candidates": len(real_eval), "train_pairs": len(encoded_train), "eval_pairs": len(encoded_eval), "dropped_equal_or_empty": len(real_train) + len(real_eval) - len(encoded_train) - len(encoded_eval), "real_counts": counts, "policy": "chosen=redacted_human_reply; rejected=robust_adapter_reply; pair_text_not_written"}
    (artifacts / "data_manifest.json").write_text(json.dumps(dataset, indent=2) + "\n", encoding="utf-8")
    (artifacts / "training_config.json").write_text(json.dumps({"base_model": str(args.base_model), "init_adapter": str(args.init_adapter), "seed": args.seed, "beta": args.beta, "training": {key: getattr(args, key) for key in ("max_length", "max_target_tokens", "batch_size", "gradient_accumulation_steps", "max_steps", "eval_every", "save_every", "learning_rate", "weight_decay", "max_grad_norm", "warmup_steps")}, "reference": "frozen copy of init_adapter", "privacy": "private redacted preference pairs held in memory only"}, indent=2) + "\n", encoding="utf-8")

    model.set_adapter("policy")
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False}); model.config.use_cache = False
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable: raise RuntimeError("no trainable policy adapter parameters")
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=args.weight_decay)
    def lr_lambda(step: int) -> float:
        if step < args.warmup_steps: return max(1e-3, step / max(1, args.warmup_steps))
        progress = (step - args.warmup_steps) / max(1, args.max_steps - args.warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, max(0.0, progress))))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    metrics = MetricsLogger(args.run_dir); accumulation = max(1, args.gradient_accumulation_steps); optimizer.zero_grad(set_to_none=True)
    rng = random.Random(args.seed); order = list(range(len(encoded_train))); step = micro = 0; losses = []; stop = False; started = time.time()
    model.train()
    while not stop:
        rng.shuffle(order)
        for index in order:
            micro += 1
            batch = collate([encoded_train[index]], tokenizer.pad_token_id, torch)
            loss, accuracy, margin = dpo_batch(model, batch, tokenizer, args, torch, train=True)
            if not torch.isfinite(loss): raise FloatingPointError(f"non-finite DPO loss at step {step}")
            (loss / accumulation).backward()
            if micro % accumulation == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm); optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True); step += 1
                value = float(loss.detach().cpu()); losses.append(value)
                metrics.step(step, loss=value, preference_accuracy=float(accuracy), preference_margin=float(margin), grad_norm=float(grad_norm.detach().cpu()), lr=float(optimizer.param_groups[0]["lr"]), memory_allocated_gib=torch.cuda.memory_allocated(args.device) / 2**30, memory_reserved_gib=torch.cuda.memory_reserved(args.device) / 2**30)
                if args.eval_every > 0 and step % args.eval_every == 0:
                    ev_loss, ev_acc, ev_margin = evaluate(model, encoded_eval, tokenizer, args, torch); metrics.log("eval_dpo_loss", ev_loss, step=step); metrics.log("eval_preference_accuracy", ev_acc, step=step); metrics.log("eval_preference_margin", ev_margin, step=step); model.train()
                if args.save_every > 0 and step % args.save_every == 0:
                    checkpoint = artifacts / f"checkpoint-{step:06d}"; model.save_pretrained(checkpoint, safe_serialization=True, selected_adapters=["policy"]); (checkpoint / "step.json").write_text(json.dumps({"step": step, "loss": value}, indent=2) + "\n")
                if args.max_steps > 0 and step >= args.max_steps: stop = True; break
        if args.max_steps <= 0 or step >= args.max_steps: break
    final_loss, final_acc, final_margin = evaluate(model, encoded_eval, tokenizer, args, torch)
    model.set_adapter("policy"); adapter = artifacts / "adapter"; model.save_pretrained(adapter, safe_serialization=True, selected_adapters=["policy"]); tokenizer.save_pretrained(adapter)
    summary = {"status": "completed", "backend": "custom_dpo_shared_quantized_reference", "base_model": str(args.base_model), "init_adapter": str(args.init_adapter), "adapter": str(adapter), "optimizer_steps": step, "train_pairs": len(encoded_train), "eval_pairs": len(encoded_eval), "initial_loss": losses[0] if losses else None, "final_loss": losses[-1] if losses else None, "eval_dpo_loss": final_loss, "eval_preference_accuracy": final_acc, "eval_preference_margin": final_margin, "elapsed_seconds": round(time.time() - started, 2), "privacy": "preference pairs were redacted and held in memory; no private SMS text was written to artifacts", "release_policy": "must re-pass robust grounding, pivot, privacy, and human-style metrics before serving"}
    (artifacts / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"event": "completed", **summary}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
