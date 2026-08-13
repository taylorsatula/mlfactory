"""Small deterministic QLoRA trainer for CausalGraph SFT.

This is intentionally separate from the generic mlfactory trainer because the
local Qwen3.5 checkpoint is a multimodal conditional-generation model and must
be loaded quantized across the available GPUs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any


def _load_rows(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
                if limit and len(rows) >= limit:
                    break
    if not rows:
        raise ValueError(f"no training rows in {path}")
    return rows


def _chat_prompt(tokenizer: Any, rendered_prompt: str) -> str:
    messages = [
        {"role": "system", "content": "Solve the symbolic state problem. Do not restate the prompt. Use at most four short derivation lines, then put exactly FINAL: YES or FINAL: NO on the last line."},
        {"role": "user", "content": rendered_prompt},
    ]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def train(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model, prepare_model_for_kbit_training

    seed = int(args.seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    base = Path(args.base_model).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    rows = _load_rows(Path(args.train_file).resolve(), args.limit)
    tokenizer = AutoTokenizer.from_pretrained(str(base), trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(base), trust_remote_code=True, quantization_config=quant,
        device_map="auto", torch_dtype=torch.bfloat16,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)
    if args.init_adapter:
        model = PeftModel.from_pretrained(model, str(Path(args.init_adapter).resolve()), is_trainable=True)
    else:
        # Qwen3.5 has both full-attention and linear-attention blocks.  Select
        # only module suffixes that are present in this checkpoint.
        # q/v-only LoRA keeps the optimizer and adapter footprint small enough
        # for a single 24 GiB RTX 3090 while still changing attention routing.
        preferred = ["q_proj", "v_proj"]
        existing = {name.rsplit(".", 1)[-1] for name, _ in model.named_modules()}
        targets = [name for name in preferred if name in existing]
        if not targets:
            raise RuntimeError("could not find standard Qwen projection modules for LoRA")
        lora = LoraConfig(
            task_type=TaskType.CAUSAL_LM, r=int(args.lora_r), lora_alpha=int(args.lora_alpha),
            lora_dropout=float(args.lora_dropout), target_modules=targets,
        )
        model = get_peft_model(model, lora)
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()

    max_length = int(args.max_length)
    max_target_tokens = int(args.max_target_tokens)
    examples = []
    for row in rows:
        prompt = _chat_prompt(tokenizer, str(row["rendered_prompt"]))
        target = str(row.get("canonical_trace") or row.get("target") or "")
        if not target:
            # Probe task records do not carry the trace in old artifacts.  The
            # caller should provide selected rows with a target field.
            raise ValueError(f"training row {row.get('id')} has no canonical_trace")
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        target_ids = tokenizer(target + (tokenizer.eos_token or ""), add_special_tokens=False, truncation=True, max_length=max_target_tokens)["input_ids"]
        if len(prompt_ids) + len(target_ids) > max_length:
            # Preserve the query and terminal instruction at the end of the
            # prompt rather than silently keeping an irrelevant prefix.
            prompt_ids = prompt_ids[-(max_length - len(target_ids)):]
        if not target_ids:
            continue
        ids = prompt_ids + target_ids
        examples.append({"input_ids": ids, "labels": [-100] * len(prompt_ids) + target_ids})
    if not examples:
        raise ValueError("all training examples became empty after tokenization")

    class Rows(Dataset):
        def __len__(self) -> int: return len(examples)
        def __getitem__(self, index: int) -> dict[str, Any]: return examples[index]

    pad = tokenizer.pad_token_id
    def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
        width = max(len(row["input_ids"]) for row in batch)
        return {
            "input_ids": torch.tensor([row["input_ids"] + [pad] * (width - len(row["input_ids"])) for row in batch], dtype=torch.long),
            "labels": torch.tensor([row["labels"] + [-100] * (width - len(row["labels"])) for row in batch], dtype=torch.long),
            "attention_mask": torch.tensor([[1] * len(row["input_ids"]) + [0] * (width - len(row["input_ids"])) for row in batch], dtype=torch.long),
        }

    loader = DataLoader(Rows(), batch_size=int(args.batch_size), shuffle=True, collate_fn=collate, generator=torch.Generator().manual_seed(seed))
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=float(args.learning_rate), weight_decay=float(args.weight_decay))
    device = next(p for p in model.parameters() if p.device.type != "meta").device
    accumulation = max(1, int(args.gradient_accumulation_steps))
    max_steps = int(args.max_steps) if args.max_steps else None
    model.train(); optimizer.zero_grad(set_to_none=True)
    losses: list[float] = []; steps = 0
    for epoch in range(max(1, int(args.epochs))):
        for batch_index, batch in enumerate(loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type="cuda" if device.type == "cuda" else "cpu", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                loss = model(**batch).loss
            (loss / accumulation).backward()
            if (batch_index + 1) % accumulation == 0 or batch_index + 1 == len(loader):
                torch.nn.utils.clip_grad_norm_(trainable, float(args.max_grad_norm))
                optimizer.step(); optimizer.zero_grad(set_to_none=True)
                steps += 1; losses.append(float(loss.detach().cpu()))
                del loss, batch
                if device.type == "cuda": torch.cuda.empty_cache()
                if steps % 10 == 0: print(json.dumps({"step": steps, "loss": losses[-1]}), flush=True)
                if max_steps and steps >= max_steps: break
        if max_steps and steps >= max_steps: break

    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output)
    tokenizer.save_pretrained(output)
    train_hash = hashlib.sha256(Path(args.train_file).read_bytes()).hexdigest()
    metadata = {
        "backend": "qlora_causal_graph", "base_model": str(base), "base_model_config": str(base / "config.json"),
        "init_adapter": str(Path(args.init_adapter).resolve()) if args.init_adapter else None,
        "training_file": str(Path(args.train_file).resolve()), "training_file_sha256": train_hash,
        "num_examples": len(examples), "optimizer_steps": steps, "initial_loss": losses[0] if losses else None,
        "final_loss": losses[-1] if losses else None, "seed": seed, "max_length": max_length,
        "lora_r": int(args.lora_r), "lora_alpha": int(args.lora_alpha), "learning_rate": float(args.learning_rate),
    }
    (output / "training_manifest.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True); parser.add_argument("--train-file", required=True); parser.add_argument("--output-dir", required=True)
    parser.add_argument("--init-adapter"); parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--epochs", type=int, default=1); parser.add_argument("--max-steps", type=int)
    parser.add_argument("--batch-size", type=int, default=1); parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-5); parser.add_argument("--weight-decay", type=float, default=0.01); parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-length", type=int, default=4096); parser.add_argument("--max-target-tokens", type=int, default=256); parser.add_argument("--limit", type=int)
    parser.add_argument("--lora-r", type=int, default=8); parser.add_argument("--lora-alpha", type=int, default=16); parser.add_argument("--lora-dropout", type=float, default=0.05)
    args = parser.parse_args(); print(json.dumps(train(args), indent=2, sort_keys=True))


if __name__ == "__main__": main()
