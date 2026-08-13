"""QLoRA trainer with a hard fixed-contract preflight."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .contract import chat_prompt, target as expected_target
from .progress import emit


def _rows(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).open(encoding="utf-8") if line.strip()]


def train(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model, prepare_model_for_kbit_training
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    torch.manual_seed(args.seed)
    base = Path(args.base_model).resolve()
    output = Path(args.output_dir).resolve()
    rows = _rows(args.train_file)
    if args.limit:
        rows = rows[: args.limit]
    tokenizer = AutoTokenizer.from_pretrained(str(base), trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    examples: list[dict[str, list[int]]] = []
    for row in rows:
        target_text = row.get("canonical_trace", "")
        if target_text != expected_target(row):
            raise ValueError(f"{row.get('id')}: target does not match fixed contract")
        prompt_ids = tokenizer(chat_prompt(tokenizer, row["rendered_prompt"]), add_special_tokens=False)["input_ids"]
        target_ids = tokenizer(target_text + (tokenizer.eos_token or ""), add_special_tokens=False)["input_ids"]
        if len(target_ids) > args.max_target_tokens:
            raise ValueError(f"{row['id']}: target exceeds token budget")
        if len(prompt_ids) + len(target_ids) > args.max_length:
            raise ValueError(f"{row['id']}: sequence exceeds context budget")
        decoded = tokenizer.decode(target_ids, skip_special_tokens=True).strip()
        if not decoded.endswith(f"FINAL: {row['canonical_answer']}"):
            raise ValueError(f"{row['id']}: decoded target lost terminal answer")
        examples.append({
            "input_ids": prompt_ids + target_ids,
            "labels": [-100] * len(prompt_ids) + target_ids,
        })
    if not examples:
        raise ValueError("no training examples")

    emit(args.dashboard_file, "training_preflight", stage=args.stage, accepted=len(examples), rejected=0)
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(base),
        trust_remote_code=True,
        quantization_config=quantization,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)
    if args.init_adapter:
        model = PeftModel.from_pretrained(model, str(Path(args.init_adapter).resolve()), is_trainable=True)
    else:
        model = get_peft_model(model, LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj"],
        ))

    class Rows(Dataset):
        def __len__(self) -> int:
            return len(examples)
        def __getitem__(self, index: int) -> dict[str, list[int]]:
            return examples[index]

    def collate(batch: list[dict[str, list[int]]]) -> dict[str, Any]:
        width = max(len(row["input_ids"]) for row in batch)
        return {
            "input_ids": torch.tensor([row["input_ids"] + [tokenizer.pad_token_id] * (width - len(row["input_ids"])) for row in batch]),
            "labels": torch.tensor([row["labels"] + [-100] * (width - len(row["labels"])) for row in batch]),
            "attention_mask": torch.tensor([[1] * len(row["input_ids"]) + [0] * (width - len(row["input_ids"])) for row in batch]),
        }

    loader = DataLoader(Rows(), batch_size=args.batch_size, shuffle=True, collate_fn=collate, generator=torch.Generator().manual_seed(args.seed))
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=0.01)
    device = next(parameter for parameter in model.parameters() if parameter.device.type != "meta").device
    expected_steps = args.max_steps or ((len(loader) + args.gradient_accumulation_steps - 1) // args.gradient_accumulation_steps)
    emit(args.dashboard_file, "stage_start", stage=args.stage, current=0, total=expected_steps)

    losses: list[float] = []
    step = 0
    model.train()
    optimizer.zero_grad(set_to_none=True)
    for batch_index, batch in enumerate(loader):
        batch = {key: value.to(device) for key, value in batch.items()}
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = model(**batch).loss
        (loss / args.gradient_accumulation_steps).backward()
        if (batch_index + 1) % args.gradient_accumulation_steps == 0 or batch_index + 1 == len(loader):
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            losses.append(float(loss.detach().cpu()))
            if step % 5 == 0 or step == expected_steps:
                emit(args.dashboard_file, "training_progress", stage=args.stage, current=step, total=expected_steps, loss=losses[-1])
            if args.max_steps and step >= args.max_steps:
                break

    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output)
    tokenizer.save_pretrained(output)
    manifest = {
        "backend": "qlora_causal_graph_round2",
        "base_model": str(base),
        "init_adapter": str(Path(args.init_adapter).resolve()) if args.init_adapter else None,
        "training_file": str(Path(args.train_file).resolve()),
        "training_file_sha256": hashlib.sha256(Path(args.train_file).read_bytes()).hexdigest(),
        "num_examples": len(examples),
        "prompt_tokens": sum(row["labels"].count(-100) for row in examples),
        "target_tokens": sum(len(row["labels"]) - row["labels"].count(-100) for row in examples),
        "optimizer_steps": step,
        "initial_loss": losses[0] if losses else None,
        "final_loss": losses[-1] if losses else None,
        "seed": args.seed,
        "max_length": args.max_length,
        "max_target_tokens": args.max_target_tokens,
        "learning_rate": args.learning_rate,
    }
    (output / "training_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    emit(args.dashboard_file, "stage_complete", stage=args.stage, current=step, total=expected_steps, final_loss=manifest["final_loss"])
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--init-adapter")
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--max-target-tokens", type=int, default=128)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dashboard-file")
    parser.add_argument("--stage", default="training")
    print(json.dumps(train(parser.parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
