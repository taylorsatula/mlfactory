"""Reusable fine-tuning utilities.

The Hugging Face backend intentionally uses a small native PyTorch loop rather
than ``Trainer`` so experiments do not require datasets or accelerate. Imports
are lazy: inference-only installations can continue using mlfactory without
training dependencies.
"""
from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any


MetricCallback = Callable[[int, dict[str, float]], None]


def build_causal_lm_examples(
    records: list[dict[str, Any]],
    prompt_template: str = "Classify this text by topic:\n\n{text}\n\nAnswer:\n",
    response_template: str = '{{"topic": "{topic}", "confidence": {confidence}, "reasoning": "{reasoning}"}}',
) -> list[dict[str, str]]:
    """Convert classification records into prompt/completion examples."""
    examples: list[dict[str, str]] = []
    for index, record in enumerate(records):
        if not record.get("text") or not record.get("topic"):
            raise ValueError(f"record {index} must contain non-empty 'text' and 'topic'")
        values = {
            "text": str(record["text"]),
            "topic": json.dumps(str(record["topic"]), ensure_ascii=False)[1:-1],
            "confidence": float(record.get("confidence", 0.0)),
            "reasoning": json.dumps(
                str(record.get("reasoning", "")), ensure_ascii=False
            )[1:-1],
        }
        examples.append({
            "prompt": prompt_template.format(**values),
            "completion": response_template.format(**values),
        })
    return examples


def train_transformers_causal_lm(
    examples: list[dict[str, str]],
    output_dir: str | Path,
    config: dict[str, Any],
    metric_callback: MetricCallback | None = None,
) -> dict[str, Any]:
    """Fine-tune a Hugging Face causal LM with full weights or a LoRA adapter.

    Required config: ``base_model``. Useful options include ``method`` (``full``
    or ``lora``), ``epochs``, ``learning_rate``, ``batch_size``,
    ``gradient_accumulation_steps``, ``max_length``, and ``device``.
    """
    try:
        import torch
        from torch.utils.data import DataLoader, Dataset
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "transformers_causal_lm requires the training extras; install with "
            "`pip install -e '.[train]'`"
        ) from exc

    if not examples:
        raise ValueError("at least one training example is required")
    base_model = config.get("base_model")
    if not base_model:
        raise ValueError("transformers_causal_lm requires 'base_model'")

    seed = int(config.get("seed", 42))
    torch.manual_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        trust_remote_code=bool(config.get("trust_remote_code", False)),
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("tokenizer must define an EOS or pad token")
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {
        "trust_remote_code": bool(config.get("trust_remote_code", False)),
    }
    dtype_name = config.get("dtype")
    if dtype_name:
        dtype = getattr(torch, str(dtype_name), None)
        if dtype is None:
            raise ValueError(f"unsupported torch dtype: {dtype_name}")
        model_kwargs["torch_dtype"] = dtype
    model = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)
    model.config.use_cache = False

    method = str(config.get("method", "lora"))
    if method == "lora":
        try:
            from peft import LoraConfig, TaskType, get_peft_model
        except ImportError as exc:
            raise RuntimeError("LoRA training requires peft; install mlfactory[train]") from exc
        target_modules = config.get("target_modules")
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=int(config.get("lora_r", 8)),
            lora_alpha=int(config.get("lora_alpha", 16)),
            lora_dropout=float(config.get("lora_dropout", 0.0)),
            target_modules=list(target_modules) if target_modules else None,
        )
        model = get_peft_model(model, lora_config)
    elif method != "full":
        raise ValueError("method must be 'full' or 'lora'")

    requested_device = str(config.get("device", "auto"))
    if requested_device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = requested_device
    model.to(device)

    max_length = int(config.get("max_length", 512))
    if max_length < 2:
        raise ValueError("max_length must be >= 2 so a completion token can be trained")

    class CompletionDataset(Dataset):
        def __len__(self) -> int:
            return len(examples)

        def __getitem__(self, index: int) -> dict[str, Any]:
            example = examples[index]
            completion_ids = tokenizer(
                example["completion"] + (tokenizer.eos_token or ""),
                add_special_tokens=False, truncation=True,
                max_length=max_length - 1,
            )["input_ids"]
            if not completion_ids:
                raise ValueError(f"example {index} produced no completion tokens")
            prompt_ids = tokenizer(
                example["prompt"], add_special_tokens=True, truncation=True,
                max_length=max_length - len(completion_ids),
            )["input_ids"]
            input_ids = prompt_ids + completion_ids
            prompt_length = len(prompt_ids)
            return {
                "input_ids": input_ids,
                "labels": [-100] * prompt_length + input_ids[prompt_length:],
            }

    def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
        width = max(len(item["input_ids"]) for item in batch)
        input_ids, labels, attention_mask = [], [], []
        for item in batch:
            padding = width - len(item["input_ids"])
            input_ids.append(item["input_ids"] + [tokenizer.pad_token_id] * padding)
            labels.append(item["labels"] + [-100] * padding)
            attention_mask.append([1] * len(item["input_ids"]) + [0] * padding)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }

    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        CompletionDataset(),
        batch_size=int(config.get("batch_size", 1)),
        shuffle=True,
        collate_fn=collate,
        generator=generator,
    )
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(config.get("learning_rate", 2e-4)),
        weight_decay=float(config.get("weight_decay", 0.0)),
    )
    accumulation = max(1, int(config.get("gradient_accumulation_steps", 1)))
    epochs = max(1, int(config.get("epochs", 1)))
    max_steps = config.get("max_steps")
    max_steps = int(max_steps) if max_steps is not None else None

    model.train()
    optimizer.zero_grad(set_to_none=True)
    step = 0
    losses: list[float] = []
    stop = False
    for epoch in range(epochs):
        for batch_index, batch in enumerate(loader):
            batch = {key: value.to(device) for key, value in batch.items()}
            output = model(**batch)
            raw_loss = output.loss
            (raw_loss / accumulation).backward()
            should_step = (batch_index + 1) % accumulation == 0 or batch_index + 1 == len(loader)
            if should_step:
                max_grad_norm = float(config.get("max_grad_norm", 1.0))
                torch.nn.utils.clip_grad_norm_(trainable, max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                loss = float(raw_loss.detach().cpu())
                losses.append(loss)
                if metric_callback:
                    metric_callback(step, {"loss": loss, "epoch": float(epoch + 1)})
                if max_steps is not None and step >= max_steps:
                    stop = True
                    break
        if stop:
            break

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    metadata = {
        "backend": "transformers_causal_lm",
        "method": method,
        "base_model": str(base_model),
        "num_examples": len(examples),
        "optimizer_steps": step,
        "initial_loss": losses[0] if losses else None,
        "final_loss": losses[-1] if losses else None,
        "device": device,
    }
    (output_dir / "mlfactory_training.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return metadata
