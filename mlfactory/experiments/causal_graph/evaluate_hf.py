"""Deterministic Hugging Face evaluation for base or QLoRA checkpoints."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

FINAL_RE = re.compile(r"FINAL\s*:\s*(YES|NO)\b", re.IGNORECASE)
SYSTEM_PROMPT = "Solve the symbolic state problem. Do not restate the prompt. Use at most four short derivation lines, then put exactly FINAL: YES or FINAL: NO on the last line."


def parse_final(text: str) -> tuple[str | None, bool]:
    matches = FINAL_RE.findall(text or "")
    return (matches[-1].upper(), False) if matches else (None, True)


def _prompt(tokenizer: Any, rendered: str) -> str:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": rendered}]
    try: return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError: return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    base = Path(args.base_model).resolve()
    tokenizer = AutoTokenizer.from_pretrained(str(base), trust_remote_code=True)
    if tokenizer.pad_token_id is None: tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(str(base), trust_remote_code=True, quantization_config=quant, device_map="auto", torch_dtype=torch.bfloat16)
    if args.adapter: model = PeftModel.from_pretrained(model, str(Path(args.adapter).resolve()))
    model.eval(); device = next(p for p in model.parameters() if p.device.type != "meta").device
    rows = [json.loads(line) for line in Path(args.input).open(encoding="utf-8") if line.strip()]
    out_path = Path(args.output); out_path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    with out_path.open("w", encoding="utf-8") as output:
        for start in range(0, len(rows), max(1, int(args.batch_size))):
            chunk = rows[start:start + max(1, int(args.batch_size))]
            texts = [_prompt(tokenizer, str(row["rendered_prompt"])) for row in chunk]
            try:
                encoded = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=int(args.max_length)).to(device)
                width = encoded["input_ids"].shape[1]
                with torch.inference_mode(): generated = model.generate(**encoded, do_sample=False, max_new_tokens=int(args.max_new_tokens), pad_token_id=tokenizer.pad_token_id)
                outputs = [tokenizer.decode(generated[i][width:], skip_special_tokens=True) for i in range(len(chunk))]
                prompt_lengths = [int(v) for v in encoded["attention_mask"].sum(dim=1).detach().cpu().tolist()]
                errors = [None] * len(chunk)
            
            except Exception as exc:
                outputs = [""] * len(chunk); prompt_lengths = [None] * len(chunk); errors = [f"{type(exc).__name__}: {exc}"] * len(chunk)
            for row, text, error, prompt_tokens in zip(chunk, outputs, errors, prompt_lengths):
                parsed, parse_failure = parse_final(text)
                target_tokens = len(tokenizer(text, add_special_tokens=False)["input_ids"]) if text else 0
                record = {
                    "example_id": row["id"], "checkpoint": args.checkpoint, "world": row["world_id"], "depth": row["depth"],
                    "relevant_nodes": row["relevant_nodes"], "distractor_nodes": row["distractor_nodes"], "binary_gate_count": row["binary_gate_count"],
                    "negation_count": row["negation_count"], "source_update_count": row["source_update_count"], "raw_model_output": text,
                    "parsed_answer": parsed, "gold_answer": row["canonical_answer"], "correct": parsed == row["canonical_answer"],
                    "parse_failure": parse_failure, "error": error, "prompt_tokens": prompt_tokens, "target_tokens": target_tokens,
                }
                records.append(record); output.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            output.flush()
            if len(records) % 25 < len(chunk) or len(records) == len(rows): print(f"evaluated {len(records)}", flush=True)
    return {"checkpoint": args.checkpoint, "n": len(records), "accuracy": sum(r["correct"] for r in records) / max(1, len(records)), "parse_failures": sum(r["parse_failure"] for r in records)}


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--base-model", required=True); p.add_argument("--input", required=True); p.add_argument("--output", required=True); p.add_argument("--adapter"); p.add_argument("--checkpoint", default="HF"); p.add_argument("--max-length", type=int, default=4096); p.add_argument("--max-new-tokens", type=int, default=512); p.add_argument("--batch-size", type=int, default=2)
    print(json.dumps(evaluate(p.parse_args()), indent=2))


if __name__ == "__main__": main()
