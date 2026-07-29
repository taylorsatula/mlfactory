#!/usr/bin/env python3
"""Paired local evaluation: stock Qwen3.5-4B vs a DFT LoRA.

The two arms share one quantized base model, raw prompts, tokenizer, sampler,
and per-row random seed. Prompts requesting roughly 800-1300 tokens are
normalized to request 1024 tokens, and generation is allowed up to 1280 tokens
so length-following is not confused with a hard 1024-token cutoff.
"""
from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import re
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from peft import PeftModel
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from eval import TokenDistribution, non_english_char_rate, repetition_rate, self_bleu
from train_dft import _median_heuristic, compute_mmd2


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def atomic_jsonl(path: Path, rows: list[dict]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def strip_reasoning(text: str) -> str:
    if "</think>" in text:
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    elif "<think>" in text:
        # Preserve the payload for forensics if generation exhausted its budget
        # inside an unterminated reasoning block. The state is recorded separately.
        before, after = text.split("<think>", 1)
        text = before if before.strip() else after
    return text.strip()


def requested_tokens(prompt: str) -> int | None:
    match = re.search(r"Length: approximately (\d+) tokens", prompt)
    return int(match.group(1)) if match else None


def use_case(prompt: str) -> str:
    match = re.search(r"Use case: ([^\n]+)", prompt)
    return match.group(1).strip() if match else "unknown"


def normalize_target(prompt: str, target: int) -> str:
    normalized, count = re.subn(
        r"Length: approximately \d+ tokens",
        f"Length: approximately {target} tokens",
        prompt,
        count=1,
    )
    if count != 1:
        raise ValueError("selected prompt has no unique Length field")
    return normalized


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "missing"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--test-file", default="pilot_full/test.jsonl")
    p.add_argument("--model", default="Qwen/Qwen3.5-4B")
    p.add_argument("--adapter", default="out_train_qwen35_h100_fixed/checkpoint-80")
    p.add_argument("--embed-model", default="nvidia/llama-embed-nemotron-8b")
    p.add_argument("--out-dir", default="out_local_pair_1024_ckpt80_20260727")
    p.add_argument("--target-tokens", type=int, default=1024)
    p.add_argument("--max-new-tokens", type=int, default=1280)
    p.add_argument("--max-prompt-tokens", type=int, default=1024)
    p.add_argument("--original-target-min", type=int, default=800)
    p.add_argument("--original-target-max", type=int, default=1300)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--prompt-mode", choices=["chat", "raw"], default="chat")
    p.add_argument(
        "--system-prompt",
        default="You write the final requested text directly. Do not include reasoning, <think> tags, or meta-commentary.",
    )
    p.add_argument("--seed", type=int, default=20260727)
    p.add_argument("--embed-device", default="cuda:0")
    p.add_argument("--embed-batch-size", type=int, default=1)
    p.add_argument("--bootstrap-samples", type=int, default=1000)
    p.add_argument("--generation-only", action="store_true")
    p.add_argument("--score-only", action="store_true")
    return p.parse_args()


def load_rows(args: argparse.Namespace) -> tuple[list[dict], str]:
    source = Path(args.test_file)
    raw_lines = source.read_text(encoding="utf-8").splitlines()
    all_rows = [json.loads(line) for line in raw_lines if line.strip()]
    selected = []
    for row_id, row in enumerate(all_rows):
        original_target = requested_tokens(row["prompt"])
        if original_target is None or not (args.original_target_min <= original_target <= args.original_target_max):
            continue
        prompt = normalize_target(row["prompt"], args.target_tokens)
        selected.append({
            "row_id": row_id,
            "use_case": use_case(prompt),
            "original_target_tokens": original_target,
            "target_tokens": args.target_tokens,
            "prompt": prompt,
            "reference": row["reference"],
        })
    if len(selected) < 2:
        raise ValueError("selection must contain at least two rows")
    return selected, sha256(source)


def load_generator(args: argparse.Namespace):
    log(f"loading tokenizer and 4-bit base {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True, local_files_only=True
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    base = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        local_files_only=True,
        quantization_config=quant,
        device_map={"": "cuda:0"},
    )
    log(f"loading DFT adapter {args.adapter}")
    model = PeftModel.from_pretrained(
        base, args.adapter, is_trainable=False, local_files_only=True
    )
    model.eval()

    lora_tensors = [
        p.detach() for name, p in model.named_parameters()
        if "lora_" in name and p.numel() > 0
    ]
    nonzero = sum(int(torch.count_nonzero(p).item()) for p in lora_tensors)
    if not lora_tensors or nonzero == 0:
        raise RuntimeError("adapter validation failed: no nonzero LoRA weights")

    probe = tokenizer("Adapter validation probe.", return_tensors="pt").to("cuda:0")
    with torch.inference_mode(), model.disable_adapter():
        stock_logits = model(**probe, logits_to_keep=1).logits.float().cpu()
    with torch.inference_mode():
        dft_logits = model(**probe, logits_to_keep=1).logits.float().cpu()
    max_logit_delta = float((stock_logits - dft_logits).abs().max())
    if not math.isfinite(max_logit_delta) or max_logit_delta <= 1e-6:
        raise RuntimeError(f"adapter validation failed: max logit delta={max_logit_delta}")
    log(
        f"adapter validated: {len(lora_tensors)} LoRA tensors, "
        f"{nonzero:,} nonzero values, probe max logit delta={max_logit_delta:.6f}"
    )
    return tokenizer, model, max_logit_delta


def generate_arm(
    model,
    tokenizer,
    prompt: str,
    arm: str,
    seed: int,
    args: argparse.Namespace,
) -> dict:
    if args.prompt_mode == "chat":
        model_input = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": args.system_prompt},
                {"role": "user", "content": prompt},
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        add_special_tokens = False
    else:
        model_input = prompt
        add_special_tokens = True
    encoded = tokenizer(
        model_input,
        return_tensors="pt",
        truncation=True,
        max_length=args.max_prompt_tokens,
        add_special_tokens=add_special_tokens,
    ).to("cuda:0")
    prompt_tokens = int(encoded.input_ids.shape[1])
    if prompt_tokens >= args.max_prompt_tokens:
        raise RuntimeError(f"prompt reached truncation cap ({prompt_tokens})")
    set_seed(seed)
    torch.cuda.reset_peak_memory_stats(0)
    start = time.time()
    context = model.disable_adapter() if arm == "stock" else contextlib.nullcontext()
    with context, torch.inference_mode():
        generated = model.generate(
            **encoded,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
    elapsed = time.time() - start
    suffix = generated[0, prompt_tokens:].detach().cpu()
    eos = False
    if tokenizer.eos_token_id is not None:
        eos = bool((suffix == tokenizer.eos_token_id).any().item())
    raw_text = tokenizer.decode(suffix, skip_special_tokens=True)
    unterminated_think = "<think>" in raw_text and "</think>" not in raw_text
    text = strip_reasoning(raw_text)
    content_tokens = len(tokenizer.encode(text, add_special_tokens=False))
    return {
        "text": text,
        "raw_text": raw_text,
        "prompt_tokens": prompt_tokens,
        "generated_tokens": int(suffix.numel()),
        "content_tokens": content_tokens,
        "unterminated_think": unterminated_think,
        "eos": eos,
        "hit_cap": int(suffix.numel()) >= args.max_new_tokens and not eos,
        "seconds": elapsed,
        "tokens_per_second": float(suffix.numel() / max(elapsed, 1e-9)),
        "peak_gpu_gib": float(torch.cuda.max_memory_allocated(0) / 2**30),
    }


def run_generation(args: argparse.Namespace, selected: list[dict], out_dir: Path) -> list[dict]:
    generations_path = out_dir / "generations.jsonl"
    completed: dict[int, dict] = {}
    if generations_path.exists():
        for line in generations_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                completed[int(row["row_id"])] = row
        log(f"resuming with {len(completed)} completed pairs")

    tokenizer, model, max_logit_delta = load_generator(args)
    for position, row in enumerate(selected, 1):
        row_id = int(row["row_id"])
        if row_id in completed:
            continue
        seed = args.seed + row_id * 1009
        log(
            f"pair {position}/{len(selected)} row={row_id} case={row['use_case']} "
            f"original_target={row['original_target_tokens']} normalized_target={args.target_tokens}"
        )
        # Back-to-back paired generation with common random numbers.
        stock = generate_arm(model, tokenizer, row["prompt"], "stock", seed, args)
        log(
            f"  stock: {stock['content_tokens']} content tokens, eos={stock['eos']}, "
            f"{stock['tokens_per_second']:.1f} tok/s"
        )
        dft = generate_arm(model, tokenizer, row["prompt"], "dft", seed, args)
        log(
            f"  dft:   {dft['content_tokens']} content tokens, eos={dft['eos']}, "
            f"{dft['tokens_per_second']:.1f} tok/s"
        )
        completed[row_id] = {
            **row,
            "seed": seed,
            "adapter_probe_max_logit_delta": max_logit_delta,
            "stock": stock,
            "dft": dft,
        }
        ordered = [completed[int(r["row_id"])] for r in selected if int(r["row_id"]) in completed]
        atomic_jsonl(generations_path, ordered)

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return [completed[int(r["row_id"])] for r in selected]


def truncate_text(tokenizer, text: str, limit: int = 256) -> str:
    ids = tokenizer.encode(text, add_special_tokens=False)[:limit]
    return tokenizer.decode(ids, skip_special_tokens=True)


def emdash_violation(rows: list[dict], arm: str) -> float:
    applicable = [r for r in rows if re.search(r"Emdash: not allowed", r["prompt"], re.I)]
    if not applicable:
        return 0.0
    return sum("—" in r[arm]["text"] for r in applicable) / len(applicable)


def summarize_arm(
    rows: list[dict], arm: str, refs: list[str], embeddings: dict[str, np.ndarray],
    token_dist: TokenDistribution, bandwidth: float, target: int, max_new: int,
) -> dict:
    texts = [r[arm]["text"] for r in rows]
    content_lengths = [int(r[arm]["content_tokens"]) for r in rows]
    emb = embeddings[arm]
    ref_emb = embeddings["reference"]
    cosine = np.sum(emb.astype(np.float32) * ref_emb.astype(np.float32), axis=1)
    abs_errors = [abs(n - target) for n in content_lengths]
    return {
        "n": len(rows),
        "rq_mmd2": compute_mmd2(emb, ref_emb, "rq", bandwidth, 1.0),
        "rbf_mmd2": compute_mmd2(emb, ref_emb, "rbf", bandwidth, 1.0),
        "l2_1gram": token_dist.l2_distance(texts, refs, n=1),
        "l2_2gram": token_dist.l2_distance(texts, refs, n=2),
        "l2_3gram": token_dist.l2_distance(texts, refs, n=3),
        "mean_reference_cosine": float(cosine.mean()),
        "median_reference_cosine": float(np.median(cosine)),
        "repetition_rate": repetition_rate(texts),
        "non_english_char_rate": non_english_char_rate(texts),
        "non_ascii_token_rate": token_dist.non_ascii_token_rate(texts),
        "self_bleu": self_bleu(texts),
        "emdash_violation_rate": emdash_violation(rows, arm),
        "avg_content_tokens": float(np.mean(content_lengths)),
        "median_content_tokens": float(np.median(content_lengths)),
        "mean_absolute_target_error": float(np.mean(abs_errors)),
        "within_20pct_target_rate": float(np.mean([e <= target * 0.2 for e in abs_errors])),
        "eos_rate": float(np.mean([r[arm]["eos"] for r in rows])),
        "cap_rate": float(np.mean([r[arm]["hit_cap"] for r in rows])),
        "avg_seconds": float(np.mean([r[arm]["seconds"] for r in rows])),
        "avg_tokens_per_second": float(np.mean([r[arm]["tokens_per_second"] for r in rows])),
        "max_peak_gpu_gib": float(max(r[arm]["peak_gpu_gib"] for r in rows)),
    }


def bootstrap_deltas(
    embeddings: dict[str, np.ndarray], rows: list[dict], bandwidth: float,
    samples: int, seed: int, target: int,
) -> dict:
    rng = np.random.default_rng(seed)
    n = len(rows)
    rq, rbf, cosine, length_error = [], [], [], []
    stock_cos = np.sum(embeddings["stock"].astype(np.float32) * embeddings["reference"].astype(np.float32), axis=1)
    dft_cos = np.sum(embeddings["dft"].astype(np.float32) * embeddings["reference"].astype(np.float32), axis=1)
    stock_err = np.array([abs(r["stock"]["content_tokens"] - target) for r in rows], dtype=np.float64)
    dft_err = np.array([abs(r["dft"]["content_tokens"] - target) for r in rows], dtype=np.float64)
    for _ in range(samples):
        idx = rng.integers(0, n, size=n)
        ref = embeddings["reference"][idx]
        stock = embeddings["stock"][idx]
        dft = embeddings["dft"][idx]
        rq.append(compute_mmd2(dft, ref, "rq", bandwidth, 1.0) - compute_mmd2(stock, ref, "rq", bandwidth, 1.0))
        rbf.append(compute_mmd2(dft, ref, "rbf", bandwidth, 1.0) - compute_mmd2(stock, ref, "rbf", bandwidth, 1.0))
        cosine.append(float(np.mean(dft_cos[idx] - stock_cos[idx])))
        length_error.append(float(np.mean(dft_err[idx] - stock_err[idx])))

    def ci(values: list[float], lower_better: bool) -> dict:
        arr = np.asarray(values)
        return {
            "mean_delta": float(arr.mean()),
            "ci95": [float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))],
            "probability_dft_better": float(np.mean(arr < 0 if lower_better else arr > 0)),
        }

    return {
        "rq_mmd2_dft_minus_stock": ci(rq, True),
        "rbf_mmd2_dft_minus_stock": ci(rbf, True),
        "reference_cosine_dft_minus_stock": ci(cosine, False),
        "absolute_length_error_dft_minus_stock": ci(length_error, True),
    }


def excerpt(text: str, limit: int = 1200) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0] + " […]"


def metric_table(stock: dict, dft: dict, target: int | None = 1024) -> str:
    metrics = [
        ("RQ MMD²", "rq_mmd2", "lower"),
        ("RBF MMD²", "rbf_mmd2", "lower"),
        ("Unigram L2", "l2_1gram", "lower"),
        ("Bigram L2", "l2_2gram", "lower"),
        ("Trigram L2", "l2_3gram", "lower"),
        ("Mean paired reference cosine", "mean_reference_cosine", "higher"),
        ("Repetition rate", "repetition_rate", "lower"),
        ("Non-English character rate", "non_english_char_rate", "lower"),
        ("Self-BLEU", "self_bleu", "context"),
        ("Em-dash violation rate", "emdash_violation_rate", "lower"),
    ]
    if target is not None:
        metrics.extend([
            ("Average content tokens", "avg_content_tokens", f"target {target}"),
            ("Mean absolute target error", "mean_absolute_target_error", "lower"),
            ("Within ±20% of target", "within_20pct_target_rate", "higher"),
            ("EOS rate", "eos_rate", "higher"),
            ("Hard-cap rate", "cap_rate", "lower"),
        ])
    lines = ["| Metric | Stock | DFT-80 | DFT minus stock | Direction |", "|---|---:|---:|---:|---|"]
    for label, key, direction in metrics:
        s, d = stock[key], dft[key]
        lines.append(f"| {label} | {s:.6f} | {d:.6f} | {d-s:+.6f} | {direction} |")
    return "\n".join(lines)


def write_report(
    args: argparse.Namespace, rows: list[dict], metrics: dict, boot: dict,
    embeddings: dict[str, np.ndarray], out_dir: Path,
) -> None:
    stock_cos = np.sum(embeddings["stock"].astype(np.float32) * embeddings["reference"].astype(np.float32), axis=1)
    dft_cos = np.sum(embeddings["dft"].astype(np.float32) * embeddings["reference"].astype(np.float32), axis=1)
    gains = dft_cos - stock_cos
    ordered = list(np.argsort(gains))
    # Explicit selection: two regressions, two central cases, and two gains.
    picks = []
    for idx in ordered[:2] + ordered[max(0, len(ordered)//2-1):len(ordered)//2+1] + ordered[-2:]:
        if int(idx) not in picks:
            picks.append(int(idx))

    lines = [
        "# Local paired 1024-token comparison: stock Qwen3.5-4B vs DFT checkpoint 80",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S %Z')}",
        "",
        "## Method",
        "",
        f"- **{len(rows)} paired prompts** selected before generation because their original requested lengths were {args.original_target_min}-{args.original_target_max} tokens.",
        f"- Each prompt's requested length was normalized to **{args.target_tokens} tokens**. All complete prompts fit below the {args.max_prompt_tokens}-token input cap.",
        f"- Both arms used the same cached `Qwen/Qwen3.5-4B` base, NF4 4-bit weights, `{args.prompt_mode}` prompt format, and sampler: temperature {args.temperature}, top-p {args.top_p}, top-k {args.top_k}.",
        "- Chat mode used the model's native template with `enable_thinking=False` and the same direct-writing system message for both arms.",
        f"- Generation cap was {args.max_new_tokens}, deliberately above the requested length so EOS and target adherence could be observed.",
        "- Stock was evaluated by disabling the LoRA; DFT used checkpoint 80. Each pair used the same per-row random seed.",
        "- Embeddings used the exact `nvidia/llama-embed-nemotron-8b` model in FP16 with normalized vectors and a reference-only bandwidth shared by both arms.",
        "- MMD and n-gram metrics are distributional. Paired reference cosine is included as a secondary conditional-fidelity diagnostic, not as the DFT objective.",
        "",
        "## Aggregate results",
        "",
        metric_table(metrics["stock"], metrics["dft"], args.target_tokens),
        "",
        f"Shared reference-only median bandwidth: `{metrics['bandwidth']:.8f}`",
        "",
        "## Paired-bootstrap uncertainty",
        "",
        "Intervals use 1,000 row-paired bootstrap resamples. For MMD and length error, negative deltas favor DFT. For cosine, positive deltas favor DFT.",
        "",
        "```json",
        json.dumps(boot, indent=2),
        "```",
        "",
        "## First-256-token sensitivity",
        "",
        "Checkpoint 80 was trained on rollouts capped at 256 tokens. This table scores the first 256 content tokens against the first 256 reference tokens, separating learned-horizon behavior from long-form extrapolation.",
        "",
        metric_table(metrics["first256_stock"], metrics["first256_dft"], None),
        "",
        "## Example pairs",
        "",
        "Examples are selected mechanically by change in paired reference cosine: two largest regressions, two central cases, and two largest gains. Excerpts are truncated for readability; `generations.jsonl` contains complete outputs.",
        "",
    ]
    for idx in picks:
        row = rows[idx]
        lines.extend([
            f"### Row {row['row_id']}: {row['use_case']} ({'gain' if gains[idx] > 0 else 'regression'} {gains[idx]:+.4f} cosine)",
            "",
            f"Original target: {row['original_target_tokens']}; normalized target: {row['target_tokens']}. "
            f"Stock content tokens: {row['stock']['content_tokens']}; DFT content tokens: {row['dft']['content_tokens']}.",
            "",
            "**Prompt excerpt**",
            "",
            "> " + excerpt(row["prompt"], 600).replace("\n", "\n> "),
            "",
            "**Reference excerpt**",
            "",
            excerpt(row["reference"]),
            "",
            "**Stock excerpt**",
            "",
            excerpt(row["stock"]["text"]),
            "",
            "**DFT-80 excerpt**",
            "",
            excerpt(row["dft"]["text"]),
            "",
        ])
    lines.extend([
        "## Limitations",
        "",
        "- This is a 26-prompt, one-seed paired long-form test, not the definitive multi-sampler sweep.",
        "- The adapter was trained with 256-token prompts and responses, so 1024-token prompting and generation test extrapolation beyond its training horizon.",
        "- References are Qwopus-cleaned FineWeb rather than an untouched human holdout.",
        "- Common random seeds reduce sampling noise but do not replace repeated-seed confidence intervals.",
        "",
    ])
    (out_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def run_scoring(args: argparse.Namespace, rows: list[dict], out_dir: Path) -> None:
    log(f"loading exact embedder {args.embed_model} on {args.embed_device}")
    embedder = SentenceTransformer(
        args.embed_model,
        device=args.embed_device,
        trust_remote_code=True,
        local_files_only=True,
        model_kwargs={"torch_dtype": torch.float16},
    )
    texts = {
        "reference": [r["reference"] for r in rows],
        "stock": [r["stock"]["text"] for r in rows],
        "dft": [r["dft"]["text"] for r in rows],
    }
    embeddings = {}
    for name, values in texts.items():
        log(f"embedding {name}: {len(values)} texts at batch size {args.embed_batch_size}")
        embeddings[name] = embedder.encode(
            values,
            batch_size=args.embed_batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
    del embedder
    gc.collect()
    torch.cuda.empty_cache()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True, local_files_only=True
    )
    first_texts = {
        name: [truncate_text(tokenizer, text, 256) for text in values]
        for name, values in texts.items()
    }
    log("reloading embedder for first-256-token sensitivity embeddings")
    embedder = SentenceTransformer(
        args.embed_model,
        device=args.embed_device,
        trust_remote_code=True,
        local_files_only=True,
        model_kwargs={"torch_dtype": torch.float16},
    )
    first_embeddings = {}
    for name, values in first_texts.items():
        first_embeddings[name] = embedder.encode(
            values,
            batch_size=args.embed_batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
    del embedder
    gc.collect()
    torch.cuda.empty_cache()

    bandwidth = _median_heuristic(embeddings["reference"], embeddings["reference"])
    first_bandwidth = _median_heuristic(first_embeddings["reference"], first_embeddings["reference"])
    token_dist = TokenDistribution(args.model)
    refs = texts["reference"]
    metrics = {
        "bandwidth": bandwidth,
        "first256_bandwidth": first_bandwidth,
        "stock": summarize_arm(rows, "stock", refs, embeddings, token_dist, bandwidth, args.target_tokens, args.max_new_tokens),
        "dft": summarize_arm(rows, "dft", refs, embeddings, token_dist, bandwidth, args.target_tokens, args.max_new_tokens),
    }

    first_rows = []
    for row, stock, dft in zip(rows, first_texts["stock"], first_texts["dft"]):
        copied = {**row, "stock": {**row["stock"], "text": stock, "content_tokens": len(tokenizer.encode(stock, add_special_tokens=False))}, "dft": {**row["dft"], "text": dft, "content_tokens": len(tokenizer.encode(dft, add_special_tokens=False))}}
        first_rows.append(copied)
    metrics["first256_stock"] = summarize_arm(first_rows, "stock", first_texts["reference"], first_embeddings, token_dist, first_bandwidth, 256, 256)
    metrics["first256_dft"] = summarize_arm(first_rows, "dft", first_texts["reference"], first_embeddings, token_dist, first_bandwidth, 256, 256)

    boot = bootstrap_deltas(embeddings, rows, bandwidth, args.bootstrap_samples, args.seed + 17, args.target_tokens)
    np.savez_compressed(out_dir / "embeddings.npz", **embeddings, **{f"first256_{k}": v for k, v in first_embeddings.items()})
    atomic_json(out_dir / "metrics.json", metrics)
    atomic_json(out_dir / "bootstrap.json", boot)

    stock_cos = np.sum(embeddings["stock"].astype(np.float32) * embeddings["reference"].astype(np.float32), axis=1)
    dft_cos = np.sum(embeddings["dft"].astype(np.float32) * embeddings["reference"].astype(np.float32), axis=1)
    scored = []
    for i, row in enumerate(rows):
        scored.append({
            **row,
            "stock_reference_cosine": float(stock_cos[i]),
            "dft_reference_cosine": float(dft_cos[i]),
            "cosine_delta_dft_minus_stock": float(dft_cos[i] - stock_cos[i]),
        })
    atomic_jsonl(out_dir / "scored_generations.jsonl", scored)
    write_report(args, scored, metrics, boot, embeddings, out_dir)
    log(f"report written to {out_dir / 'REPORT.md'}")


def main() -> int:
    args = parse_args()
    if args.generation_only and args.score_only:
        raise ValueError("--generation-only and --score-only are mutually exclusive")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("CUDA GPU required")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    selected, dataset_hash = load_rows(args)
    adapter_file = Path(args.adapter) / "adapter_model.safetensors"
    manifest = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "args": vars(args),
        "selected_row_ids": [r["row_id"] for r in selected],
        "selected_use_cases": {case: sum(r["use_case"] == case for r in selected) for case in sorted({r["use_case"] for r in selected})},
        "dataset_sha256": dataset_hash,
        "adapter_sha256": sha256(adapter_file),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "transformers": package_version("transformers"),
        "peft": package_version("peft"),
        "bitsandbytes": package_version("bitsandbytes"),
        "sentence_transformers": package_version("sentence-transformers"),
        "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "script_sha256": sha256(__file__),
    }
    atomic_json(out_dir / "manifest.json", manifest)
    log(f"selected {len(selected)} rows: {manifest['selected_use_cases']}")

    generations_path = out_dir / "generations.jsonl"
    if args.score_only:
        if not generations_path.exists():
            raise FileNotFoundError(generations_path)
        rows = [json.loads(line) for line in generations_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        rows = run_generation(args, selected, out_dir)
    if not args.generation_only:
        run_scoring(args, rows, out_dir)

    files = [p for p in out_dir.iterdir() if p.is_file() and p.name != "SHA256SUMS"]
    with open(out_dir / "SHA256SUMS", "w", encoding="utf-8") as f:
        for path in sorted(files):
            f.write(f"{sha256(path)}  {path.name}\n")
    log("local paired evaluation complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
