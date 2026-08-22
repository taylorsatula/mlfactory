#!/usr/bin/env python3
"""Native mlfactory baseline trajectory collection driver.

Reads prompts.jsonl, calls the configured OpenAI-compatible endpoint, and writes
append-safe JSONL output under the run directory. Expects to be invoked from a
plugin that has already started a model server and supplies --base-url.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import time
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openai import OpenAI
from transformers import AutoTokenizer

from mlfactory.core.metrics import MetricsLogger


SAMPLING_PROFILES = {
    "general": {
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "presence_penalty": 1.5,
        "repeat_penalty": 1.0,
        "repetition_penalty": 1.0,
    },
    "coding": {
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "presence_penalty": 0.0,
        "repeat_penalty": 1.0,
        "repetition_penalty": 1.0,
    },
}


def provider_sampling_params(profile: dict, provider: str) -> dict:
    params = {"top_k": profile["top_k"], "presence_penalty": profile["presence_penalty"]}
    if provider == "llama":
        params["repeat_penalty"] = profile["repeat_penalty"]
    else:
        params["repetition_penalty"] = profile["repetition_penalty"]
    return params


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sample_id(prompt_id: str, seed: int) -> str:
    return f"{prompt_id}_s{seed}"


def deterministic_base_seed(prompt_id: str, seed_offset: int) -> int:
    digest = hashlib.sha256(prompt_id.encode("utf-8")).hexdigest()
    return (int(digest, 16) % (2**31)) + seed_offset


def load_jsonl(path: Path) -> list[dict]:
    records = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def write_json_atomic(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def select_stratified_subset(prompts: list[dict], minimum: int, seed_offset: int) -> list[str]:
    groups: dict[tuple[str, str], list[dict]] = {}
    for p in prompts:
        key = (p["category"], p["difficulty"])
        groups.setdefault(key, []).append(p)
    order = sorted(groups.keys())
    selected: list[str] = []
    round_idx = 0
    while len(selected) < minimum:
        added = False
        for key in order:
            items = groups[key]
            if round_idx < len(items):
                pid = items[round_idx]["prompt_id"]
                if pid not in selected:
                    selected.append(pid)
                    added = True
                    if len(selected) >= minimum:
                        break
        if not added:
            break
        round_idx += 1
    return selected


def build_sample_plan(prompts: list[dict], stratified_extras: int, seed_offset: int) -> list[tuple[dict, int]]:
    stratified = set(select_stratified_subset(prompts, minimum=50, seed_offset=seed_offset))
    stratified_seeds = [10**9 + seed_offset + i for i in range(stratified_extras)]
    plan: list[tuple[dict, int]] = []
    for p in prompts:
        base_seed = deterministic_base_seed(p["prompt_id"], seed_offset)
        plan.append((p, base_seed))
        if p["prompt_id"] in stratified:
            for extra in stratified_seeds:
                plan.append((p, base_seed + extra))
    return plan


def count_prompt_tokens(tokenizer: Any, messages: list[dict]) -> int:
    text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    return len(tokenizer.encode(text))


def call_model(
    client: OpenAI,
    tokenizer: Any,
    prompt: dict,
    seed: int,
    base_url: str,
    model_name: str,
    max_model_len: int,
    max_output_tokens: int,
    request_timeout: float,
    time_budget_seconds: float,
    margin_tokens: int,
    provider: str,
) -> dict:
    messages = [
        {"role": "system", "content": prompt["system_prompt"]},
        {"role": "user", "content": prompt["prompt_text"]},
    ]
    prompt_tokens = count_prompt_tokens(tokenizer, messages)
    context_headroom = max(1, max_model_len - prompt_tokens - margin_tokens)
    max_tokens = min(max_output_tokens, context_headroom)

    profile = SAMPLING_PROFILES.get(prompt["sampling_profile"], SAMPLING_PROFILES["general"])
    sampling_params = {
        "temperature": profile["temperature"],
        "top_p": profile["top_p"],
        "max_tokens": max_tokens,
        "seed": seed,
        "extra_body": provider_sampling_params(profile, provider),
    }

    sid = sample_id(prompt["prompt_id"], seed)
    start_time = time.perf_counter()
    start_iso = now_iso()
    error = None
    raw_chunks: list[dict] = []
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    finish_reason: str | None = None
    usage = None
    time_budget_hit = False

    try:
        with client.chat.completions.create(
            model=model_name,
            messages=messages,
            timeout=request_timeout,
            stream=True,
            stream_options={"include_usage": True},
            **sampling_params,
        ) as stream:
            for chunk in stream:
                raw_chunks.append(chunk.model_dump(mode="json"))
                if chunk.usage is not None:
                    usage = chunk.usage
                if not chunk.choices:
                    continue
                choice = chunk.choices[0]
                if choice.finish_reason:
                    finish_reason = choice.finish_reason
                delta = choice.delta
                if delta:
                    if delta.content:
                        content_parts.append(delta.content)
                    if getattr(delta, "reasoning_content", None):
                        reasoning_parts.append(delta.reasoning_content)
                if time.perf_counter() - start_time > time_budget_seconds:
                    time_budget_hit = True
                    break
    except Exception as e:
        error = {
            "type": type(e).__name__,
            "message": str(e),
            "traceback": traceback.format_exc(),
        }

    elapsed = time.perf_counter() - start_time
    content = "".join(content_parts)
    reasoning = "".join(reasoning_parts) if reasoning_parts else None
    raw_output = f"<think>\n{reasoning}\n</think>\n\n{content}" if reasoning else content

    if usage is not None and usage.completion_tokens is not None:
        completion_tokens = usage.completion_tokens
        total_tokens = usage.total_tokens
    else:
        completion_tokens = len(tokenizer.encode(content, add_special_tokens=False))
        total_tokens = prompt_tokens + completion_tokens

    truncated = False
    if finish_reason == "length":
        truncated = True
    elif time_budget_hit and finish_reason not in ("stop", "eos"):
        truncated = True
        finish_reason = finish_reason or "time_budget"

    return {
        "prompt_id": prompt["prompt_id"],
        "sample_id": sid,
        "seed": seed,
        "sampling_profile": prompt["sampling_profile"],
        "sampling_params": sampling_params,
        "messages": messages,
        "prompt_tokens": prompt_tokens,
        "max_model_len": max_model_len,
        "max_tokens": max_tokens,
        "requested_at": start_iso,
        "duration_seconds": round(elapsed, 4),
        "raw_model_output": raw_output if not error else None,
        "reasoning_content": reasoning if not error else None,
        "final_answer_content": (content if content else raw_output) if not error else None,
        "completion_tokens": completion_tokens if not error else None,
        "total_tokens": total_tokens if not error else None,
        "finish_reason": finish_reason if not error else None,
        "truncated": truncated if not error else None,
        "time_budget_hit": time_budget_hit if not error else None,
        "error": error,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:3090/v1")
    parser.add_argument("--model-name", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--provider", default="llama")
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--max-output-tokens", type=int, default=16000)
    parser.add_argument("--margin-tokens", type=int, default=10)
    parser.add_argument("--request-timeout", type=float, default=1200.0)
    parser.add_argument("--time-budget-seconds", type=float, default=900.0)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--stratified-extras", type=int, default=3)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    prompts = load_jsonl(args.prompts)
    if not prompts:
        raise SystemExit("No prompts found.")

    run_dir = args.run_dir
    artifacts_dir = run_dir / "artifacts"
    logs_dir = run_dir / "logs"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    generations_path = artifacts_dir / "generations.jsonl"
    raw_dir = artifacts_dir / "raw_responses"
    raw_dir.mkdir(exist_ok=True)

    metrics = MetricsLogger(run_dir, echo=True)

    plan = build_sample_plan(prompts, args.stratified_extras, args.seed_offset)
    if args.max_samples:
        plan = plan[: args.max_samples]

    existing_ids = set()
    if generations_path.exists():
        for rec in load_jsonl(generations_path):
            existing_ids.add(rec["sample_id"])
    existing_raw = {p.stem for p in raw_dir.glob("*.json")}

    print(f"Prompts: {len(prompts)} | Planned samples: {len(plan)} | Already done: {len(existing_ids)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    client = OpenAI(base_url=args.base_url, api_key="none", max_retries=2)

    stats = Counter({"generated": 0, "skipped": 0, "errors": 0})
    try:
        for prompt, seed in plan:
            sid = sample_id(prompt["prompt_id"], seed)
            if sid in existing_ids and sid in existing_raw:
                stats["skipped"] += 1
                continue
            if args.dry_run:
                print(f"Would generate {sid} for {prompt['prompt_id']} ({prompt['category']})")
                stats["generated"] += 1
                continue

            record = call_model(
                client=client,
                tokenizer=tokenizer,
                prompt=prompt,
                seed=seed,
                base_url=args.base_url,
                model_name=args.model_name,
                max_model_len=args.max_model_len,
                max_output_tokens=args.max_output_tokens,
                request_timeout=args.request_timeout,
                time_budget_seconds=args.time_budget_seconds,
                margin_tokens=args.margin_tokens,
                provider=args.provider,
            )
            if record.get("raw_chunks"):
                # raw_chunks is not currently captured; keep field consistent.
                record["raw_response_path"] = None
            append_jsonl(generations_path, record)
            existing_ids.add(sid)
            existing_raw.add(sid)
            stats["generated"] += 1
            if record.get("error"):
                stats["errors"] += 1
                metrics.event("sample_error", {"sample_id": sid, "error": record["error"]["type"]})
                print(f"ERROR {sid}: {record['error']['type']}: {record['error']['message'][:120]}")
            else:
                comp = record.get("completion_tokens")
                trunc = " [TRUNCATED]" if record.get("truncated") else ""
                print(f"OK {sid}: {comp} tokens{trunc} in {record['duration_seconds']:.1f}s")
    except KeyboardInterrupt:
        print("Interrupted. Flushing metrics and exiting.")
    finally:
        metrics.event("collect_done", {"stats": dict(stats), "planned": len(plan)})
        print("Stats:", dict(stats))


if __name__ == "__main__":
    main()
