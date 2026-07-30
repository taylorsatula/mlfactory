#!/usr/bin/env python3
"""Baseline trajectory collection driver.

Reads prompts.jsonl, calls the local vLLM server sequentially, and writes
append-safe JSONL output.  The script is resumable and idempotent: it never
overwrites an existing successful sample.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openai import APIError, OpenAI
from transformers import AutoTokenizer


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_URL = os.environ.get("ACE_BASE_URL", "http://127.0.0.1:3090/v1")
API_KEY = os.environ.get("ACE_API_KEY", "none")
MODEL_NAME = os.environ.get("ACE_MODEL_NAME", "Qwen/Qwen3.5-4B")
MAX_MODEL_LEN = int(os.environ.get("ACE_MAX_MODEL_LEN", "32768"))
MAX_OUTPUT_TOKENS = int(os.environ.get("ACE_MAX_OUTPUT_TOKENS", "16000"))
MARGIN_TOKENS = int(os.environ.get("ACE_MARGIN_TOKENS", "10"))
PROVIDER = os.environ.get("ACE_PROVIDER", "vllm").lower()
RAW_DIR = Path(os.environ.get("ACE_RAW_DIR", "raw_responses"))
GENERATIONS_PATH = Path(os.environ.get("ACE_GENERATIONS_PATH", "generations.jsonl"))
PROMPTS_PATH = Path(os.environ.get("ACE_PROMPTS_PATH", "prompts.jsonl"))
MANIFEST_PATH = Path(os.environ.get("ACE_MANIFEST_PATH", "manifest.json"))
REQUEST_TIMEOUT = float(os.environ.get("ACE_REQUEST_TIMEOUT", "1200"))
TIME_BUDGET_SECONDS = float(os.environ.get("ACE_TIME_BUDGET_SECONDS", "900"))
STRATIFIED_EXTRAS = int(os.environ.get("ACE_STRATIFIED_EXTRAS", "3"))
SEED_OFFSET = int(os.environ.get("ACE_SEED_OFFSET", "0"))
STRATIFIED_SEEDS = [10**9 + SEED_OFFSET + i for i in range(STRATIFIED_EXTRAS)]

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


def provider_sampling_params(profile: dict) -> dict:
    """Return the extra_body sampling params keyed for the active backend."""
    params = {
        "top_k": profile["top_k"],
        "presence_penalty": profile["presence_penalty"],
    }
    if PROVIDER == "llama":
        params["repeat_penalty"] = profile["repeat_penalty"]
    else:
        params["repetition_penalty"] = profile["repetition_penalty"]
    return params


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sample_id(prompt_id: str, seed: int) -> str:
    return f"{prompt_id}_s{seed}"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    records = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def append_jsonl(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def write_json_atomic(path: Path, data: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def deterministic_base_seed(prompt_id: str) -> int:
    digest = hashlib.sha256(prompt_id.encode("utf-8")).hexdigest()
    return (int(digest, 16) % (2**31)) + SEED_OFFSET


def select_stratified_subset(prompts: list[dict], minimum: int = 50) -> list[str]:
    """Deterministically select >=minimum prompt IDs covering category x difficulty."""
    groups: dict[tuple[str, str], list[dict]] = {}
    for p in prompts:
        key = (p["category"], p["difficulty"])
        groups.setdefault(key, []).append(p)
    # sort groups deterministically
    order = sorted(groups.keys())
    selected: list[str] = []
    round_idx = 0
    while len(selected) < minimum:
        added_this_round = False
        for key in order:
            items = groups[key]
            if round_idx < len(items):
                pid = items[round_idx]["prompt_id"]
                if pid not in selected:
                    selected.append(pid)
                    added_this_round = True
                    if len(selected) >= minimum:
                        break
        if not added_this_round:
            break
        round_idx += 1
    return selected


def build_sample_plan(prompts: list[dict]) -> list[tuple[dict, int]]:
    """Return (prompt, seed) pairs: one per prompt plus 3 extras for stratified subset."""
    stratified = set(select_stratified_subset(prompts))
    plan: list[tuple[dict, int]] = []
    for p in prompts:
        base_seed = deterministic_base_seed(p["prompt_id"])
        plan.append((p, base_seed))
        if p["prompt_id"] in stratified:
            for extra in STRATIFIED_SEEDS:
                plan.append((p, base_seed + extra))
    return plan


def count_prompt_tokens(tokenizer: Any, messages: list[dict]) -> int:
    # Some tokenizers return a list of Encoding objects; normalize by encoding the
    # rendered prompt string.
    text = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    return len(tokenizer.encode(text))


def call_model(
    client: OpenAI,
    tokenizer: Any,
    prompt: dict,
    seed: int,
) -> dict:
    messages = [
        {"role": "system", "content": prompt["system_prompt"]},
        {"role": "user", "content": prompt["prompt_text"]},
    ]
    prompt_tokens = count_prompt_tokens(tokenizer, messages)
    context_headroom = max(1, MAX_MODEL_LEN - prompt_tokens - MARGIN_TOKENS)
    max_tokens = min(MAX_OUTPUT_TOKENS, context_headroom)

    profile = SAMPLING_PROFILES.get(prompt["sampling_profile"], SAMPLING_PROFILES["general"])
    sampling_params = {
        "temperature": profile["temperature"],
        "top_p": profile["top_p"],
        "max_tokens": max_tokens,
        "seed": seed,
        "extra_body": provider_sampling_params(profile),
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
            model=MODEL_NAME,
            messages=messages,
            timeout=REQUEST_TIMEOUT,
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
                if time.perf_counter() - start_time > TIME_BUDGET_SECONDS:
                    time_budget_hit = True
                    break
    except Exception as e:
        error = {
            "type": type(e).__name__,
            "message": str(e),
            "traceback": traceback.format_exc(),
        }

    end_time = time.perf_counter()
    elapsed = end_time - start_time

    content = "".join(content_parts)
    reasoning = "".join(reasoning_parts) if reasoning_parts else None

    if reasoning:
        raw_output = f"<think>\n{reasoning}\n</think>\n\n{content}"
    else:
        raw_output = content

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

    record: dict = {
        "prompt_id": prompt["prompt_id"],
        "sample_id": sid,
        "seed": seed,
        "sampling_profile": prompt["sampling_profile"],
        "sampling_params": sampling_params,
        "messages": messages,
        "prompt_tokens": prompt_tokens,
        "max_model_len": MAX_MODEL_LEN,
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

    # Save the streamed chunks for forensic recovery.
    if raw_chunks:
        raw_path = RAW_DIR / f"{sid}.json"
        write_json_atomic(raw_path, raw_chunks)
        record["raw_response_path"] = str(raw_path)

    return record


def ensure_model_revision() -> dict:
    path = Path("model_revision.json")
    if path.exists():
        with path.open() as f:
            return json.load(f)
    return {"model_id": MODEL_NAME, "revision": "unknown"}


def build_manifest(prompts: list[dict], plan: list[tuple[dict, int]]) -> dict:
    rev = ensure_model_revision()
    try:
        nv = subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"], text=True)
    except Exception as e:
        nv = str(e)
    try:
        pkgs = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    except Exception as e:
        pkgs = str(e)
    sp_hash = prompts[0].get("system_prompt_hash", "") if prompts else ""
    return {
        "created_at": now_iso(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "vllm_base_url": BASE_URL,
        "model_name": MODEL_NAME,
        "model_revision": rev.get("revision"),
        "tokenizer_revision": rev.get("revision"),
        "max_model_len": MAX_MODEL_LEN,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "provider": PROVIDER,
        "system_prompt_hash": sp_hash,
        "gpu_info": nv.strip().splitlines(),
        "inference_command": str(Path("launch_vllm.sh").resolve()),
        "package_freeze": pkgs.splitlines(),
        "prompt_count": len(prompts),
        "planned_samples": len(plan),
        "stratified_subset_size": len(select_stratified_subset(prompts)),
        "file_hashes": {
            "prompts.jsonl": sha256_file(PROMPTS_PATH) if PROMPTS_PATH.exists() else None,
            "generations.jsonl": sha256_file(GENERATIONS_PATH) if GENERATIONS_PATH.exists() else None,
        },
    }


def main() -> None:
    global PROMPTS_PATH, GENERATIONS_PATH, MANIFEST_PATH, RAW_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", type=Path, default=PROMPTS_PATH)
    parser.add_argument("--out-dir", type=Path, default=Path("."))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    prompts = load_jsonl(args.prompts)
    if not prompts:
        raise SystemExit("No prompts found.")

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    PROMPTS_PATH = args.prompts
    GENERATIONS_PATH = out_dir / "generations.jsonl"
    MANIFEST_PATH = out_dir / "manifest.json"
    RAW_DIR = out_dir / "raw_responses"
    RAW_DIR.mkdir(exist_ok=True)

    plan = build_sample_plan(prompts)
    if args.max_samples:
        plan = plan[: args.max_samples]

    existing_ids = set()
    if GENERATIONS_PATH.exists():
        for rec in load_jsonl(GENERATIONS_PATH):
            existing_ids.add(rec["sample_id"])
    existing_raw = {p.stem for p in RAW_DIR.glob("*.json")}

    print(f"Prompts: {len(prompts)} | Planned samples: {len(plan)} | Already done: {len(existing_ids)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    client = OpenAI(base_url=BASE_URL, api_key=API_KEY, max_retries=2)

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

            record = call_model(client, tokenizer, prompt, seed)
            append_jsonl(GENERATIONS_PATH, record)
            existing_ids.add(sid)
            existing_raw.add(sid)
            stats["generated"] += 1
            if record.get("error"):
                stats["errors"] += 1
                print(f"ERROR {sid}: {record['error']['type']}: {record['error']['message'][:120]}")
            else:
                comp = record.get("completion_tokens")
                trunc = " [TRUNCATED]" if record.get("truncated") else ""
                print(f"OK {sid}: {comp} tokens{trunc} in {record['duration_seconds']:.1f}s")
    except KeyboardInterrupt:
        print("Interrupted. Flushing manifest and exiting.")
    finally:
        manifest = build_manifest(prompts, plan)
        write_json_atomic(MANIFEST_PATH, manifest)
        print("Manifest written to", MANIFEST_PATH)
        print("Stats:", dict(stats))


if __name__ == "__main__":
    main()
