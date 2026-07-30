#!/usr/bin/env python3
"""
Build a 10K/500 pilot dataset from fineweb, using local Qwopus for cleaning,
prompt generation, and outline extraction.

Control flags (use_case, style, target_length, emdash_allowed) are extracted
from the cleaned text rather than guessed, matching the article's protocol.

Run a small dry-run first:
  python build_pilot.py --dry-run 10 --out-dir pilot_dry
Then full build:
  python build_pilot.py --out-dir pilot_full
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path

from datasets import load_dataset
from openai import OpenAI
from transformers import AutoTokenizer


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str) -> None:
    print(f"[{now()}] {msg}", flush=True)


def extract_json_block(text: str) -> dict | None:
    """Try to extract a JSON object from model output, with fallbacks."""
    text = text.strip()
    # strip markdown fences
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # find first { ... } block
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return None


SENTENCE_END_RE = re.compile(r"[.!?]['\"\)\]}]*(?:\s+|$)")


def truncate_to_last_sentence(text: str, min_chars: int = 200) -> str | None:
    """Return text truncated to the last complete sentence."""
    text = text.strip()
    if not text:
        return None
    if re.search(r"[.!?]['\"\)\]}]*\s*$", text):
        return text
    for m in reversed(list(SENTENCE_END_RE.finditer(text))):
        truncated = text[: m.end()].strip()
        if len(truncated) >= min_chars:
            return truncated
    return None


class QwopusClient:
    def __init__(self, base_url: str = "http://localhost:3090/v1", model: str = "Qwopus3.6-27b", api_key: str = "dummy"):
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=300)
        self.model = model

    def chat(self, system: str, user: str, max_tokens: int = 1024, temperature: float = 0.7) -> str:
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=1.0,
                extra_body={"top_k": 0, "repeat_penalty": 1.0},
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            log(f"api error: {e}")
            return ""


def clean_document(client: QwopusClient, text: str) -> str:
    system = (
        "You are a data-cleaning assistant. Given a web document, return only the main written content. "
        "Remove navigation menus, ads, contact information, image captions, footers, legal notices, "
        "and any repeated boilerplate. Preserve all substantive sections of the article. "
        "Output only the cleaned text, no commentary."
    )
    text = text[:8000]
    user = f"Clean this web document:\n\n{text}"
    out = client.chat(system, user, max_tokens=2048, temperature=0.3)
    return out


def generate_prompt_and_metadata(client: QwopusClient, text: str) -> dict:
    """Generate prompt, use_case, and style in a single call. Returns dict or {}."""
    system = (
        "You write prompts and metadata for text-generation models. Given a piece of writing, produce:\n"
        "1. A concise user prompt that could plausibly produce the writing.\n"
        "2. The use_case: one of [blog post, news article, educational article, website copy, personal essay, product description].\n"
        "3. The style: one of [conversational, clear and informative, professional, essayistic, warm, concise and modern].\n"
        "Return ONLY a JSON object with keys: prompt, use_case, style."
    )
    user = f"Analyze this text:\n\n{text[:2500]}"
    out = client.chat(system, user, max_tokens=512, temperature=0.7)
    parsed = extract_json_block(out)
    if parsed and all(k in parsed for k in ("prompt", "use_case", "style")):
        return {
            "prompt": str(parsed["prompt"]).strip(),
            "use_case": str(parsed["use_case"]).strip(),
            "style": str(parsed["style"]).strip(),
        }
    # fallback: just return raw prompt, generic metadata
    return {"prompt": out, "use_case": "blog post", "style": "clear and informative"}


def generate_outline(client: QwopusClient, text: str) -> str:
    system = (
        "You write outlines for responses. Given a piece of writing, produce a brief bullet outline of the response, "
        "including any key facts or quotes. Output only the outline."
    )
    user = f"Outline this text:\n\n{text[:2500]}"
    return client.chat(system, user, max_tokens=512, temperature=0.7)


def add_control_flags(tokenizer, text: str, metadata: dict) -> dict:
    ids = tokenizer.encode(text, add_special_tokens=False)
    has_emdash = "\u2014" in text or "--" in text
    return {
        "target_length": len(ids),
        "emdash_allowed": has_emdash,
        "use_case": metadata.get("use_case", "blog post"),
        "style": metadata.get("style", "clear and informative"),
    }


def format_prompt(prompt: str, flags: dict, outline: str) -> str:
    parts = [
        f"Use case: {flags['use_case']}",
        f"Style: {flags['style']}",
        f"Length: approximately {flags['target_length']} tokens",
        f"Emdash: {'allowed' if flags['emdash_allowed'] else 'not allowed'}",
    ]
    if outline and random.random() < 0.75:
        parts.append(f"Outline:\n{outline}")
    parts.append(f"Prompt:\n{prompt}")
    return "\n\n".join(parts)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="pilot_full", help="output directory")
    p.add_argument("--train-size", default=10000, type=int)
    p.add_argument("--test-size", default=500, type=int)
    p.add_argument("--fineweb-sample", default="CC-MAIN-2024-10", help="fineweb sample name")
    p.add_argument("--dry-run", default=0, type=int, help="if >0, only process N samples and print timing estimate")
    p.add_argument("--seed", default=42, type=int)
    p.add_argument("--resume", action="store_true", help="resume from existing partial outputs")
    p.add_argument("--tokenizer", default="qwen/Qwen3.6-27B", help="HF tokenizer for length computation")
    args = p.parse_args()

    random.seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    total = args.dry_run if args.dry_run > 0 else args.train_size + args.test_size

    log("loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    log("loading fineweb...")
    ds = load_dataset("HuggingFaceFW/fineweb", name=args.fineweb_sample, split="train", streaming=True)
    ds = ds.shuffle(seed=args.seed)
    rows = []
    for i, row in enumerate(ds):
        if i >= total:
            break
        rows.append(row)

    log(f"selected {len(rows)} documents")

    client = QwopusClient()
    processed: list[dict] = []
    partial_path = out_dir / "partial.jsonl"
    if args.resume and partial_path.exists():
        with open(partial_path) as f:
            processed = [json.loads(line) for line in f if line.strip()]
        log(f"resumed with {len(processed)} partial rows")

    start = time.time()
    last_pause = start
    accepted = len(processed)
    for idx, row in enumerate(rows[len(processed):], start=len(processed)):
        # cool-down: pause 5 min every 3 hours of wall time
        if time.time() - last_pause >= 3 * 3600:
            log("cool-down: pausing for 5 minutes after 3 hours of processing")
            time.sleep(300)
            last_pause = time.time()
            log("cool-down complete, resuming")

        raw = row.get("text", "")
        if not raw or len(raw) < 200:
            continue

        cleaned = clean_document(client, raw)
        cleaned = truncate_to_last_sentence(cleaned, min_chars=200)
        if not cleaned:
            continue

        metadata = generate_prompt_and_metadata(client, cleaned)
        if not metadata.get("prompt"):
            continue

        outline = generate_outline(client, cleaned)
        if not outline:
            continue

        flags = add_control_flags(tokenizer, cleaned, metadata)
        final_prompt = format_prompt(metadata["prompt"], flags, outline)

        record = {
            "raw": raw,
            "cleaned": cleaned,
            "prompt": final_prompt,
            "reference": cleaned,
            "flags": flags,
            "outline": outline,
            "original_prompt": metadata["prompt"],
        }
        processed.append(record)
        accepted += 1

        with open(partial_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        if (idx + 1) % 10 == 0:
            elapsed = time.time() - start
            rate = elapsed / (idx + 1)
            log(f"processed {idx+1}/{total} docs, {rate:.1f}s/doc, accepted {accepted}, ETA {rate*(total-idx-1)/3600:.1f}h")

    if args.dry_run > 0:
        elapsed = time.time() - start
        rate = elapsed / len(processed)
        log(f"dry run complete: {len(processed)} docs in {elapsed/60:.1f}min, {rate:.1f}s/doc")
        log(f"full {args.train_size+args.test_size} build ETA: {rate*(args.train_size+args.test_size)/3600:.1f}h")
        if processed:
            log("sample record:")
            print(json.dumps(processed[0], indent=2)[:2000])
        return 0

    random.shuffle(processed)
    train = processed[: args.train_size]
    test = processed[args.train_size : args.train_size + args.test_size]

    with open(out_dir / "train.jsonl", "w", encoding="utf-8") as f:
        for r in train:
            f.write(json.dumps({"prompt": r["prompt"], "reference": r["reference"]}) + "\n")
    with open(out_dir / "test.jsonl", "w", encoding="utf-8") as f:
        for r in test:
            f.write(json.dumps({"prompt": r["prompt"], "reference": r["reference"]}) + "\n")

    log(f"wrote {len(train)} train / {len(test)} test to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
