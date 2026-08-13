#!/usr/bin/env python3
"""CPU-only schema, privacy, quality, and near-duplicate gate for synthetic SMS."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PII_PATTERNS = {
    "email": re.compile(r"\b[^\s@]+@[^\s@]+\.[A-Za-z]{2,}\b"),
    "url": re.compile(r"(?:https?://|www\.)\S+|\b[A-Za-z0-9.-]+\.(?:com|net|org|io|co)\b", re.I),
    "phone": re.compile(r"(?<!\d)(?:\+?1[\s.-]?)?(?:\(\d{3}\)|\d{3})[\s.-]\d{3}[\s.-]\d{4}(?!\d)"),
    "zip": re.compile(r"(?<!\d)\d{5}(?:-\d{4})?(?!\d)"),
    "payment_or_secret": re.compile(r"\b(?:venmo|zelle|cash\s*app|paypal|routing|account\s*(?:number|#)|pass(?:word|code)|access\s*code|door\s*code|security\s*code)\b", re.I),
    "unresolved_marker": re.compile(r"<[A-Z][A-Z0-9_-]{1,}>|\{\{.*?\}\}", re.S),
}
BAD_TEXT = re.compile(r"(?:<think>|</think>|as an ai|language model|```|^\s*(?:assistant|owner|taylor)\s*:\s*)", re.I | re.M)


def parse_json_response(text: str) -> dict[str, Any] | None:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned[3:].strip()
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3].strip()
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            value = json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError:
            return None
    return value if isinstance(value, dict) else None


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.casefold()).strip()


def ngram_set(text: str, n: int = 5) -> set[str]:
    text = normalize(text)
    return {text[index : index + n] for index in range(max(0, len(text) - n + 1))}


def jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def forbidden_customer_names(threads_dir: Path | None, allowlist: set[str]) -> set[str]:
    if not threads_dir:
        return set()
    names: set[str] = set()
    for path in threads_dir.glob("*.json"):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
            name = normalize(str(row.get("customer", {}).get("name") or ""))
            if name and name not in allowlist:
                names.add(name)
        except Exception:
            continue
    return names


def token_count(tokenizer: Any | None, text: str) -> int:
    if tokenizer is None:
        return max(1, len(text.split()))
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def main() -> int:
    parser = argparse.ArgumentParser(description="Filter fictional synthetic SMS candidates")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, default=None)
    parser.add_argument("--real-threads", type=Path, default=None)
    parser.add_argument("--standard-min", type=int, default=8)
    parser.add_argument("--standard-max", type=int, default=63)
    parser.add_argument("--long-min", type=int, default=64)
    parser.add_argument("--long-max", type=int, default=220)
    parser.add_argument("--target-frequency-cap", type=int, default=100)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(args.output_dir, stat.S_IRWXU)
    catalog = json.loads(args.catalog.read_text(encoding="utf-8"))
    allow = {normalize(x) for values in catalog.get("fictional_allowlist", {}).values() for x in values}
    allow.update({"taylor", "owner"})
    forbidden_names = forbidden_customer_names(args.real_threads, allow)

    tokenizer = None
    if args.tokenizer:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)

    accepted_path = args.output_dir / "accepted_examples.jsonl"
    reasons: Counter[str] = Counter()
    scenario_counts: Counter[str] = Counter()
    length_counts: Counter[str] = Counter()
    target_counts: Counter[str] = Counter()
    exact_seen: set[str] = set()
    buckets: dict[tuple[int, str], list[set[str]]] = defaultdict(list)
    input_count = 0
    accepted_count = 0

    with args.input.open("r", encoding="utf-8", errors="ignore") as source, accepted_path.open("w", encoding="utf-8") as output:
        for line in source:
            input_count += 1
            try:
                candidate = json.loads(line)
                raw = str(candidate.get("raw_response") or "")
            except Exception:
                reasons["invalid_candidate_json"] += 1
                continue
            value = parse_json_response(raw)
            if value is None and isinstance(candidate.get("messages"), list):
                value = {"messages": candidate["messages"], "target": raw}
            if value is None:
                reasons["malformed_response"] += 1
                continue
            messages = value.get("messages")
            target = value.get("target")
            if not isinstance(messages, list) or not isinstance(target, str):
                reasons["schema"] += 1
                continue
            if len(messages) not in {3, 5, 7}:
                reasons["turn_count"] += 1
                continue
            clean_messages: list[dict[str, str]] = []
            valid_roles = True
            for index, message in enumerate(messages):
                if not isinstance(message, dict):
                    valid_roles = False
                    break
                expected = "customer" if index % 2 == 0 else "owner"
                role = normalize(str(message.get("role") or ""))
                text = str(message.get("text") or "").strip()
                if role != expected or not text:
                    valid_roles = False
                    break
                clean_messages.append({"role": role, "text": re.sub(r"\s+", " ", text)})
            if not valid_roles or clean_messages[-1]["role"] != "customer":
                reasons["role_order"] += 1
                continue
            target = re.sub(r"\s+", " ", target).strip()
            target = re.sub(r"^\s*(?:assistant|owner|taylor)\s*:\s*", "", target, flags=re.I)
            if not target or BAD_TEXT.search(target) or any(BAD_TEXT.search(item["text"]) for item in clean_messages):
                reasons["bad_text"] += 1
                continue
            combined = " ".join([item["text"] for item in clean_messages] + [target])
            pii_reason = next((name for name, pattern in PII_PATTERNS.items() if pattern.search(combined)), None)
            if pii_reason:
                reasons[pii_reason] += 1
                continue
            lowered = normalize(combined)
            if any(name in lowered for name in forbidden_names):
                reasons["real_customer_name_overlap"] += 1
                continue
            band = str(candidate.get("length_band") or "standard")
            count = token_count(tokenizer, target)
            lo, hi = (args.long_min, args.long_max) if band == "long" else (args.standard_min, args.standard_max)
            if count < lo or count > hi:
                reasons[f"{band}_length"] += 1
                continue
            exact_key = hashlib.sha256(normalize(" || ".join(item["text"] for item in clean_messages) + " => " + target).encode()).hexdigest()
            if exact_key in exact_seen:
                reasons["exact_duplicate"] += 1
                continue
            exact_seen.add(exact_key)
            target_key = normalize(target)
            if target_counts[target_key] >= args.target_frequency_cap:
                reasons["target_frequency_cap"] += 1
                continue
            sketch = ngram_set(combined)
            bucket_key = (len(sketch) // 20, target_key[:12])
            if any(jaccard(sketch, prior) >= 0.90 for prior in buckets[bucket_key]):
                reasons["near_duplicate"] += 1
                continue
            buckets[bucket_key].append(sketch)
            target_counts[target_key] += 1
            scenario_id = str(candidate.get("scenario_id") or "unknown")
            scenario_counts[scenario_id] += 1
            length_counts[band] += 1
            accepted_count += 1
            record = {
                "example_id": str(candidate.get("candidate_id") or f"accepted-{accepted_count:06d}"),
                "scenario_id": scenario_id,
                "category": str(candidate.get("category") or "unknown"),
                "business": str(candidate.get("business") or "fictional business"),
                "length_band": band,
                "messages": clean_messages,
                "target": target,
                "target_tokens": count,
                "teacher_adapter_sha256": candidate.get("teacher_adapter_sha256"),
                "source_seed": candidate.get("seed"),
            }
            output.write(json.dumps(record, ensure_ascii=False) + "\n")

    report = {
        "status": "completed",
        "input_candidates": input_count,
        "accepted": accepted_count,
        "rejected": sum(reasons.values()),
        "rejection_reasons": dict(reasons),
        "scenario_counts": dict(scenario_counts),
        "length_counts": dict(length_counts),
        "real_name_terms_checked": len(forbidden_names),
        "accepted_examples": str(accepted_path),
        "privacy": "Generated data was filtered locally; no external reviewer or remote service was called.",
    }
    (args.output_dir / "privacy_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"event": "completed", **report}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
