#!/usr/bin/env python3
"""Deterministically filter fictional grounded voice candidates.

The filter is intentionally stricter than a style score: targets must match the
visible role order, pass privacy and unsupported-action checks against their
visible state, stay within real-channel length bands, and be sufficiently
unduplicated.  No private corpus is read by this stage.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import stat
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mlfactory.experiments.voice.privacy_review_synthetic import PII_PATTERNS
from mlfactory.experiments.voice.voice_safety import response_violation

BAD_TEXT = re.compile(r"(?:<think>|</think>|as an ai|language model|synthetic data|scenario|hidden state|```|^\s*(?:assistant|owner|taylor|business representative)\s*:)", re.I | re.M)
META_TEXT = re.compile(r"\b(?:verified state|prompt contract|system instruction|according to the scenario|as requested by the prompt)\b", re.I)
GENERIC_TARGET = re.compile(r"^(?:i can help with that\.?|sure\.?|okay\.?|got it\.?|let me know\.?|sounds good\.?)$", re.I)


def norm(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").casefold()).strip()


def ngrams(value: str, n: int = 5) -> set[str]:
    value = norm(value)
    return {value[i : i + n] for i in range(max(0, len(value) - n + 1))}


def jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def word_count(value: str) -> int:
    return len(norm(value).split())


def clean_target(value: str) -> str:
    value = re.sub(r"<think>.*?</think>", "", str(value or ""), flags=re.I | re.S)
    value = re.sub(r"^\s*(?:assistant|owner|taylor|business representative)\s*:\s*", "", value, flags=re.I)
    return re.sub(r"\s+", " ", value).strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-per-context", type=int, default=2)
    parser.add_argument("--target-frequency-cap", type=int, default=16)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    args.output_dir.chmod(stat.S_IRWXU)

    reasons: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    length_counts: Counter[str] = Counter()
    target_counts: Counter[str] = Counter()
    context_counts: Counter[str] = Counter()
    accepted: list[dict[str, Any]] = []
    seen_candidate: set[str] = set()
    family_sketches: dict[str, list[set[str]]] = defaultdict(list)
    input_count = 0

    for path in args.input:
        with path.open("r", encoding="utf-8", errors="ignore") as source:
            for line in source:
                input_count += 1
                try:
                    candidate = json.loads(line)
                except json.JSONDecodeError:
                    reasons["invalid_json"] += 1
                    continue
                candidate_id = str(candidate.get("candidate_id") or "")
                if not candidate_id or candidate_id in seen_candidate:
                    reasons["duplicate_candidate"] += 1
                    continue
                seen_candidate.add(candidate_id)
                messages = candidate.get("messages")
                target = clean_target(str(candidate.get("raw_response") or candidate.get("target") or ""))
                if not isinstance(messages, list) or not target:
                    reasons["schema"] += 1
                    continue
                if len(messages) < 3 or len(messages) > 9 or len(messages) % 2 == 0:
                    reasons["turn_count"] += 1
                    continue
                valid = True
                clean_messages: list[dict[str, str]] = []
                for index, message in enumerate(messages):
                    expected = "customer" if index % 2 == 0 else "owner"
                    if not isinstance(message, dict) or str(message.get("role")) != expected or not str(message.get("text") or "").strip():
                        valid = False
                        break
                    text = re.sub(r"\s+", " ", str(message["text"])).strip()
                    clean_messages.append({"role": expected, "text": text})
                if not valid:
                    reasons["role_order"] += 1
                    continue
                combined = " ".join(row["text"] for row in clean_messages) + " " + target
                if BAD_TEXT.search(combined) or META_TEXT.search(combined):
                    reasons["meta_or_bad_text"] += 1
                    continue
                pii = next((name for name, pattern in PII_PATTERNS.items() if pattern.search(combined)), None)
                if pii:
                    reasons[f"privacy_{pii}"] += 1
                    continue
                mode = str(candidate.get("mode") or "business_reply")
                state = candidate.get("verified_state") if isinstance(candidate.get("verified_state"), dict) else {}
                violation = response_violation(target, mode, state)
                if violation:
                    reasons[f"grounding_{violation}"] += 1
                    continue
                band = str(candidate.get("length_band") or "standard")
                words = word_count(target)
                limits = {"short": (2, 32), "standard": (6, 65), "detailed": (25, 120)}
                lo, hi = limits.get(band, limits["standard"])
                if words < lo or words > hi:
                    reasons[f"length_{band}"] += 1
                    continue
                if GENERIC_TARGET.fullmatch(target):
                    reasons["generic_target"] += 1
                    continue
                target_key = norm(target)
                if target_counts[target_key] >= args.target_frequency_cap:
                    reasons["target_frequency_cap"] += 1
                    continue
                context_key = hashlib.sha256(norm(json.dumps({"messages": clean_messages, "state": state}, sort_keys=True, ensure_ascii=False)).encode()).hexdigest()
                if context_counts[context_key] >= args.max_per_context:
                    reasons["context_frequency_cap"] += 1
                    continue
                family = str(candidate.get("family") or "unknown")
                sketch = ngrams(target)
                if any(jaccard(sketch, prior) >= 0.94 for prior in family_sketches[family]):
                    reasons["near_duplicate_target"] += 1
                    continue
                family_sketches[family].append(sketch)
                target_counts[target_key] += 1
                context_counts[context_key] += 1
                family_counts[family] += 1
                split = str(candidate.get("split") or "train")
                split_counts[split] += 1
                length_counts[band] += 1
                accepted.append({
                    "example_id": candidate_id,
                    "scenario_id": str(candidate.get("scenario_id") or "unknown"),
                    "family": family,
                    "category": family,
                    "domain": str(candidate.get("domain") or "general_business"),
                    "split": split,
                    "mode": mode,
                    "length_band": band,
                    "style": str(candidate.get("style") or "unknown"),
                    "topic_terms": list(candidate.get("topic_terms") or []),
                    "messages": clean_messages,
                    "target": target,
                    "verified_state": state,
                    "source": "fictional_grounded_base_teacher",
                    "teacher": "frozen_base_model",
                    "candidate_sha256": hashlib.sha256(line.encode("utf-8")).hexdigest(),
                })

    train = [row for row in accepted if row["split"] == "train"]
    evaluation = [row for row in accepted if row["split"] == "eval"]
    replay = [row for row in accepted if row["mode"] in {"casual_sms", "general_question"}]
    for name, values in (("accepted.jsonl", accepted), ("train.jsonl", train), ("eval.jsonl", evaluation), ("replay.jsonl", replay)):
        with (args.output_dir / name).open("w", encoding="utf-8") as stream:
            for row in values:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = {
        "status": "completed",
        "input_candidates": input_count,
        "accepted": len(accepted),
        "rejected": sum(reasons.values()),
        "rejection_reasons": dict(reasons),
        "split_counts": dict(split_counts),
        "family_counts": dict(family_counts),
        "length_counts": dict(length_counts),
        "train": len(train),
        "eval": len(evaluation),
        "replay": len(replay),
        "max_per_context": args.max_per_context,
        "target_frequency_cap": args.target_frequency_cap,
        "policy": "fictional_visible_state_only_deterministic_grounding_gate",
        "private_data": "not read",
    }
    (args.output_dir / "filter_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"event": "completed", **report}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
