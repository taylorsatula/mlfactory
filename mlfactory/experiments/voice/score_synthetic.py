#!/usr/bin/env python3
"""Score, select, and split synthetic SMS with local DFT-style diagnostics."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Make direct execution robust when launched by a shell from outside the repo.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from typing import Any


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.casefold()).strip()


def load_records(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def read_redacted_reference(threads_dir: Path, seed: int, test_fraction: float) -> list[str]:
    """Read only redacted held-out Taylor targets into memory; never write them."""
    from mlfactory.experiments.voice.train_voice import EXCLUDED_CUSTOMERS, redact_text, customer_key

    refs: list[str] = []
    for path in sorted(threads_dir.glob("*.json")):
        thread = json.loads(path.read_text(encoding="utf-8"))
        if customer_key(thread) in EXCLUDED_CUSTOMERS:
            continue
        split_value = int.from_bytes(hashlib.sha256(f"{seed}:{path.name}".encode()).digest()[:8], "big") / 2**64
        if split_value >= test_fraction:
            continue
        customer = thread.get("customer", {})
        for message in thread.get("messages", []):
            if str(message.get("from")) == "taylor":
                text = redact_text(str(message.get("text") or ""), customer).strip()
                if text and not any(token in text for token in ("<NAME>", "<PHONE>", "<EMAIL>", "<URL>", "<ADDRESS>", "<ZIP>")):
                    refs.append(text)
    return refs


def simple_metrics(texts: list[str], references: list[str], tokenizer: Any | None) -> dict[str, float]:
    def count_tokens(text: str) -> int:
        return len(tokenizer(text, add_special_tokens=False)["input_ids"]) if tokenizer else len(text.split())

    lengths = [count_tokens(text) for text in texts]
    ref_lengths = [count_tokens(text) for text in references]
    non_ascii = sum(any(ord(char) > 127 for char in text) for text in texts) / max(1, len(texts))
    repeated = 0
    for text in texts:
        words = normalize(text).split()
        if len(words) >= 4 and any(words[i] == words[i + 1] == words[i + 2] for i in range(len(words) - 2)):
            repeated += 1
    ref_mean = sum(ref_lengths) / max(1, len(ref_lengths))
    return {
        "count": float(len(texts)),
        "mean_target_tokens": sum(lengths) / max(1, len(lengths)),
        "p95_target_tokens": sorted(lengths)[min(len(lengths) - 1, int(len(lengths) * 0.95))] if lengths else 0.0,
        "reference_mean_target_tokens": ref_mean,
        "non_ascii_rate": non_ascii,
        "repetition_rate": repeated / max(1, len(texts)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Score and select synthetic SMS examples")
    parser.add_argument("--accepted", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--reference-threads", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--target-count", type=int, default=5000)
    parser.add_argument("--eval-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--embed-model", default=None)
    parser.add_argument("--embed-device", default="cuda:0")
    parser.add_argument("--embed-batch-size", type=int, default=32)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    rows = load_records(args.accepted)
    references = read_redacted_reference(args.reference_threads, seed=42, test_fraction=0.2)
    if not rows:
        raise RuntimeError("no accepted synthetic records")
    if not references:
        raise RuntimeError("no held-out redacted reference targets available")

    # DFT-style witness rewards are optional so the pipeline remains usable if
    # an embedding checkpoint is not cached locally. The lexical diagnostics
    # and hard privacy gates still run in that case.
    witness: list[float] = [0.0] * len(rows)
    embedding_status = "not_requested"
    if args.embed_model:
        try:
            from mlfactory.core.embeddings import embedder
            from mlfactory.experiments.dft.train_dft import compute_mmd_witness_rewards
            encoder = embedder(args.embed_model, device=args.embed_device)
            embedding_status = "dft_mmd_witness"
            for start in range(0, len(rows), 256):
                chunk = rows[start : start + 256]
                values = compute_mmd_witness_rewards(
                    [row["target"] for row in chunk],
                    references[: min(256, len(references))],
                    encoder,
                    kernel="rq",
                    bandwidth="median",
                    rq_alpha=1.0,
                )
                witness[start : start + len(chunk)] = [float(value) for value in values]
                print(json.dumps({"event": "scored", "records": min(start + 256, len(rows))}), flush=True)
        except Exception as exc:
            embedding_status = f"unavailable:{type(exc).__name__}"
            print(json.dumps({"event": "embedding_fallback", "error": str(exc)}), flush=True)

    texts = [row["target"] for row in rows]
    base_metrics = simple_metrics(texts, references, tokenizer)
    # Distributional witness is centered per chunk; use it as a tie-breaker,
    # while length-band and scenario quotas prevent generic short replies from
    # dominating the selected set.
    for row, score in zip(rows, witness):
        row["dft_witness"] = float(score)
        row["quality_score"] = float(score)

    target_count = min(args.target_count, len(rows))
    desired_long = min(sum(row.get("length_band") == "long" for row in rows), round(target_count * 0.20))
    desired_standard = target_count - desired_long
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row.get("category") or "unknown"), str(row.get("length_band") or "standard"))].append(row)
    for values in grouped.values():
        values.sort(key=lambda row: (-float(row["quality_score"]), str(row.get("example_id"))))

    # Allocate quotas proportionally by category within each length band.
    selected: list[dict[str, Any]] = []
    for band, desired in (("long", desired_long), ("standard", desired_standard)):
        groups = {key: values for key, values in grouped.items() if key[1] == band}
        total_available = sum(len(values) for values in groups.values())
        if not total_available:
            continue
        allocations: dict[tuple[str, str], int] = {}
        fractions: list[tuple[float, tuple[str, str]]] = []
        for key, values in groups.items():
            exact = desired * len(values) / total_available
            allocations[key] = min(len(values), int(exact))
            fractions.append((exact - int(exact), key))
        remaining = desired - sum(allocations.values())
        for _fraction, key in sorted(fractions, reverse=True):
            if remaining <= 0:
                break
            if allocations[key] < len(groups[key]):
                allocations[key] += 1
                remaining -= 1
        for key, values in groups.items():
            selected.extend(values[: allocations[key]])

    if len(selected) < target_count:
        existing = {row["example_id"] for row in selected}
        for row in sorted(rows, key=lambda item: (-float(item["quality_score"]), str(item.get("example_id")))):
            if row["example_id"] not in existing:
                selected.append(row)
                existing.add(row["example_id"])
                if len(selected) >= target_count:
                    break
    selected = selected[:target_count]
    selected.sort(key=lambda row: str(row["example_id"]))

    scored_path = args.output_dir / "dft_scores.jsonl"
    with scored_path.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps({
                "example_id": row.get("example_id"),
                "scenario_id": row.get("scenario_id"),
                "length_band": row.get("length_band"),
                "target_tokens": row.get("target_tokens"),
                "dft_witness": row.get("dft_witness", 0.0),
                "quality_score": row.get("quality_score", 0.0),
            }) + "\n")

    selected_path = args.output_dir / "selected_examples.jsonl"
    train_path = args.output_dir / "train.jsonl"
    eval_path = args.output_dir / "eval.jsonl"
    train_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    with selected_path.open("w", encoding="utf-8") as selected_file, train_path.open("w", encoding="utf-8") as train_file, eval_path.open("w", encoding="utf-8") as eval_file:
        for row in selected:
            selected_file.write(json.dumps(row, ensure_ascii=False) + "\n")
            group = f"{row.get('scenario_id')}:{row.get('source_seed')}"
            is_eval = int(hashlib.sha256(group.encode()).hexdigest()[:8], 16) % 10 == 0
            destination = eval_rows if is_eval else train_rows
            destination.append(row)
            (eval_file if is_eval else train_file).write(json.dumps(row, ensure_ascii=False) + "\n")

    report = {
        "status": "completed",
        "input_accepted": len(rows),
        "selected": len(selected),
        "train": len(train_rows),
        "eval": len(eval_rows),
        "long_selected": sum(row.get("length_band") == "long" for row in selected),
        "standard_selected": sum(row.get("length_band") != "long" for row in selected),
        "categories": dict(Counter(str(row.get("category") or "unknown") for row in selected)),
        "reference_targets_in_memory": len(references),
        "embedding_status": embedding_status,
        "metrics": base_metrics,
        "selection_policy": "hard privacy gates first; proportional category/length quotas; DFT witness tie-break; deterministic grouped holdout",
    }
    (args.output_dir / "selection_manifest.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "metrics.json").write_text(json.dumps(report["metrics"], indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"event": "completed", **report}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
