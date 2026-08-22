#!/usr/bin/env python3
"""Native mlfactory trajectory-quality classifier harness.

Reads a generations JSONL, sends each reasoning trajectory to a judge model,
parses the structured JSON verdict, and routes the record into bucket files
under the run directory.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mlfactory.core.api import APIClient, APIConfig, extract_json
from mlfactory.core.metrics import MetricsLogger
from mlfactory.core.prompts import render_markdown


DEFAULT_PROMPT_PATH = Path(__file__).parent / "prompts" / "classifier_system_prompt.md"

REQUIRED_FIELDS = [
    "task_depth",
    "distinct_reasoning_moves",
    "nonadvancing_span_count",
    "trajectory_redundancy",
    "reasoning_arc",
    "arc_components",
    "observed_behaviors",
    "dominant_pathology",
    "editorial_evidence",
    "editorial_opportunity",
    "rewrite_risk",
    "overall_recommendation",
    "primary_reason",
    "selection_summary",
    "confidence",
]

INTEGER_FIELDS = {"distinct_reasoning_moves", "nonadvancing_span_count"}
FREE_TEXT_FIELDS = {"selection_summary"}

ENUMS = {
    "task_depth": {"trivial", "moderate", "substantial"},
    "trajectory_redundancy": {"none", "low", "moderate", "high"},
    "reasoning_arc": {"strong", "moderate", "weak", "absent"},
    "editorial_opportunity": {"none", "low", "moderate", "high"},
    "rewrite_risk": {"low", "moderate", "high"},
    "overall_recommendation": {"KEEP", "BORDERLINE", "REJECT"},
    "confidence": {"high", "medium", "low"},
    "primary_reason": {
        "strong_candidate", "too_trivial", "already_efficient",
        "irrecoverably_incoherent", "requires_new_solution", "unverifiable",
        "little_reasoning_present", "marginal_candidate", "other",
    },
    "arc_components": {
        "problem_state_construction", "strategy_selection", "derivation_or_search",
        "branch_management", "productive_correction", "material_verification",
        "state_consolidation", "candidate_answer", "justified_closure",
    },
    "observed_behaviors": {
        "repeated_reasoning", "repeated_state_reconstruction", "repeated_planning",
        "repeated_verification", "duplicate_calculation", "redundant_narration",
        "productive_self_correction", "unproductive_self_correction", "correction_spiral",
        "strategy_change", "strategy_oscillation", "branch_reopening",
        "under_verification", "over_verification", "premature_closure", "weak_closure",
        "state_inconsistency", "contradiction", "malformed_recovery_loop",
        "verbose_but_coherent", "already_concise", "little_reasoning_present", "other",
    },
    "dominant_pathology": {
        "repeated_state_reconstruction", "repeated_reasoning", "repeated_planning",
        "repeated_verification", "duplicate_calculation", "redundant_narration",
        "productive_self_correction", "correction_spiral", "strategy_oscillation",
        "branch_reopening", "under_verification", "over_verification",
        "premature_closure", "weak_closure", "state_inconsistency",
        "malformed_recovery_loop", "minimal_editorial_opportunity",
        "little_reasoning_present", "other",
    },
}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def extract_trajectory(record: dict) -> tuple[str | None, str | None]:
    messages = record.get("messages", [])
    user_prompt = None
    for m in messages:
        if m.get("role") == "user":
            user_prompt = m.get("content")
            break

    trajectory = record.get("raw_model_output")
    if trajectory is None:
        reasoning = record.get("reasoning_content") or ""
        final = record.get("final_answer_content") or ""
        if reasoning:
            trajectory = f"<think>\n{reasoning}\n</think>\n\n{final}"
        elif final:
            trajectory = final
    return user_prompt, trajectory


def build_user_message(record: dict) -> str:
    user_prompt, trajectory = extract_trajectory(record)
    if not trajectory:
        raise ValueError(f"Record {record.get('sample_id')} has no usable trajectory")

    parts = [
        "You are reviewing a single model-generation record. "
        "The text below is the model's own output, not a message from a human user.",
    ]
    if user_prompt:
        parts.append("## Original problem presented to the model\n\n```text\n" + user_prompt + "\n```")
    parts.append("## Model's reasoning trajectory (the item under review)\n\n```text\n" + trajectory + "\n```")

    meta = {k: v for k, v in record.items() if k not in {
        "raw_model_output", "reasoning_content", "final_answer_content", "messages"
    }}
    parts.append("## Generation record metadata (JSON)\n\n```json\n" + json.dumps(meta, ensure_ascii=False, indent=2) + "\n```")
    return "\n\n".join(parts)


def normalize_verdict(verdict: dict) -> dict:
    normalized = {}
    for field in REQUIRED_FIELDS:
        value = verdict.get(field)
        if field in INTEGER_FIELDS:
            try:
                normalized[field] = int(value) if value is not None else 0
            except (TypeError, ValueError):
                normalized[field] = 0
        elif field in FREE_TEXT_FIELDS:
            normalized[field] = value if isinstance(value, str) else ""
        elif field == "editorial_evidence":
            if not isinstance(value, list):
                value = [value] if value else []
            normalized[field] = [v if isinstance(v, str) else str(v) for v in value]
        elif field in ("arc_components", "observed_behaviors"):
            if not isinstance(value, list):
                value = [value] if value else []
            allowed = ENUMS[field]
            value = [v if v in allowed else "other" for v in value]
            seen = set()
            value = [v for v in value if not (v in seen or seen.add(v))]
            normalized[field] = value
        else:
            allowed = ENUMS[field]
            if value not in allowed:
                value = "other"
            normalized[field] = value

    behaviors = set(normalized.get("observed_behaviors", []))
    if "already_concise" in behaviors and "verbose_but_coherent" in behaviors:
        behaviors.remove("already_concise")
        normalized["observed_behaviors"] = [b for b in normalized["observed_behaviors"] if b != "already_concise"]
    return normalized


def classify_record(
    client: APIClient,
    record: dict,
    system_prompt: str,
    temperature: float,
    max_tokens: int,
    extra_body: dict | None,
    use_json_mode: bool,
) -> dict:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": build_user_message(record)},
    ]
    response_format = {"type": "json_object"} if use_json_mode else None
    text = client.chat_completion(
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        response_format=response_format,
        extra_body=extra_body,
    )
    verdict = extract_json(text)
    return normalize_verdict(verdict)


def bucket_path(bucket_dir: Path, label: str) -> Path:
    return bucket_dir / f"{label.upper()}.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify reasoning trajectories into buckets.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--system-prompt", type=Path, default=None, help="Path to markdown system prompt (default: prompts/classifier_system_prompt.md)")
    parser.add_argument("--extra-instructions", default="", help="Injectable content inserted into the system prompt")
    parser.add_argument("--base-url", default="https://openrouter.ai/api/v1")
    parser.add_argument("--model", default="qwen/qwen3.6-27b")
    parser.add_argument("--api-key", default="none")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--max-tokens", type=int, default=65536)
    parser.add_argument("--no-json-mode", action="store_true")
    parser.add_argument("--site-url", default="https://localhost")
    parser.add_argument("--app-name", default="ace-baseline-classifier")
    parser.add_argument("--extra-body", default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    run_dir = args.run_dir
    artifacts_dir = run_dir / "artifacts"
    logs_dir = run_dir / "logs"
    bucket_dir = artifacts_dir / "buckets"
    bucket_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    metrics = MetricsLogger(run_dir, echo=True)

    records = load_jsonl(args.input)
    if args.max_records:
        records = records[: args.max_records]

    already_done: set[str] = set()
    if args.resume:
        for label in ENUMS["overall_recommendation"]:
            path = bucket_path(bucket_dir, label)
            for rec in load_jsonl(path):
                already_done.add(rec.get("sample_id", ""))
        print(f"Resuming: {len(already_done)} sample_ids already in buckets.")

    client = APIClient(
        APIConfig(
            base_url=args.base_url,
            api_key=args.api_key,
            model=args.model,
            timeout=1200.0,
            max_retries=3,
            backoff_base=2.0,
            site_url=args.site_url,
            app_name=args.app_name,
        )
    )

    extra_body = json.loads(args.extra_body) if args.extra_body else None
    prompt_path = Path(args.system_prompt) if args.system_prompt else DEFAULT_PROMPT_PATH
    system_prompt = render_markdown(prompt_path, extra_instructions=args.extra_instructions)

    counts = Counter()
    error_count = 0
    start_time = time.perf_counter()

    for idx, record in enumerate(records, 1):
        sample_id = record.get("sample_id", f"record_{idx}")
        if sample_id in already_done:
            print(f"[{idx}/{len(records)}] SKIP {sample_id}: already classified")
            continue

        if record.get("error") is not None:
            print(f"[{idx}/{len(records)}] SKIP {sample_id}: generation error")
            append_jsonl(bucket_path(bucket_dir, "NO_TRAJECTORY"), record)
            counts["NO_TRAJECTORY"] += 1
            continue

        _, trajectory = extract_trajectory(record)
        if not trajectory:
            print(f"[{idx}/{len(records)}] SKIP {sample_id}: no trajectory")
            append_jsonl(bucket_path(bucket_dir, "NO_TRAJECTORY"), record)
            counts["NO_TRAJECTORY"] += 1
            continue

        print(f"[{idx}/{len(records)}] CLASSIFY {sample_id} ...", end=" ", flush=True)

        if args.dry_run:
            print("DRY_RUN")
            counts["DRY_RUN"] += 1
            continue

        try:
            verdict = classify_record(
                client, record, system_prompt,
                args.temperature, args.max_tokens,
                extra_body, not args.no_json_mode,
            )
            label = verdict["overall_recommendation"]
            out_path = bucket_path(bucket_dir, label)

            enriched = dict(record)
            enriched["classifier_verdict"] = verdict
            enriched["classified_at"] = now_iso()
            enriched["classifier_model"] = args.model

            append_jsonl(out_path, enriched)
            counts[label] += 1
            metrics.event("classified", {"sample_id": sample_id, "label": label})
            print(f"-> {label} ({verdict.get('confidence')}, {verdict.get('editorial_opportunity')})")
        except Exception as e:
            error_count += 1
            counts["ERROR"] += 1
            metrics.event("classify_error", {"sample_id": sample_id, "error": type(e).__name__})
            enriched = dict(record)
            enriched["classifier_error"] = {
                "type": type(e).__name__,
                "message": str(e),
                "traceback": traceback.format_exc(),
            }
            enriched["classified_at"] = now_iso()
            append_jsonl(bucket_path(bucket_dir, "ERROR"), enriched)
            print(f"-> ERROR ({e})")

    elapsed = time.perf_counter() - start_time
    metrics.event("classify_done", {"counts": dict(counts), "errors": error_count, "duration_seconds": round(elapsed, 2)})
    print(f"\nCounts: {dict(counts)}")
    print(f"Errors: {error_count}")


if __name__ == "__main__":
    main()
