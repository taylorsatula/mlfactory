#!/usr/bin/env python3
"""Native mlfactory trajectory stratification harness.

Annotates KEEP-bucket reasoning trajectories and persists queryable tags in
SQLite under the run directory.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sqlite3
import sys
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mlfactory.core.api import APIClient, APIConfig, extract_json
from mlfactory.core.metrics import MetricsLogger
from mlfactory.core.prompts import render_markdown

from classify import extract_trajectory, load_jsonl


DEFAULT_PROMPT_PATH = Path(__file__).parent / "prompts" / "stratify_system_prompt.md"

POSITIVE = {
    "coherent_arc", "nontrivial_reasoning", "useful_struggle", "productive_self_correction",
    "useful_branch_exploration", "productive_strategy_change", "meaningful_verification",
    "effective_state_consolidation", "stable_representation", "appropriate_uncertainty",
    "evidence_before_commitment", "justified_closure", "effective_action_state_integration",
    "efficient_progression",
}
NEGATIVE = {
    "redundant_narration", "repeated_planning", "repeated_state_reconstruction",
    "duplicate_calculation", "redundant_verification", "under_verification", "branch_reopening",
    "strategy_oscillation", "correction_spiral", "premature_commitment", "weak_closure",
    "overextended_closure", "action_state_disconnect", "representation_churn",
    "state_inconsistency", "unresolved_material_error", "incomplete_arc",
}
TRANSFORMATIONS = {
    "SPAN_REMOVAL", "STATE_CONSOLIDATION", "VERIFICATION_CALIBRATION",
    "CORRECTION_PRESERVATION", "BRANCH_RESOLUTION", "STRATEGY_TRANSITION",
    "COMMITMENT_SEQUENCING", "CLOSURE_CALIBRATION", "ACTION_STATE_INTEGRATION",
    "REPRESENTATION_NORMALIZATION",
}
TRANSFORMATION_MIXTURE_WEIGHTS: dict[str, float] = {
    "SPAN_REMOVAL": 1.0,
    "STATE_CONSOLIDATION": 1.0,
    "VERIFICATION_CALIBRATION": 1.0,
    "CORRECTION_PRESERVATION": 1.0,
    "BRANCH_RESOLUTION": 0.5,
    "STRATEGY_TRANSITION": 0.5,
    "CLOSURE_CALIBRATION": 0.5,
    "COMMITMENT_SEQUENCING": 0.25,
    "ACTION_STATE_INTEGRATION": 0.25,
    "REPRESENTATION_NORMALIZATION": 0.25,
}
TRANSFORMATION_WITNESS_MAP: dict[str, set[str]] = {
    "SPAN_REMOVAL": {"redundant_narration", "repeated_planning", "duplicate_calculation", "redundant_verification", "overextended_closure"},
    "STATE_CONSOLIDATION": {"repeated_state_reconstruction", "duplicate_calculation"},
    "VERIFICATION_CALIBRATION": {"redundant_verification", "under_verification", "weak_closure", "overextended_closure"},
    "CORRECTION_PRESERVATION": {"productive_self_correction", "useful_struggle", "correction_spiral"},
    "BRANCH_RESOLUTION": {"useful_branch_exploration", "branch_reopening", "strategy_oscillation"},
    "STRATEGY_TRANSITION": {"productive_strategy_change"},
    "COMMITMENT_SEQUENCING": {"premature_commitment"},
    "CLOSURE_CALIBRATION": {"weak_closure", "overextended_closure"},
    "ACTION_STATE_INTEGRATION": {"action_state_disconnect"},
    "REPRESENTATION_NORMALIZATION": {"representation_churn"},
}
CONFIDENCE = {"high", "medium", "low"}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def validate_annotation(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("response is not a JSON object")
    required = {
        "trajectory_arc_summary", "transformation_support_labels",
        "requires_new_reasoning_for_full_repair", "confidence",
    }
    missing = required - value.keys()
    if missing:
        raise ValueError(f"missing fields: {sorted(missing)}")
    if not isinstance(value["trajectory_arc_summary"], str) or not value["trajectory_arc_summary"].strip():
        raise ValueError("trajectory_arc_summary must be a non-empty string")
    for field, allowed in (
        ("positive_attributes_observed", POSITIVE),
        ("negative_attributes_observed", NEGATIVE),
        ("transformation_support_labels", TRANSFORMATIONS),
    ):
        labels = value.get(field, [])
        if not isinstance(labels, list) or any(not isinstance(x, str) for x in labels):
            raise ValueError(f"{field} must be an array of strings")
        unknown = set(labels) - allowed
        if unknown:
            raise ValueError(f"unknown {field}: {sorted(unknown)}")
        value[field] = list(dict.fromkeys(labels))
    for field in ("requires_new_reasoning_for_full_repair",):
        if type(value[field]) is not bool:
            raise ValueError(f"{field} must be Boolean")
    if value["confidence"] not in CONFIDENCE:
        raise ValueError("invalid confidence")

    value["trajectory_preserving_transformation_supported"] = len(value["transformation_support_labels"]) > 0
    value["mixture_priority_score"] = sum(TRANSFORMATION_MIXTURE_WEIGHTS.get(label, 0.0) for label in value["transformation_support_labels"])

    positive_set = set(value.get("positive_attributes_observed", []))
    negative_set = set(value.get("negative_attributes_observed", []))
    audit_flags = []
    for label in value["transformation_support_labels"]:
        witnesses = TRANSFORMATION_WITNESS_MAP.get(label, set())
        if witnesses and not ((positive_set & witnesses) or (negative_set & witnesses)):
            audit_flags.append(f"{label}_without_witness")
    value["audit_flags"] = audit_flags

    output_keys = required | {"trajectory_preserving_transformation_supported", "mixture_priority_score", "audit_flags"}
    for optional_field in ("positive_attributes_observed", "negative_attributes_observed"):
        if optional_field in value:
            output_keys.add(optional_field)
    return {key: value[key] for key in output_keys}


def problem_from(record: dict) -> str:
    for message in record.get("messages", []):
        if message.get("role") == "user" and isinstance(message.get("content"), str):
            return message["content"]
    return ""


def render_prompt(template: str, record: dict) -> str:
    _, trajectory = extract_trajectory(record)
    if not trajectory:
        raise ValueError("record has no raw reasoning trajectory")
    if "{{PROBLEM}}" not in template or "{{TRAJECTORY}}" not in template:
        raise ValueError("prompt template must contain {{PROBLEM}} and {{TRAJECTORY}}")
    return template.replace("{{PROBLEM}}", problem_from(record)).replace("{{TRAJECTORY}}", trajectory)


def create_schema(db: sqlite3.Connection) -> None:
    db.executescript("""
    PRAGMA foreign_keys = ON;
    PRAGMA journal_mode = WAL;
    CREATE TABLE IF NOT EXISTS annotations (
        sample_id TEXT PRIMARY KEY,
        prompt_id TEXT,
        trajectory_arc_summary TEXT NOT NULL,
        trajectory_preserving_transformation_supported INTEGER NOT NULL CHECK (trajectory_preserving_transformation_supported IN (0,1)),
        requires_new_reasoning_for_full_repair INTEGER NOT NULL CHECK (requires_new_reasoning_for_full_repair IN (0,1)),
        confidence TEXT NOT NULL CHECK (confidence IN ('high','medium','low')),
        positive_attributes_json TEXT NOT NULL,
        negative_attributes_json TEXT NOT NULL,
        transformation_labels_json TEXT NOT NULL,
        mixture_priority_score REAL NOT NULL DEFAULT 0.0,
        audit_flags_json TEXT NOT NULL DEFAULT '[]',
        classifier_verdict_json TEXT,
        source_record_json TEXT NOT NULL,
        raw_response TEXT NOT NULL,
        model TEXT NOT NULL,
        tagged_at TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS positive_attributes (
        sample_id TEXT NOT NULL REFERENCES annotations(sample_id) ON DELETE CASCADE,
        label TEXT NOT NULL,
        PRIMARY KEY (sample_id, label)
    );
    CREATE TABLE IF NOT EXISTS negative_attributes (
        sample_id TEXT NOT NULL REFERENCES annotations(sample_id) ON DELETE CASCADE,
        label TEXT NOT NULL,
        PRIMARY KEY (sample_id, label)
    );
    CREATE TABLE IF NOT EXISTS transformation_labels (
        sample_id TEXT NOT NULL REFERENCES annotations(sample_id) ON DELETE CASCADE,
        label TEXT NOT NULL,
        PRIMARY KEY (sample_id, label)
    );
    CREATE TABLE IF NOT EXISTS errors (
        sample_id TEXT PRIMARY KEY,
        prompt_id TEXT,
        error_type TEXT NOT NULL,
        error_message TEXT NOT NULL,
        traceback TEXT NOT NULL,
        source_record_json TEXT NOT NULL,
        failed_at TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_annotations_prompt_id ON annotations(prompt_id);
    CREATE INDEX IF NOT EXISTS idx_transform_label ON transformation_labels(label);
    CREATE INDEX IF NOT EXISTS idx_positive_label ON positive_attributes(label);
    CREATE INDEX IF NOT EXISTS idx_negative_label ON negative_attributes(label);
    """)
    cols = {row[1] for row in db.execute("PRAGMA table_info(annotations)")}
    if "mixture_priority_score" not in cols:
        db.execute("ALTER TABLE annotations ADD COLUMN mixture_priority_score REAL NOT NULL DEFAULT 0.0")
    if "audit_flags_json" not in cols:
        db.execute("ALTER TABLE annotations ADD COLUMN audit_flags_json TEXT NOT NULL DEFAULT '[]'")
    if "classifier_verdict_json" not in cols:
        db.execute("ALTER TABLE annotations ADD COLUMN classifier_verdict_json TEXT")


def save_annotation(db: sqlite3.Connection, record: dict, annotation: dict, raw: str, model: str) -> None:
    sid = str(record["sample_id"])
    with db:
        db.execute("DELETE FROM errors WHERE sample_id = ?", (sid,))
        db.execute("""
            INSERT OR REPLACE INTO annotations (
                sample_id, prompt_id, trajectory_arc_summary,
                trajectory_preserving_transformation_supported, requires_new_reasoning_for_full_repair,
                confidence, positive_attributes_json, negative_attributes_json, transformation_labels_json,
                mixture_priority_score, audit_flags_json, classifier_verdict_json,
                source_record_json, raw_response, model, tagged_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            sid, record.get("prompt_id"), annotation["trajectory_arc_summary"],
            int(annotation["trajectory_preserving_transformation_supported"]),
            int(annotation["requires_new_reasoning_for_full_repair"]), annotation["confidence"],
            json.dumps(annotation.get("positive_attributes_observed", [])),
            json.dumps(annotation.get("negative_attributes_observed", [])),
            json.dumps(annotation["transformation_support_labels"]),
            float(annotation["mixture_priority_score"]),
            json.dumps(annotation["audit_flags"]),
            json.dumps(record.get("classifier_verdict"), ensure_ascii=False) if record.get("classifier_verdict") is not None else None,
            json.dumps(record, ensure_ascii=False), raw, model, now_iso(),
        ))
        for table in ("positive_attributes", "negative_attributes", "transformation_labels"):
            db.execute(f"DELETE FROM {table} WHERE sample_id = ?", (sid,))
        for table, field in (
            ("positive_attributes", "positive_attributes_observed"),
            ("negative_attributes", "negative_attributes_observed"),
            ("transformation_labels", "transformation_support_labels"),
        ):
            db.executemany(f"INSERT INTO {table} (sample_id,label) VALUES (?,?)", ((sid, x) for x in annotation.get(field, [])))


def call_model(
    client: APIClient,
    model: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
    extra_body: dict | None,
    json_mode: bool,
    retries: int,
    backoff: float,
) -> tuple[dict, str]:
    last: Exception | None = None
    for attempt in range(retries):
        try:
            response_format = {"type": "json_object"} if json_mode else None
            text = client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format,
                extra_body=extra_body,
            )
            return validate_annotation(extract_json(text)), text
        except Exception as exc:
            last = exc
            if attempt + 1 < retries:
                wait = backoff * 2 ** attempt
                print(f"retry {attempt + 1}/{retries}: {exc} (sleep {wait}s)", file=sys.stderr)
                time.sleep(wait)
    raise last or RuntimeError("all attempts failed")


def _process_one(
    args: argparse.Namespace,
    template: str,
    client: APIClient,
    record: dict,
    index: int,
    total: int,
    done: set[str],
    progress_lock: threading.Lock,
) -> str:
    sid = str(record.get("sample_id", f"record_{index}"))
    if sid in done and not args.force:
        with progress_lock:
            print(f"[{index}/{total}] SKIP {sid}")
        return "skip"

    db = sqlite3.connect(args.database)
    extra_body = json.loads(args.extra_body) if args.extra_body else None
    try:
        prompt = render_prompt(template, record)
        with progress_lock:
            print(f"[{index}/{total}] TAG {sid} ...", end=" ", flush=True)
        annotation, raw = call_model(
            client, args.model, prompt, args.temperature, args.max_tokens,
            extra_body, not args.no_json_mode, args.retries, args.backoff,
        )
        save_annotation(db, record, annotation, raw, args.model)
        with progress_lock:
            print(f"OK ({len(annotation['transformation_support_labels'])} transformations)")
        return "tag"
    except Exception as exc:
        with db:
            db.execute("INSERT OR REPLACE INTO errors VALUES (?,?,?,?,?,?,?)", (
                sid, record.get("prompt_id"), type(exc).__name__, str(exc), traceback.format_exc(),
                json.dumps(record, ensure_ascii=False), now_iso()))
        with progress_lock:
            print(f"ERROR: {exc}")
        return "error"
    finally:
        db.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Stratify KEEP trajectories into a SQLite database.")
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--prompt", type=Path, default=None, help="Path to markdown prompt template")
    ap.add_argument("--extra-instructions", default="", help="Injectable content inserted into the prompt")
    ap.add_argument("--base-url", default="https://openrouter.ai/api/v1")
    ap.add_argument("--api-key", default="none")
    ap.add_argument("--model", default="qwen/qwen3.6-27b")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--extra-body", default="")
    ap.add_argument("--no-json-mode", action="store_true")
    ap.add_argument("--offset", type=int, default=0, help="Number of records to skip from the start of the input.")
    ap.add_argument("--max-records", type=int)
    ap.add_argument("--max-workers", type=int, default=1)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--backoff", type=float, default=2.0)
    ap.add_argument("--loop-delay", type=float, default=0.3)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    run_dir = args.run_dir
    artifacts_dir = run_dir / "artifacts"
    logs_dir = run_dir / "logs"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    metrics = MetricsLogger(run_dir, echo=True)

    prompt_path = Path(args.prompt) if args.prompt else DEFAULT_PROMPT_PATH
    template = render_markdown(prompt_path, extra_instructions=args.extra_instructions)

    records = load_jsonl(args.input)
    if args.offset:
        records = records[args.offset:]
    if args.max_records is not None:
        records = records[:args.max_records]

    database = artifacts_dir / "stratification.sqlite3"
    database.parent.mkdir(parents=True, exist_ok=True)
    args.database = database

    main_db = sqlite3.connect(database)
    create_schema(main_db)
    done = {row[0] for row in main_db.execute("SELECT sample_id FROM annotations")}
    main_db.close()

    client = APIClient(
        APIConfig(
            base_url=args.base_url,
            api_key=args.api_key,
            model=args.model,
            timeout=1200.0,
            max_retries=0,
        )
    )
    progress_lock = threading.Lock()
    total = len(records)

    if args.dry_run:
        prompt = render_prompt(template, records[0])
        print(prompt)
        return

    results: list[str] = []
    if args.max_workers > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {
                executor.submit(_process_one, args, template, client, record, i, total, done, progress_lock): i
                for i, record in enumerate(records, 1)
            }
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())
    else:
        db = sqlite3.connect(database)
        extra_body = json.loads(args.extra_body) if args.extra_body else None
        try:
            for i, record in enumerate(records, 1):
                sid = str(record.get("sample_id", f"record_{i}"))
                if sid in done and not args.force:
                    results.append("skip")
                    print(f"[{i}/{total}] SKIP {sid}")
                    continue
                try:
                    prompt = render_prompt(template, record)
                    print(f"[{i}/{total}] TAG {sid} ...", end=" ", flush=True)
                    annotation, raw = call_model(
                        client, args.model, prompt, args.temperature, args.max_tokens,
                        extra_body, not args.no_json_mode, args.retries, args.backoff,
                    )
                    save_annotation(db, record, annotation, raw, args.model)
                    results.append("tag")
                    metrics.event("stratified", {"sample_id": sid, "transformations": len(annotation["transformation_support_labels"])})
                    print(f"OK ({len(annotation['transformation_support_labels'])} transformations)")
                    if args.loop_delay:
                        time.sleep(args.loop_delay)
                except Exception as exc:
                    results.append("error")
                    metrics.event("stratify_error", {"sample_id": sid, "error": type(exc).__name__})
                    with db:
                        db.execute("INSERT OR REPLACE INTO errors VALUES (?,?,?,?,?,?,?)", (
                            sid, record.get("prompt_id"), type(exc).__name__, str(exc), traceback.format_exc(),
                            json.dumps(record, ensure_ascii=False), now_iso()))
                    print(f"ERROR: {exc}")
        finally:
            db.close()

    tagged = results.count("tag")
    skipped = results.count("skip")
    errors = results.count("error")
    metrics.event("stratify_done", {"tagged": tagged, "skipped": skipped, "errors": errors})
    print(f"Done: tagged={tagged} skipped={skipped} errors={errors} database={database}")


if __name__ == "__main__":
    main()
