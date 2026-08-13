#!/usr/bin/env python3
"""Conservative GLM review harness for the private voice corpus.

Raw corpus files are never modified. Outputs live under voice/data/review by
default (gitignored) and contain local source mappings, model decisions, prompt
hashes, and response hashes for auditability.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import copy
import hashlib
import json
import os
import random
import re
import string
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Allow direct execution from any working directory without installation.
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mlfactory.core.secrets import SecretsStore

HERE = Path(__file__).resolve().parent
DEFAULT_PROMPT = HERE / "prompts" / "review_session.md"
DEFAULT_CASES = HERE / "smoke_cases.json"
DEFAULT_THREADS = HERE / "data" / "threads"
DEFAULT_OUTPUT_ROOT = HERE / "data" / "review"
BASE_URL = "https://gw.lunaroute.com/v1"
MODEL = "glm-5.2-vision"
# The review is constrained extraction/classification; hidden reasoning adds
# latency without useful signal. Lunaroute's active model maps "off" to "none".
REASONING_EFFORT = "none"
EXCLUDED_CUSTOMERS = {
    "taylor satula",
    "josiah quality care exteriors",
    "annika rettstadt",
    "charlie rettstadt",
}
ACTIONS = {"KEEP", "PSEUDONYMIZE", "EXCLUDE_SESSION", "HUMAN_REVIEW"}
SESSION_ACTIONS = {"KEEP", "EXCLUDE", "HUMAN_REVIEW"}
SESSION_REASONS = {
    "NORMAL", "PERSONAL_OR_FAMILY", "VENDOR_OR_INTERNAL", "ROLE_REVERSED",
    "WRONG_IDENTITY", "NON_TEXT_PAYLOAD", "PERVASIVE_SENSITIVE", "AMBIGUOUS",
}
SENSITIVE_CATEGORIES = {
    "PHONE", "EMAIL", "ADDRESS", "ACCOUNT_ID", "ACCESS_CODE",
    "PRIVATE_URL", "PAYMENT_HANDLE",
}


def response_schema() -> dict[str, Any]:
    """Documentation schema; JSON-object mode is validated locally."""
    substitution = {
        "type": "object",
        "additionalProperties": False,
        "required": ["source", "occurrence", "category"],
        "properties": {
            "source": {"type": "string", "minLength": 1},
            "occurrence": {"type": "integer", "minimum": 1},
            "category": {"type": "string", "enum": sorted(SENSITIVE_CATEGORIES)},
        },
    }
    message = {
        "type": "object",
        "additionalProperties": False,
        "required": ["message_id", "action", "substitutions", "confidence"],
        "properties": {
            "message_id": {"type": "string"},
            "action": {"type": "string", "enum": sorted(ACTIONS)},
            "substitutions": {"type": "array", "items": substitution},
            "confidence": {"type": "string", "enum": ["HIGH", "MEDIUM", "LOW"]},
        },
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["session_id", "session_action", "session_reason", "messages"],
        "properties": {
            "session_id": {"type": "string"},
            "session_action": {"type": "string", "enum": sorted(SESSION_ACTIONS)},
            "session_reason": {"type": "string", "enum": sorted(SESSION_REASONS)},
            "messages": {"type": "array", "minItems": 1, "items": message},
        },
    }


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def count_occurrences(text: str, source: str) -> int:
    if not source:
        return 0
    count = start = 0
    while True:
        pos = text.find(source, start)
        if pos < 0:
            return count
        count += 1
        start = pos + len(source)


def _stable_bytes(category: str, source: str) -> bytes:
    return hashlib.sha256(f"voice-pseudonym-v1\0{category}\0{source}".encode("utf-8")).digest()


def _shape_map(source: str, alphabet: str, digits_only: bool = False) -> str:
    digest = _stable_bytes("SHAPE", source)
    out = []
    j = 0
    for char in source:
        if char.isdigit() or (not digits_only and char.isalpha()):
            pool = string.digits if char.isdigit() else (string.ascii_uppercase if char.isupper() else string.ascii_lowercase)
            out.append(pool[digest[j % len(digest)] % len(pool)])
            j += 1
        else:
            out.append(char)
    result = "".join(out)
    if result == source and result:
        tail = "1" if result[-1] != "1" else "2"
        result = result[:-1] + tail
    return result


def synthetic_replacement(category: str, source: str) -> str:
    """Generate stable realistic dummy content; never emit training placeholders."""
    digest = _stable_bytes(category, source)
    number = int.from_bytes(digest[:4], "big")
    if category == "PHONE":
        digits = "".join(c for c in source if c.isdigit())
        area = digits[-10:-7] if len(digits) >= 10 else "256"
        fake10 = f"{area}555{100 + number % 100:04d}"
        fake_digits = ("1" + fake10) if len(digits) == 11 and digits.startswith("1") else fake10[-len(digits):]
        iterator = iter(fake_digits)
        return "".join(next(iterator) if c.isdigit() else c for c in source)
    if category == "EMAIL":
        return f"customer{number % 10000:04d}@example.com"
    if category == "ADDRESS":
        first = ["Juniper", "Maple", "Cedar", "Willow", "Oak", "Pine", "Magnolia", "Dogwood"]
        second = ["Hollow", "Crest", "Brook", "Bend", "Meadow", "Ridge", "Grove", "Point"]
        suffix = ["Lane", "Drive", "Road", "Court", "Avenue", "Circle"]
        street = f"{100 + number % 9800} {first[digest[4] % len(first)]} {second[digest[5] % len(second)]} {suffix[digest[6] % len(suffix)]}"
        locality = source[source.find(","):] if "," in source else ""
        return street + locality
    if category == "PRIVATE_URL":
        return f"https://payments.example.com/session/demo-{digest.hex()[:10]}"
    if category == "PAYMENT_HANDLE":
        prefix = "@" if source.startswith("@") else ""
        return f"{prefix}customer{number % 100000:05d}"
    if category == "ACCESS_CODE":
        return _shape_map(source, string.digits, digits_only=True)
    if category == "ACCOUNT_ID":
        return _shape_map(source, string.ascii_letters + string.digits)
    raise ValueError(f"unsupported substitution category: {category}")


def enforce_local_policy(review: dict[str, Any], session: dict[str, Any]) -> list[dict[str, str]]:
    """Add deterministic pseudonyms and enforce session-level disposition."""
    adjustments: list[dict[str, str]] = []
    excluded = review.get("session_action") == "EXCLUDE"
    for row in review.get("messages", []):
        if not isinstance(row, dict):
            continue
        mid = str(row.get("message_id"))
        if "substitutions" not in row:
            row["substitutions"] = []
            adjustments.append({"message_id": mid, "rule": "ADD_EMPTY_SUBSTITUTIONS"})
        if "confidence" not in row and excluded:
            row["confidence"] = "HIGH"
            adjustments.append({"message_id": mid, "rule": "ADD_EXCLUSION_CONFIDENCE"})
        if excluded and row.get("action") != "EXCLUDE_SESSION":
            row["action"] = "EXCLUDE_SESSION"
            adjustments.append({"message_id": mid, "rule": "PROPAGATE_SESSION_EXCLUSION"})
        elif row.get("substitutions") and row.get("action") == "KEEP":
            row["action"] = "PSEUDONYMIZE"
            adjustments.append({"message_id": mid, "rule": "SUBSTITUTION_REQUIRES_ACTION"})
    return adjustments


def locate_occurrence(text: str, source: str, occurrence: int) -> tuple[int, int]:
    cursor = 0
    start = -1
    for _ in range(occurrence):
        start = text.find(source, cursor)
        if start < 0:
            raise ValueError(f"source occurrence not found: {source!r} #{occurrence}")
        cursor = start + len(source)
    return start, start + len(source)


def placeholder_token(category: str, source: str) -> str:
    digest = hashlib.sha256(f"voice-placeholder-v1\0{category}\0{source}".encode("utf-8")).hexdigest()[:12]
    return f"[[{category}:{digest}]]"


def apply_substitutions(text: str, substitutions: list[dict[str, Any]]) -> str:
    spans = []
    for substitution in substitutions:
        start, end = locate_occurrence(text, substitution["source"], substitution["occurrence"])
        spans.append((start, end, substitution["replacement"]))
    spans.sort()
    if any(a_end > b_start for (_, a_end, _), (b_start, _, _) in zip(spans, spans[1:])):
        raise ValueError("overlapping substitutions")
    result = text
    for start, end, replacement in reversed(spans):
        result = result[:start] + replacement + result[end:]
    return result


def build_placeholder_session(session: dict[str, Any], review: dict[str, Any]) -> dict[str, Any]:
    decisions = {row["message_id"]: row for row in review["messages"]}
    session_usable = review["session_action"] == "KEEP"
    messages = []
    for message in session["messages"]:
        decision = decisions[message["message_id"]]
        prepared = [
            {
                **substitution,
                "replacement": placeholder_token(substitution["category"], substitution["source"]),
            }
            for substitution in decision["substitutions"]
        ]
        placeholder_text = apply_substitutions(message["text"], prepared)
        messages.append({
            "message_id": message["message_id"],
            "role": message["role"],
            "text": placeholder_text,
            "original_text_hash": sha256_text(message["text"]),
            "changed": placeholder_text != message["text"],
            "review_action": decision["action"],
            "target_eligible": bool(
                session_usable
                and message["role"] == "taylor"
                and decision["action"] in {"KEEP", "PSEUDONYMIZE"}
            ),
        })
    return {
        "artifact_policy": "INTERMEDIATE_PLACEHOLDERS_NOT_FOR_TRAINING",
        "session_id": session["session_id"],
        "usable": session_usable,
        "messages": messages,
    }


def validate_review(review: Any, session: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(review, dict):
        raise ValueError("review is not an object")
    if review.get("session_id") != session["session_id"]:
        raise ValueError("session_id mismatch")
    if review.get("session_action") not in SESSION_ACTIONS:
        raise ValueError("invalid session_action")
    if review.get("session_reason") not in SESSION_REASONS:
        raise ValueError("invalid session_reason")
    by_id = {m["message_id"]: m for m in session["messages"]}
    rows = review.get("messages")
    if not isinstance(rows, list) or len(rows) != len(by_id):
        raise ValueError("message result count mismatch")
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("message review is not an object")
        mid = row.get("message_id")
        if mid not in by_id or mid in seen:
            raise ValueError(f"unknown or duplicate message_id: {mid}")
        seen.add(mid)
        action = row.get("action")
        if action not in ACTIONS or row.get("confidence") not in {"HIGH", "MEDIUM", "LOW"}:
            raise ValueError(f"invalid action/confidence for {mid}")
        substitutions = row.get("substitutions")
        if not isinstance(substitutions, list):
            raise ValueError(f"substitutions is not an array: {mid}")
        if substitutions and action not in {"PSEUDONYMIZE", "EXCLUDE_SESSION", "HUMAN_REVIEW"}:
            raise ValueError(f"substitutions require pseudonymize/review/exclusion action: {mid}")
        if action == "PSEUDONYMIZE" and not substitutions:
            raise ValueError(f"PSEUDONYMIZE requires at least one span: {mid}")
        text = by_id[mid]["text"]
        spans = []
        for substitution in substitutions:
            source = substitution.get("source")
            occurrence = substitution.get("occurrence")
            category = substitution.get("category")
            if category not in SENSITIVE_CATEGORIES or "replacement" in substitution:
                raise ValueError(f"invalid substitution category or model-supplied replacement: {mid}")
            if not isinstance(source, str) or not isinstance(occurrence, int):
                raise ValueError(f"invalid substitution source/occurrence: {mid}")
            if count_occurrences(text, source) < occurrence:
                raise ValueError(f"substitution source not found at occurrence {occurrence}: {mid}")
            start, end = locate_occurrence(text, source, occurrence)
            spans.append((start, end))
        spans.sort()
        if any(a1 > b0 for (_, a1), (b0, _) in zip(spans, spans[1:])):
            raise ValueError(f"overlapping substitutions: {mid}")
    if seen != set(by_id):
        raise ValueError("missing message IDs")
    if review["session_action"] == "EXCLUDE" and any(r["action"] != "EXCLUDE_SESSION" for r in rows):
        raise ValueError("EXCLUDE session must propagate to all messages")
    return review


def call_lunaroute(api_key: str, prompt: str, session: dict[str, Any], model: str = MODEL, retries: int = 0) -> tuple[dict[str, Any], dict[str, Any]]:
    """Call Lunaroute using the same streaming shape as the known-good Pi client."""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": json.dumps(session, ensure_ascii=False, separators=(",", ":"))},
        ],
        # GLM's strict grammar decoder is disproportionately slow for this
        # nested per-message schema. JSON-object mode plus local validation is
        # faster and still fail-closed.
        "response_format": {"type": "json_object"},
        "reasoning_effort": REASONING_EFFORT,
        "max_completion_tokens": 8192,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    req_data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    last: Exception | None = None
    for attempt in range(retries + 1):
        req = urllib.request.Request(
            f"{BASE_URL}/chat/completions",
            data=req_data,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        )
        try:
            started = time.monotonic()
            last_progress = started
            first_chunk_seconds: float | None = None
            content_parts: list[str] = []
            reasoning_chars = 0
            usage: dict[str, Any] = {}
            finish_reason = None
            with urllib.request.urlopen(req, timeout=60) as resp:
                for raw_line in resp:
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    event = json.loads(data)
                    if first_chunk_seconds is None:
                        first_chunk_seconds = time.monotonic() - started
                    if isinstance(event.get("usage"), dict):
                        usage = event["usage"]
                    for choice in event.get("choices", []):
                        if choice.get("finish_reason") is not None:
                            finish_reason = choice["finish_reason"]
                        delta = choice.get("delta") or {}
                        reasoning = delta.get("reasoning_content") or delta.get("reasoning")
                        if isinstance(reasoning, str):
                            reasoning_chars += len(reasoning)
                        content = delta.get("content")
                        if isinstance(content, str):
                            content_parts.append(content)
                    now = time.monotonic()
                    if now - last_progress >= 10:
                        print(
                            f"[stream {session['session_id']}] {now - started:.0f}s "
                            f"reasoning_chars={reasoning_chars} content_chars={sum(map(len, content_parts))}",
                            flush=True,
                        )
                        last_progress = now
            duration = time.monotonic() - started
            content = "".join(content_parts)
            if not content.strip():
                raise ValueError(
                    f"empty streamed content; finish_reason={finish_reason!r} "
                    f"reasoning_chars={reasoning_chars} usage={usage}"
                )
            try:
                parsed = json.loads(content)
            except json.JSONDecodeError as exc:
                lo = max(0, exc.pos - 160)
                hi = min(len(content), exc.pos + 160)
                raise ValueError(
                    f"invalid streamed JSON at char {exc.pos}; finish_reason={finish_reason!r}; "
                    f"content_chars={len(content)}; near={content[lo:hi]!r}"
                ) from exc
            metadata = {
                "duration_seconds": round(duration, 3),
                "first_chunk_seconds": round(first_chunk_seconds or duration, 3),
                "reasoning_chars": reasoning_chars,
                "finish_reason": finish_reason,
                "usage": usage,
                "response_hash": sha256_text(content),
                "request_hash": hashlib.sha256(req_data).hexdigest(),
            }
            return parsed, metadata
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")[:4000]
            last = RuntimeError(f"HTTP {exc.code}: {body}")
        except (urllib.error.URLError, TimeoutError, KeyError, ValueError) as exc:
            last = exc
        if attempt < retries:
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Lunaroute request failed: {last}")


def synthetic_sessions(path: Path) -> list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]]:
    cases = json.loads(path.read_text(encoding="utf-8"))
    result = []
    for case in cases:
        messages = [
            {"message_id": f"m{i:03d}", "role": m["role"], "text": m["text"]}
            for i, m in enumerate(case["messages"])
        ]
        session = {"session_id": case["case_id"], "messages": messages}
        result.append((session, case["expect"], {"case_id": case["case_id"]}))
    return result


def parse_ts(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def local_bucket(messages: list[dict[str, Any]]) -> str:
    text = "\n".join(m["text"] for m in messages)
    if re.search(r"(?:\b\d{3}[-.) ]*\d{3}[- ]*\d{4}\b|\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b|https?://)", text, re.I):
        return "structured_pii"
    if re.search(r"\b(?:will|i['’]ll|can be there|come by|check on|let you know|price|quote|cost|\$\d|paid|payment|tomorrow|appointment)\b", text, re.I):
        return "commitment"
    if sum(len(m["text"]) for m in messages) > 1000:
        return "long"
    return "clean"


def real_sessions(threads_dir: Path) -> list[tuple[dict[str, Any], None, dict[str, Any]]]:
    result = []
    public_index = 0
    for path in sorted(threads_dir.glob("*.json")):
        thread = json.loads(path.read_text(encoding="utf-8"))
        name = str(thread.get("customer", {}).get("name") or "").strip().lower()
        if name in EXCLUDED_CUSTOMERS:
            continue
        sessions: list[list[tuple[int, dict[str, Any]]]] = []
        current: list[tuple[int, dict[str, Any]]] = []
        previous: datetime | None = None
        for source_index, message in enumerate(thread["messages"]):
            ts = parse_ts(message["ts"])
            if previous is not None and (ts - previous).total_seconds() > 96 * 3600:
                if current:
                    sessions.append(current)
                current = []
            current.append((source_index, message))
            previous = ts
        if current:
            sessions.append(current)
        for source_session_index, source_messages in enumerate(sessions):
            if not any(m["from"] == "taylor" for _, m in source_messages):
                continue
            public_index += 1
            messages = [
                {"message_id": f"m{i:03d}", "role": m["from"], "text": m["text"]}
                for i, (_, m) in enumerate(source_messages)
            ]
            session = {"session_id": f"session-{public_index:05d}", "messages": messages}
            source = {
                "source_file": path.name,
                "source_session_index": source_session_index,
                "source_message_indices": [i for i, _ in source_messages],
                "bucket": local_bucket(messages),
            }
            result.append((session, None, source))
    return result


def select_stratified(rows: list[tuple[dict[str, Any], Any, dict[str, Any]]], limit: int, seed: int) -> list[tuple[dict[str, Any], Any, dict[str, Any]]]:
    if limit < 0 or limit >= len(rows):
        return rows
    rng = random.Random(seed)
    groups: dict[str, list] = {}
    for row in rows:
        groups.setdefault(row[2].get("bucket", "clean"), []).append(row)
    for values in groups.values():
        rng.shuffle(values)
    selected = []
    order = ["structured_pii", "commitment", "long", "clean"]
    while len(selected) < limit and any(groups.get(k) for k in order):
        for key in order:
            if groups.get(key) and len(selected) < limit:
                selected.append(groups[key].pop())
    return selected


def evaluate_synthetic(review: dict[str, Any], expected: dict[str, Any], session: dict[str, Any]) -> list[str]:
    failures = []
    if "session_action" in expected and review["session_action"] != expected["session_action"]:
        failures.append(f"session_action expected {expected['session_action']} got {review['session_action']}")
    if "allowed_session_reasons" in expected and review["session_reason"] not in expected["allowed_session_reasons"]:
        failures.append(f"session_reason {review['session_reason']} not allowed")
    rows = {row["message_id"]: row for row in review["messages"]}
    for i, action in enumerate(expected.get("actions", [])):
        got = rows[f"m{i:03d}"]["action"]
        if got != action:
            failures.append(f"m{i:03d} action expected {action} got {got}")
    found = {s["source"] for row in review["messages"] for s in row["substitutions"]}
    for source in expected.get("required_substitution_sources", []):
        if source not in found:
            failures.append(f"missing exact substitution source: {source!r}")
    for source in expected.get("forbidden_substitution_sources", []):
        if source in found:
            failures.append(f"excessive substitution source: {source!r}")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["synthetic", "corpus"], required=True)
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--retries", type=int, default=0, help="bounded retries per API request")
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--threads", type=Path, default=DEFAULT_THREADS)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", action="store_true", help="resume a fixed --output directory")
    args = parser.parse_args()

    prompt = args.prompt.read_text(encoding="utf-8")
    prompt_hash = sha256_text(prompt)
    store = SecretsStore(Path.cwd() / ".mlfactory" / "secrets.yaml")
    api_key = store.get("LUNAROUTE_API_KEY")
    if not api_key:
        raise SystemExit("LUNAROUTE_API_KEY not found in .mlfactory/secrets.yaml")

    rows = synthetic_sessions(args.cases) if args.mode == "synthetic" else real_sessions(args.threads)
    if args.mode == "corpus":
        rows = select_stratified(rows, args.limit, args.seed)
    elif args.limit >= 0:
        rows = rows[:args.limit]

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output = args.output or DEFAULT_OUTPUT_ROOT / f"{args.mode}-{len(rows)}-{stamp}"
    config = {
        "mode": args.mode,
        "session_ids_hash": sha256_text("\n".join(row[0]["session_id"] for row in rows)),
        "requested": len(rows),
        "seed": args.seed,
        "model": args.model,
        "reasoning_effort": REASONING_EFFORT,
        "prompt_hash": prompt_hash,
    }
    config_path = output / "run_config.json"
    if output.exists():
        if not args.resume:
            raise SystemExit(f"output already exists; use --resume or a new path: {output}")
        if not config_path.is_file() or json.loads(config_path.read_text(encoding="utf-8")) != config:
            raise SystemExit("resume configuration does not match existing run_config.json")
    else:
        output.mkdir(parents=True)
        os.chmod(output, 0o700)
        config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        os.chmod(config_path, 0o600)
    results_path = output / "results.jsonl"
    errors_path = output / "errors.jsonl"
    results_path.touch(exist_ok=True)
    errors_path.touch(exist_ok=True)
    os.chmod(results_path, 0o600)
    os.chmod(errors_path, 0o600)

    def process(row):
        session, expected, source = row
        review, api_meta = call_lunaroute(api_key, prompt, session, model=args.model, retries=args.retries)
        model_review = copy.deepcopy(review)
        policy_adjustments = enforce_local_policy(review, session)
        try:
            review = validate_review(review, session)
        except Exception as exc:
            raise ValueError(
                f"review validation failed: {exc}; review={json.dumps(review, ensure_ascii=False)}"
            ) from exc
        placeholder_session = build_placeholder_session(session, review)
        failures = evaluate_synthetic(review, expected, session) if expected is not None else []
        return {
            "session_id": session["session_id"],
            "source": source,
            "input_hash": sha256_text(json.dumps(session, ensure_ascii=False, sort_keys=True)),
            "review": review,
            "model_review": model_review if policy_adjustments else None,
            "policy_adjustments": policy_adjustments,
            "placeholder_session": placeholder_session,
            "api": api_meta,
            "synthetic_failures": failures,
        }

    completed = [json.loads(line) for line in results_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    completed_ids = {row["session_id"] for row in completed}
    pending_rows = [row for row in rows if row[0]["session_id"] not in completed_ids]
    errors = []
    if completed:
        print(f"[resume] retained {len(completed)}/{len(rows)} completed sessions", flush=True)
    with results_path.open("a", encoding="utf-8") as results_handle, errors_path.open("a", encoding="utf-8") as errors_handle:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            futures = {pool.submit(process, row): row for row in pending_rows}
            for future in concurrent.futures.as_completed(futures):
                row = futures[future]
                try:
                    result = future.result()
                    completed.append(result)
                    completed_ids.add(result["session_id"])
                    results_handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                    results_handle.flush()
                    os.fsync(results_handle.fileno())
                    print(f"[done {len(completed)}/{len(rows)}] {result['session_id']}", flush=True)
                except Exception as exc:
                    error = {"session_id": row[0]["session_id"], "source": row[2], "error": str(exc), "failed_utc": datetime.now(UTC).isoformat()}
                    errors.append(error)
                    errors_handle.write(json.dumps(error, ensure_ascii=False) + "\n")
                    errors_handle.flush()
                    os.fsync(errors_handle.fileno())
                    processed = len(completed) + len(errors)
                    print(f"[error {processed}/{len(rows)}] {row[0]['session_id']}: {exc}", flush=True)

    completed.sort(key=lambda x: x["session_id"])
    errors.sort(key=lambda x: x["session_id"])

    detections_path = output / "detections.jsonl"
    placeholders_path = output / "placeholder_sessions.jsonl"
    with detections_path.open("w", encoding="utf-8") as handle:
        for row in completed:
            handle.write(json.dumps({
                "session_id": row["session_id"],
                "source": row["source"],
                "input_hash": row["input_hash"],
                "review": row["review"],
                "model_review": row["model_review"],
                "policy_adjustments": row["policy_adjustments"],
                "api": row["api"],
            }, ensure_ascii=False) + "\n")
    with placeholders_path.open("w", encoding="utf-8") as handle:
        for row in completed:
            handle.write(json.dumps({
                "source": row["source"],
                **row["placeholder_session"],
            }, ensure_ascii=False) + "\n")
    marker_path = output / "PLACEHOLDERS_NOT_FOR_TRAINING.txt"
    marker_path.write_text(
        "placeholder_sessions.jsonl is a retained intermediate and MUST NOT be used for training.\n"
        "Run transform_placeholders.py to create a separate pseudonymized corpus.\n",
        encoding="utf-8",
    )
    for path in (results_path, errors_path, detections_path, placeholders_path, marker_path):
        os.chmod(path, 0o600)

    actions = Counter()
    reasons = Counter()
    synthetic_failures = []
    total_usage = Counter()
    for row in completed:
        reasons[row["review"]["session_reason"]] += 1
        for message in row["review"]["messages"]:
            actions[message["action"]] += 1
        synthetic_failures.extend((row["session_id"], f) for f in row["synthetic_failures"])
        for key, value in row["api"]["usage"].items():
            if isinstance(value, int):
                total_usage[key] += value
    summary = {
        "mode": args.mode,
        "requested": len(rows),
        "completed": len(completed),
        "errors": len(rows) - len(completed),
        "error_events": sum(1 for line in errors_path.read_text(encoding="utf-8").splitlines() if line.strip()),
        "synthetic_failure_count": len(synthetic_failures),
        "synthetic_failures": synthetic_failures,
        "actions": dict(actions),
        "session_reasons": dict(reasons),
        "usage": dict(total_usage),
        "model": args.model,
        "reasoning_effort": REASONING_EFFORT,
        "base_url": BASE_URL,
        "prompt_path": str(args.prompt),
        "prompt_hash": prompt_hash,
        "seed": args.seed,
        "created_utc": stamp,
    }
    summary_path = output / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.chmod(summary_path, 0o600)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"output: {output}")
    return 1 if len(completed) != len(rows) or synthetic_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
