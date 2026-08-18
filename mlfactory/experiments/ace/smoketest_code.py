#!/usr/bin/env python3
"""Generate code-arm software-engineering problems via Lunaroute.

By default this is a 10-example smoketest: one envelope per code domain,
with blind random task/friction draws and seeds 5000-5009. Use --count for a
larger corpus.

Usage:
    python3.14 smoketest_code.py

Exits 0 only when all planned records exist. Prints problems to stdout and
appends frozen records to data/code_smoketest.jsonl. Existing valid records
are used as the resume checkpoint, so failed or interrupted seeds are retried
on the next invocation.
"""
import argparse, json, requests, re, time, sys, os
from datetime import datetime
from pathlib import Path

from mlfactory.core.madlibz import (
    sample_envelope, authoring_messages, freeze_authored, CODE_DOMAIN_PROFILES,
)
from mlfactory.core.api import extract_json

LUNAROUTE_URL = "https://gw.lunaroute.com/v1/chat/completions"
LUNAROUTE_MODELS_URL = "https://gw.lunaroute.com/v1/models"
API_KEY = os.environ.get("LUNAROUTE_API_KEY")
MODEL_PREFERENCE = [
    "glm-5.2-vision-ballast",
    "glm-5.2-vision",
    "glm-5.2-vision-background",
]
DOMAINS = sorted(CODE_DOMAIN_PROFILES.keys())
STREAM_TIMEOUT = 30
MAX_TOKENS = 32000
MAX_RETRIES = 5
RETRY_BACKOFF = 20
N = 10
OUT_PATH = Path(__file__).parent / "data" / "code_smoketest.jsonl"


def now():
    return datetime.now().strftime("%H:%M:%S")


def require_api_key():
    if not API_KEY:
        raise RuntimeError("LUNAROUTE_API_KEY must be set")


def discover_and_pick_model():
    """Query GET /v1/models and pick a -background variant."""
    try:
        r = requests.get(LUNAROUTE_MODELS_URL,
                         headers={"Authorization": f"Bearer {API_KEY}"},
                         timeout=15)
        r.raise_for_status()
        active = {m["id"] for m in r.json().get("data", [])}
        print(f"[{now()}] Active models: {sorted(active)}")
    except Exception as e:
        print(f"[{now()}] WARNING: model discovery failed ({e}), trying preferences directly")
        active = set()

    for m in MODEL_PREFERENCE:
        if not active or m in active:
            return m
    if active:
        fallback = sorted(active)[0]
        print(f"[{now()}] WARNING: no preferred model active, falling back to {fallback}")
        return fallback
    raise RuntimeError("no models available")


def call_streaming(messages, model_id):
    resp = requests.post(
        LUNAROUTE_URL,
        headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
        json={"model": model_id, "messages": messages, "temperature": 0.8,
              "max_tokens": MAX_TOKENS, "stream": True},
        stream=True, timeout=STREAM_TIMEOUT,
    )
    # Let the retry loop inspect HTTPError.response.status_code. This covers
    # rate limiting and every 5xx response instead of only selected strings.
    resp.raise_for_status()

    content_parts, reasoning_parts = [], []
    for line in resp.iter_lines():
        if not line:
            continue
        decoded = line.decode("utf-8", "replace")
        if not decoded.startswith("data: "):
            continue
        payload_str = decoded[6:]
        if payload_str.strip() == "[DONE]":
            break
        try:
            chunk = json.loads(payload_str)
        except json.JSONDecodeError:
            continue
        choices = chunk.get("choices", [])
        if not choices:
            continue
        delta = choices[0].get("delta", {})
        content = delta.get("content") or ""
        reasoning = delta.get("reasoning") or delta.get("reasoning_content") or ""
        if content:
            content_parts.append(content)
        if reasoning:
            reasoning_parts.append(reasoning)

    return "".join(content_parts), "".join(reasoning_parts)


def is_retryable_request_error(exc):
    """Return whether a request failure is transient enough to retry."""
    if not isinstance(exc, requests.exceptions.RequestException):
        return False
    response = getattr(exc, "response", None)
    status = response.status_code if response is not None else None
    return status is None or status in (408, 429) or status >= 500


def author_with_retries(messages, model_id):
    """Call the authoring endpoint and return parsed JSON, with bounded retries."""
    last_error = "unknown generation failure"
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            content, reasoning = call_streaming(messages, model_id)
            if not content and not reasoning:
                raise RuntimeError("empty response")
            authored = extract_authored(content, reasoning)
            if authored:
                return authored
            last_error = "no valid JSON in response"
        except requests.exceptions.RequestException as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if not is_retryable_request_error(exc):
                raise
        except RuntimeError as exc:
            last_error = str(exc)

        if attempt < MAX_RETRIES:
            delay = RETRY_BACKOFF * attempt
            print(f"  ⚠ attempt {attempt}/{MAX_RETRIES} failed: {last_error}; retrying in {delay}s")
            time.sleep(delay)
    raise RuntimeError(f"failed after {MAX_RETRIES} attempts: {last_error}")


def extract_authored(content, reasoning):
    for src in [content, reasoning]:
        if not src.strip():
            continue
        try:
            return extract_json(src)
        except Exception:
            m = re.search(r'\{[\s\S]*\}', src)
            if m:
                try:
                    return json.loads(m.group())
                except json.JSONDecodeError:
                    pass
    return None


def load_existing_records():
    """Load prior records keyed by seed; the JSONL is the resume checkpoint."""
    records = {}
    if not OUT_PATH.exists():
        return records
    with OUT_PATH.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"invalid JSON in {OUT_PATH} at line {line_number}; repair it before resuming"
                ) from exc
            seed = record.get("seed")
            if not isinstance(seed, int):
                raise RuntimeError(f"record at {OUT_PATH}:{line_number} has no integer seed")
            if seed in records:
                raise RuntimeError(f"duplicate seed {seed} in {OUT_PATH}")
            records[seed] = record
    return records


def append_record(record):
    """Durably append one record before treating its seed as complete."""
    with OUT_PATH.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def main():
    global OUT_PATH
    parser = argparse.ArgumentParser(description="Generate ACE code-arm examples via Lunaroute")
    parser.add_argument("--count", type=int, default=N)
    parser.add_argument("--start-seed", type=int, default=5000)
    parser.add_argument("--output", type=Path, default=OUT_PATH)
    parser.add_argument("--corpus", default="code-smoketest")
    args = parser.parse_args()
    if args.count <= 0:
        parser.error("--count must be positive")

    OUT_PATH = args.output
    count = args.count

    print(f"[{now()}] === Code-arm generation: {count} problems ===")
    require_api_key()
    model_id = discover_and_pick_model()
    print(f"[{now()}] Using model: {model_id}")

    plan = [
        (args.start_seed + i, DOMAINS[i % len(DOMAINS)])
        for i in range(count)
    ]
    print(f"[{now()}] Plan: seeds {args.start_seed}-{args.start_seed + count - 1}; domains cycle across {len(DOMAINS)} code domains")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    records_by_seed = load_existing_records()
    failures = 0
    if records_by_seed:
        planned_existing = sum(seed in records_by_seed for seed, _ in plan)
        print(f"[{now()}] Resuming with {planned_existing}/{count} planned records already present")

    for i, (seed, domain) in enumerate(plan):
        print(f"\n{'─'*70}")
        print(f"[{i+1}/{N}] seed={seed} domain={domain}")
        if seed in records_by_seed:
            print(f"  ↩ already frozen; skipping seed {seed}")
            continue

        env = sample_envelope(seed=seed, domain=domain, mode="code")
        print(f"  persona: {env.persona}")
        print(f"  stakes:  {env.stakes}")
        print(f"  task:    {env.task_kind}")
        print(f"  friction:{env.friction}")

        messages = authoring_messages(env)
        t0 = time.time()
        try:
            authored = author_with_retries(messages, model_id)
        except Exception as e:
            print(f"  ❌ generation failed: {e}")
            failures += 1
            continue
        elapsed = time.time() - t0
        print(f"  authored JSON received in {elapsed:.1f}s")

        try:
            record = freeze_authored(env, authored, model=model_id, corpus=args.corpus)
        except ValueError as e:
            print(f"  ❌ FAIL: freeze rejected: {e}")
            failures += 1
            continue

        # Persist before adding the seed to the in-memory checkpoint. A crash
        # after this point can only cause a harmless duplicate if the file is
        # externally copied; a normal restart will see this seed as complete.
        append_record(record)
        records_by_seed[seed] = record
        prob = record["problem"]
        print(f"\n  PROSE:\n  {record['prose']}")
        print(f"\n  Surface Q: {record['surface_question']}")
        print(f"  Task:      {prob.get('task_kind')} (asked: {env.task_kind})")
        print(f"  Friction:  {prob.get('friction')} (asked: {env.friction})")
        print(f"  What:      {prob.get('what_must_be_produced')}")
        print(f"  Where:     {prob.get('where_the_difficulty_lives')}")
        print(f"  Why:       {prob.get('why_it_requires_work')}")

    completed = sum(seed in records_by_seed for seed, _ in plan)
    print(f"\n{'='*70}")
    if failures == 0 and completed == count:
        print(f"[{now()}] ✅ Code generation passed — {count}/{count} generated")
    else:
        print(f"[{now()}] ⚠️  Completed {completed}/{count}, {failures} failures")
    print(f"{'='*70}")
    sys.exit(0 if completed == count else 1)


if __name__ == "__main__":
    main()
