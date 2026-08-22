#!/usr/bin/env python3
"""Generate 300 madlibz problems overnight via Lunaroute.

Usage:
    cd mlfactory/experiments/ace
    nohup python3 generate_corpus_300.py > data/corpus_run.log 2>&1 &

- Blind draws across both domains (natural genus/detectability distribution).
- Streams responses to detect dead connections fast.
- Saves incrementally — each record written immediately, nothing lost on crash.
- Retries transient failures (5xx, timeouts, connection errors) with backoff.
- Queries /v1/models on startup per the Lunaroute convention.
"""
import json, requests, re, time, sys, os, signal
from datetime import datetime
from pathlib import Path

from mlfactory.core.madlibz import (
    sample_envelope, authoring_messages, freeze_authored,
    DOMAIN_PROFILES, ANOMALY_GENUSES, DETECTABILITY_GRANULARS,
)
from mlfactory.core.api import extract_json

# ── Config ──────────────────────────────────────────────────────────────
N = 1000
BASE_SEED = 1000           # starting seed; each problem gets base_seed + i
DOMAINS = list(DOMAIN_PROFILES.keys())
ACE_DIR = Path(__file__).parent
OUTPUT = ACE_DIR / "data" / "corpus_300_a.jsonl"
PROGRESS = ACE_DIR / "data" / "corpus_300_a_progress.json"

LUNAROUTE_URL = "https://gw.lunaroute.com/v1/chat/completions"
LUNAROUTE_MODELS_URL = "https://gw.lunaroute.com/v1/models"
API_KEY = os.environ.get("LUNAROUTE_API_KEY", "lr_f360bcbeeb8a6aa0d0e07a28f82809bb42d863549ded922d244b5b0d128c5124")
MAX_TOKENS = 32000
STREAM_TIMEOUT = 30        # per-chunk read timeout (catches dead connections fast)
MAX_RETRIES = 5
RETRY_BACKOFF = 20         # seconds between retries

# Prefer -ballast, fall back to plain, then -background
MODEL_PREFERENCE = ["glm-5.2-vision-ballast", "glm-5.2-vision", "glm-5.2-vision-background"]

# ── Globals for graceful shutdown ───────────────────────────────────────
shutdown_requested = False

def handle_signal(signum, frame):
    global shutdown_requested
    print(f"\n[{now()}] Signal {signum} received, finishing current problem then stopping...")
    shutdown_requested = True

signal.signal(signal.SIGTERM, handle_signal)
signal.signal(signal.SIGINT, handle_signal)


def now():
    return datetime.now().strftime("%H:%M:%S")


def discover_models():
    """Query /v1/models and return set of active model IDs."""
    try:
        r = requests.get(LUNAROUTE_MODELS_URL,
                         headers={"Authorization": f"Bearer {API_KEY}"},
                         timeout=15)
        r.raise_for_status()
        data = r.json()
        active = {m["id"] for m in data.get("data", [])}
        print(f"[{now()}] Active models: {sorted(active)}")
        return active
    except Exception as e:
        print(f"[{now()}] WARNING: could not query models ({e}), using preferences as-is")
        return set(MODEL_PREFERENCE)


def pick_model(active_models):
    """Pick the best available model from preference list."""
    for m in MODEL_PREFERENCE:
        if m in active_models:
            return m
    # If none of our preferred models are active, use whatever is available
    if active_models:
        fallback = sorted(active_models)[0]
        print(f"[{now()}] WARNING: no preferred model active, falling back to {fallback}")
        return fallback
    raise RuntimeError("no models available on Lunaroute")


def call_streaming(messages, model_id):
    """Stream a chat completion. Returns (content, reasoning, usage) or raises."""
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model_id,
        "messages": messages,
        "temperature": 0.8,
        "max_tokens": MAX_TOKENS,
        "stream": True,
    }

    resp = requests.post(
        LUNAROUTE_URL, headers=headers, json=payload,
        stream=True, timeout=STREAM_TIMEOUT,
    )
    if resp.status_code >= 500:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
    resp.raise_for_status()

    content_parts = []
    reasoning_parts = []
    first_token_time = None
    t0 = time.time()

    for line in resp.iter_lines():
        if shutdown_requested:
            raise KeyboardInterrupt("shutdown requested")
        if not line:
            continue
        decoded = line.decode("utf-8")
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
        # Handle both "reasoning" (zai format) and "reasoning_content" (openai format)
        c = delta.get("content", "")
        r = delta.get("reasoning", "") or delta.get("reasoning_content", "")
        if c:
            if first_token_time is None:
                first_token_time = time.time()
            content_parts.append(c)
        elif r:
            if first_token_time is None:
                first_token_time = time.time()
            reasoning_parts.append(r)

    elapsed = time.time() - t0
    content = "".join(content_parts)
    reasoning = "".join(reasoning_parts)
    ttft = (first_token_time - t0) if first_token_time else None

    return content, reasoning, elapsed, ttft


def extract_authored(content, reasoning):
    """Try to extract the authored JSON from content or reasoning."""
    for src_name, src in [("content", content), ("reasoning", reasoning)]:
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


def load_progress():
    """Load progress file if it exists (for resume)."""
    if os.path.exists(PROGRESS):
        with open(PROGRESS) as f:
            return json.load(f)
    return {"completed": 0, "failed": [], "skipped_seeds": []}


def save_progress(progress):
    with open(PROGRESS, "w") as f:
        json.dump(progress, f, indent=2)


def append_record(record):
    with open(OUTPUT, "a") as f:
        f.write(json.dumps(record) + "\n")


def main():
    print(f"[{now()}] === Corpus generation A: {N} problems ===")
    print(f"[{now()}] Output: {OUTPUT}")
    print(f"[{now()}] Max tokens per call: {MAX_TOKENS}")

    # Discover active models
    active = discover_models()
    model_id = pick_model(active)
    print(f"[{now()}] Using model: {model_id}")

    # Resume support
    progress = load_progress()
    start_idx = progress["completed"]
    if start_idx > 0:
        print(f"[{now()}] Resuming from index {start_idx} ({start_idx}/{N} already done)")

    success_count = start_idx
    fail_count = len(progress["failed"])

    for i in range(start_idx, N):
        if shutdown_requested:
            print(f"[{now()}] Stopping at index {i} (shutdown requested)")
            break

        seed = BASE_SEED + i
        domain = DOMAINS[i % len(DOMAINS)]  # alternate domains for balance

        print(f"\n[{now()}] [{i+1}/{N}] seed={seed} domain={domain}")

        env = sample_envelope(seed=seed, domain=domain)  # blind genus/detectability
        print(f"  persona: {env.persona}")
        print(f"  genus: {env.genus} | detect: {env.detectability}")

        messages = authoring_messages(env)

        # Retry loop
        authored = None
        last_error = None
        for attempt in range(MAX_RETRIES):
            if shutdown_requested:
                break
            try:
                content, reasoning, elapsed, ttft = call_streaming(messages, model_id)
                ttft_str = f"{ttft:.1f}s" if ttft else "N/A"
                print(f"  attempt {attempt+1}: content={len(content)} reasoning={len(reasoning)} "
                      f"elapsed={elapsed:.1f}s ttft={ttft_str}")

                if not content and not reasoning:
                    last_error = "empty response"
                    print(f"  ⚠ empty response, retrying...")
                    time.sleep(RETRY_BACKOFF)
                    continue

                authored = extract_authored(content, reasoning)
                if authored:
                    break
                else:
                    last_error = "no JSON found"
                    print(f"  ⚠ no JSON in response, retrying...")
                    # Show a snippet for debugging
                    preview = content[:200] if content else reasoning[:200]
                    print(f"    preview: {preview!r}")
                    time.sleep(RETRY_BACKOFF)

            except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError,
                    requests.exceptions.ChunkedEncodingError) as e:
                last_error = f"{type(e).__name__}"
                print(f"  attempt {attempt+1}: {last_error}, retrying in {RETRY_BACKOFF}s...")
                time.sleep(RETRY_BACKOFF)

            except RuntimeError as e:
                last_error = str(e)
                if "500" in str(e) or "502" in str(e) or "503" in str(e):
                    print(f"  attempt {attempt+1}: {last_error}, retrying in {RETRY_BACKOFF}s...")
                    time.sleep(RETRY_BACKOFF)
                else:
                    raise

        if authored is None:
            print(f"  ❌ FAILED after {MAX_RETRIES} attempts: {last_error}")
            progress["failed"].append({"index": i, "seed": seed, "domain": domain, "error": last_error})
            fail_count += 1
            save_progress(progress)
            continue

        try:
            record = freeze_authored(env, authored, model=model_id, corpus="corpus-300-a")
            append_record(record)
            success_count += 1
            progress["completed"] = i + 1
            save_progress(progress)
            print(f"  ✅ Problem {i+1} frozen (hash: {record['surface_hash'][:12]}...)")
        except Exception as e:
            print(f"  ❌ freeze failed: {e}")
            progress["failed"].append({"index": i, "seed": seed, "domain": domain, "error": str(e)})
            fail_count += 1
            save_progress(progress)

    # Final summary
    print(f"\n{'='*60}")
    print(f"[{now()}] === RUN COMPLETE ===")
    print(f"  Successes: {success_count}/{N}")
    print(f"  Failures:  {fail_count}")
    if progress["failed"]:
        print(f"  Failed indices: {[f['index'] for f in progress['failed']]}")
    print(f"  Output: {OUTPUT}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
