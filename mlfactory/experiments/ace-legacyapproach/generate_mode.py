#!/usr/bin/env python3
"""Parameterized corpus generator for clean or thrash mode.

Usage:
    python3 generate_mode.py --mode clean --start-seed 3000 --count 375 \
        --output data/clean_375_a.jsonl --progress data/clean_375_a_progress.json \
        --corpus clean-375-a
"""
import argparse, json, requests, re, time, sys, os, signal
from datetime import datetime
from pathlib import Path

from mlfactory.core.madlibz import (
    sample_envelope, authoring_messages, freeze_authored,
    DOMAIN_PROFILES, THRASH_DOMAIN_PROFILES,
)
from mlfactory.core.api import extract_json

# ── Config ──────────────────────────────────────────────────────────────
LUNAROUTE_URL = "https://gw.lunaroute.com/v1/chat/completions"
LUNAROUTE_MODELS_URL = "https://gw.lunaroute.com/v1/models"
API_KEY = os.environ.get("LUNAROUTE_API_KEY", "lr_f360bcbeeb8a6aa0d0e07a28f82809bb42d863549ded922d244b5b0d128c5124")
MAX_TOKENS = 32000
STREAM_TIMEOUT = 30
MAX_RETRIES = 5
RETRY_BACKOFF = 20
MODEL_PREFERENCE = ["glm-5.2-vision-ballast", "glm-5.2-vision", "glm-5.2-vision-background"]

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
    for m in MODEL_PREFERENCE:
        if m in active_models:
            return m
    if active_models:
        fallback = sorted(active_models)[0]
        print(f"[{now()}] WARNING: no preferred model active, falling back to {fallback}")
        return fallback
    raise RuntimeError("no models available on Lunaroute")


def call_streaming(messages, model_id):
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


def load_progress(progress_path):
    if os.path.exists(progress_path):
        with open(progress_path) as f:
            return json.load(f)
    return {"completed": 0, "failed": [], "skipped_seeds": []}


def save_progress(progress_path, progress):
    with open(progress_path, "w") as f:
        json.dump(progress, f, indent=2)


def append_record(output_path, record):
    with open(output_path, "a") as f:
        f.write(json.dumps(record) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["clean", "thrash"])
    parser.add_argument("--start-seed", type=int, required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--progress", required=True)
    parser.add_argument("--corpus", required=True)
    args = parser.parse_args()

    mode = args.mode
    N = args.count
    BASE_SEED = args.start_seed
    OUTPUT = Path(args.output)
    PROGRESS = Path(args.progress)
    corpus_name = args.corpus

    # Pick domain pool based on mode
    if mode == "clean":
        DOMAINS = list(DOMAIN_PROFILES.keys())
    else:  # thrash
        DOMAINS = list(THRASH_DOMAIN_PROFILES.keys())

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    print(f"[{now()}] === Corpus generation {corpus_name}: {N} problems (mode={mode}) ===")
    print(f"[{now()}] Output: {OUTPUT}")
    print(f"[{now()}] Seed range: {BASE_SEED}–{BASE_SEED + N - 1}")
    print(f"[{now()}] Domains: {len(DOMAINS)} ({mode} pool)")
    print(f"[{now()}] Max tokens per call: {MAX_TOKENS}")

    active = discover_models()
    model_id = pick_model(active)
    print(f"[{now()}] Using model: {model_id}")

    progress = load_progress(PROGRESS)
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
        domain = DOMAINS[i % len(DOMAINS)]

        print(f"\n[{now()}] [{i+1}/{N}] seed={seed} domain={domain}")

        env = sample_envelope(seed=seed, domain=domain, mode=mode)

        if mode == "clean":
            print(f"  persona: {env.persona}")
            print(f"  texture: {env.texture}")
        else:
            print(f"  persona: {env.persona}")
            print(f"  load: {env.load_type} | amp: {env.amplifier}")

        messages = authoring_messages(env)

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
            save_progress(PROGRESS, progress)
            continue

        try:
            record = freeze_authored(env, authored, model=model_id, corpus=corpus_name)
            append_record(OUTPUT, record)
            success_count += 1
            progress["completed"] = i + 1
            save_progress(PROGRESS, progress)
            print(f"  ✅ Problem {i+1} frozen (hash: {record['surface_hash'][:12]}...)")
        except Exception as e:
            print(f"  ❌ freeze failed: {e}")
            progress["failed"].append({"index": i, "seed": seed, "domain": domain, "error": str(e)})
            fail_count += 1
            save_progress(PROGRESS, progress)

    print(f"\n{'='*60}")
    print(f"[{now()}] === RUN COMPLETE ({corpus_name}) ===")
    print(f"  Mode:      {mode}")
    print(f"  Successes: {success_count}/{N}")
    print(f"  Failures:  {fail_count}")
    if progress["failed"]:
        print(f"  Failed indices: {[f['index'] for f in progress['failed']]}")
    print(f"  Output: {OUTPUT}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
