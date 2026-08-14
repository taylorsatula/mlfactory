#!/usr/bin/env python3
"""Smoketest: generate 5 random clean (texture-based) madlibz problems via Lunaroute.

    python3 smoketest_clean.py

Clean-arm prompts carry no planted anomaly; their difficulty is interpretive
(texture-driven) and must invite reasoning toward a concrete commitment.

Exits 0 on success, 1 on failure. Prints problems to stdout and appends
frozen records to data/clean_smoketest.jsonl.
"""
import json, requests, re, time, sys, os, random
from datetime import datetime
from pathlib import Path

from mlfactory.core.madlibz import (
    sample_envelope, authoring_messages, freeze_authored,
    DOMAIN_PROFILES, TEXTURES,
)
from mlfactory.core.api import extract_json

LUNAROUTE_URL = "https://gw.lunaroute.com/v1/chat/completions"
LUNAROUTE_MODELS_URL = "https://gw.lunaroute.com/v1/models"
API_KEY = os.environ.get("LUNAROUTE_API_KEY", "lr_f360bcbeeb8a6aa0d0e07a28f82809bb42d863549ded922d244b5b0d128c5124")
MODEL_PREFERENCE = ["glm-5.2-vision-ballast", "glm-5.2-vision", "glm-5.2-vision-background"]
DOMAINS = sorted(DOMAIN_PROFILES.keys())
STREAM_TIMEOUT = 30
MAX_TOKENS = 10000
N = 5
OUT_PATH = Path(__file__).parent / "data" / "clean_smoketest.jsonl"


def now():
    return datetime.now().strftime("%H:%M:%S")


def discover_and_pick_model():
    """Per AGENTS.md: query GET /v1/models on first use; prefer -ballast."""
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
        return sorted(active)[0]
    raise RuntimeError("no models available")


def call_streaming(messages, model_id):
    resp = requests.post(
        LUNAROUTE_URL,
        headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
        json={"model": model_id, "messages": messages, "temperature": 0.9,
              "max_tokens": MAX_TOKENS, "stream": True},
        stream=True, timeout=STREAM_TIMEOUT,
    )
    if resp.status_code >= 500:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
    resp.raise_for_status()

    content_parts, reasoning_parts = [], []
    for line in resp.iter_lines():
        if not line or not line.decode("utf-8").startswith("data: "):
            continue
        payload_str = line.decode("utf-8")[6:]
        if payload_str.strip() == "[DONE]":
            break
        try:
            chunk = json.loads(payload_str)
            choices = chunk.get("choices", [])
            if not choices:
                continue
            delta = choices[0].get("delta", {})
            c = delta.get("content", "")
            r = delta.get("reasoning", "") or delta.get("reasoning_content", "")
            if c: content_parts.append(c)
            elif r: reasoning_parts.append(r)
        except json.JSONDecodeError:
            pass

    return "".join(content_parts), "".join(reasoning_parts)


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


def main():
    print(f"[{now()}] === Clean-arm smoketest: {N} problems ===")
    model_id = discover_and_pick_model()
    print(f"[{now()}] Using model: {model_id}")

    rng = random.SystemRandom()
    plan = [(rng.randrange(100000, 999999), rng.choice(DOMAINS)) for _ in range(N)]
    print(f"[{now()}] Random plan (seed, domain): {plan}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    records = []
    failures = 0

    for i, (seed, domain) in enumerate(plan):
        print(f"\n{'─'*70}")
        print(f"[{i+1}/{N}] seed={seed} domain={domain}")

        env = sample_envelope(seed=seed, domain=domain, mode="clean")
        print(f"  persona: {env.persona}")
        print(f"  stakes:  {env.stakes}")
        print(f"  texture: {env.texture}")

        messages = authoring_messages(env)
        t0 = time.time()

        try:
            content, reasoning = call_streaming(messages, model_id)
        except Exception as e:
            print(f"  ❌ API call failed: {e}")
            failures += 1
            continue
        elapsed = time.time() - t0
        print(f"  content: {len(content)} chars, reasoning: {len(reasoning)} chars ({elapsed:.1f}s)")

        authored = extract_authored(content, reasoning)
        if not authored:
            print(f"  ❌ FAIL: no valid JSON in response")
            failures += 1
            continue

        try:
            record = freeze_authored(env, authored, model=model_id, corpus="clean-smoketest")
        except ValueError as e:
            print(f"  ❌ FAIL: freeze rejected: {e}")
            failures += 1
            continue

        records.append(record)
        rsn = record["reasoning"]
        print(f"\n  PROSE:\n  {record['prose']}")
        print(f"\n  Surface Q: {record['surface_question']}")
        print(f"  Texture:   {rsn.get('texture')} (asked: {env.texture})")
        print(f"  Why hard:  {rsn.get('why_sustained_reasoning_needed')}")
        print(f"  Target:    {rsn.get('decision_target')}")

    if records:
        with OUT_PATH.open("a", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"\n[{now()}] Wrote {len(records)} records to {OUT_PATH}")

    print(f"\n{'='*70}")
    if failures == 0 and len(records) == N:
        print(f"[{now()}] ✅ Clean smoketest passed — {N}/{N} generated")
    else:
        print(f"[{now()}] ⚠️  Completed {len(records)}/{N}, {failures} failures")
    print(f"{'='*70}")
    sys.exit(0 if records else 1)


if __name__ == "__main__":
    main()
