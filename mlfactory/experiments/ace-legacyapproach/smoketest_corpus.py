#!/usr/bin/env python3
"""Smoketest: generate 2 madlibz problems via Lunaroute to verify the pipeline.

    python3 smoketest_corpus.py

Exits 0 on success, 1 on failure. Prints problems to stdout.
"""
import json, requests, re, time, sys, os
from datetime import datetime

from mlfactory.core.madlibz import (
    sample_envelope, authoring_messages, freeze_authored,
    DOMAIN_PROFILES,
)
from mlfactory.core.api import extract_json

LUNAROUTE_URL = "https://gw.lunaroute.com/v1/chat/completions"
LUNAROUTE_MODELS_URL = "https://gw.lunaroute.com/v1/models"
API_KEY = os.environ.get("LUNAROUTE_API_KEY", "lr_f360bcbeeb8a6aa0d0e07a28f82809bb42d863549ded922d244b5b0d128c5124")
MODEL_PREFERENCE = ["glm-5.2-vision-ballast", "glm-5.2-vision", "glm-5.2-vision-background"]
DOMAINS = list(DOMAIN_PROFILES.keys())
STREAM_TIMEOUT = 30
MAX_TOKENS = 10000

PLAN = [
    (42, "diner_breakfast_shift"),
    (99, "field_ecology_survey"),
]


def now():
    return datetime.now().strftime("%H:%M:%S")


def discover_and_pick_model():
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
        json={"model": model_id, "messages": messages, "temperature": 0.8,
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
    print(f"[{now()}] === Smoketest: 2 problems ===")
    model_id = discover_and_pick_model()
    print(f"[{now()}] Using model: {model_id}")

    for i, (seed, domain) in enumerate(PLAN):
        print(f"\n{'─'*60}")
        print(f"[{i+1}/2] seed={seed} domain={domain}")

        env = sample_envelope(seed=seed, domain=domain)
        print(f"  persona: {env.persona}")
        print(f"  genus: {env.genus} | detect: {env.detectability}")

        messages = authoring_messages(env)
        t0 = time.time()

        content, reasoning = call_streaming(messages, model_id)
        elapsed = time.time() - t0
        print(f"  content: {len(content)} chars, reasoning: {len(reasoning)} chars ({elapsed:.1f}s)")

        authored = extract_authored(content, reasoning)
        if not authored:
            print(f"  ❌ FAIL: no valid JSON in response")
            sys.exit(1)

        record = freeze_authored(env, authored, model=model_id, corpus="smoketest")
        print(f"  ✅ Frozen (hash: {record['surface_hash'][:12]}...)")

        # Print the problem
        anom = record["anomaly"]
        print(f"\n  PROSE:\n  {record['prose']}")
        print(f"\n  Surface Q: {record['surface_question']}")
        print(f"  Anomaly:   {anom['what_is_wrong']}")

    print(f"\n{'='*60}")
    print(f"[{now()}] ✅ Smoketest passed — pipeline is good to go")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
