#!/usr/bin/env python3
"""Generate 20 thrash prompts via Lunaroute (non-ballast), then run through Qwopus."""
import json, requests, re, time, sys, os, random
from pathlib import Path

from mlfactory.core.madlibz import (
    sample_envelope, authoring_messages, freeze_authored,
    THRASH_DOMAIN_PROFILES,
)
from mlfactory.core.api import extract_json

# --- Lunaroute config (non-ballast) ---
LUNAROUTE_URL = "https://gw.lunaroute.com/v1/chat/completions"
LUNAROUTE_MODELS_URL = "https://gw.lunaroute.com/v1/models"
API_KEY = os.environ.get("LUNAROUTE_API_KEY", "lr_f360bcbeeb8a6aa0d0e07a28f82809bb42d863549ded922d244b5b0d128c5124")
MODEL_PREFERENCE = ["glm-5.2-vision"]  # non-ballast as requested

# --- Qwopus config ---
QWOPUS_URL = "http://127.0.0.1:3090/v1/chat/completions"
QWOPUS_KEY = "de401e0064756cf372297a5a4068a8ff3aa00e96efea7d85a2504f753e6d3763"
QWOPUS_MODEL = "Qwopus 27B Fusion MTP"

DOMAINS = sorted(THRASH_DOMAIN_PROFILES.keys())
N = 20
OUT_PROMPTS = Path(__file__).parent / "data" / "thrash_prompts_20.jsonl"
OUT_TRACES = Path(__file__).parent / "data" / "thrash_traces_20.jsonl"


def now():
    from datetime import datetime
    return datetime.now().strftime("%H:%M:%S")


def discover_lunaroute_model():
    try:
        r = requests.get(LUNAROUTE_MODELS_URL,
                         headers={"Authorization": f"Bearer {API_KEY}"},
                         timeout=15)
        r.raise_for_status()
        active = {m["id"] for m in r.json().get("data", [])}
        print(f"[{now()}] Active Lunaroute models: {sorted(active)}")
    except Exception as e:
        print(f"[{now()}] WARNING: model discovery failed ({e})")
        active = set()
    for m in MODEL_PREFERENCE:
        if not active or m in active:
            return m
    if active:
        return sorted(active)[0]
    raise RuntimeError("no models available")


def call_lunaroute_streaming(messages, model_id):
    resp = requests.post(
        LUNAROUTE_URL,
        headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
        json={"model": model_id, "messages": messages, "temperature": 0.9,
              "max_tokens": 10000, "stream": True},
        stream=True, timeout=60,
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


def extract_json_safe(content, reasoning):
    for src in [content, reasoning]:
        if not src or not src.strip():
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


def call_qwopus(prose):
    resp = requests.post(
        QWOPUS_URL,
        headers={"Authorization": f"Bearer {QWOPUS_KEY}", "Content-Type": "application/json"},
        json={
            "model": QWOPUS_MODEL,
            "messages": [{"role": "user", "content": prose}],
            "temperature": 0.7,
            "max_tokens": 16384,
            "stream": False,
        },
        timeout=600,
    )
    resp.raise_for_status()
    data = resp.json()
    choice = data["choices"][0]
    return {
        "content": choice["message"].get("content", ""),
        "reasoning": choice["message"].get("reasoning_content", "") or choice["message"].get("reasoning", ""),
        "finish_reason": choice.get("finish_reason", ""),
        "usage": data.get("usage", {}),
    }


def main():
    print(f"[{now()}] === Phase 1: Generate {N} thrash prompts via Lunaroute ===")
    lunaroute_model = discover_lunaroute_model()
    print(f"[{now()}] Using Lunaroute model: {lunaroute_model}")

    rng = random.Random(777)  # reproducible
    plan = [(rng.randrange(100000, 999999), rng.choice(DOMAINS)) for _ in range(N)]

    OUT_PROMPTS.parent.mkdir(parents=True, exist_ok=True)
    records = []
    failures = 0

    for i, (seed, domain) in enumerate(plan):
        print(f"\n[{i+1}/{N}] seed={seed} domain={domain}")
        env = sample_envelope(seed=seed, domain=domain, mode="thrash")
        print(f"  persona: {env.persona}")
        print(f"  stakes:  {env.stakes}")
        print(f"  load:    {env.load_type} | amp: {env.amplifier}")

        messages = authoring_messages(env)
        try:
            content, reasoning = call_lunaroute_streaming(messages, lunaroute_model)
        except Exception as e:
            print(f"  ❌ API call failed: {e}")
            failures += 1
            continue

        authored = extract_json_safe(content, reasoning)
        if not authored:
            print(f"  ❌ No valid JSON")
            failures += 1
            continue

        try:
            record = freeze_authored(env, authored, model=lunaroute_model, corpus="thrash-20")
        except ValueError as e:
            print(f"  ❌ Freeze rejected: {e}")
            failures += 1
            continue

        records.append(record)
        print(f"  ✅ Frozen: {record['prose'][:80]}...")
        time.sleep(1)

    with OUT_PROMPTS.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"\n[{now()}] Phase 1 done: {len(records)}/{N} prompts saved to {OUT_PROMPTS}")

    # Phase 2: Run through Qwopus
    print(f"\n{'='*70}")
    print(f"[{now()}] === Phase 2: Run {len(records)} prompts through Qwopus ===")

    results = []
    for i, rec in enumerate(records):
        print(f"\n[{i+1}/{len(records)}] seed={rec['seed']} domain={rec['domain']}")
        print(f"  Surface Q: {rec['surface_question']}")
        t0 = time.time()
        try:
            trace = call_qwopus(rec["prose"])
            elapsed = time.time() - t0
            print(f"  Reasoning: {len(trace['reasoning'])} chars | Content: {len(trace['content'])} chars | Finish: {trace['finish_reason']} ({elapsed:.1f}s)")
            rec["trace"] = trace
            rec["trace_elapsed_s"] = round(elapsed, 1)
        except Exception as e:
            print(f"  ❌ Qwopus error: {e}")
            rec["trace"] = {"error": str(e)}
        results.append(rec)

    with OUT_TRACES.open("w", encoding="utf-8") as f:
        for rec in results:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"\n[{now()}] Phase 2 done: {len(results)} traces saved to {OUT_TRACES}")
    print(f"[{now()}] Failures in generation: {failures}")


if __name__ == "__main__":
    main()
