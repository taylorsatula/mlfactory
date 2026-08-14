#!/usr/bin/env python3
"""Run the 5 clean smoketest prompts through local Qwopus (port 3090) sequentially.
Captures full reasoning traces for manual assessment."""
import json, requests, time, sys
from pathlib import Path

MODEL_URL = "http://127.0.0.1:3090/v1/chat/completions"
API_KEY = "de401e0064756cf372297a5a4068a8ff3aa00e96efea7d85a2504f753e6d3763"
INPUT = Path(__file__).parent / "data" / "clean_smoketest.jsonl"
OUTPUT = Path(__file__).parent / "data" / "clean_traces.jsonl"

def call_model(prose: str, surface_question: str) -> dict:
    messages = [
        {"role": "user", "content": prose}
    ]
    resp = requests.post(
        MODEL_URL,
        headers={"Authorization": f"Bearer {API_KEY}"},
        json={
            "model": "Qwopus 27B Fusion MTP",
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 16384,
            "stream": False,
        },
        timeout=300,
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
    records = [json.loads(l) for l in INPUT.read_text().splitlines() if l.strip()]
    print(f"Loaded {len(records)} clean prompts")
    results = []
    for i, rec in enumerate(records):
        print(f"\n{'='*70}")
        print(f"[{i+1}/{len(records)}] seed={rec['seed']} domain={rec['domain']} texture={rec['envelope']['texture']}")
        print(f"  Prompt: {rec['surface_question']}")
        t0 = time.time()
        try:
            result = call_model(rec["prose"], rec["surface_question"])
            elapsed = time.time() - t0
            print(f"  Response: {len(result['content'])} chars content, {len(result['reasoning'])} chars reasoning ({elapsed:.1f}s)")
            print(f"  Finish: {result['finish_reason']} | Tokens: {result['usage']}")
            rec["trace"] = result
            rec["trace_elapsed_s"] = round(elapsed, 1)
        except Exception as e:
            print(f"  ERROR: {e}")
            rec["trace"] = {"error": str(e)}
        results.append(rec)

    OUTPUT.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in results) + "\n")
    print(f"\n{'='*70}")
    print(f"Saved {len(results)} traces to {OUTPUT}")

if __name__ == "__main__":
    main()
