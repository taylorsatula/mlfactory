#!/usr/bin/env python3
"""Run ACE rewrite on sampled reasoning traces via Lunaroute.

Usage:
    cd /home/admin/mlfactory
    python3 run_ace_rewrite_lunaroute.py

Environment:
    LUNAROUTE_API_KEY   API key (or read from .mlfactory/secrets.yaml)
    LUNAROUTE_BASE_URL  Defaults to https://gw.lunaroute.com/v1
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import requests
import yaml

from mlfactory.core.api import APIClient, APIConfig


REPO_ROOT = Path(__file__).resolve().parent
SKILL_PATH = REPO_ROOT / "agents" / "skills" / "ace-rewrite" / "SKILL.md"
EXAMPLES_PATH = REPO_ROOT / "agents" / "skills" / "ace-rewrite" / "EXAMPLES.md"
DEFAULT_TRACES_PATH = (
    REPO_ROOT
    / "mlfactory"
    / "experiments"
    / "ace"
    / "data"
    / "thrash_400_qwen38_traces.jsonl"
)
DEFAULT_BASE_URL = "https://gw.lunaroute.com/v1"
DEFAULT_MODEL = "glm-5.2-vision"
DEFAULT_OUT_DIR = REPO_ROOT / "runs" / "ace_rewrite_lunaroute"


def load_secrets() -> dict:
    secrets_path = REPO_ROOT / ".mlfactory" / "secrets.yaml"
    if secrets_path.exists():
        return yaml.safe_load(secrets_path.read_text()) or {}
    return {}


def get_api_key() -> str | None:
    key = os.environ.get("LUNAROUTE_API_KEY")
    if key:
        return key
    return load_secrets().get("LUNAROUTE_API_KEY")


def get_base_url() -> str:
    return os.environ.get("LUNAROUTE_BASE_URL", DEFAULT_BASE_URL)


def pick_model(base_url: str, api_key: str, prefer_ballast: bool = False) -> str:
    """Query /v1/models and return a working model.

    Empirically, the *-ballast variant on Lunaroute is currently being routed to
    the background tier and often returns empty content for long prompts. We
    therefore default to the plain glm-5.2-vision unless --ballast is set.
    """
    resp = requests.get(
        f"{base_url.rstrip('/')}/models",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=30,
    )
    resp.raise_for_status()
    models = [m["id"] for m in resp.json().get("data", [])]
    if prefer_ballast:
        ballast = [m for m in models if m.endswith("-ballast")]
        if ballast:
            return ballast[0]
    if "glm-5.2-vision" in models:
        return "glm-5.2-vision"
    if models:
        return models[0]
    raise RuntimeError("No models returned by Lunaroute")


def build_system_prompt() -> str:
    if not SKILL_PATH.exists():
        raise FileNotFoundError(SKILL_PATH)
    if not EXAMPLES_PATH.exists():
        raise FileNotFoundError(EXAMPLES_PATH)
    skill = SKILL_PATH.read_text()
    examples = EXAMPLES_PATH.read_text()
    return f"{skill}\n\n---\n\n{examples}"


def build_user_prompt(record: dict) -> str:
    trace_content = record.get("trace", {}).get("content", "")
    return f"""Apply the ACE rewrite rules in the system prompt to the reasoning trace below.

Maintain the original trajectory. Do not invent reasoning, repair errors, or change the conclusion. Return the rewritten trace and the fixed-schema metadata.

Problem context:
{record.get('prose', '')}

Surface question:
{record.get('surface_question', '')}

Reasoning trace to rewrite:
{trace_content}

Return your response in this exact format:

<rewritten_trace>
[rewritten trace text]
</rewritten_trace>

```json
{{
  "rewrite_status": "rewritten",
  "transformations_applied": ["span_removal"],
  "source_reasoning_added": false,
  "conclusion_changed": false,
  "branch_order_changed": false,
  "judgment_flags": []
}}
```
"""


def load_traces(path: Path, n: int, seed: int | None = None, indices: list[int] | None = None) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(path)
    lines = path.read_text().splitlines()
    records = [json.loads(line) for line in lines if line.strip()]
    records = [r for r in records if r.get("trace", {}).get("content", "").strip()]
    if indices is not None:
        chosen = []
        for idx in indices:
            if 0 <= idx < len(records):
                chosen.append(records[idx])
            else:
                raise IndexError(f"Line index {idx} out of range (0-{len(records) - 1})")
        return chosen
    if seed is not None:
        random.seed(seed)
    if len(records) < n:
        raise ValueError(f"Only {len(records)} non-empty traces available, requested {n}")
    return random.sample(records, n)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ACE rewrite via Lunaroute")
    parser.add_argument("--traces", type=Path, default=DEFAULT_TRACES_PATH)
    parser.add_argument("-n", "--count", type=int, default=3, help="Number of random traces to rewrite")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for sampling")
    parser.add_argument("--indices", type=int, nargs="+", default=None, help="Specific 0-based line indices to use")
    parser.add_argument("--model", type=str, default=None, help="Override model name")
    parser.add_argument("--ballast", action="store_true", help="Prefer a *-ballast model if available")
    parser.add_argument("--base-url", type=str, default=None, help="Override Lunaroute base URL")
    parser.add_argument("--max-tokens", type=int, default=20000)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("-o", "--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--skip-models-query", action="store_true", help="Do not query /v1/models (use --model)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    api_key = get_api_key()
    if not api_key:
        print("Error: LUNAROUTE_API_KEY not found in env or .mlfactory/secrets.yaml", file=sys.stderr)
        return 1

    base_url = args.base_url or get_base_url()

    if args.model:
        model = args.model
    elif args.skip_models_query:
        model = DEFAULT_MODEL
    else:
        try:
            model = pick_model(base_url, api_key, prefer_ballast=args.ballast)
        except Exception as exc:
            print(f"Warning: could not query models ({exc}); falling back to {DEFAULT_MODEL}", file=sys.stderr)
            model = DEFAULT_MODEL

    print(f"Base URL: {base_url}")
    print(f"Model:    {model}")

    client = APIClient(
        APIConfig(
            base_url=base_url,
            api_key=api_key,
            model=model,
            timeout=600.0,
        )
    )

    system_prompt = build_system_prompt()
    records = load_traces(args.traces, args.count, seed=args.seed, indices=args.indices)

    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    run_dir = args.out_dir / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "system_prompt.md").write_text(system_prompt)
    config = {
        "model": model,
        "base_url": base_url,
        "traces_path": str(args.traces),
        "count": len(records),
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "traces": [],
    }

    for i, record in enumerate(records, 1):
        seed = record.get("seed", "unknown")
        domain = record.get("domain", "unknown")
        print(f"[{i}/{len(records)}] Rewriting seed {seed} ({domain}) ...")
        user_prompt = build_user_prompt(record)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        start = time.time()
        response = None
        attempts = 0
        while attempts < 3:
            response = client.chat_completion(
                messages=messages,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
            )
            attempts += 1
            if response is not None and response.strip():
                break
            print(f"  attempt {attempts} returned empty/None, retrying...")
            time.sleep(2 ** attempts)
        elapsed = time.time() - start
        if response is None or not response.strip():
            print(f"  FAILED after {attempts} attempts")
            continue
        print(f"  -> {len(response):,} chars in {elapsed:.1f}s")

        out_file = run_dir / f"rewrite_{i:02d}_seed_{seed}_{domain}.md"
        out_file.write_text(response)

        meta = {
            "index": i,
            "seed": seed,
            "domain": domain,
            "input_chars": len(record.get("trace", {}).get("content", "")),
            "output_chars": len(response),
            "elapsed_s": round(elapsed, 2),
            "output_file": str(out_file),
        }
        config["traces"].append(meta)

    (run_dir / "config.json").write_text(json.dumps(config, indent=2))
    print(f"\nDone. Outputs written to: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
