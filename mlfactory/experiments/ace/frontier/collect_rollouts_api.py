#!/usr/bin/env python3
"""Rollout collector via an OpenAI-compatible llama-server endpoint.

Same row schema, prompt construction, and strict objective_check as
``collect_rollouts.py`` (imported from there), but generation goes through
``/v1/chat/completions`` so the substrate can be a GGUF quant + MTP
self-speculation instead of HF bf16.

2026-08-24 ruling: q8_0 + MTP is accepted for collection/calibration while
the ACE effect itself is unproven (~8x faster). Rows record their substrate
(``backend``/``quant`` fields); they do NOT pool with bf16 batches —
calibration across substrates happens at the pool level, and q8-banded
prompts get their bands re-verified under bf16 at training time.

Caveat on determinism: seeds are recorded per request, but llama.cpp
continuous batching does not guarantee bit-stable outputs for a given seed
(batch composition varies). Resume still skips done (pid, sample_i) keys.

Run:
  .venv/bin/python -m mlfactory.experiments.ace.frontier.collect_rollouts_api \
      --candidates data/acegen_probe_b2.jsonl --out data/acegen_probe_b2_s0.jsonl \
      --port 3091 --candidate-range 0:24
"""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib import request as urlrequest

from mlfactory.experiments.ace.frontier.collect_rollouts import (
    already_done, load_candidates, objective_check, solver_prompt, wilson,
)

THINK_OPEN = "<" + "think>"
THINK_CLOSE = "<" + "/think>"

ACE = Path(__file__).resolve().parent.parent


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def chat_completion(base_url: str, prompt: str, *, seed: int,
                    max_tokens: int, temperature: float, top_p: float,
                    timeout: float = 3600.0) -> dict:
    body = json.dumps({
        "model": "default",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "seed": seed,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": True},
    }).encode("utf-8")
    req = urlrequest.Request(
        base_url.rstrip("/") + "/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json"})
    with urlrequest.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--candidates", type=Path,
                   default=ACE / "data" / "acegen_probe_b2.jsonl")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--base-url", type=str, default=None,
                   help="Overrides http://127.0.0.1:<port>")
    p.add_argument("--port", type=int, default=3091)
    p.add_argument("--n-samples", type=int, default=8)
    p.add_argument("--sample-start", type=int, default=0)
    p.add_argument("--candidate-range", type=str, default=None)
    p.add_argument("--max-new", type=int, default=26000)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--seed-base", type=int, default=58000,
                   help="Same seed space as collect_rollouts.py")
    p.add_argument("--quant", type=str, default="unknown",
                   help="GGUF quant label recorded on every row")
    p.add_argument("--backend", type=str, default="llama.cpp",
                   help="Inference backend label recorded on every row")
    return p.parse_args()


def main() -> None:
    cfg = parse_args()
    base_url = cfg.base_url or f"http://127.0.0.1:{cfg.port}"
    cands = load_candidates(cfg.candidates)
    if cfg.candidate_range:
        start, end = map(int, cfg.candidate_range.split(":"))
        cands = cands[start:end]
    cfg.out.parent.mkdir(parents=True, exist_ok=True)
    done = already_done(cfg.out)
    print(json.dumps({
        "start": now(), "backend": cfg.backend, "quant": cfg.quant,
        "base_url": base_url, "n_candidates": len(cands),
        "candidate_range": cfg.candidate_range,
        "n_samples": cfg.n_samples, "sample_start": cfg.sample_start,
        "max_new": cfg.max_new, "enable_thinking": True, "steering": False,
        "already_done": len(done), "out": str(cfg.out),
    }), flush=True)

    for rec in cands:
        pid = int(rec["provenance"]["proposal_id"])
        missing = [i for i in range(cfg.sample_start,
                                    cfg.sample_start + cfg.n_samples)
                   if (pid, i) not in done]
        if not missing:
            print(json.dumps({"skip": pid, "reason": "complete"}), flush=True)
            continue

        prompt = solver_prompt(rec)
        gold = rec["problem"]["reference_answer"]
        for sample_i in missing:
            seed = cfg.seed_base + pid * 17 + sample_i
            t0 = time.time()
            resp = chat_completion(
                base_url, prompt, seed=seed, max_tokens=cfg.max_new,
                temperature=cfg.temperature, top_p=cfg.top_p)
            dt = time.time() - t0
            choice = resp["choices"][0]
            text = choice["message"].get("content") or ""
            # llama-server parses the CoT into reasoning_content and strips
            # the template's think tags. Re-wrap so objective_check sees the
            # same shape collect_rollouts.py produces from a raw HF decode:
            # "..." + visible answer.
            rc = choice["message"].get("reasoning_content")
            if rc:
                text = THINK_OPEN + rc + THINK_CLOSE + "\n\n" + text
            usage = resp.get("usage", {})
            n_new = usage.get("completion_tokens", 0)
            truncated = (n_new >= cfg.max_new
                         or choice.get("finish_reason") == "length")
            check = objective_check(text, gold, rec)
            row = {
                "proposal_id": pid,
                "sample_i": sample_i,
                "sample_id": f"p{pid:02d}_s{sample_i}",
                "envelope_hash": rec["envelope_hash"],
                "surface_hash": rec["surface_hash"],
                "seed_candidate": rec["seed"],
                "seed_sample": seed,
                "domain": rec["domain"],
                "task": rec["envelope"]["objective_task"],
                "search_topology": rec["envelope"]["search_topology"],
                "verifier_kind": rec["envelope"]["verifier_kind"],
                "surface_question": rec["surface_question"],
                "reference_answer": gold,
                "n_prompt_tokens": usage.get("prompt_tokens", 0),
                "n_new_tokens": n_new,
                "truncated": truncated,
                "elapsed_s": dt,
                "backend": cfg.backend,
                "quant": cfg.quant,
                "model_id": "Qwen3.5-9B",
                "enable_thinking": True,
                "steering": False,
                "sampling": {
                    "temperature": cfg.temperature,
                    "top_p": cfg.top_p,
                    "max_new_tokens": cfg.max_new,
                    "do_sample": True,
                },
                "correct": check["correct"],
                "match_mode": check["match_mode"],
                "extracted_answer_line": check["extracted_answer_line"],
                "has_think_close": check["has_think_close"],
                "visible_chars": check["visible_chars"],
                "completion": text,
                "collected_at": now(),
            }
            with cfg.out.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            done.add((pid, sample_i))
            print(json.dumps({
                "proposal_id": pid, "sample_i": sample_i,
                "domain": rec["domain"], "correct": check["correct"],
                "match_mode": check["match_mode"], "truncated": truncated,
                "think_closed": check["has_think_close"],
                "n_new": n_new, "dt_s": round(dt, 1),
            }), flush=True)

    all_rows = [json.loads(l) for l in cfg.out.read_text().splitlines() if l.strip()]
    by_p: dict[int, list[dict]] = {}
    for r in all_rows:
        by_p.setdefault(r["proposal_id"], []).append(r)
    per = []
    tot_ok = tot_n = tot_trunc = 0
    for rec in cands:
        pid = int(rec["provenance"]["proposal_id"])
        rs = by_p.get(pid, [])
        k = sum(r["correct"] for r in rs)
        n = len(rs)
        tot_ok += k; tot_n += n; tot_trunc += sum(r["truncated"] for r in rs)
        lo, hi = wilson(k, n)
        per.append({"proposal_id": pid, "domain": rec["domain"], "n": n,
                    "correct": k, "acc": (k / n) if n else None,
                    "ci": [lo, hi],
                    "n_truncated": sum(r["truncated"] for r in rs)})
    summary = {"created_at": now(), "candidates": str(cfg.candidates),
               "artifact": str(cfg.out), "backend": cfg.backend,
               "quant": cfg.quant, "n_rows": tot_n, "n_correct": tot_ok,
               "acc": (tot_ok / tot_n) if tot_n else None,
               "n_truncated": tot_trunc, "per_proposal": per}
    cfg.out.with_name(cfg.out.stem + "_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"done": True, "n_rows": tot_n, "acc": summary["acc"],
                      "n_truncated": tot_trunc}), flush=True)


if __name__ == "__main__":
    main()
