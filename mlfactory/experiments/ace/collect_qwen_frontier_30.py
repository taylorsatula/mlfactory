#!/usr/bin/env python3
"""8 unsteered, reasoning-enabled Qwen3.5-9B rollouts on the frozen 30.

No steering, no rewriting, no stratification. Thinking stays open
(``enable_thinking=True``). Completions run to the model's own stop token.
``--max-new`` (default 26000) is only a backstop, not a target length.
The 26k cap was chosen after tail analysis of 32k-capped traces showed
terminal verbatim loops contribute no information: a trace that cannot
finish within 26k is recorded as a model-flaw truncation.

Objective outcome for ace-gen candidates is the strict per-family
verifier in ``gen/`` (structural check against ``problem.reference_answer``
with format-tolerant extraction). Legacy madlibz candidates fall back to
the soft substring/numeric check (advisory only).

Run:
  CUDA_VISIBLE_DEVICES=1 .venv/bin/python -u collect_qwen_frontier_30.py
"""
from __future__ import annotations

import argparse
import json
import math
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from steering_controller import (
    MODEL_PATH, STOP_TOKEN_IDS, build_prompt_ids, freeze_base_model,
    generate_batch,
)
from gen import adversary, assign, certify, grid, hypothesis, machine

GEN_CHECKS = {
    "certify": certify.check,
    "machine": machine.check,
    "assign": assign.check,
    "adversary": adversary.check,
    "grid": grid.check,
    "hypothesis": hypothesis.check,
}

ACE = Path(__file__).resolve().parent
CANDIDATES = ACE / "data" / "madlibz_verifiable_frontier_30.jsonl"
DEFAULT_OUT = ACE / "data" / "qwen35_frontier30_reasoning_x8.jsonl"

NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")
ANSWER_RE = re.compile(r"answer\s*:\s*([^\n]*)", re.IGNORECASE)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_candidates(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise RuntimeError(f"no candidates in {path}")
    return rows


def solver_prompt(rec: dict) -> str:
    prose = (rec.get("prose") or "").strip()
    q = (rec.get("surface_question") or "").strip()
    parts = []
    if prose:
        parts.append(prose)
    if q and q not in prose:
        parts.append(q)
    parts.append(
        "Solve the problem. Show your reasoning. End with a single line "
        "'Answer: <your final answer>'."
    )
    return "\n\n".join(parts)


def visible_answer(text: str) -> str:
    if "</think>" in text:
        return text.split("</think>")[-1].strip()
    return text.strip()


def last_answer_line(text: str) -> str | None:
    hits = ANSWER_RE.findall(text)
    return hits[-1].strip() if hits else None


def norm(s: str) -> str:
    s = s.lower().replace("→", "->").replace("—", "-").replace("–", "-")
    s = re.sub(r"[`$\\]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip(" \t\n.,;:")


def parse_number(s: str) -> float | None:
    m = NUM_RE.search(s.replace(",", ""))
    if not m:
        return None
    try:
        return float(m.group())
    except ValueError:
        return None


def objective_check(completion: str, gold: str, rec: dict | None = None) -> dict:
    """Strict gen-family verifier when the candidate is ace-gen; soft
    substring/numeric fallback for legacy madlibz candidates."""
    vis = visible_answer(completion)
    line = last_answer_line(vis) or last_answer_line(completion)
    meta = {
        "extracted_answer_line": line,
        "visible_chars": len(vis),
        "has_think_close": "</think>" in completion,
    }
    if rec is not None and rec.get("domain") in GEN_CHECKS:
        ok = bool(GEN_CHECKS[rec["domain"]](completion, gold, rec.get("knobs")))
        return {"correct": ok, "match_mode": "gen_strict_v2", **meta}
    ng, nv, nl = norm(gold), norm(vis), norm(line or "")
    gold_n = parse_number(gold)
    pred_n = parse_number(line) if line else parse_number(vis[-200:])
    mode = "none"
    ok = False
    if ng and nv and nv == ng:
        ok, mode = True, "exact_visible"
    elif ng and nl and nl == ng:
        ok, mode = True, "exact_answer_line"
    elif ng and len(ng) >= 4 and ng in nv:
        ok, mode = True, "gold_in_visible"
    elif ng and nl and len(ng) >= 4 and ng in nl:
        ok, mode = True, "gold_in_answer_line"
    elif (gold_n is not None and pred_n is not None
          and math.isclose(gold_n, pred_n, rel_tol=0, abs_tol=0.011)):
        # only claim numeric match when the gold is itself a number
        if norm(re.sub(r"[^\d.\-]", "", gold)) and len(norm(gold)) <= 12:
            ok, mode = True, "numeric"
    return {
        "correct": ok,
        "match_mode": mode,
        **meta,
    }


def already_done(out: Path) -> set[tuple[int, int]]:
    done: set[tuple[int, int]] = set()
    if not out.exists():
        return done
    for line in out.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        done.add((int(rec["proposal_id"]), int(rec["sample_i"])))
    return done


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, center - half), min(1.0, center + half))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--candidates", type=Path, default=CANDIDATES)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--n-samples", type=int, default=8)
    p.add_argument("--sample-start", type=int, default=0,
                   help="First sample_i (pass 2 uses 8 so seeds do not collide).")
    p.add_argument("--candidate-range", type=str, default=None,
                   help="Slice like '0:15' or '15:30' to split work across GPUs.")
    p.add_argument("--max-new", type=int, default=26000)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--seed-base", type=int, default=58000)
    return p.parse_args()


def main() -> None:
    cfg = parse_args()
    cands = load_candidates(cfg.candidates)
    if cfg.candidate_range:
        start, end = map(int, cfg.candidate_range.split(":"))
        cands = cands[start:end]
    cfg.out.parent.mkdir(parents=True, exist_ok=True)
    done = already_done(cfg.out)
    print(json.dumps({
        "start": now(), "model": MODEL_PATH, "n_candidates": len(cands),
        "candidate_range": cfg.candidate_range,
        "n_samples": cfg.n_samples, "sample_start": cfg.sample_start,
        "max_new": cfg.max_new,
        "enable_thinking": True, "steering": False,
        "already_done": len(done), "out": str(cfg.out),
    }), flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    freeze_base_model(model)

    for rec in cands:
        pid = int(rec["provenance"]["proposal_id"])
        missing = [i for i in range(cfg.sample_start,
                                    cfg.sample_start + cfg.n_samples)
                   if (pid, i) not in done]
        if not missing:
            print(json.dumps({"skip": pid, "reason": "complete"}), flush=True)
            continue

        prompt = solver_prompt(rec)
        n_prompt = len(build_prompt_ids(tok, prompt, enable_thinking=True))
        gold = rec["problem"]["reference_answer"]
        # Sequential, not batched: long reasoning traces at 32k would OOM
        # an 8-wide batch on a 24 GB card. Per-sample seeds so resume is
        # bit-stable for unfinished samples.
        for sample_i in missing:
            seed = cfg.seed_base + pid * 17 + sample_i
            t0 = time.time()
            seqs, _ = generate_batch(
                model, tok, prompt, n=1,
                max_new_tokens=cfg.max_new, controller=None,
                do_sample=True, temperature=cfg.temperature, top_p=cfg.top_p,
                seed=seed, enable_thinking=True)
            dt = time.time() - t0
            ids = seqs[0]
            text = tok.decode(ids[n_prompt:], skip_special_tokens=False)
            for stop in ("<|im_end|>", "<|endoftext|>"):
                if text.endswith(stop):
                    text = text[: -len(stop)]
            n_new = len(ids) - n_prompt
            truncated = n_new >= cfg.max_new
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
                "n_prompt_tokens": n_prompt,
                "n_new_tokens": n_new,
                "truncated": truncated,
                "elapsed_s": dt,
                "model": MODEL_PATH,
                "model_id": "Qwen3.5-9B",
                "enable_thinking": True,
                "steering": False,
                "sampling": {
                    "temperature": cfg.temperature,
                    "top_p": cfg.top_p,
                    "max_new_tokens": cfg.max_new,
                    "do_sample": True,
                    "stop_token_ids": STOP_TOKEN_IDS,
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
                "domain": rec["domain"],
                "correct": check["correct"],
                "match_mode": check["match_mode"],
                "truncated": truncated,
                "think_closed": check["has_think_close"],
                "n_new": n_new, "dt_s": round(dt, 1),
                "n_prompt": n_prompt,
            }), flush=True)

    # summary over whatever is on disk (supports resume)
    all_rows = [json.loads(l) for l in cfg.out.read_text().splitlines() if l.strip()]
    by_p: dict[int, list[dict]] = {}
    for r in all_rows:
        by_p.setdefault(r["proposal_id"], []).append(r)
    per = []
    tot_ok = tot_n = tot_trunc = tot_close = 0
    for rec in cands:
        pid = int(rec["provenance"]["proposal_id"])
        rs = by_p.get(pid, [])
        k = sum(r["correct"] for r in rs)
        n = len(rs)
        tot_ok += k
        tot_n += n
        tot_trunc += sum(r["truncated"] for r in rs)
        tot_close += sum(r["has_think_close"] for r in rs)
        lo, hi = wilson(k, n)
        per.append({
            "proposal_id": pid, "domain": rec["domain"],
            "task": rec["envelope"]["objective_task"],
            "verifier_kind": rec["envelope"]["verifier_kind"],
            "n": n, "correct": k, "acc": (k / n) if n else None,
            "ci": [lo, hi],
            "n_truncated": sum(r["truncated"] for r in rs),
            "n_think_closed": sum(r["has_think_close"] for r in rs),
            "mean_new_tokens": (sum(r["n_new_tokens"] for r in rs) / n) if n else None,
        })
    summary = {
        "created_at": now(),
        "candidates": str(cfg.candidates),
        "artifact": str(cfg.out),
        "model": MODEL_PATH,
        "n_rows": tot_n,
        "n_correct": tot_ok,
        "acc": (tot_ok / tot_n) if tot_n else None,
        "ci": list(wilson(tot_ok, tot_n)),
        "n_truncated": tot_trunc,
        "n_think_closed": tot_close,
        "enable_thinking": True,
        "steering": False,
        "max_new": cfg.max_new,
        "per_proposal": per,
    }
    cfg.out.with_name(cfg.out.stem + "_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"done": True, "n_rows": tot_n, "acc": summary["acc"],
                      "n_truncated": tot_trunc, "n_think_closed": tot_close}),
          flush=True)


if __name__ == "__main__":
    main()
