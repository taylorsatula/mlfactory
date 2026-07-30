#!/usr/bin/env python3
"""
DFT Phase-0 evaluation harness.

Measures distributional match between a generator's outputs and a reference corpus
using the author's protocol: MMD^2 over sentence embeddings, unigram L2 distance,
and LLM-as-judge pairwise quality (JMQ). Plus slop diagnostics.

Built for own consumption: dense, data-first, scriptable.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from openai import OpenAI
from transformers import AutoTokenizer

from mlfactory.core.api import APIConfig, Judge, run_judge_pairwise
from mlfactory.core.embeddings import Embedder


# ---------------------------------------------------------------------------
# utilities
# ---------------------------------------------------------------------------

def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str) -> None:
    print(f"[{now()}] {msg}", flush=True)


def load_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def split_sentences(text: str) -> list[str]:
    # crude but consistent with the blog's repetition diagnostic
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def first_word(sentence: str) -> str:
    m = re.search(r"\b\w+\b", sentence)
    return m.group(0).lower() if m else ""


# ---------------------------------------------------------------------------
# generation client
# ---------------------------------------------------------------------------

class Generator:
    def __init__(self, base_url: str, model: str, api_key: str = "dummy", system_prompt: str | None = None):
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=600)
        self.model = model
        self.system_prompt = system_prompt

    def generate(self, prompts: list[str], temperature: float, max_tokens: int, top_p: float = 1.0, top_k: int = 0,
                 repeat_penalty: float = 1.0, presence_penalty: float = 0.0, frequency_penalty: float = 0.0) -> list[str]:
        # Always pass sampler overrides so server defaults (temp 0.2, repeat-penalty 1.09, top_k 20) do not leak in.
        outs: list[str] = []
        for p in prompts:
            extra: dict = {
                "top_k": top_k,
                "repeat_penalty": repeat_penalty,
                "presence_penalty": presence_penalty,
                "frequency_penalty": frequency_penalty,
            }
            try:
                if self.system_prompt is not None:
                    resp = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": self.system_prompt},
                            {"role": "user", "content": p},
                        ],
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        extra_body=extra,
                    )
                    outs.append(resp.choices[0].message.content)
                else:
                    resp = self.client.completions.create(
                        model=self.model,
                        prompt=p,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        extra_body=extra,
                    )
                    outs.append(resp.choices[0].text)
            except Exception as e:
                log(f"generation failed: {e}")
                outs.append("")
        return outs


# ---------------------------------------------------------------------------
# embedding + MMD
# ---------------------------------------------------------------------------

def median_heuristic(X: np.ndarray, Y: np.ndarray) -> float:
    """Median pairwise distance between two embedding sets."""
    Z = np.concatenate([X, Y], axis=0)
    # sample up to 2000 points to keep this cheap
    if Z.shape[0] > 2000:
        idx = np.random.choice(Z.shape[0], 2000, replace=False)
        Z = Z[idx]
    dists = np.linalg.norm(Z[:, None, :] - Z[None, :, :], axis=2)
    return float(np.median(dists[dists > 0]))


def rbf_kernel(X: np.ndarray, Y: np.ndarray | None, bandwidth: float) -> np.ndarray:
    # X: (m,d), Y: (n,d) or None for self
    if Y is None:
        dists = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
    else:
        dists = np.linalg.norm(X[:, None, :] - Y[None, :, :], axis=2)
    return np.exp(-(dists ** 2) / (2.0 * bandwidth ** 2))


def mmd2_unbiased(X: np.ndarray, Y: np.ndarray, bandwidth: float) -> float:
    """U-statistic MMD^2 (diagonals excluded). Can be negative."""
    m, n = X.shape[0], Y.shape[0]
    Kxx = rbf_kernel(X, None, bandwidth)
    Kyy = rbf_kernel(Y, None, bandwidth)
    Kxy = rbf_kernel(X, Y, bandwidth)
    np.fill_diagonal(Kxx, 0.0)
    np.fill_diagonal(Kyy, 0.0)
    term1 = Kxx.sum() / (m * (m - 1))
    term2 = Kyy.sum() / (n * (n - 1))
    term3 = Kxy.sum() / (m * n)
    return float(term1 + term2 - 2.0 * term3)


# ---------------------------------------------------------------------------
# token-level L2 distance
# ---------------------------------------------------------------------------

class TokenDistribution:
    def __init__(self, tokenizer_name: str):
        log(f"loading tokenizer {tokenizer_name}")
        self.tok = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)

    def l2_distance(self, hyps: list[str], refs: list[str], n: int = 1) -> float:
        """L2 distance between empirical n-gram token distributions."""
        hyp_counts = Counter()
        ref_counts = Counter()
        total_hyp = 0
        total_ref = 0
        for text, is_ref in [(h, False) for h in hyps] + [(r, True) for r in refs]:
            ids = self.tok.encode(text, add_special_tokens=False)
            if n == 1:
                grams = [(i,) for i in ids]
            else:
                grams = [tuple(ids[i : i + n]) for i in range(len(ids) - n + 1)]
            if is_ref:
                ref_counts.update(grams)
                total_ref += len(grams)
            else:
                hyp_counts.update(grams)
                total_hyp += len(grams)
        vocab = set(hyp_counts.keys()) | set(ref_counts.keys())
        p = np.array([hyp_counts.get(g, 0) / max(total_hyp, 1) for g in vocab], dtype=np.float64)
        q = np.array([ref_counts.get(g, 0) / max(total_ref, 1) for g in vocab], dtype=np.float64)
        return float(np.linalg.norm(p - q))

    def non_ascii_token_rate(self, texts: list[str]) -> float:
        bad = 0
        for text in texts:
            ids = self.tok.encode(text, add_special_tokens=False)
            for i in ids:
                # heuristic: decode single id and check if non-ascii appears
                s = self.tok.decode([i], skip_special_tokens=True)
                if any(ord(c) > 127 for c in s):
                    bad += 1
                    break
        return bad / max(len(texts), 1)


# ---------------------------------------------------------------------------
# diagnostics
# ---------------------------------------------------------------------------

def repetition_rate(texts: list[str]) -> float:
    """% texts with >=3 consecutive sentences starting with the same word."""
    bad = 0
    for text in texts:
        sents = split_sentences(text)
        if len(sents) < 3:
            continue
        for i in range(len(sents) - 2):
            w0, w1, w2 = first_word(sents[i]), first_word(sents[i + 1]), first_word(sents[i + 2])
            if w0 and w0 == w1 == w2:
                bad += 1
                break
    return bad / max(len(texts), 1)


def non_english_char_rate(texts: list[str]) -> float:
    # blog's metric: outputs containing non-English characters
    bad = 0
    for text in texts:
        # any CJK / Hangul / Arabic / Cyrillic / etc.
        if re.search(r"[\u4e00-\u9fff\uac00-\ud7af\u0600-\u06ff\u0400-\u04ff\u0370-\u03ff]", text):
            bad += 1
    return bad / max(len(texts), 1)


def self_bleu(texts: list[str], n: int = 4) -> float:
    """Approximate self-BLEU using character n-grams (cheaper, adequate for diversity)."""
    from collections import defaultdict

    scores = []
    for i, hyp in enumerate(texts):
        hyp_chars = list(hyp)
        hyp_grams = set(tuple(hyp_chars[j : j + n]) for j in range(len(hyp_chars) - n + 1))
        if not hyp_grams:
            continue
        overlaps = []
        for j, ref in enumerate(texts):
            if i == j:
                continue
            ref_chars = list(ref)
            ref_grams = Counter(tuple(ref_chars[k : k + n]) for k in range(len(ref_chars) - n + 1))
            if not ref_grams:
                continue
            clipped = sum(min(ref_grams[g], 1) for g in hyp_grams)
            overlaps.append(clipped / len(hyp_grams))
        if overlaps:
            scores.append(np.mean(overlaps))
    return float(np.mean(scores)) if scores else 0.0


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description="DFT Phase-0 eval harness")
    p.add_argument("--test-file", required=True, help="JSONL with prompt, reference fields")
    p.add_argument("--gen-url", default="http://localhost:3090/v1", help="OpenAI-compatible generation endpoint")
    p.add_argument("--gen-model", default="Qwopus3.6-27b", help="model id for generation")
    p.add_argument("--tokenizer", default="qwen/Qwen3.6-27B", help="HF tokenizer name for L2 diagnostics")
    p.add_argument("--embed-model", default="nvidia/llama-embed-nemotron-8b", help="sentence-transformers embedding model")
    p.add_argument("--embed-device", default="cuda:0", help="device for embedder")
    p.add_argument("--judge-url", default=None, help="OpenAI-compatible judge endpoint (defaults to gen-url)")
    p.add_argument("--judge-model", default=None, help="judge model id (defaults to gen-model)")
    p.add_argument("--system-prompt", default="You write the final requested text directly. Do not include reasoning, <think> tags, or meta-commentary.", help="system prompt for chat-completions generation mode")
    p.add_argument("--temps", default="0.7,1.0", help="comma-separated temperatures to sweep")
    p.add_argument("--top-p", default=1.0, type=float)
    p.add_argument("--top-k", default=0, type=int, help="0 disables top-k (matches article's full-vocab sampling); >0 enables")
    p.add_argument("--repeat-penalty", default=1.0, type=float, help="1.0 disables repetition penalty (server default is 1.09)")
    p.add_argument("--presence-penalty", default=0.0, type=float)
    p.add_argument("--frequency-penalty", default=0.0, type=float)
    p.add_argument("--max-tokens", default=512, type=int)
    p.add_argument("--n", default=None, type=int, help="limit to first N test samples")
    p.add_argument("--out-dir", default="./out", help="where to write results")
    p.add_argument("--judge-criterion", default="overall quality", help="criterion for JMQ")
    p.add_argument("--judge-samples", default=50, type=int, help="max samples to judge (expensive)")
    p.add_argument("--seed", default=42, type=int)
    p.add_argument("--strip-think", action=argparse.BooleanOptionalAction, default=True, help="remove <think>...</think> blocks from hypotheses before scoring")
    args = p.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_jsonl(args.test_file)
    if args.n:
        rows = rows[: args.n]
    prompts = [r["prompt"] for r in rows]
    refs = [r["reference"] for r in rows]

    log(f"loaded {len(prompts)} test samples")
    log(f"refs token min/max/avg: {min(len(r) for r in refs):d} / {max(len(r) for r in refs):d} / {sum(len(r) for r in refs)/len(refs):.0f} chars")

    embedder = Embedder(args.embed_model, device=args.embed_device)
    token_dist = TokenDistribution(args.tokenizer)
    judge = None
    if args.judge_url or args.gen_url:
        judge = Judge(
            APIConfig(
                base_url=args.judge_url or args.gen_url,
                model=args.judge_model or args.gen_model,
                timeout=300,
            )
        )

    generator = Generator(args.gen_url, args.gen_model, system_prompt=args.system_prompt)

    ref_embeddings = embedder.encode(refs)
    # fixed bandwidth from full reference set
    bw_ref = median_heuristic(ref_embeddings, ref_embeddings) or 1.0
    log(f"reference MMD bandwidth (median heuristic): {bw_ref:.4f}")

    temps = [float(t) for t in args.temps.split(",")]
    results: list[dict] = []

    for temp in temps:
        log(f"generating at T={temp}")
        hyps = generator.generate(
            prompts,
            temperature=temp,
            max_tokens=args.max_tokens,
            top_p=args.top_p,
            top_k=args.top_k,
            repeat_penalty=args.repeat_penalty,
            presence_penalty=args.presence_penalty,
            frequency_penalty=args.frequency_penalty,
        )
        if args.strip_think:
            def strip_think(text: str) -> str:
                if "</think>" in text:
                    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
                elif "<think>" in text:
                    # truncated reasoning: drop from think tag onward
                    text = text.split("<think>")[0]
                return text.strip()
            hyps = [strip_think(h) for h in hyps]
            log(f"stripped think blocks; avg hyp chars after strip: {sum(len(h) for h in hyps)/max(len(hyps),1):.0f}")

        log(f"computing metrics at T={temp}")
        hyp_embeddings = embedder.encode(hyps)
        bw = median_heuristic(hyp_embeddings, ref_embeddings)
        mmd = mmd2_unbiased(hyp_embeddings, ref_embeddings, bw)
        l2_1 = token_dist.l2_distance(hyps, refs, n=1)
        l2_2 = token_dist.l2_distance(hyps, refs, n=2)
        l2_3 = token_dist.l2_distance(hyps, refs, n=3)
        rep = repetition_rate(hyps)
        noneng = non_english_char_rate(hyps)
        nonascii = token_dist.non_ascii_token_rate(hyps)
        sbleu = self_bleu(hyps)

        jmq = None
        if judge:
            jn = min(args.judge_samples, len(prompts))
            log(f"judging {jn} pairs at T={temp}")
            result = run_judge_pairwise(
                judge,
                prompts[:jn],
                hyps[:jn],
                refs[:jn],
                criterion=args.judge_criterion,
                seed=args.seed,
            )
            jmq = result.jmq

        res = {
            "temperature": temp,
            "n": len(hyps),
            "mmd2": mmd,
            "l2_1gram": l2_1,
            "l2_2gram": l2_2,
            "l2_3gram": l2_3,
            "jmq": jmq,
            "repetition_rate": rep,
            "non_english_char_rate": noneng,
            "non_ascii_token_rate": nonascii,
            "self_bleu": sbleu,
            "avg_hyp_chars": sum(len(h) for h in hyps) / max(len(hyps), 1),
        }
        results.append(res)
        log(json.dumps(res, indent=2))

        # write completions
        with open(out_dir / f"completions_T{temp}.jsonl", "w", encoding="utf-8") as f:
            for r, h in zip(rows, hyps):
                f.write(json.dumps({"prompt": r["prompt"], "reference": r["reference"], "hypothesis": h}) + "\n")

    summary = {
        "config": vars(args),
        "results": results,
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    log(f"results written to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
