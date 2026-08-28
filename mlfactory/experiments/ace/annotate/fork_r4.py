#!/usr/bin/env python3
"""R4 — concentrated forks on detector-nominated onset states.

The only annotation-sidestep rung that spends rental compute. Design
(pre-registered 2026-08-27, frozen — the attendance runbook forbids
changes to any of it):

  States   10 per class (CYCLE/LOOP/MUSE), distinct pids, earliest
           conf=clear onset per trace; fork point = onset_abs - 6
           (lead-time curve: earliest reliable detection window 4-8
           tokens pre-onset; lab_notes/2026-08-27-lookback-k5-rec-results.md).
  Arms     noop / toward_healthy (+lam d) / toward_diverge (-lam d).
           Saved directions point FROM divergence TOWARD healthy
           (directions_annot_clear_merged.npz sign_convention), so
           +lam is the therapeutic arm, -lam the causal-sufficiency arm.
  Layers   focal per class: CYCLE L18, LOOP L2, MUSE L17 (probe-best).
           Hook fires from the fork position onward: last prefix
           position in prefill, every decode step after.
  lam      0.05 x median onset residual norm at the focal layer across
           the merged captures (half the 0.1||h|| intervention bound).
  Seeds    m = 24 paired seeds per arm (identical seed sequence across
           arms); production runs use sub-batch 1, and batch seeds
           derive from sha256(state_id) so they are stable across
           processes and resumes.
  Model    bf16 HF Qwen3.5-9B, sdpa with the FLASH_ATTENTION backend
           forced (deterministic: two concurrent processes, same seed,
           4096 tokens -> identical SHA256, verified 2026-08-27; the
           MATH backend is also deterministic but materializes q x kv,
           whose prefill spike (~2x steady footprint) makes two
           concurrent processes OOM on fork >~9.5k states)
           (default backend is call-to-call nondeterministic, R11).
  Scoring  frontier.collect_rollouts.objective_check against the trace
           row's reference_answer; backstop cap 26000 absolute tokens.

Modes:
  --plan          build the fork plan (via datasave) and exit
  --run           execute the plan (resume-safe by (state, arm, seed))

Generation is full-forward batched (GRPO-proven pattern) with an
adaptive sub-batch sized to the prefix length (production runs use
sub-batch 1: two concurrent processes per box beat batched decode on
this model).

Run as a module from the repo root:
  CUDA_VISIBLE_DEVICES=0 python -m mlfactory.experiments.ace.annotate.fork_r4 --run
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoModelForCausalLM, AutoTokenizer

from mlfactory.core.datasave import datasave
from mlfactory.experiments.ace.annotate.capture_activations import (
    char_to_token, load_annotations, load_candidates, load_corpus)
from mlfactory.experiments.ace.core.steering_controller import (
    MODEL_PATH, PAD_TOKEN_ID, STOP_TOKEN_IDS, build_prompt_ids)
from mlfactory.experiments.ace.frontier.collect_rollouts import (
    objective_check, solver_prompt)

HERE = Path(__file__).resolve().parent
ACE = HERE.parent
DATA = ACE / "data"
PLAN_PATH = Path(__file__).resolve().parents[4] / "artifacts" / "fork_plan_r4.jsonl"

FOCAL = {"CYCLE": 18, "LOOP": 2, "MUSE": 17}
FORK_BACK = 6
ARMS = ("noop", "toward_healthy", "toward_diverge")
M_PER_ARM = 24
SUB_BATCH = 4
N_STATES_PER_CLASS = 10
PLAN_SEED = 484000
RUN_SEED_BASE = 484100
CAP_ABS = 26000
TEMPERATURE = 0.8
TOP_P = 0.95
LAM_FRACTION = 0.05  # half the 0.1*||h|| intervention bound
DIRECTIONS = DATA / "steering_directions" / "directions_annot_clear_merged.npz"

# merged-corpus sources (same pairing as the R1-R3 merged scoring)
CORPUS_GROUPS = [
    {"corpus": [DATA / "xsub_q8.jsonl",
                DATA / "xsub_bf16_gpu0.jsonl", DATA / "xsub_bf16_gpu1.jsonl"],
     "annotations": DATA / "annotations_xsub_pass1.jsonl"},
    {"corpus": [DATA / "annot_b2_q8.jsonl"],
     "annotations": DATA / "annotations_b2_pass1.jsonl"},
]
CANDIDATES = [DATA / "xsub_candidates.jsonl", DATA / "acegen_live_b2.jsonl"]
CAPTURE_DIRS = [DATA / "annot_captures_xsub_lb", DATA / "annot_captures_b2_lb"]


# ---------------------------------------------------------------- plan

def median_onset_norms() -> dict[str, float]:
    """Median ||h|| at the focal layer over onset positions, merged
    captures. Sets the lam scale per class."""
    acc: dict[str, list[float]] = defaultdict(list)
    for d in CAPTURE_DIRS:
        for f in sorted(d.glob("*.npz")):
            z = np.load(f, allow_pickle=True)
            res = z["residuals"]  # (n_layers, n_pos, hidden) fp16
            pt = json.loads(z["pos_table"].tobytes().decode())
            for p in pt:
                if p["kind"] != "onset" or p["class"] not in FOCAL:
                    continue
                L = FOCAL[p["class"]]
                v = res[L, p["pos_idx"]].astype(np.float32)
                acc[p["class"]].append(float(np.linalg.norm(v)))
    return {cls: float(np.median(v)) for cls, v in acc.items()}


def resolve_fork_candidates(tok) -> dict[str, list[dict]]:
    """All conf=clear onsets with token positions, per class. One entry
    per (trace, annotation): substrate/pid/sample_i/onset_abs/horizon."""
    cands = load_candidates(CANDIDATES)
    out: dict[str, list[dict]] = defaultdict(list)
    for group in CORPUS_GROUPS:
        corpus = load_corpus(group["corpus"])
        annots = load_annotations(group["annotations"])
        for key in sorted(annots.keys()):
            sub, pid, sid = key
            row = corpus[key]
            cand = cands.get(row["surface_hash"])
            if cand is None:
                continue
            prompt = solver_prompt({**cand, **cand.get("provenance", {})})
            n_prompt = len(build_prompt_ids(tok, prompt, enable_thinking=True))
            c_enc = tok(row["completion"][:CAP_ABS * 4],
                        add_special_tokens=False, return_offsets_mapping=True)
            offsets = c_enc["offset_mapping"][:CAP_ABS]
            n_comp = min(len(c_enc["input_ids"]), CAP_ABS)
            n_total = n_prompt + n_comp
            for a in annots[key]:
                if a.get("conf") != "clear":
                    continue
                t = char_to_token(offsets, a["start_char"], find_ge=True)
                if t < 0 or t >= n_comp:
                    continue
                onset_abs = n_prompt + t
                fork_abs = onset_abs - FORK_BACK
                if fork_abs <= n_prompt or fork_abs >= n_total:
                    continue
                out[a["class"]].append({
                    "substrate": sub, "pid": pid, "sample_i": sid,
                    "onset_abs": onset_abs, "fork_abs": fork_abs,
                    "horizon": n_total - fork_abs,
                    "reference_answer": row["reference_answer"],
                    "surface_hash": row["surface_hash"],
                    "domain": cand["domain"],
                    "knobs": cand.get("knobs"),
                    "outcome": row["correct"],
                    "prompt_text": prompt, "completion": row["completion"],
                    "n_prompt": n_prompt,
                })
    return out


def build_plan() -> None:
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    norms = median_onset_norms()
    print("median onset residual norms at focal layers:",
          {k: round(v, 2) for k, v in norms.items()})
    cands = resolve_fork_candidates(tok)
    rng = np.random.default_rng(PLAN_SEED)
    plan = []
    for cls in ("CYCLE", "LOOP", "MUSE"):
        entries = cands.get(cls, [])
        # earliest onset per trace, then one trace per pid
        by_trace: dict[tuple, dict] = {}
        for e in entries:
            k = (e["substrate"], e["pid"], e["sample_i"])
            if k not in by_trace or e["onset_abs"] < by_trace[k]["onset_abs"]:
                by_trace[k] = e
        by_pid: dict[int, dict] = {}
        for k in sorted(by_trace):
            e = by_trace[k]
            by_pid.setdefault(e["pid"], e)
        pool = [by_pid[p] for p in sorted(by_pid)]
        idx = rng.permutation(len(pool))[:N_STATES_PER_CLASS]
        lam = LAM_FRACTION * norms[cls]
        for i, j in enumerate(sorted(int(x) for x in idx)):
            e = pool[j]
            plan.append({
                "state_id": f"r4_{cls.lower()}_{i:02d}",
                "class": cls, "layer": FOCAL[cls], "lam": round(lam, 4),
                **e,
            })
        print(f"{cls}: {len(entries)} clear onsets -> {len(by_trace)} traces "
              f"-> {len(by_pid)} pids -> picked {len(idx)}; lam={lam:.3f}")
    plan.sort(key=lambda r: r["state_id"])
    datasave("fork_plan_r4.jsonl", plan,
             title="R4 fork plan — detector-nominated onset states",
             description=("30 fork states (10/class, distinct pids, earliest "
                          "conf=clear onset per trace, fork point onset-6). "
                          "Pre-registered design: arms noop/toward_healthy/"
                          "toward_diverge, m=24 paired seeds, focal layers "
                          "CYCLE L18/LOOP L2/MUSE L17, lam=0.05x median "
                          "onset residual norm. Built from the merged xsub+b2 "
                          "corpus annotations (conf=clear)."),
             tags=["r4", "fork-plan", "annotation", "merged"])
    print(f"plan: {len(plan)} states -> {PLAN_PATH}")


# ----------------------------------------------------------------- run

def load_plan() -> list[dict]:
    return [json.loads(l) for l in PLAN_PATH.open()]


_DECODE_STEPS = [0]


def _empty_cache_hook(_m, _inp, out):
    """Periodic empty_cache during decode. Prefill frees its blocks
    into the allocator at prefill end, but they stay RESERVED for the
    whole continuation (the allocator returns them only on
    empty_cache) — with two concurrent processes the co-process then
    sees the spike as resident and may not be able to load/prefill.
    Returning them every 128 decode steps keeps the steady footprint
    near weights+KV. No effect on numerics: frees only unused cached
    blocks."""
    h = out[0] if isinstance(out, tuple) else out
    if h.shape[1] == 1:
        _DECODE_STEPS[0] += 1
        if _DECODE_STEPS[0] % 128 == 0:
            torch.cuda.empty_cache()


def make_hook(model, layer: int, delta: torch.Tensor, st: dict):
    """Steer the residual stream from the fork position onward.

    Position-aware, keyed on the current forward call's span
    (st['start'], st['end'], updated by the runner per prefill chunk):
      * prefill chunk (seq_len > 1): add delta at the fork position's
        index within the chunk if the chunk contains it — the replay is
        untouched everywhere else;
      * decode step (seq_len == 1): every generated position is >= fork,
        so add delta unconditionally.
    """
    def hook(_m, _inp, out):
        h = out[0] if isinstance(out, tuple) else out
        d = delta.to(h.dtype)  # directions are fp32; model is bf16 —
        # bf16 + fp32 promotes to fp32 and the next linear raises
        if h.shape[1] > 1:
            idx = st["fork"] - 1 - st["start"]
            if 0 <= idx < h.shape[1]:
                h = h.clone()
                h[:, idx] = h[:, idx] + d
        else:
            h = h + d
        return (h,) + out[1:] if isinstance(out, tuple) else h
    return model.model.layers[layer].register_forward_hook(hook)


def batch_for(fork_abs: int, cap: int) -> int:
    """Sub-batch size cap for a full-forward generation.

    Conservative VRAM-based caps from the MATH-backend era, kept as a
    backstop; production runs pass sub-batch 1, which this only
    clamps downward.
    """
    if fork_abs > 14000:
        limit = 1
    elif fork_abs > 8000:
        limit = 2
    else:
        limit = 4
    return min(cap, limit)


@torch.no_grad()
def gen_batch(model, arm_delta, layer, prefix_ids: list[int], fork_abs: int,
              max_new: int, seed: int, batch: int):
    """One sub-batch of continuations from the fork point.

    Full-forward batched generation over the shared prefix — the exact
    pattern proven by steering_controller.generate_batch in GRPO (no
    cache surgery: this model's VL-wrapper generate path rejects
    manual past_key_values + short input). One monolithic forward over
    the prefix, then autoregressive decode inside generate(). The hook
    (steered arms) adds delta at the fork position during the prefill
    forward and at every decode step, so steering starts exactly at the
    fork position. Identical seed + identical batch composition across
    arms keeps the coupled sampling draws pair-matched. Returns list of
    new-token id lists (positions >= fork), trimmed at the first stop
    token."""
    if seed is not None:
        torch.manual_seed(seed)
    st = {"start": 0, "end": fork_abs, "fork": fork_abs}
    handle = None
    if arm_delta is not None:
        handle = make_hook(model, layer, arm_delta, st)
    gc_handle = model.model.layers[0].register_forward_hook(
        _empty_cache_hook)
    stop = set(STOP_TOKEN_IDS)
    try:
        x = torch.tensor([prefix_ids] * batch, dtype=torch.long,
                         device=model.device)
        mask = torch.ones_like(x)
        kwargs = dict(input_ids=x, attention_mask=mask,
                      max_new_tokens=max_new, do_sample=True,
                      temperature=TEMPERATURE, top_p=TOP_P,
                      eos_token_id=STOP_TOKEN_IDS, pad_token_id=PAD_TOKEN_ID)
        out = model.generate(**kwargs)
        del x
        rows = []
        for row in out.tolist():
            row = row[fork_abs:]
            for i, t in enumerate(row):
                if t in stop:
                    row = row[:i + 1]
                    break
            rows.append(row)
        return rows
    finally:
        gc_handle.remove()
        if handle is not None:
            handle.remove()


def run(args) -> None:
    plan = load_plan()
    if args.only:
        keep = set(args.only.split(","))
        plan = [p for p in plan if p["state_id"] in keep]
    if args.max_horizon:
        plan = [p for p in plan if p["horizon"] <= args.max_horizon]
    m = args.m or M_PER_ARM
    sub = args.sub_batch or SUB_BATCH
    out_path = Path(args.out)
    done: set[tuple] = set()
    if out_path.exists():
        for l in out_path.open():
            r = json.loads(l)
            done.add((r["state_id"], r["arm"], r["seed_i"]))
    total_rows = len(plan) * len(ARMS) * m
    print(f"states: {len(plan)} | arms: {len(ARMS)} | m: {m} | "
          f"rows target: {total_rows} | already done: {len(done)}",
          flush=True)

    dirs = np.load(DIRECTIONS)
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    # Load retries: with two concurrent processes per box, a weight-load
    # chunk can collide with the other process's prefill spike.
    # Backoff lets the spike pass.
    for attempt in range(3):
        try:
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_PATH, dtype=torch.bfloat16, device_map="cuda",
                attn_implementation="sdpa")
            break
        except torch.cuda.OutOfMemoryError:
            if attempt == 2:
                raise
            print(f"model load OOM (attempt {attempt + 1}/3), backing "
                  f"off {30 * (attempt + 1)}s for the co-process spike "
                  f"to pass", flush=True)
            torch.cuda.empty_cache()
            time.sleep(30 * (attempt + 1))
    model.eval()

    directions = {cls: torch.tensor(
        dirs[f"dir_{cls.lower()}_L{FOCAL[cls]}"], dtype=torch.float32,
        device=model.device).to(model.dtype) for cls in FOCAL}

    fout = out_path.open("a")
    rows_written = 0
    for si, st in enumerate(plan):
        prefix = build_prompt_ids(tok, st["prompt_text"], enable_thinking=True)
        c_ids = tok(st["completion"][:CAP_ABS * 4],
                    add_special_tokens=False)["input_ids"][:CAP_ABS]
        prefix_ids = (prefix + c_ids)[: st["fork_abs"]]
        assert len(prefix_ids) == st["fork_abs"], \
            f"prefix rebuild mismatch for {st['state_id']}"
        max_new = CAP_ABS - st["fork_abs"]
        d = directions[st["class"]]
        arm_deltas = {"noop": None,
                      "toward_healthy": st["lam"] * d,
                      "toward_diverge": -st["lam"] * d}
        for arm in ARMS:
            seeds = list(range(m))
            pending = [s for s in seeds if (st["state_id"], arm, s) not in done]
            if not pending:
                continue
            for bi in range(0, len(pending), sub):
                batch_seeds = pending[bi:bi + sub]
                bs = min(batch_for(st["fork_abs"], sub), len(batch_seeds))
                # batch_seed must be stable across processes/resumes —
                # str hash() is per-process randomized (PYTHONHASHSEED)
                batch_seed = (RUN_SEED_BASE
                              + int.from_bytes(hashlib.sha256(
                                    st["state_id"].encode()).digest()[:4],
                                    "big") % 100000 * 97 + bi)
                t0 = time.time()
                try:
                    seqs = gen_batch(model, arm_deltas[arm], st["layer"],
                                     prefix_ids, st["fork_abs"], max_new,
                                     batch_seed, bs)
                except torch.cuda.OutOfMemoryError:
                    if bs == 1:
                        # batch-1 OOM: nothing left to halve. The usual
                        # cause under two-process concurrency is the
                        # other process's prefill spike; backoff lets
                        # it pass. Loud raise if it persists — a silent
                        # skip would drop the row.
                        seqs = None
                        for wait_s in (25, 60):
                            print(f"batch-1 OOM at {st['state_id']} {arm}, "
                                  f"retrying after {wait_s}s", flush=True)
                            torch.cuda.empty_cache()
                            time.sleep(wait_s)
                            try:
                                seqs = gen_batch(model, arm_deltas[arm],
                                                 st["layer"], prefix_ids,
                                                 st["fork_abs"], max_new,
                                                 batch_seed, 1)
                                break
                            except torch.cuda.OutOfMemoryError:
                                continue
                        if seqs is None:
                            raise torch.cuda.OutOfMemoryError(
                                f"persistent OOM: {st['state_id']} {arm}")
                    else:
                        # one retry at half batch; if that also OOMs it raises
                        seqs = []
                        for half in (batch_seeds[:len(batch_seeds)//2],
                                       batch_seeds[len(batch_seeds)//2:]):
                            if half:
                                seqs += gen_batch(model, arm_deltas[arm],
                                                  st["layer"], prefix_ids,
                                                  st["fork_abs"], max_new,
                                                  batch_seed,
                                                  min(bs // 2, len(half)))
                # Release prefill blocks immediately, not at sub-batch
                # end: with two concurrent processes the co-process
                # must see ~17GB resident, not a cached spike, during
                # the decode phase.
                torch.cuda.empty_cache()
                for seed_i, new_ids in zip(batch_seeds, seqs):
                    text = tok.decode(new_ids, skip_special_tokens=False)
                    check = objective_check(
                        text, st["reference_answer"],
                        {"domain": st["domain"], "knobs": st.get("knobs")})
                    row = {
                        "state_id": st["state_id"], "class": st["class"],
                        "substrate": st["substrate"], "pid": st["pid"],
                        "sample_i": st["sample_i"], "arm": arm,
                        "lam": st["lam"], "layer": st["layer"],
                        "fork_token": st["fork_abs"],
                        "seed_i": seed_i, "seed_batch": batch_seed,
                        "n_new": len(new_ids),
                        "hit_cap": len(new_ids) >= max_new,
                        "correct": bool(check["correct"]),
                        "match_mode": check.get("match_mode"),
                        "extracted_answer_line": check.get(
                            "extracted_answer_line"),
                        "elapsed_s": round(time.time() - t0, 1),
                        "completion": text,
                    }
                    fout.write(json.dumps(row) + "\n")
                    fout.flush()
                    rows_written += 1
                dt = time.time() - t0
                n_tok = sum(len(s) for s in seqs)
                print(f"[{si+1}/{len(plan)}] {st['state_id']} {arm} "
                      f"seeds {batch_seeds[0]}-{batch_seeds[-1]}: "
                      f"{len(seqs)} cont, {n_tok} tok, {dt:.0f}s "
                      f"({n_tok/max(dt,1e-6):.0f} tok/s), "
                      f"done {len(done)+rows_written}/{total_rows}",
                      flush=True)
    fout.close()
    print(f"run complete: {rows_written} new rows -> {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--out", default=str(DATA / "fork_r4_results.jsonl"))
    ap.add_argument("--m", type=int, default=0, help="samples per arm (0=24)")
    ap.add_argument("--sub-batch", type=int, default=0)
    ap.add_argument("--max-horizon", type=int, default=0,
                    help="only states with horizon <= this (smoke)")
    ap.add_argument("--only", default="", help="comma-sep state_ids")
    args = ap.parse_args()
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        if args.plan:
            build_plan()
        elif args.run:
            run(args)
        else:
            ap.error("pass --plan or --run")


if __name__ == "__main__":
    main()
