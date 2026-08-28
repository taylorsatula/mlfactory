"""R1 capture: teacher-forced activations at annotated span positions.

For every trace with resolved annotations in the given annotations file
(default: data/annotations_xsub_pass1.jsonl):
  * rebuild the served stream (solver_prompt from candidate records +
    build_prompt_ids, bit-verified against recorded n_prompt_tokens),
  * teacher-force the completion through the local HF bf16 model in chunks
    (DynamicCache continuation, use_cache=True — single-token cached decode
    crashes in this transformers build, so chunks are merged up to >= 2 tokens),
  * hook every decoder layer and gather the residual stream rows at the
    requested token positions,
  * snapshot the DeltaNet recurrent state of the REC_LAYERS subset at each
    chunk boundary that matches a requested position,
  * save one .npz per trace under the capture dir plus a manifest.

Capture dirs are per-corpus-tag: --tag xsub (default) writes to
data/annot_captures/, any other tag to data/annot_captures_<tag>/. Corpora
with overlapping (pid, sample_i) keys MUST use different tags — npz
filenames would collide and the resume guard would silently mix traces.

Position model (per annotation): pre_onset (token before span start),
onset (first token of span), mid, end (last token of span), lookback
positions lb_<k> at k tokens before onset (k in LOOKBACK_KS, skipped if
the token falls inside any annotated span), plus depth-matched control
positions outside all annotated spans — four per (annotation, anchor
decile), where anchor deciles are the onset decile plus the deciles of
that annotation's retained lookback positions. Controls carry an
"anchor" field ("onset" vs "lb"); the recurrent-state channel keeps
onset + onset-anchor controls only (lookback is a residual-channel
analysis).

All positions are completion-relative token indices; absolute token index
= n_prompt + pos.

Run:
    CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    .venv/bin/python -m mlfactory.experiments.ace.annotate.capture_activations \
    [--annotations FILE] [--corpus F1 F2 ...] [--candidates F1 ...] [--tag TAG] [--limit N]
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

from mlfactory.experiments.ace.core.steering_controller import MODEL_PATH, build_prompt_ids
from mlfactory.experiments.ace.frontier.collect_rollouts import solver_prompt

HERE = Path(__file__).resolve().parent
ACE = HERE.parent
DATA = ACE / "data"
DEFAULT_ANNOTATIONS = DATA / "annotations_xsub_pass1.jsonl"
DEFAULT_CANDIDATES = [DATA / "xsub_candidates.jsonl"]

DEFAULT_CORPUS_FILES = [
    DATA / "xsub_q8.jsonl",
    DATA / "xsub_bf16_gpu0.jsonl",
    DATA / "xsub_bf16_gpu1.jsonl",
]

N_LAYERS = 32
LINEAR_LAYERS = [i for i in range(N_LAYERS) if i % 4 != 3]
# Layers with strongest b1-map separability (LAYER_HYPOTHESES.md):
# rec_2 +0.759, rec_12 +0.635, rec_9 +0.603, rec_20 -0.686, rec_8 -0.574
REC_LAYERS = [2, 8, 9, 12, 20]
LOOKBACK_KS = (2, 4, 8, 16, 32, 64)
CONTROLS_PER_ANNOTATION = 4
MIN_CHUNK = 2
MAX_TOKENS = 26000  # backstop cap from OPERATIONS.md


def substrate_of(row: dict) -> str:
    return "q8" if str(row.get("quant", "")).startswith("Q8") else "bf16"


def load_corpus(files: list[Path]) -> dict[tuple, dict]:
    rows = {}
    for path in files:
        for line in path.open():
            r = json.loads(line)
            rows[(substrate_of(r), r["proposal_id"], r["sample_i"])] = r
    return rows


def load_candidates(files: list[Path]) -> dict[str, dict]:
    out = {}
    for path in files:
        for line in path.open():
            c = json.loads(line)
            out[c["surface_hash"]] = c
    return out


def load_annotations(path: Path) -> dict[tuple, list[dict]]:
    by_trace = defaultdict(list)
    for line in path.open():
        a = json.loads(line)
        if a.get("start_char") is None or a.get("end_char") is None:
            continue  # unresolved quotes: keep in JSONL for manual review
        key = (a["substrate"], a["pid"], a["sample_i"])
        by_trace[key].append(a)
    return by_trace


def char_to_token(offsets: list[tuple[int, int]], pos: int, find_ge: bool) -> int:
    """Map a char offset to a completion-relative token index.

    find_ge=True: first token whose span covers or follows pos (span starts).
    find_ge=False: last token whose span starts at or before pos (span ends).
    Returns -1 if no token matches.
    """
    if find_ge:
        for i, (s, e) in enumerate(offsets):
            if s == e:  # special token
                continue
            if s >= pos:
                return i
            if s < pos < e:
                return i
        return -1
    last = -1
    for i, (s, e) in enumerate(offsets):
        if s == e:
            continue
        if s <= pos:
            last = i
        else:
            break
    return last


def decile_of(t: int, n_tokens: int) -> int:
    return min(t * 10 // max(n_tokens, 1), 9)


def build_positions(annotations: list[dict], offsets: list[tuple[int, int]],
                    n_tokens: int, rng) -> tuple[list[dict], list[int]]:
    """Resolve annotation char spans to token positions; add lookback
    positions and depth-matched controls.

    Core kinds (pre_onset/onset/mid/end) are recorded first so they win
    pos_table priority on token collisions; lookback lb_<k> positions are
    added second, skipping tokens inside any annotated span or already
    used. Controls: four per (annotation, anchor decile) — the onset
    decile plus the deciles of the annotation's retained lookback
    positions — outside all spans, tagged anchor="onset" or "lb".

    Returns (position_records, sorted unique token positions to capture).
    """
    span_tok_ranges = []
    span_by_ann: dict[int, tuple[int, int]] = {}
    for idx, a in enumerate(annotations):
        t_start = char_to_token(offsets, a["start_char"], find_ge=True)
        t_end = char_to_token(offsets, a["end_char"], find_ge=False)
        if t_start < 0 or t_end < 0 or t_end < t_start:
            continue
        span_tok_ranges.append((t_start, t_end))
        span_by_ann[idx] = (t_start, t_end)

    if not span_tok_ranges:
        return [], []

    def in_span(t: int) -> bool:
        return any(s <= t <= e for s, e in span_tok_ranges)

    records = []
    used_tokens = set()
    for idx, (t_start, t_end) in span_by_ann.items():
        a = annotations[idx]
        t_mid = (t_start + t_end) // 2
        for kind, t in (("pre_onset", t_start - 1), ("onset", t_start),
                        ("mid", t_mid), ("end", t_end)):
            if 0 <= t < n_tokens:
                records.append({"kind": kind, "ann_idx": idx, "token": t,
                                "class": a["class"], "conf": a["conf"],
                                "decile": decile_of(t, n_tokens)})
                used_tokens.add(t)

    lookback_tokens: dict[int, list[int]] = {}
    for idx, (t_start, _t_end) in span_by_ann.items():
        a = annotations[idx]
        for k in LOOKBACK_KS:
            t = t_start - k
            if t < 0 or t >= n_tokens or in_span(t) or t in used_tokens:
                continue
            records.append({"kind": f"lb_{k}", "ann_idx": idx, "token": t,
                            "class": a["class"], "conf": a["conf"],
                            "decile": decile_of(t, n_tokens)})
            lookback_tokens.setdefault(idx, []).append(t)
            used_tokens.add(t)

    # depth-matched controls: same decile as the anchor position, outside
    # all spans and all used tokens, each control tied to its annotation
    for idx, (t_start, _t_end) in span_by_ann.items():
        onset_dec = decile_of(t_start, n_tokens)
        anchors = {onset_dec} | {decile_of(t, n_tokens)
                                 for t in lookback_tokens.get(idx, [])}
        for d in sorted(anchors):
            lo = d * n_tokens // 10
            hi = (d + 1) * n_tokens // 10
            candidates = [c for c in range(max(lo, 1), hi)
                          if not in_span(c) and c not in used_tokens]
            if not candidates:
                continue
            picks = rng.choice(len(candidates),
                               size=min(CONTROLS_PER_ANNOTATION, len(candidates)),
                               replace=False)
            for p in picks:
                c_tok = candidates[int(p)]
                records.append({"kind": "control", "ann_idx": idx,
                                "token": c_tok, "class": "control",
                                "conf": "clear", "decile": d,
                                "anchor": "onset" if d == onset_dec else "lb"})
                used_tokens.add(c_tok)

    uniq = sorted({r["token"] for r in records})
    return records, uniq


@torch.no_grad()
def capture_trace(model, tok, row: dict, cand: dict, annotations: list[dict],
                  rng) -> dict | None:
    prompt = solver_prompt({**cand, **cand.get("provenance", {})})
    p_ids = build_prompt_ids(tok, prompt, enable_thinking=True)
    n_prompt = len(p_ids)

    comp = row["completion"][:MAX_TOKENS * 4]  # char pre-trim; token cap below
    c_enc = tok(comp, add_special_tokens=False, return_offsets_mapping=True)
    offsets = c_enc["offset_mapping"]
    c_ids = c_enc["input_ids"][:MAX_TOKENS]
    offsets = offsets[: len(c_ids)]
    n_comp = len(c_ids)

    records, positions = build_positions(annotations, offsets, n_comp, rng)
    if not positions:
        return None

    abs_positions = [n_prompt + p for p in positions]
    pos_index = {t: i for i, t in enumerate(abs_positions)}

    ids = p_ids + c_ids
    x = torch.tensor([ids], dtype=torch.long, device=model.device)
    n_total = len(ids)

    # Chunk boundaries: at every capture position (state snapshot BEFORE the
    # token at that position is processed) and at the end.
    bounds = sorted(set(abs_positions))
    chunks = []
    prev = 0
    last_b = None
    for b in bounds + [n_total]:
        if last_b is not None and b - prev < MIN_CHUNK:
            last_b = b
            continue
        if b > prev:
            chunks.append((prev, b))
            prev = b
        last_b = b
    if prev < n_total:
        chunks.append((prev, n_total))
    # ensure final chunk reaches n_total
    if chunks[-1][1] != n_total:
        chunks[-1] = (chunks[-1][0], n_total)

    residuals = {}
    rec_states = {i: {} for i in REC_LAYERS}

    def make_resid_hook(layer_idx: int):
        def hook(_m, _inp, out):
            h = out[0] if isinstance(out, tuple) else out
            rows = []
            keep = []
            chunk_start = hook_state["start"]
            for i, t in enumerate(abs_positions):
                local = t - chunk_start
                if 0 <= local < h.shape[1]:
                    rows.append(h[0, local].detach().float().cpu())
                    keep.append(i)
            if rows:
                residuals.setdefault(layer_idx, {})[chunk_start] = (keep, rows)
        return hook

    def make_rec_hook(layer_idx: int):
        def hook(_m, _inp, _out):
            # fires after the layer processed its chunk; state now reflects
            # tokens up to hook_state["end"]
            try:
                st = current_cache.layers[layer_idx].recurrent_states[0]
            except (AttributeError, IndexError):
                return
            rec_states[layer_idx][hook_state["end"]] = st.detach().cpu().numpy().astype(np.float16)
        return hook

    hook_state = {"start": 0, "end": 0}
    handles = [model.model.layers[i].register_forward_hook(make_resid_hook(i))
               for i in range(N_LAYERS)]
    handles += [model.model.layers[i].register_forward_hook(make_rec_hook(i))
                for i in REC_LAYERS]

    current_cache = DynamicCache(config=model.config)
    try:
        for (a, b) in chunks:
            hook_state["start"] = a
            hook_state["end"] = b
            model.model(input_ids=x[:, a:b], past_key_values=current_cache,
                        use_cache=True)
        del current_cache
    finally:
        for h in handles:
            h.remove()

    # assemble residuals: (n_layers, n_positions, hidden) fp16
    n_pos = len(abs_positions)
    hidden = None
    resid_arr = np.zeros((N_LAYERS, n_pos, 0), dtype=np.float16)
    layers_out = []
    for li in range(N_LAYERS):
        rows_gathered = [None] * n_pos
        for _chunk_start, (keep, rows) in residuals.get(li, {}).items():
            for slot, row_vec in zip(keep, rows):
                rows_gathered[slot] = row_vec
        missing = [i for i, v in enumerate(rows_gathered) if v is None]
        if missing:
            print(f"  WARN layer {li}: missing positions {missing[:5]}...")
            continue
        layers_out.append(torch.stack(rows_gathered).numpy().astype(np.float16))
    if len(layers_out) != N_LAYERS:
        print(f"  WARN: only {len(layers_out)}/{N_LAYERS} layers complete")
    resid_arr = np.stack(layers_out, axis=0) if layers_out else resid_arr

    # per-position metadata
    pos_table = []
    for i, t in enumerate(abs_positions):
        r = [rec for rec in records if rec["token"] == t - n_prompt]
        rec0 = r[0] if r else {}
        pos_table.append({
            "pos_idx": i, "token_abs": t, "token_comp": t - n_prompt,
            "kind": rec0.get("kind"), "class": rec0.get("class"),
            "conf": rec0.get("conf"), "ann_idx": rec0.get("ann_idx"),
            "decile": rec0.get("decile"), "anchor": rec0.get("anchor"),
        })

    # recurrent states: onset + onset-anchor controls (lookback analysis is
    # residual-channel only; missing anchor = pre-lookback capture = onset)
    rec_keep = [i for i, p in enumerate(pos_table)
                if p["kind"] == "onset"
                or (p["kind"] == "control"
                    and p.get("anchor", "onset") == "onset")]
    rec_arrays = {}
    for li in REC_LAYERS:
        snaps = rec_states[li]
        if snaps:
            state_shape = tuple(next(iter(snaps.values())).shape[1:])
            arr = np.zeros((len(rec_keep),) + state_shape, dtype=np.float16)
            snap_bounds = sorted(snaps.keys())
            for row, i in enumerate(rec_keep):
                t = abs_positions[i]
                # snapshot at boundary b == state before token b; take the
                # latest snapshot at or before the position's token
                best = None
                for b in snap_bounds:
                    if b <= t:
                        best = b
                if best is not None:
                    arr[row] = snaps[best].astype(np.float16)
        else:
            arr = np.zeros((len(rec_keep), 0), dtype=np.float16)
        rec_arrays[f"rec_L{li}"] = arr

    return {
        "residuals": resid_arr,
        "positions": np.array(abs_positions, dtype=np.int32),
        "pos_table": pos_table,
        "rec_states": rec_arrays,
        "rec_pos_idx": np.array(rec_keep, dtype=np.int32),
        "n_prompt": n_prompt,
        "n_comp": n_comp,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only", default="", help="comma-sep keys: q8:140:7")
    ap.add_argument("--annotations", default=str(DEFAULT_ANNOTATIONS))
    ap.add_argument("--corpus", nargs="+", default=None,
                    help="rollout jsonl files (default: the xsub files)")
    ap.add_argument("--candidates", nargs="+", default=None,
                    help="candidate jsonl files (default: xsub_candidates.jsonl)")
    ap.add_argument("--tag", default="xsub",
                    help="capture dir: annot_captures (xsub) or annot_captures_<tag>. "
                         "Corpora with overlapping (pid, sample_i) MUST use "
                         "different tags — npz names would collide.")
    args = ap.parse_args()

    OUT_DIR = DATA / ("annot_captures" if args.tag == "xsub"
                      else f"annot_captures_{args.tag}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    corpus = load_corpus([Path(f) for f in args.corpus]
                         if args.corpus else DEFAULT_CORPUS_FILES)
    cands = load_candidates([Path(f) for f in args.candidates]
                            if args.candidates else DEFAULT_CANDIDATES)
    annots = load_annotations(Path(args.annotations))

    todo = sorted(annots.keys())
    if args.only:
        keep = set()
        for spec in args.only.split(","):
            sub, pid, sid = spec.split(":")
            keep.add((sub, int(pid), int(sid)))
        todo = [k for k in todo if k in keep]
    if args.limit:
        todo = todo[: args.limit]

    done = [k for k in todo if (OUT_DIR / f"{k[0]}_p{k[1]}_s{k[2]}.npz").exists()]
    todo = [k for k in todo if k not in done]
    print(f"traces with resolved annotations: {len(annots)}; "
          f"already captured: {len(done)}; todo: {len(todo)}", flush=True)
    if not todo:
        return

    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=torch.bfloat16, device_map="cuda",
        attn_implementation="sdpa")
    model.eval()

    manifest = []
    for j, key in enumerate(todo):
        sub, pid, sid = key
        row = corpus[key]
        cand = cands.get(row["surface_hash"])
        if cand is None:
            print(f"  SKIP {key}: no candidate record", flush=True)
            continue
        rng = np.random.default_rng(1000 + pid * 17 + sid)
        t0 = time.time()
        try:
            out = capture_trace(model, tok, row, cand, annots[key], rng)
        except Exception as e:  # noqa: BLE001
            print(f"  FAIL {key}: {type(e).__name__}: {e}", flush=True)
            torch.cuda.empty_cache()
            continue
        if out is None:
            print(f"  SKIP {key}: no resolvable positions", flush=True)
            continue
        fname = OUT_DIR / f"{sub}_p{pid}_s{sid}.npz"
        np.savez(fname,
                 residuals=out["residuals"],
                 positions=out["positions"],
                 n_prompt=np.int32(out["n_prompt"]),
                 n_comp=np.int32(out["n_comp"]),
                 pos_table=json.dumps(out["pos_table"]).encode(),
                 rec_pos_idx=out["rec_pos_idx"],
                 **out["rec_states"])
        manifest.append({
            "substrate": sub, "pid": pid, "sample_i": sid,
            "file": fname.name, "n_positions": len(out["positions"]),
            "n_annotations": len(annots[key]),
            "outcome": row["correct"] and "correct" or (
                "cap" if row["truncated"] else "wrong"),
            "n_prompt": out["n_prompt"], "n_comp": out["n_comp"],
            "elapsed_s": round(time.time() - t0, 1),
        })
        print(f"[{j+1}/{len(todo)}] {sub} p{pid} s{sid}: "
              f"{len(out['positions'])} positions, "
              f"{time.time()-t0:.1f}s", flush=True)
        del out
        torch.cuda.empty_cache()

    mfile = OUT_DIR / "capture_manifest.jsonl"
    with mfile.open("a") as f:
        for m in manifest:
            f.write(json.dumps(m) + "\n")
    print(f"wrote {len(manifest)} capture files; manifest: {mfile}")


if __name__ == "__main__":
    main()
