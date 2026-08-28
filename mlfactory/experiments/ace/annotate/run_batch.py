"""Batch annotation pass over the xsub corpus — applies the locked v5
protocol (framing C) to every plan pair.

Resume-safe: a pair is done when out/<pass>/<pair_id>/C.json exists.
Consolidation (--consolidate-only) is CPU-only and re-runnable: it
re-reads the raw outputs and writes one row per flag to
data/annotations_xsub_<pass>.jsonl with resolved char offsets.

Usage (from repo root):
    .venv/bin/python -m mlfactory.experiments.ace.annotate.run_batch \
        --pass pass1 [--workers 6] [--limit N]
    .venv/bin/python -m mlfactory.experiments.ace.annotate.run_batch \
        --pass pass2                # double-annotation subset only
    .venv/bin/python -m mlfactory.experiments.ace.annotate.run_batch \
        --pass pass1 --consolidate-only
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from mlfactory.core.secrets import SecretsStore
from mlfactory.experiments.ace.annotate.pilot import run_pilot as P

HERE = Path(__file__).resolve().parent
ACE = HERE.parent
DATA = ACE / "data"
DEFAULT_CORPUS_FILES = [
    DATA / "xsub_q8.jsonl",
    DATA / "xsub_bf16_gpu0.jsonl",
    DATA / "xsub_bf16_gpu1.jsonl",
]


def load_corpus(files: list[Path]) -> dict[tuple, dict]:
    rows = {}
    for f in files:
        for line in f.open():
            r = json.loads(line)
            sub = "q8" if str(r.get("quant", "")).startswith("Q8") else "bf16"
            rows[(sub, r["proposal_id"], r["sample_i"])] = r
    return rows


def load_prompt_framings(prompt_version: str) -> tuple[str, dict[str, str]]:
    pv = (HERE / "pilot" / f"prompt_{prompt_version}.md").read_text()
    system = pv.split("## SYSTEM (all framings)")[1].split("## USER")[0].strip()
    fr = {}
    for name in ("A (single trace)", "C (pair, compare & contrast)"):
        fr[name[0]] = pv.split(f"## USER — framing {name}")[1]
        fr[name[0]] = fr[name[0]].split("## USER")[0] if "## USER" in fr[name[0]] else fr[name[0]]
    return system, fr


def substitute(template: str, subs: dict[str, str]) -> str:
    out = template
    for k, v in subs.items():
        out = out.replace("{" + k + "}", str(v))
    assert "{surface_question}" not in out and "{trace_1}" not in out \
        and "{trace_2}" not in out, "unsubstituted placeholder in prompt"
    return out


def call_with_retry(key: str, model: str, msgs: list[dict]) -> dict:
    """Retry policy for overnight unattended runs: 429 gets up to 4 tries
    with exponential backoff (rate limits clear); anything else retries
    once (transient gateway/timeout noise)."""
    attempt = 0
    while True:
        try:
            return P.call_api(key, model, msgs)
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < 4:
                wait = 60 * (2 ** attempt)
                print(f"    429 rate-limited; backing off {wait}s "
                      f"(attempt {attempt + 1}/4)", flush=True)
                time.sleep(wait)
                attempt += 1
                continue
            if attempt < 1:
                print(f"    HTTP {e.code}; retrying once", flush=True)
                time.sleep(10)
                attempt += 1
                continue
            raise RuntimeError(f"call failed: HTTP {e.code}") from e
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            if attempt < 1:
                print(f"    api error: {e}; retrying once", flush=True)
                time.sleep(10)
                attempt += 1
                continue
            raise RuntimeError(f"call failed: {e}") from e


def annotate_one(key: str, model: str, framing: str, system: str, template: str,
                 subs: dict, labels: dict[int, str], texts: dict[int, str],
                 pair_id: str, outdir: Path, filename: str) -> str:
    """Run one framing; write <filename>.json + digest; return summary line."""
    user = substitute(template, subs)
    msgs = P.build_messages(framing, system, user)
    t0 = time.time()
    resp = call_with_retry(key, model, msgs)
    dt = time.time() - t0
    text = resp["choices"][0]["message"].get("content") or ""
    finish = resp["choices"][0].get("finish_reason")
    meta = {"model": model, "framing": framing, "pair_id": pair_id,
            "trace_labels": {str(k): v for k, v in labels.items()},
            "dt_s": dt, "finish_reason": finish,
            "out_tokens": resp.get("usage", {}).get("completion_tokens"),
            "in_tokens": resp.get("usage", {}).get("prompt_tokens")}
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / f"{filename}.json").write_text(json.dumps({"meta": meta, "text": text}, indent=2))
    flags = P.parse_flags(text, texts, labels)
    (outdir / f"{filename}.digest.txt").write_text(P.render_digest(pair_id, meta, flags))
    n = sum(1 for f in flags if f["class"] != "NONE")
    res = sum(1 for f in flags if f["class"] != "NONE" and f.get("span_chars") is not None)
    return f"{pair_id}/{framing}: flags={n} resolved={res} dt={dt:.0f}s finish={finish}"


def run_pair(pair: dict, corpus: dict, system: str, fr: dict, key: str,
             model: str, outdir: Path) -> str:
    (sa, sb) = (pair["samples"][0], pair["samples"][1])
    ra = corpus[(pair["substrate"], pair["pid"], sa["sample_i"])]
    rb = corpus[(pair["substrate"], pair["pid"], sb["sample_i"])]
    labels = {1: f"s{sa['sample_i']}", 2: f"s{sb['sample_i']}"}
    texts = {1: ra["completion"], 2: rb["completion"]}
    outcomes = {labels[1]: sa["outcome"], labels[2]: sb["outcome"]}

    if (outdir / "C.json").exists():
        return f"{pair['pair_id']}: already done"
    msgs_summary = annotate_one(
        key, model, "C", system, fr["C"],
        {"surface_question": ra["surface_question"],
         "trace_1": ra["completion"], "trace_2": rb["completion"]},
        labels, texts, pair["pair_id"], outdir, "C")

    # A-fallback: if C produced no usable flags (empty content or hit the
    # thinking wall), annotate each trace alone.
    c_text = json.loads((outdir / "C.json").read_text())
    c_finish = c_text["meta"]["finish_reason"]
    c_flags = P.parse_flags(c_text["text"], texts, labels)
    if not c_text["text"].strip() or (c_finish == "length" and not c_flags):
        for tag, (li, row) in (("A1", (1, ra)), ("A2", (2, rb))):
            a_labels = {1: labels[li]}
            a_texts = {1: row["completion"]}
            annotate_one(
                key, model, "A", system, fr["A"],
                {"surface_question": row["surface_question"],
                 "trace_1": row["completion"]},
                a_labels, a_texts, pair["pair_id"], outdir, tag)
        msgs_summary += f" | A-fallback (C finish={c_finish})"
    # record outcomes for consolidation
    (outdir / "outcomes.json").write_text(json.dumps(outcomes))
    return msgs_summary


def consolidate(pass_name: str, tag: str,
                corpus: dict[tuple, dict]) -> Path:
    base = HERE / "out" / f"{tag}_{pass_name}"
    out_path = DATA / f"annotations_{tag}_{pass_name}.jsonl"
    rows = []
    for d in sorted(base.iterdir()):
        meta_files = sorted(d.glob("[AC]*.json"))
        meta_files = [f for f in meta_files if f.name != "outcomes.json"]
        for mf in meta_files:
            obj = json.loads(mf.read_text())
            meta, text = obj["meta"], obj["text"]
            labels = {int(k): v for k, v in meta["trace_labels"].items()}
            sub = meta["pair_id"].split("_p")[0][1:]
            pid = int(meta["pair_id"].split("_p")[1].split("_")[0])
            texts = {k: corpus[(sub, pid, int(v[1:]))]["completion"]
                     for k, v in labels.items()}
            outcomes = json.loads((d / "outcomes.json").read_text()) \
                if (d / "outcomes.json").exists() else {}
            flags = P.parse_flags(text, texts, labels)
            for f in flags:
                if f["class"] == "NONE":
                    continue
                tnum = int(f["trace"])
                rows.append({
                    "pass": pass_name, "pair_id": meta["pair_id"],
                    "framing": meta["framing"], "substrate": sub,
                    "pid": pid, "sample_i": int(labels[tnum][1:]),
                    "trace_label": labels[tnum],
                    "outcome": outcomes.get(labels[tnum]),
                    "class": f["class"], "conf": f.get("conf"),
                    "span_res": f.get("span_res", "UNRESOLVED"),
                    "start_char": f.get("span_s0") if f.get("span_s0", -1) >= 0 else None,
                    "end_char": f.get("span_e1") if f.get("span_s0", -1) >= 0 else None,
                    "span_chars": f.get("span_chars"),
                    "start_quote": f.get("start"), "end_quote": f.get("end"),
                    "basis": f.get("basis"),
                })
    with out_path.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pass", dest="pass_name", default="pass1")
    ap.add_argument("--tag", default="xsub",
                    help="corpus tag for out-dir and output filenames")
    ap.add_argument("--plan", default=str(DATA / "annotation_plan_xsub.jsonl"))
    ap.add_argument("--corpus", nargs="*", default=None,
                    help="corpus jsonl files (default: the xsub files)")
    ap.add_argument("--prompt-version", default="v5")
    ap.add_argument("--model", default="glm-5.2-vision")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0, help="smoke: first N pairs only")
    ap.add_argument("--consolidate-only", action="store_true")
    args = ap.parse_args()

    corpus = load_corpus([Path(f) for f in args.corpus]
                         if args.corpus else DEFAULT_CORPUS_FILES)

    if args.consolidate_only:
        out = consolidate(args.pass_name, args.tag, corpus)
        print(f"consolidated -> {out}")
        return

    plan = [json.loads(l) for l in Path(args.plan).open()]
    if args.pass_name != "pass1":
        plan = [p for p in plan if p["double"]]
    system, fr = load_prompt_framings(args.prompt_version)
    key = SecretsStore(P.REPO / ".mlfactory" / "secrets.yaml").get("LUNAROUTE_API_KEY")
    if not key:
        raise SystemExit("LUNAROUTE_API_KEY not found")

    base = HERE / "out" / f"{args.tag}_{args.pass_name}"
    todo = [p for p in plan
            if not (base / p["pair_id"] / "C.json").exists()]
    if args.limit:
        todo = todo[:args.limit]
    print(f"[{args.pass_name}] plan={len(plan)} todo={len(todo)} "
          f"(already done: {len(plan) - len(todo) if not args.limit else '?'})", flush=True)

    def work(pair):
        try:
            return run_pair(pair, corpus, system, fr, key, args.model,
                            base / pair["pair_id"])
        except Exception as e:  # keep the pool alive; resume covers it
            return f"{pair['pair_id']}: FAILED {type(e).__name__}: {e}"

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(work, p): p for p in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            print(f"[{i}/{len(todo)}] {fut.result()}", flush=True)

    out = consolidate(args.pass_name, args.tag, corpus)
    n = sum(1 for _ in out.open())
    print(f"consolidated {n} flag rows -> {out}")


if __name__ == "__main__":
    main()
