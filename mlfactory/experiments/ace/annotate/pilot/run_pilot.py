"""Annotation pilot runner — hill-climb harness for probe-point prompts.

Sends xsub traces to a Lunaroute annotator under framing variants
(A: single trace, B: unlabeled pair, C: compare-&-contrast pair),
parses ⟦-anchored flag lines, auto-resolves quoted span boundaries
against the trace text, and prints/saves a digest so the operator reads
only flagged points, never whole traces.

Usage:
    .venv/bin/python -m mlfactory.experiments.ace.annotate.pilot.run_pilot \
        --prompt-version v1 --model glm-5.2-vision \
        --pid 140 --success 7 --fail 1
"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from mlfactory.core.secrets import SecretsStore

HERE = Path(__file__).resolve().parent
ACE = HERE.parent.parent  # .../experiments/ace
REPO = ACE.parent.parent.parent  # mlfactory repo root (cwd-independent)
ANCHOR = "\u27e6"  # ⟦
BASE_URL = "https://gw.lunaroute.com/v1"
SHUFFLE_SEED = 42  # fixed: pair order is stable across prompt versions

# a flag may wrap across lines or be concatenated without a newline;
# split on the class head itself, wherever it occurs
BLOCK_RE = re.compile(r"(?=⟦(?:MUSE|CYCLE|LOOP|NONE)⟧)")
FLAG_HEAD_RE = re.compile(r"^⟦(MUSE|CYCLE|LOOP|NONE)⟧")
FIELD_RE = {
    "trace": re.compile(r"trace=(\d+)"),
    "conf": re.compile(r"conf=(clear|probable)"),
    "start": re.compile(r'start="((?:[^"\\]|\\.)*)"'),
    "end": re.compile(r'end="((?:[^"\\]|\\.)*)"'),
    "basis": re.compile(r"basis=(.*?)(?=\s+[a-z_]+=\S|\s*$)"),
    "reason": re.compile(r"reason=(.*)"),
}


def load_row(quant: str, pid: int, sample_i: int) -> dict:
    path = ACE / "data" / "xsub_q8.jsonl" if quant == "Q8_0-MTP" else None
    assert path and path.exists(), f"no data file for {quant}"
    for line in path.open():
        r = json.loads(line)
        if int(r["proposal_id"]) == pid and int(r["sample_i"]) == sample_i and r["quant"] == quant:
            return r
    raise KeyError((quant, pid, sample_i))


def build_messages(framing: str, system: str, user: str) -> list[dict]:
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def call_api(key: str, model: str, messages: list[dict], max_tokens: int = 65536) -> dict:
    # No temperature override: Lunaroute models run at their designed default.
    # Large max_tokens: GLM thinks long; the flags must not be truncated.
    body = {"model": model, "messages": messages, "max_tokens": max_tokens}
    req = urllib.request.Request(
        f"{BASE_URL}/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=1800) as resp:
        return json.load(resp)


def _norm_map(text: str) -> tuple[str, list[int]]:
    """Whitespace-collapsed text + map from collapsed index to original."""
    chars, pos = [], []
    for idx, ch in enumerate(text):
        if not ch.isspace():
            chars.append(ch)
            pos.append(idx)
    return "".join(chars), pos


def resolve_span(text: str, start_q: str, end_q: str) -> tuple[str, int, int]:
    """Resolve a span's quotes as a pair: the end quote must occur after
    the start quote. Repetitive traces make single quotes ambiguous; the
    pair ordering usually picks the right occurrence. Exact match first,
    whitespace-normalized as fallback (marked OK-NORM). Paraphrase is NOT
    rescued — an unresolved quote is a real annotation-quality signal.
    """
    sq, eq = (start_q or "").strip(), (end_q or "").strip()
    if not sq or not eq:
        return "UNRESOLVED", -1, -1
    # exact pass: try start hits in order, pair with first end after it
    i = text.find(sq)
    while i != -1:
        j = text.find(eq, i + len(sq))
        if j != -1:
            return "OK", i, j + len(eq)
        i = text.find(sq, i + 1)
    # normalized pass
    norm_t, pos = _norm_map(text)
    nsq, neq = "".join(sq.split()), "".join(eq.split())
    i = norm_t.find(nsq)
    while i != -1:
        j = norm_t.find(neq, i + len(nsq))
        if j != -1:
            return "OK-NORM", pos[i], pos[j + len(neq) - 1] + 1
        i = norm_t.find(nsq, i + 1)
    return "UNRESOLVED", -1, -1


def parse_flags(text: str, trace_texts: dict[int, str], trace_labels: dict[int, str]) -> list[dict]:
    """Parse ⟦-anchored flag blocks (a flag may wrap across lines)."""
    out = []
    blocks = [b.strip() for b in BLOCK_RE.split(text) if b.strip().startswith("⟦")]
    for block in blocks:
        hm = FLAG_HEAD_RE.match(block)
        if not hm:
            continue
        cls = hm.group(1)
        row = {"class": cls, "raw": block[:400]}
        for name, rx in FIELD_RE.items():
            fm = rx.search(block)
            row[name] = fm.group(1).strip() if fm else None
        tnum = int(row["trace"]) if row["trace"] else 1
        row["trace_which"] = trace_labels.get(tnum, "?")
        if cls != "NONE":
            t = trace_texts.get(tnum, "")
            stat, s0, e1 = resolve_span(t, row["start"], row["end"])
            row["span_res"] = stat
            if stat.startswith("OK") and e1 > s0:
                row["span_chars"] = e1 - s0
                span = t[s0:e1]
                row["span_head"] = span[:80].replace("\n", " ")
                row["span_tail"] = span[-80:].replace("\n", " ")
            else:
                row["span_chars"] = None
        out.append(row)
    return out


def render_digest(task: str, meta: dict, flags: list[dict]) -> str:
    L = [f"===== {task} =====", f"model={meta['model']} dt_s={meta['dt_s']:.1f} "
         f"out_tokens={meta.get('out_tokens', '?')} finish={meta.get('finish_reason')}", ""]
    if not flags:
        if meta.get("finish_reason") == "length":
            L.append("!! NO FLAGS: thinking ran to the token budget (finish_reason=length)")
        else:
            L.append("!! NO ⟦-ANCHORED FLAG LINES FOUND — prompt format failure")
        return "\n".join(L)
    n_flag = sum(1 for f in flags if f["class"] != "NONE")
    resolved = sum(1 for f in flags if f["class"] != "NONE" and f.get("span_chars") is not None)
    L.append(f"flags={n_flag} fully-resolved={resolved} "
             + " ".join(f"{k}={sum(1 for f in flags if f['class']==k)}"
                        for k in ('MUSE', 'CYCLE', 'LOOP', 'NONE')))
    for f in flags:
        if f["class"] == "NONE":
            L.append(f"  {ANCHOR}NONE trace={f['trace']} ({f['trace_which']}) reason={f.get('reason')}")
            continue
        ok = f.get("span_chars") is not None
        stat = f.get("span_res", "UNRESOLVED") if not ok else ("RESOLVED" if f.get("span_res") == "OK" else "RESOLVED-NORM")
        L.append(f"  {ANCHOR}{f['class']} trace={f['trace']} ({f['trace_which']}) conf={f.get('conf')} [{stat}]"
                 + (f" span={f['span_chars']}ch" if f.get("span_chars") else ""))
        if ok:
            L.append(f"    head: {f['span_head']}")
            L.append(f"    tail: {f['span_tail']}")
        else:
            L.append(f"    start-quote: {(f.get('start') or '')[:100]}")
            L.append(f"    end-quote:   {(f.get('end') or '')[:100]}")
        L.append(f"    basis: {(f.get('basis') or '')[:200]}")
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt-version", default="v1")
    ap.add_argument("--model", default="glm-5.2-vision")
    ap.add_argument("--quant", default="Q8_0-MTP")
    ap.add_argument("--pid", type=int, default=140)
    ap.add_argument("--success", type=int, default=7)
    ap.add_argument("--fail", type=int, default=1)
    ap.add_argument("--only", default="", help="comma-sep subset: A1,A2,B,C")
    args = ap.parse_args()

    pv = Path(HERE / f"prompt_{args.prompt_version}.md").read_text()
    system = pv.split("## SYSTEM (all framings)")[1].split("## USER")[0].strip()
    fr = {}
    for name in ("A (single trace)", "B (pair, unlabeled)", "C (pair, compare & contrast)"):
        fr[name[0]] = pv.split(f"## USER — framing {name}")[1]
        fr[name[0]] = fr[name[0]].split("## USER")[0] if "## USER" in fr[name[0]] else fr[name[0]]

    succ = load_row(args.quant, args.pid, args.success)
    fail = load_row(args.quant, args.pid, args.fail)
    q = succ["surface_question"]

    # fixed shuffle: which physical sample is TRACE 1 vs 2
    pair_order = [fail, succ] if (args.pid + SHUFFLE_SEED) % 2 else [succ, fail]
    label_of = {1: f"p{args.pid}s{pair_order[0]['sample_i']}", 2: f"p{args.pid}s{pair_order[1]['sample_i']}"}
    traces_pair = {1: pair_order[0]["completion"], 2: pair_order[1]["completion"]}

    tasks = {}
    tasks["A1"] = ("A", {"surface_question": q, "trace_1": succ["completion"]},
                   {1: f"p{args.pid}s{args.success}"}, {1: succ["completion"]})
    tasks["A2"] = ("A", {"surface_question": q, "trace_1": fail["completion"]},
                   {1: f"p{args.pid}s{args.fail}"}, {1: fail["completion"]})
    tasks["B"] = ("B", {"surface_question": q, "trace_1": traces_pair[1], "trace_2": traces_pair[2]},
                  label_of, traces_pair)
    tasks["C"] = ("C", {"surface_question": q, "trace_1": traces_pair[1], "trace_2": traces_pair[2]},
                  label_of, traces_pair)
    if args.only:
        keep = set(args.only.split(","))
        tasks = {k: v for k, v in tasks.items() if k in keep}

    key = SecretsStore(REPO / ".mlfactory" / "secrets.yaml").get("LUNAROUTE_API_KEY")
    if not key:
        raise SystemExit("LUNAROUTE_API_KEY not found in secrets store")
    outdir = HERE / "out" / f"{args.prompt_version}_p{args.pid}s{args.success}s{args.fail}"
    outdir.mkdir(parents=True, exist_ok=True)

    def run_one(item):
        name, (framing, subs, labels, texts) = item
        user = fr[framing]
        for k, v in subs.items():
            user = user.replace("{" + k + "}", str(v))
        msgs = build_messages(framing, system, user)
        t0 = time.time()
        resp = call_api(key, args.model, msgs)
        dt = time.time() - t0
        text = resp["choices"][0]["message"].get("content") or ""
        finish = resp["choices"][0].get("finish_reason")
        meta = {"model": args.model, "framing": framing, "task": name,
                "trace_labels": {str(k): v for k, v in labels.items()},
                "dt_s": dt, "finish_reason": finish,
                "out_tokens": resp.get("usage", {}).get("completion_tokens"),
                "in_tokens": resp.get("usage", {}).get("prompt_tokens")}
        if finish == "length":
            print(f"!! {name}: finish_reason=length — output truncated; raise max_tokens")
        (outdir / f"{name}.json").write_text(json.dumps({"meta": meta, "text": text}, indent=2))
        flags = parse_flags(text, texts, labels)
        digest = render_digest(name, meta, flags)
        (outdir / f"{name}.digest.txt").write_text(digest)
        return digest

    with ThreadPoolExecutor(max_workers=6) as ex:
        for d in ex.map(run_one, list(tasks.items())):
            print(d, "\n")


if __name__ == "__main__":
    main()
