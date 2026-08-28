#!/usr/bin/env python3
"""R4v2 judge — LLM reading of three-branch fork windows.

The R4v2 readout (principal ruling 2026-08-28): the intervention's
effect lives in the ~2048 tokens after the fork, so an LLM judge reads
the three blind branches (noop / toward_healthy / toward_diverge) and
assesses WHAT the intervention changed. This is an evaluation of
yielded tokens to interpret the effect of the intervention — not a
training reward (REWARD_POLICY scope note, written back 2026-08-28).

Blinding: per triplet the arms are assigned to labels A/B/C by a
stable permutation seeded from sha256(state_id:seed_i). The judge
never sees arm names; the label->arm map is recorded per pass for
unblinding at analysis time.

Position-bias cancellation: the judge carries a residual preference
for position A (measured 2026-08-28: v1 rubric 6/6 position-A wins on
a 6-permutation probe; v2 4/6). Each triplet is therefore judged with
an ENSEMBLE of three label assignments — cyclic rotations of the base
permutation, so every arm occupies every label position exactly once
— and analysis averages ranks across passes. Additive position bias
cancels exactly under that balance.

Input:  one or more rollout JSONLs (fork_r4v2 row schema).
Output: verdicts JSONL, resume-safe by (state_id, seed_i).

Run as a module from the repo root:
  python -m mlfactory.experiments.ace.annotate.judge_r4v2 \
      --rollouts data/fork_r4v2_pilot.jsonl --out data/judge_r4v2_pilot.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from mlfactory.core.secrets import SecretsStore
from mlfactory.experiments.ace.annotate.pilot import run_pilot as P

HERE = Path(__file__).resolve().parent
ACE = HERE.parent
DATA = ACE / "data"
ARMS = ("noop", "toward_healthy", "toward_diverge")
LABELS = ("A", "B", "C")
DEFAULT_MODEL = "glm-5.2-vision"

SYSTEM = "You are an expert evaluator of reasoning traces."

TEMPLATE = """A language model is solving a multi-step problem. Below are three continuation runs that all start from the same point in its reasoning. They are presented in random order; the presentation order carries no information about their quality.

Context just before the continuation point:
<CONTEXT>
{prefix_tail}
</CONTEXT>

Continuation A:
<A>
{A}
</A>

Continuation B:
<B>
{B}
</B>

Continuation C:
<C>
{C}
</C>

Assess each continuation's reasoning behavior: what is the model doing, and is it making durable progress on the problem (committing to decisions, eliminating possibilities, converging toward an answer) or spinning (re-deriving content it already derived without keeping the result, re-enumerating candidates it already enumerated, fixating on output format instead of substance)? Judge each continuation on its own content only.

Then:
1. For each continuation: one-sentence characterization, plus classify it as "progress", "mixed", or "spinning".
2. Rank the continuations from most to least productive reasoning behavior. Ties share a rank group.
3. Note the specific differences between the continuations that drove your ranking.

Respond with JSON only, exactly this shape:
{{"A": {{"characterization": "...", "mode": "progress"}}, "B": {{"characterization": "...", "mode": "mixed"}}, "C": {{"characterization": "...", "mode": "spinning"}}, "ranking": [["A"], ["B", "C"]], "differences": "..."}}"""


def blind_assignments(state_id: str, seed_i: int) -> list[dict[str, str]]:
    """Three arm -> label assignments: cyclic rotations of the base
    permutation, so each arm sits at each label position exactly once
    (exact cancellation of additive position bias)."""
    h = hashlib.sha256(f"{state_id}:{seed_i}".encode()).digest()
    idx = int.from_bytes(h[:4], "big")
    perms = [("A", "B", "C"), ("A", "C", "B"), ("B", "A", "C"),
             ("B", "C", "A"), ("C", "A", "B"), ("C", "B", "A")]
    base = perms[idx % 6]
    out = []
    for rot in range(3):
        labels = base[rot:] + base[:rot]   # cyclic rotation of positions
        out.append(dict(zip(ARMS, labels)))
    return out


def load_triplets(paths: list[Path]) -> dict[tuple, dict[str, dict]]:
    rows: dict[tuple, dict[str, dict]] = {}
    for p in paths:
        for l in p.open():
            r = json.loads(l)
            key = (r["state_id"], r["seed_i"])
            rows.setdefault(key, {})[r["arm"]] = r
    complete = {k: v for k, v in rows.items()
                if all(a in v for a in ARMS)}
    print(f"rows: {sum(len(v) for v in rows.values())} | "
          f"triplets: {len(rows)} | complete: {len(complete)}")
    return complete


def judge_one(key, arms_rows, model: str, k: str) -> dict:
    state_id, seed_i = key
    t0 = time.time()
    passes = []
    for amap in blind_assignments(state_id, seed_i):
        fields = {"prefix_tail": arms_rows["noop"]["prefix_tail"]}
        for arm, lab in amap.items():
            fields[lab] = arms_rows[arm]["window"]
        prompt = TEMPLATE.format(**fields)
        resp = P.call_api(k, model,
                          [{"role": "system", "content": SYSTEM},
                           {"role": "user", "content": prompt}])
        text = resp["choices"][0]["message"]["content"]
        parsed = None
        try:
            s = text.strip()
            if s.startswith("```"):
                s = s.split("```")[1]
                if s.startswith("json"):
                    s = s[4:]
            parsed = json.loads(s)
        except (json.JSONDecodeError, IndexError):
            pass
        passes.append({
            "label_to_arm": {lab: arm for arm, lab in amap.items()},
            "raw": text,
            "parsed": parsed,
        })
    return {
        "state_id": state_id, "seed_i": seed_i,
        "model": model,
        "elapsed_s": round(time.time() - t0, 1),
        "passes": passes,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", required=True,
                    help="comma-sep rollout JSONL paths")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only-state", default="")
    ap.add_argument("--seeds", default="")
    args = ap.parse_args()

    key = SecretsStore(P.REPO / ".mlfactory" / "secrets.yaml").get(
        "LUNAROUTE_API_KEY")
    triplets = load_triplets([Path(p) for p in args.rollouts.split(",")])
    if args.only_state:
        triplets = {k: v for k, v in triplets.items()
                    if k[0] in set(args.only_state.split(","))}
    if args.seeds:
        keep = {int(s) for s in args.seeds.split(",")}
        triplets = {k: v for k, v in triplets.items() if k[1] in keep}

    out_path = Path(args.out)
    done: set[tuple] = set()
    if out_path.exists():
        for l in out_path.open():
            r = json.loads(l)
            done.add((r["state_id"], r["seed_i"]))
    pending = sorted(k for k in triplets if k not in done)
    if args.limit:
        pending = pending[:args.limit]
    print(f"pending: {len(pending)} | model: {args.model}", flush=True)

    fout = out_path.open("a")
    for i, k in enumerate(pending):
        for attempt in range(3):
            try:
                verdict = judge_one(k, triplets[k], args.model, key)
                break
            except Exception as e:  # network / provider hiccups
                if attempt == 2:
                    raise
                print(f"  retry {attempt + 1} for {k}: {e}", flush=True)
                time.sleep(5 * (attempt + 1))
        fout.write(json.dumps(verdict) + "\n")
        fout.flush()
        ok = sum(1 for p in verdict["passes"] if p["parsed"])
        print(f"[{i+1}/{len(pending)}] {k[0]} seed {k[1]}: "
              f"{ok}/3 passes parsed ({verdict['elapsed_s']}s)", flush=True)
    fout.close()
    print(f"judge run complete: {len(pending)} verdicts -> {out_path}")


if __name__ == "__main__":
    main()
