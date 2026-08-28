"""R0 instrumentation check: double-annotator span agreement.

Compares pass1 and pass2 annotations on the double-annotated subset
(five pairs, ten traces — the instrumentation sample from the plan).
Annotations are noisy measurements; this quantifies the noise so probe
results can be read against it (ANNOTATION_SIDESTEP kill condition K4:
poor agreement indicts the rubric before it indicts the models).

Metrics per trace and pooled:
  - Jaccard over matched span pairs (greedy best-overlap matching,
    same class required; a match needs char-overlap IoU >= 0.3 OR one
    span contained in the other)
  - class-confusion counts among matched spans
  - missed/extra counts (spans present in only one pass)

CPU-only; reads data/annotations_xsub_pass{1,2}.jsonl.

Usage: .venv/bin/python -m mlfactory.experiments.ace.annotate.r0_agreement
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ACE = HERE.parent
DATA = ACE / "data"
IOU_MATCH = 0.3


def load_pass(path: Path) -> dict[tuple, list[dict]]:
    by_trace = defaultdict(list)
    if not path.exists():
        return by_trace
    for line in path.open():
        a = json.loads(line)
        if a.get("start_char") is None or a.get("end_char") is None:
            continue
        by_trace[(a["substrate"], a["pid"], a["sample_i"])].append(a)
    return by_trace


def iou(s1: tuple[int, int], s2: tuple[int, int]) -> float:
    inter = max(0, min(s1[1], s2[1]) - max(s1[0], s2[0]))
    union = max(s1[1], s2[1]) - min(s1[0], s2[0])
    return inter / union if union > 0 else 0.0


def contained(s1: tuple[int, int], s2: tuple[int, int]) -> bool:
    return (s1[0] <= s2[0] and s2[1] <= s1[1]) or (s2[0] <= s1[0] and s1[1] <= s2[1])


def match_spans(a: list[dict], b: list[dict]) -> dict:
    """Greedy best-IoU matching of pass-a spans to pass-b spans (same class)."""
    b_used = set()
    matched, class_conf, a_only = [], [], 0
    for sa in a:
        ra = (sa["start_char"], sa["end_char"])
        best = None
        for j, sb in enumerate(b):
            if j in b_used or sb["class"] != sa["class"]:
                continue
            rb = (sb["start_char"], sb["end_char"])
            score = iou(ra, rb)
            if contained(ra, rb):
                score = max(score, IOU_MATCH)  # containment counts as a match
            if best is None or score > best[0]:
                best = (score, j, rb)
        if best and best[0] >= IOU_MATCH:
            b_used.add(best[1])
            matched.append((sa, b[best[1]], best[0]))
        else:
            # check class-confused matches (same place, different class)
            conf = None
            for j, sb in enumerate(b):
                if j in b_used:
                    continue
                rb = (sb["start_char"], sb["end_char"])
                if iou(ra, rb) >= IOU_MATCH or contained(ra, rb):
                    conf = sb
                    b_used.add(j)
                    break
            if conf:
                class_conf.append((sa, conf))
            else:
                a_only += 1
    b_only = len(b) - len(b_used)
    return {"matched": matched, "class_conf": class_conf,
            "a_only": a_only, "b_only": b_only}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="xsub",
                    help="corpus tag (reads annotations_<tag>_pass{1,2}.jsonl)")
    args = ap.parse_args()
    p1 = load_pass(DATA / f"annotations_{args.tag}_pass1.jsonl")
    p2 = load_pass(DATA / f"annotations_{args.tag}_pass2.jsonl")
    both = sorted(set(p1) & set(p2))
    if not both:
        raise SystemExit("no double-annotated traces found in both passes")

    print(f"traces double-annotated: {len(both)}")
    pooled = {"matched": 0, "class_conf": 0, "p1_only": 0, "p2_only": 0,
              "n1": 0, "n2": 0, "boundary_diffs": []}
    for key in both:
        a, b = p1[key], p2[key]
        m = match_spans(a, b)
        pooled["matched"] += len(m["matched"])
        pooled["class_conf"] += len(m["class_conf"])
        pooled["p1_only"] += m["a_only"]
        pooled["p2_only"] += m["b_only"]
        pooled["n1"] += len(a)
        pooled["n2"] += len(b)
        for sa, sb, sc in m["matched"]:
            pooled["boundary_diffs"].append(
                abs(sa["start_char"] - sb["start_char"]) +
                abs(sa["end_char"] - sb["end_char"]))
        jac = (len(m["matched"]) /
               max(1, len(m["matched"]) + m["a_only"] + m["b_only"]))
        print(f"  {key[0]} p{key[1]} s{key[2]:>2}: "
              f"pass1={len(a)} pass2={len(b)} matched={len(m['matched'])} "
              f"class-confused={len(m['class_conf'])} "
              f"p1-only={m['a_only']} p2-only={m['b_only']} jaccard={jac:.2f}")

    n_union = pooled["matched"] + pooled["p1_only"] + pooled["p2_only"]
    print(f"\n== pooled ==")
    print(f"spans: pass1={pooled['n1']} pass2={pooled['n2']}")
    print(f"matched={pooled['matched']} class-confused={pooled['class_conf']} "
          f"pass1-only={pooled['p1_only']} pass2-only={pooled['p2_only']}")
    print(f"span Jaccard (agreement): {pooled['matched'] / max(1, n_union):.3f}")
    if pooled["boundary_diffs"]:
        bd = sorted(pooled["boundary_diffs"])
        print(f"matched-span boundary drift (chars, both ends summed): "
              f"median={bd[len(bd)//2]} p90={bd[int(len(bd)*0.9)]}")
    print("\nK4 verdict guide: Jaccard >= 0.5 with low class-confusion = "
          "rubric usable; < 0.3 or heavy confusion = fix rubric, not data.")


if __name__ == "__main__":
    main()
