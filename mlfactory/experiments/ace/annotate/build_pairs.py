"""Build the annotation pair manifest from existing verified rollout pools.

Pairs same-prompt success/failure traces for the span-annotation workstream
(see RUBRIC.md in this directory). Also emits truncated (cap-hit) failure
traces as standalone loop-annotation targets — those are excluded from
pairing (truncation confound) but are prime class-a/loop material.

Pools scanned (see ace/data/):
  probe_b1        HF bf16 rollouts (substrate hf-bf16)   — primary
  frontier pass1  HF bf16 rollouts (substrate hf-bf16)   — primary
  b2 pool, r1-r4  llama.cpp Q8_0-MTP rollouts (q8-mtp)   — secondary,
                  teacher-forceable on bf16 but off-policy tokens

Manifest rows are pointers (source file + sample_id), not trace copies —
the annotator joins back to the source rows. Rows never duplicate data.

Run:  .venv/bin/python -m mlfactory.experiments.ace.annotate.build_pairs
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ACE_ROOT = Path(__file__).resolve().parent.parent
DATA = ACE_ROOT / "data"
OUT_PATH = DATA / "annotate_pairs_p1.jsonl"

POOLS = [
    # name, substrate, source files
    ("probe_b1", "hf-bf16", ["probe_rollouts_b1.jsonl"]),
    ("frontier_p1", "hf-bf16", ["frontier_rollouts_pass1.jsonl"]),
    ("b2_pool", "q8-mtp", ["acegen_b2_pool_gpu0.jsonl", "acegen_b2_pool_gpu1.jsonl"]),
    ("b2_r1", "q8-mtp", ["b2/r1_rollouts.jsonl"]),
    ("b2_r2", "q8-mtp", ["b2/r2_rollouts.jsonl"]),
    ("b2_r3", "q8-mtp", ["b2/r3_rollouts.jsonl"]),
    ("b2_r4", "q8-mtp", ["b2/r4_rollouts.jsonl"]),
]


def load_pool(files: list[str]) -> list[dict]:
    rows = []
    for rel in files:
        path = DATA / rel
        if not path.exists():
            print(f"  WARNING: missing pool file {path}")
            continue
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            r["_source"] = str(path.relative_to(ACE_ROOT))
            rows.append(r)
    return rows


def trace_ref(r: dict) -> dict:
    return {
        "source": r["_source"],
        "sample_id": r["sample_id"],
        "sample_i": r.get("sample_i"),
        "n_new_tokens": r["n_new_tokens"],
    }


def main() -> None:
    manifest: list[dict] = []
    stats: dict[str, dict] = {}

    for pool_name, substrate, files in POOLS:
        rows = load_pool(files)
        if not rows:
            continue
        by_prompt: dict[str, dict[str, list[dict]]] = {}
        for r in rows:
            bucket = "win" if r.get("correct") else ("trunc" if r.get("truncated") else "lose")
            by_prompt.setdefault(r["proposal_id"], {"win": [], "lose": [], "trunc": []})[bucket].append(r)

        n_pairs = 0
        for pid, buckets in sorted(by_prompt.items()):
            wins = sorted(buckets["win"], key=lambda r: r["n_new_tokens"])
            loses = sorted(buckets["lose"], key=lambda r: r["n_new_tokens"])
            for w, l in zip(wins, loses):
                n_pairs += 1
                manifest.append({
                    "kind": "pair",
                    "pair_id": f"{pool_name}:{pid}:p{n_pairs}",
                    "pool": pool_name,
                    "substrate": substrate,
                    "proposal_id": pid,
                    "domain": w.get("domain"),
                    "task": w.get("task"),
                    "win": trace_ref(w),
                    "lose": trace_ref(l),
                })
            # cap-hit failures: standalone loop targets (truncation confound
            # bars them as pair members, not as annotation material)
            for t in buckets["trunc"]:
                manifest.append({
                    "kind": "loop",
                    "pair_id": f"{pool_name}:{pid}:loop{t['sample_i']}",
                    "pool": pool_name,
                    "substrate": substrate,
                    "proposal_id": pid,
                    "domain": t.get("domain"),
                    "task": t.get("task"),
                    "trace": trace_ref(t),
                })

        n_loop = sum(1 for m in manifest if m["pool"] == pool_name and m["kind"] == "loop")
        n_pair_pool = sum(1 for m in manifest if m["pool"] == pool_name and m["kind"] == "pair")
        stats[pool_name] = {
            "rows": len(rows),
            "substrate": substrate,
            "prompts": len(by_prompt),
            "pairs": n_pair_pool,
            "loop_targets": n_loop,
        }

    # ordering: hf-bf16 pairs first (primary substrate), then q8; loops last.
    # within pairs, shorter traces first — cheap to annotate and capture.
    def sort_key(m: dict) -> tuple:
        kind_rank = 0 if m["kind"] == "pair" else 1
        sub_rank = 0 if m.get("substrate") == "hf-bf16" else 1
        length = (
            m["win"]["n_new_tokens"] + m["lose"]["n_new_tokens"]
            if m["kind"] == "pair" else m["trace"]["n_new_tokens"]
        )
        return (kind_rank, sub_rank, length)

    manifest.sort(key=sort_key)

    OUT_PATH.write_text("".join(json.dumps(m) + "\n" for m in manifest))
    sha = hashlib.sha256(OUT_PATH.read_bytes()).hexdigest()

    sidecar = {
        "name": "annotate_pairs_p1",
        "title": "Annotation pair manifest pass 1",
        "description": (
            "Same-prompt success/failure pairs (kind=pair) and cap-hit loop "
            "targets (kind=loop) drawn from probe_b1, frontier pass1, and b2 "
            "pools for the span-annotation workstream. Pointer rows only — "
            "join to source by sample_id. hf-bf16 pairs first, then q8-mtp."
        ),
        "format": "jsonl",
        "tags": ["annotate", "pairs", "pass1"],
        "caveats": "q8-mtp pairs are off-policy tokens for the bf16 capture model",
        "sensitivity": None,
        "data_schema": None,
        "path": str(OUT_PATH),
        "sha256": sha,
        "size_bytes": OUT_PATH.stat().st_size,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    sidecar_path = OUT_PATH.with_suffix(".jsonl.meta.json")
    sidecar_path.write_text(json.dumps(sidecar, indent=2) + "\n")

    print(f"\nWrote {OUT_PATH.relative_to(ACE_ROOT)} ({len(manifest)} rows)")
    print(f"{'pool':14s} {'substrate':9s} {'rows':>5s} {'prompts':>8s} {'pairs':>6s} {'loops':>6s}")
    for name, s in stats.items():
        print(f"{name:14s} {s['substrate']:9s} {s['rows']:5d} {s['prompts']:8d} {s['pairs']:6d} {s['loop_targets']:6d}")
    n_pair = sum(1 for m in manifest if m["kind"] == "pair")
    n_loop = sum(1 for m in manifest if m["kind"] == "loop")
    n_pair_hf = sum(1 for m in manifest if m["kind"] == "pair" and m["substrate"] == "hf-bf16")
    print(f"\nTotals: {n_pair} pairs ({n_pair_hf} on hf-bf16), {n_loop} loop targets")


if __name__ == "__main__":
    main()
