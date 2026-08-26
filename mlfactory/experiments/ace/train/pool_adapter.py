#!/usr/bin/env python3
"""Pool adapter for GRPO on the calibrated b2 46-prompt LIVE pool.

Loads a calibrated pool JSONL, builds solver items via
``frontier.collect_rollouts.solver_prompt``, dispatches strict per-family
verifiers through ``gen.calibrate.CHECK``, and makes the deterministic
stratified train/holdout split (~1/3 held out per family) that a run
writes as ``split_manifest.json`` into its run dir.

Pure CPU; no model imports. Preview a split without a GPU:

  python -m mlfactory.experiments.ace.train.pool_adapter \
      --pool data/acegen_live_b2.jsonl
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from mlfactory.experiments.ace.frontier.collect_rollouts import solver_prompt
from mlfactory.experiments.ace.gen.calibrate import CHECK

ACE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_POOL = ACE_DIR / "data" / "acegen_live_b2.jsonl"


def load_pool(path: Path) -> list[dict]:
    """Raw pool rows (calibration candidates format), order preserved."""
    rows = [json.loads(l) for l in Path(path).read_text().splitlines()
            if l.strip()]
    if not rows:
        raise RuntimeError(f"empty pool: {path}")
    pids = [r["provenance"]["proposal_id"] for r in rows]
    if len(set(pids)) != len(pids):
        raise RuntimeError(f"duplicate proposal_id in pool: {path}")
    for r in rows:
        if r["domain"] not in CHECK:
            raise RuntimeError(f"pool family {r['domain']!r} has no CHECK")
    return rows


def make_items(rows: list[dict], pids: list[int] | None = None) -> list[dict]:
    """Pool rows -> GRPO items.

    Item keys are a superset of the legacy ``core.problems`` item shape
    (``id``/``family``/``prompt``/``gold``) so the trainer can treat both
    uniformly; ``knobs``/``pid`` mark a pool item (CHECK-based verify).
    """
    items = []
    for r in rows:
        pid = r["provenance"]["proposal_id"]
        if pids is not None and pid not in pids:
            continue
        items.append({
            "id": f"{r['domain']}-p{pid}",
            "pid": pid,
            "family": r["domain"],
            "prompt": solver_prompt(r),
            "gold": r["problem"]["reference_answer"],
            "knobs": r.get("knobs", {}),
        })
    if pids is not None and len(items) != len(set(pids)):
        have = {it["pid"] for it in items}
        missing = sorted(set(pids) - have)
        raise RuntimeError(f"pool is missing requested pids: {missing}")
    return items


def verify_item(item: dict, completion: str) -> bool:
    """Strict terminal reward for one completion.

    Pool items go through the family's strict ``check()`` (format-tolerant
    extraction lives in the checker); legacy arithmetic items fall back to
    ``core.problems.verify``. This is the ONLY reward source (REWARD_POLICY).
    """
    if "knobs" in item:
        return bool(CHECK[item["family"]](completion, item["gold"],
                                          item["knobs"]))
    from mlfactory.experiments.ace.core.problems import verify
    return bool(verify(completion, item["gold"]))


def stratified_split(items: list[dict],
                     holdout_frac: float = 1 / 3) -> tuple[list[dict], list[dict]]:
    """Deterministic stratified train/holdout split.

    No RNG: items are sorted by (family, pid) and each family holds out
    ``round(n * holdout_frac)`` items at evenly spaced positions. The same
    pool always yields the same split; the run writes it to a manifest so
    the split itself is evidence.
    """
    by_fam: dict[str, list[dict]] = defaultdict(list)
    for it in sorted(items, key=lambda x: (x["family"], x.get("pid", 0))):
        by_fam[it["family"]].append(it)
    train, holdout = [], []
    for fam in sorted(by_fam):
        fam_items = by_fam[fam]
        n = len(fam_items)
        n_hold = max(1, round(n * holdout_frac)) if n > 1 else 0
        hold_idx = ({int(j * n / n_hold) for j in range(n_hold)}
                    if n_hold else set())
        for i, it in enumerate(fam_items):
            (holdout if i in hold_idx else train).append(it)
    return train, holdout


def write_split_manifest(path: Path, train: list[dict], holdout: list[dict],
                         pool_path: Path, extra: dict | None = None) -> dict:
    def side(items):
        return {
            "n": len(items),
            "pids": [it.get("pid") for it in items],
            "ids": [it["id"] for it in items],
            "families": {f: sum(1 for it in items if it["family"] == f)
                         for f in sorted({it["family"] for it in items})},
        }
    manifest = {
        "pool": str(pool_path),
        "rule": "deterministic stratified: sort (family, pid), hold out "
                "round(n/3) per family at evenly spaced positions",
        "train": side(train),
        "holdout": side(holdout),
    }
    if extra:
        manifest.update(extra)
    Path(path).write_text(json.dumps(manifest, indent=2))
    return manifest


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pool", type=Path, default=DEFAULT_POOL)
    p.add_argument("--manifest-out", type=Path, default=None,
                   help="write the split manifest here (preview mode)")
    return p.parse_args()


def main() -> None:
    cfg = parse_args()
    items = make_items(load_pool(cfg.pool))
    train, holdout = stratified_split(items)
    print(f"pool: {cfg.pool}  ({len(items)} items)")
    for fam in sorted({it['family'] for it in items}):
        nt = sum(1 for it in train if it["family"] == fam)
        nh = sum(1 for it in holdout if it["family"] == fam)
        print(f"  {fam:12} train={nt:2d}  holdout={nh:2d}")
    print(f"total: train={len(train)}  holdout={len(holdout)}")
    if cfg.manifest_out:
        write_split_manifest(cfg.manifest_out, train, holdout, cfg.pool)
        print(f"manifest: {cfg.manifest_out}")


if __name__ == "__main__":
    main()
