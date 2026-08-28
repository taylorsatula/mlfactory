"""One-off helper: write a datasave-schema .meta.json sidecar for a
data file. Computes sha256 + size; takes name/title/description/tags
from argv. Run from repo root.

Used by the b2 overnight runbook (Phase 5) to sidecar files written by
collectors/batch scripts that predate datasave or write directly.

Usage:
  .venv/bin/python -m mlfactory.experiments.ace.annotate.write_meta \
    --path data/annot_b2_q8.jsonl --name annot_b2_q8 \
    --title "..." --description "..." --tags rollouts q8 b2
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ACE = HERE.parent


def sha_size(path: Path) -> tuple[str, int]:
    h = hashlib.sha256()
    n = 0
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
            n += len(chunk)
    return h.hexdigest(), n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True, help="data file (relative to ace/ or absolute)")
    ap.add_argument("--name", required=True)
    ap.add_argument("--title", required=True)
    ap.add_argument("--description", required=True)
    ap.add_argument("--tags", nargs="*", default=[])
    ap.add_argument("--caveats", default=None)
    args = ap.parse_args()

    p = Path(args.path)
    if not p.is_absolute():
        p = ACE / "data" / p.name if "/" not in args.path else ACE / args.path
    dig, size = sha_size(p)
    meta = {
        "name": args.name,
        "title": args.title,
        "description": args.description,
        "format": "jsonl",
        "tags": args.tags,
        "caveats": args.caveats,
        "sensitivity": None,
        "data_schema": None,
        "path": str(p),
        "sha256": dig,
        "size_bytes": size,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    out = p.with_suffix(p.suffix + ".meta.json")
    out.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {out} ({size} bytes, sha256={dig[:12]})")


if __name__ == "__main__":
    main()
