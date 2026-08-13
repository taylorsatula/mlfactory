#!/usr/bin/env python3
"""Create a new realistic pseudonymized corpus from retained placeholders.

This stage never calls a model. Detection decisions and placeholder text remain
unchanged, so pseudonym-generation rules can be iterated independently.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mlfactory.experiments.voice.review_corpus import placeholder_token, synthetic_replacement

PLACEHOLDER_RE = re.compile(r"\[\[([A-Z_]+):([0-9a-f]{12})\]\]")
POISON_RE = re.compile(r"<(?:PHONE|EMAIL|ADDRESS|ACCOUNT_ID|ACCESS_CODE|PRIVATE_URL|CUSTOMER_NAME)>")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("review_dir", type=Path, help="directory containing detections.jsonl and placeholder_sessions.jsonl")
    parser.add_argument("--version", default="v1", help="output version suffix; use a new value for each transformation policy")
    args = parser.parse_args()

    review_dir = args.review_dir.resolve()
    detections_path = review_dir / "detections.jsonl"
    placeholders_path = review_dir / "placeholder_sessions.jsonl"
    if not detections_path.is_file() or not placeholders_path.is_file():
        raise SystemExit("review directory must contain detections.jsonl and placeholder_sessions.jsonl")

    output_path = review_dir / f"pseudonymized_sessions-{args.version}.jsonl"
    mapping_path = review_dir / f"pseudonym_map-{args.version}.jsonl"
    summary_path = review_dir / f"transform_summary-{args.version}.json"
    for path in (output_path, mapping_path, summary_path):
        if path.exists():
            raise SystemExit(f"refusing to overwrite retained artifact: {path}; choose a new --version")

    detections = {}
    for line in detections_path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        detections[row["session_id"]] = row

    mapping: dict[str, dict] = {}
    output_rows = []
    changed_messages = usable_sessions = 0
    for line in placeholders_path.read_text(encoding="utf-8").splitlines():
        session = json.loads(line)
        sid = session["session_id"]
        detection = detections.get(sid)
        if detection is None:
            raise ValueError(f"placeholder session has no detection record: {sid}")
        decisions = {row["message_id"]: row for row in detection["review"]["messages"]}
        transformed_messages = []
        for message in session["messages"]:
            decision = decisions[message["message_id"]]
            text = message["text"]
            for substitution in decision["substitutions"]:
                token = placeholder_token(substitution["category"], substitution["source"])
                replacement = synthetic_replacement(substitution["category"], substitution["source"])
                if token not in text:
                    raise ValueError(f"expected placeholder missing: {sid}/{message['message_id']} {token}")
                text = text.replace(token, replacement)
                existing = mapping.get(token)
                record = {
                    "placeholder": token,
                    "category": substitution["category"],
                    "source": substitution["source"],
                    "pseudonym": replacement,
                }
                if existing is not None and existing != record:
                    raise ValueError(f"inconsistent pseudonym mapping: {token}")
                mapping[token] = record
            if PLACEHOLDER_RE.search(text):
                raise ValueError(f"unresolved placeholder remains: {sid}/{message['message_id']}")
            if POISON_RE.search(text):
                raise ValueError(f"training-poison placeholder remains: {sid}/{message['message_id']}")
            transformed = {**message, "text": text}
            transformed["changed"] = text != message["text"]
            changed_messages += int(transformed["changed"])
            transformed_messages.append(transformed)
        usable_sessions += int(session["usable"])
        output_rows.append({
            "artifact_policy": "PSEUDONYMIZED_TRAINING_CANDIDATE",
            "transform_version": args.version,
            "source": session["source"],
            "session_id": sid,
            "usable": session["usable"],
            "messages": transformed_messages,
        })

    with output_path.open("x", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with mapping_path.open("x", encoding="utf-8") as handle:
        for token in sorted(mapping):
            handle.write(json.dumps(mapping[token], ensure_ascii=False) + "\n")
    for path in (output_path, mapping_path):
        os.chmod(path, 0o600)

    summary = {
        "transform_version": args.version,
        "input_detections": str(detections_path),
        "input_placeholders": str(placeholders_path),
        "sessions": len(output_rows),
        "usable_sessions": usable_sessions,
        "changed_messages": changed_messages,
        "unique_pseudonyms": len(mapping),
        "output": str(output_path),
        "output_sha256": sha256_file(output_path),
        "mapping": str(mapping_path),
        "mapping_sha256": sha256_file(mapping_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    os.chmod(summary_path, 0o600)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
