"""Run-attached lab notes.

A lab note is a single timestamped line appended to ``runs/<run_id>/notes.jsonl``.
The intent is capture-cost reduction: cheaper than a sticky note, structured
only by being a JSONL record, and searchable across the whole runs/ tree so
"what have I tried" is an ``rg`` away. Notes are recorded as manifest
``FileRecord``s with ``role="note"`` so they are hashed and provenance-linked
like any other artifact.

This is the frequent, cheap, mid-flight end of the notebook axis. The rare,
high-cost, end-of-session end is ``session_notes/`` (see AGENTS.md).
"""
from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

from mlfactory.core.manifest import FileRecord, RunManifest, sha256_file


NOTES_FILENAME = "notes.jsonl"
# Sidecar label-card suffix. Matches mlfactory.core.datasave.META_SUFFIX so a
# scientist browsing the run folder sees a consistent label format for every
# file (data artifacts get their label from datasave; notes.jsonl gets its label
# here). We define the constant locally rather than importing it from datasave
# so the notes feature stays self-contained and does not depend on datasave
# being present.
META_SUFFIX = ".meta.json"

# Fixed label for the notes.jsonl container. Per-note content (the actual
# reasoning) lives inside the jsonl; this labels the file as a whole.
_NOTES_TITLE = "Lab notes"
_NOTES_DESCRIPTION = (
    "Free-text reasoning appended to this run mid-flight — hypothesis changes, "
    "unexpected results, parameter changes with rationale, dead ends, resumption "
    "points. See AGENTS.md 'Lab notes'. Searchable via `mlfactory notes --grep`."
)
_NOTES_TAGS = ["note"]


def _resolve_run_dir(registry, run_id: str) -> Path:
    """Resolve a run's on-disk directory from the registry.

    The manifest stores ``source.path`` as ``<run_dir>/source.tar.gz``, so the
    run directory is its parent. Falls back to ``runs/<run_id>`` when no
    manifest is registered (e.g. a fresh ``mlfactory init`` run that was never
    registered) so notes can still be attached.
    """
    manifest = registry.get(run_id) if registry is not None else None
    if manifest is not None and manifest.source is not None and manifest.source.path:
        return Path(manifest.source.path).parent
    return Path("runs") / run_id


def _resolve_run_dir_local(run_id: str, runs_dir: Path = Path("runs")) -> Path:
    """Resolve a run directory without a registry, for read/search only."""
    candidate = runs_dir / run_id
    if candidate.is_dir():
        return candidate
    return candidate


def _notes_path(run_dir: Path) -> Path:
    return run_dir / NOTES_FILENAME


def _get_author() -> str:
    """Best-effort author identity. Overridable via MLFACTORY_AUTHOR."""
    env = os.environ.get("MLFACTORY_AUTHOR")
    if env:
        return env
    try:
        import subprocess

        out = subprocess.check_output(
            ["git", "config", "user.name"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        if out:
            return out
    except Exception:
        pass
    try:
        return os.getlogin()
    except Exception:
        return "unknown"


def append_note(
    registry,
    run_id: str,
    text: str,
    author: str | None = None,
) -> dict:
    """Append one lab note to ``runs/<run_id>/notes.jsonl``.

    Creates the file if absent (appends a single line — no trailing-newline
    games). Records the file as a manifest ``FileRecord(role="note")`` and
    rewrites the manifest so the note is hashed and provenance-linked. Returns
    the note record that was written.

    The note text is stored verbatim; no length or structure is enforced.
    """
    if not text or not text.strip():
        raise ValueError("note text must not be empty")

    run_dir = _resolve_run_dir(registry, run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    path = _notes_path(run_dir)

    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "author": author or _get_author(),
        "run_id": run_id,
        "text": text,
    }

    # Append exactly one line. JSONL is deliberately robust to interleaved
    # appends from concurrent processes (one line == one write).
    line = json.dumps(record, ensure_ascii=False)
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
        f.flush()
        os.fsync(f.fileno())

    # Provenance-link the note file into the manifest, hashing the whole file.
    # We re-hash on every append: the sha256 reflects the notes file as it now
    # stands, and updating the FileRecord (matched by path) keeps the manifest
    # honest about the current content.
    manifest = registry.get(run_id) if registry is not None else None
    if manifest is not None:
        _record_note_file(manifest, run_dir, path)
        _write_note_sidecar(manifest, run_dir, path)
        manifest.write(run_dir / "manifest.json")
        registry.register(manifest)
    else:
        # No manifest (standalone script): still drop a sidecar label so the
        # folder-browsing UX matches datasave's convention.
        _write_note_sidecar(None, run_dir, path)

    return record


def _record_note_file(manifest: RunManifest, run_dir: Path, path: Path) -> None:
    """Insert or refresh the notes.jsonl FileRecord on the manifest."""
    fr = FileRecord(
        path=str(path.resolve()),
        sha256=sha256_file(path),
        role="note",
        size_bytes=path.stat().st_size,
    )
    # Replace any existing note record for the same path, else append.
    manifest.logs = [l for l in manifest.logs if l.path != fr.path]
    manifest.logs.append(fr)


def _write_note_sidecar(manifest: RunManifest | None, run_dir: Path, path: Path) -> Path:
    """Write/refresh ``notes.jsonl.meta.json`` — the label card next to the notes.

    Field shape mirrors ``mlfactory.core.datasave._build_meta`` so a scientist
    browsing the run folder sees a consistent label format for every file, and
    so a future ``Registry.find_artifacts`` inclusion of notes needs no schema
    work. The label describes the *container* (the file); per-note content lives
    inside the jsonl with its own per-line ``ts``.
    """
    sidecar = path.with_name(path.name + META_SUFFIX)
    sha = sha256_file(path)
    size = path.stat().st_size
    meta: dict = {
        "name": path.stem,
        "title": _NOTES_TITLE,
        "description": _NOTES_DESCRIPTION,
        "format": "jsonl",
        "tags": list(_NOTES_TAGS),
        "caveats": None,
        "sensitivity": None,
        "data_schema": {
            "line": "json object",
            "fields": {"ts": "iso8601", "author": "str", "run_id": "str", "text": "str"},
        },
        "path": str(path.resolve()),
        "sha256": sha,
        "size_bytes": size,
        # created_at = birth of the notes file (first append), preserved across
        # refreshes by not overwriting an existing sidecar's created_at.
    }
    if sidecar.exists():
        try:
            old = json.loads(sidecar.read_text(encoding="utf-8"))
            meta["created_at"] = old.get("created_at")
        except (json.JSONDecodeError, OSError):
            meta["created_at"] = datetime.now(timezone.utc).isoformat()
    else:
        meta["created_at"] = datetime.now(timezone.utc).isoformat()
    if manifest is not None:
        meta["run_id"] = manifest.run_id
        meta["stage"] = manifest.stage
        meta["git_commit"] = manifest.git.commit
        meta["parent_runs"] = list(manifest.parent_runs)
    tmp = sidecar.with_suffix(sidecar.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False, default=str)
        f.flush()
    tmp.replace(sidecar)
    return sidecar


def read_notes(run_dir: Path) -> list[dict]:
    """Read all notes for a run directory, oldest first. Returns [] if none."""
    path = _notes_path(run_dir)
    if not path.exists():
        return []
    out = []
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                # Tolerate a malformed line rather than aborting the whole read.
                out.append({"_malformed": True, "lineno": lineno, "raw": line})
    return out


def search_notes(
    pattern: str,
    runs_dir: Path = Path("runs"),
    ignore_case: bool = True,
) -> list[tuple[str, dict]]:
    """Search note text across all runs. Returns (run_id, note) tuples.

    Uses a plain regex over each note's ``text`` field (not the raw file) so
    matches are per-note and structured. ``ignore_case`` defaults True because
    "what have I tried" searches are casual.
    """
    flags = re.IGNORECASE if ignore_case else 0
    rx = re.compile(pattern, flags)
    results: list[tuple[str, dict]] = []
    if not runs_dir.exists():
        return results
    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        for note in read_notes(run_dir):
            text = note.get("text", "")
            if isinstance(text, str) and rx.search(text):
                results.append((run_dir.name, note))
    return results
