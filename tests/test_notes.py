"""Tests for run-attached lab notes (mlfactory/core/notes.py)."""
from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

from mlfactory.core.manifest import RunManifest, SourceArchive
from mlfactory.core.notes import (
    append_note,
    read_notes,
    search_notes,
)
from mlfactory.core.registry import Registry


def _make_registry_and_run(td: Path, run_id: str = "test.notes.001") -> tuple[Registry, Path, str]:
    """Create a registry, a run dir with a source archive (so source.path resolves),
    register the manifest, and return (registry, run_dir, run_id)."""
    db = td / "registry.db"
    registry = Registry(db)
    run_dir = td / "runs" / run_id
    run_dir.mkdir(parents=True)
    # source.tar.gz must exist so Path(source.path).parent resolves to run_dir.
    (run_dir / "source.tar.gz").write_bytes(b"")
    manifest = RunManifest(
        run_id=run_id,
        stage="train",
        status="completed",
        source=SourceArchive(path=str(run_dir / "source.tar.gz"), sha256="0" * 64),
    )
    registry.register(manifest)
    return registry, run_dir, run_id


def test_append_creates_file_and_records_in_manifest() -> None:
    with tempfile.TemporaryDirectory() as td:
        registry, run_dir, run_id = _make_registry_and_run(Path(td))
        record = append_note(registry, run_id, "lr 3e-5 diverges step 800", author="alice")

        # File exists with one JSONL line.
        notes_path = run_dir / "notes.jsonl"
        assert notes_path.exists()
        lines = notes_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1

        # The returned record carries the text and author verbatim.
        import json
        written = json.loads(lines[0])
        assert written["text"] == "lr 3e-5 diverges step 800"
        assert written["author"] == "alice"
        assert written["run_id"] == run_id
        assert "ts" in written

        # Manifest in registry carries the note as a hashed FileRecord.
        manifest = registry.get(run_id)
        note_records = [l for l in manifest.logs if l.role == "note"]
        assert len(note_records) == 1
        assert note_records[0].path == str(notes_path.resolve())

        # Hash matches actual file content.
        actual = hashlib.sha256(notes_path.read_bytes()).hexdigest()
        assert note_records[0].sha256 == actual

        # Sidecar label card exists and matches the datasave meta schema shape,
        # so a scientist browsing the folder sees a consistent label and a
        # future find_artifacts inclusion needs no schema work.
        sidecar = run_dir / "notes.jsonl.meta.json"
        assert sidecar.exists()
        meta = json.loads(sidecar.read_text(encoding="utf-8"))
        for key in ("name", "title", "description", "format", "tags", "path",
                   "sha256", "size_bytes", "created_at", "run_id", "stage"):
            assert key in meta, f"sidecar missing datasave-aligned key {key!r}"
        assert meta["title"] == "Lab notes"
        assert meta["format"] == "jsonl"
        assert meta["sha256"] == actual
        assert meta["run_id"] == run_id


def test_append_multiple_notes_rehashes_each_time() -> None:
    with tempfile.TemporaryDirectory() as td:
        registry, run_dir, run_id = _make_registry_and_run(Path(td))
        append_note(registry, run_id, "first note", author="a")
        first_created = json.loads((run_dir / "notes.jsonl.meta.json").read_text())["created_at"]
        append_note(registry, run_id, "second note", author="b")

        notes = read_notes(run_dir)
        assert len(notes) == 2
        assert notes[0]["text"] == "first note"
        assert notes[1]["text"] == "second note"

        # Manifest carries exactly one note FileRecord (refreshed, not duplicated).
        manifest = registry.get(run_id)
        note_records = [l for l in manifest.logs if l.role == "note"]
        assert len(note_records) == 1

        # Its hash reflects the two-line file.
        actual = hashlib.sha256((run_dir / "notes.jsonl").read_bytes()).hexdigest()
        assert note_records[0].sha256 == actual

        # Sidecar created_at is the file's birth (first append), preserved
        # across refreshes — not the last append.
        second_created = json.loads((run_dir / "notes.jsonl.meta.json").read_text())["created_at"]
        assert second_created == first_created


def test_read_notes_empty_run() -> None:
    with tempfile.TemporaryDirectory() as td:
        assert read_notes(Path(td) / "runs" / "nope") == []


def test_search_notes_across_runs() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        # Two runs, each with a note.
        for rid, text in [("runA", "diverges at step 800"), ("runB", "plateau at step 1200")]:
            rdir = root / "runs" / rid
            rdir.mkdir(parents=True)
            (rdir / "notes.jsonl").write_text(
                f'{{"ts":"t","author":"x","run_id":"{rid}","text":{chr(34)+text+chr(34)}}}\n',
                encoding="utf-8",
            )
        hits = search_notes("diverge", runs_dir=root / "runs")
        assert len(hits) == 1
        assert hits[0][0] == "runA"
        assert "diverges" in hits[0][1]["text"]

        # Case-insensitive default matches DIVERGE.
        hits_ci = search_notes("DIVERGE", runs_dir=root / "runs")
        assert len(hits_ci) == 1
        # Disabling case sensitivity drops the match.
        hits_cs = search_notes("DIVERGE", runs_dir=root / "runs", ignore_case=False)
        assert hits_cs == []


def test_append_rejects_empty_text() -> None:
    with tempfile.TemporaryDirectory() as td:
        registry, _, run_id = _make_registry_and_run(Path(td))
        for empty in ["", "   ", "\n\t"]:
            try:
                append_note(registry, run_id, empty)
            except ValueError:
                continue
            raise AssertionError(f"expected ValueError for {empty!r}")


def test_author_fallback_chain(monkeypatch) -> None:
    """author resolves MLFACTORY_AUTHOR > git user.name > os.getlogin()."""
    with tempfile.TemporaryDirectory() as td:
        registry, run_dir, run_id = _make_registry_and_run(Path(td))

        # No env var, no git config in this tmpdir -> os.getlogin() path.
        import os
        monkeypatch.delenv("MLFACTORY_AUTHOR", raising=False)
        # Force a chdir to a non-git dir so git config lookup fails cleanly.
        monkeypatch.chdir(td)
        rec = append_note(registry, run_id, "fallback test")
        assert rec["author"]  # something non-empty

        # Env var takes precedence.
        monkeypatch.setenv("MLFACTORY_AUTHOR", "env-author")
        rec2 = append_note(registry, run_id, "env author test")
        assert rec2["author"] == "env-author"

        # Explicit author argument beats env var.
        rec3 = append_note(registry, run_id, "explicit", author="explicit-author")
        assert rec3["author"] == "explicit-author"
