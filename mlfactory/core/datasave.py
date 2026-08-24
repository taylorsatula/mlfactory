"""Lab-notebook data saving for mlfactory.

Every datum an experiment writes to disk goes through :func:`datasave`. It writes
the payload to the run's ``artifacts/`` directory, attaches a human-readable
**title** and a two-sentence **description** (the lab-notebook "label" for that
datum), drops a sidecar ``<file>.meta.json`` label card next to the data so a
scientist browsing the folder can read what each file is, and registers a
:class:`~mlfactory.core.manifest.FileRecord` (sha256 + metadata) into the run
manifest so the registry can find it across runs.

Lab model
---------
A laboratory records two kinds of metadata for a generated datum:

* **Provenance** — who/what ran it, when, on what hardware, with what code, and
  derived from what. This does not vary per artifact, so the run manifest
  captures it once (git commit, environment, hardware, parent runs, spec).
  ``datasave`` pulls it automatically; the caller never re-enters it.

* **Meaning** — what a scientist writes on the sample label: a title, a short
  description, the data format/schema, tags for the drawer, caveats ("do not
  use for training"), and sensitivity (restricted / PII). This varies per
  artifact and only the caller knows it. ``datasave`` requires ``title`` and
  ``description``; the rest are optional.

So: ``title`` and ``description`` are the only required arguments. They are the
only things the code cannot derive on its own. Provenance is inherited from the
run manifest; everything else is optional labelling.

Model checkpoints are saved with :func:`mlfactory.core.artifacts.save_checkpoint`
(which accepts the same ``title``/``description``); ``datasave`` covers every
other data format: json, jsonl, csv/tsv, text, yaml, numpy (.npy/.npz), parquet,
and raw bytes.

Example
-------
::

    from mlfactory.core.datasave import DataSaver, finalize_artifacts

    saver = DataSaver(self.run_dir, self.manifest)
    saver.save("chunks.jsonl", chunk_records,
               title="Chunked corpus",
               description="Input text split into fixed-size chunks for "
                           "classification. Each row carries word/sentence "
                           "statistics for one chunk.",
               tags=["corpus", "sample"], format="jsonl")
    ...
    # in finalize(): one line replaces the rglob + sha256 + FileRecord boilerplate
    finalize_artifacts(self.manifest, self.run_dir)
"""
from __future__ import annotations

import csv as _csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mlfactory.core.manifest import FileRecord, RunManifest, sha256_file

# Suffix appended to data files for their sidecar label card.
META_SUFFIX = ".meta.json"

# Per-artifact metadata fields stored on FileRecord (besides path/sha/role/size).
META_FIELDS: tuple[str, ...] = (
    "name", "title", "description", "format", "tags",
    "caveats", "sensitivity", "data_schema", "created_at",
)


# ---------------------------------------------------------------------------
# path / format resolution
# ---------------------------------------------------------------------------

def _resolve_run_dir(manifest: RunManifest | None, run_dir: str | Path | None) -> Path:
    if run_dir is not None:
        return Path(run_dir)
    if manifest is not None:
        return (
            Path(manifest.source.path).parent
            if manifest.source
            else Path("runs") / manifest.run_id
        )
    return Path(".")


def _resolve_dest(path: str | Path, run_dir: Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return run_dir / "artifacts" / p


def _detect_format(path: Path, data: Any, fmt: str) -> str:
    if fmt != "auto":
        return fmt
    suffix = path.suffix.lower()
    if suffix in {".jsonl", ".ndjson"}:
        return "jsonl"
    if suffix == ".json":
        return "json"
    if suffix == ".csv":
        return "csv"
    if suffix == ".tsv":
        return "tsv"
    if suffix in {".yaml", ".yml"}:
        return "yaml"
    if suffix == ".npy":
        return "numpy"
    if suffix == ".npz":
        return "npz"
    if suffix == ".parquet":
        return "parquet"
    if isinstance(data, (bytes, bytearray)):
        return "bytes"
    if isinstance(data, str):
        return "text"
    if isinstance(data, list):
        return "jsonl"
    return "json"


def _coerce_suffix(dest: Path, fmt: str) -> Path:
    """Ensure ``dest``'s suffix matches ``fmt`` for formats that append one.

    ``np.save``/``np.savez``/``pq.write_table`` silently append their own suffix
    when missing, which would write to a different path than the one we register
    and label. Coerce the path first so the sidecar and manifest agree.
    """
    desired = {
        "numpy": ".npy",
        "npz": ".npz",
        "parquet": ".parquet",
    }
    if fmt in desired and dest.suffix.lower() != desired[fmt]:
        return dest.with_suffix(desired[fmt])
    return dest


# ---------------------------------------------------------------------------
# payload writers
# ---------------------------------------------------------------------------

def _write_json_atomic(path: Path, data: Any, sort_keys: bool = True) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, sort_keys=sort_keys, default=str)
        f.flush()
    tmp.replace(path)


def _write_payload(
    dest: Path, data: Any, fmt: str, *, append: bool, sort_keys: bool
) -> None:
    if fmt == "json":
        _write_json_atomic(dest, data, sort_keys=sort_keys)
    elif fmt == "jsonl":
        mode = "a" if append else "w"
        with open(dest, mode, encoding="utf-8") as f:
            for row in data:
                f.write(
                    json.dumps(row, ensure_ascii=False, sort_keys=sort_keys, default=str)
                    + "\n"
                )
    elif fmt in ("csv", "tsv"):
        delim = "\t" if fmt == "tsv" else ","
        with open(dest, "w", encoding="utf-8", newline="") as f:
            if data and isinstance(data[0], dict):
                writer = _csv.DictWriter(f, fieldnames=list(data[0].keys()), delimiter=delim)
                writer.writeheader()
                writer.writerows(data)
            else:
                writer = _csv.writer(f, delimiter=delim)
                writer.writerows(data)
    elif fmt == "yaml":
        import yaml

        dest.write_text(
            yaml.safe_dump(data, sort_keys=sort_keys, allow_unicode=True),
            encoding="utf-8",
        )
    elif fmt == "text":
        dest.write_text(data, encoding="utf-8")
    elif fmt == "bytes":
        dest.write_bytes(bytes(data))
    elif fmt == "numpy":
        import numpy as np

        np.save(dest, data, allow_pickle=False)
    elif fmt == "npz":
        import numpy as np

        if isinstance(data, dict):
            np.savez_compressed(dest, **data)
        else:
            np.savez_compressed(dest, data=data)
    elif fmt == "parquet":
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as exc:  # pragma: no cover - exercised when pyarrow absent
            raise RuntimeError(
                "datasave: parquet output requires pyarrow — install with "
                "`python3.14 -m pip install pyarrow`"
            ) from exc
        table = data if isinstance(data, pa.Table) else pa.Table.from_pylist(data)
        pq.write_table(table, dest)
    else:
        raise ValueError(
            f"datasave: unknown format {fmt!r}; use one of "
            "json|jsonl|csv|tsv|text|yaml|numpy|npz|parquet|bytes "
            "(checkpoints use save_checkpoint)"
        )


# ---------------------------------------------------------------------------
# metadata + sidecar + manifest registration
# ---------------------------------------------------------------------------

def _build_meta(
    dest: Path,
    sha: str,
    size: int,
    *,
    name: str,
    title: str,
    description: str,
    fmt: str,
    tags: list[str] | None,
    caveats: str | None,
    sensitivity: str | None,
    schema: dict | None,
    manifest: RunManifest | None,
) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "name": name,
        "title": title,
        "description": description,
        "format": fmt,
        "tags": tags or [],
        "caveats": caveats,
        "sensitivity": sensitivity,
        "data_schema": schema,
        "path": str(dest.resolve()),
        "sha256": sha,
        "size_bytes": size,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    if manifest is not None:
        meta["run_id"] = manifest.run_id
        meta["stage"] = manifest.stage
        meta["git_commit"] = manifest.git.commit
        meta["parent_runs"] = list(manifest.parent_runs)
    return meta


def _write_meta_sidecar(dest: Path, meta: dict[str, Any]) -> Path:
    """Write ``<dest>.meta.json`` — the label card next to the data file."""
    sidecar = dest.with_name(dest.name + META_SUFFIX)
    _write_json_atomic(sidecar, meta, sort_keys=False)
    return sidecar


def _is_empty_meta(val: Any) -> bool:
    return val is None or (isinstance(val, (list, str, tuple)) and len(val) == 0)


def _register_artifact(
    manifest: RunManifest,
    dest: Path,
    sha: str,
    size: int,
    meta: dict[str, Any],
    role_rel: str,
) -> None:
    """Upsert a FileRecord for ``dest`` into manifest.artifacts (by path).

    On upsert (e.g. an append to an existing jsonl), metadata fields the new
    save omits are inherited from the existing record so appending to a file
    never silently drops its title/tags/caveats. ``created_at`` is preserved
    from the first save (the artifact's birth time, not the last append).
    """
    target = dest.resolve().as_posix()
    for i, existing in enumerate(manifest.artifacts):
        if Path(existing.path).resolve().as_posix() == target:
            for k in META_FIELDS:
                if _is_empty_meta(meta.get(k)):
                    meta[k] = getattr(existing, k)
            # created_at records the artifact's creation, not the last append.
            if existing.created_at:
                meta["created_at"] = existing.created_at
            manifest.artifacts[i] = FileRecord(
                path=str(dest.resolve()),
                sha256=sha,
                role=f"artifact:{role_rel}",
                size_bytes=size,
                **{k: meta.get(k) for k in META_FIELDS},
            )
            return
    manifest.artifacts.append(
        FileRecord(
            path=str(dest.resolve()),
            sha256=sha,
            role=f"artifact:{role_rel}",
            size_bytes=size,
            **{k: meta.get(k) for k in META_FIELDS},
        )
    )


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

def datasave(
    path: str | Path,
    data: Any,
    *,
    title: str,
    description: str,
    manifest: RunManifest | None = None,
    run_dir: str | Path | None = None,
    name: str | None = None,
    format: str = "auto",  # noqa: A002 - mirror a "format" kwarg on purpose
    tags: list[str] | None = None,
    caveats: str | None = None,
    sensitivity: str | None = None,
    schema: dict[str, Any] | None = None,
    append: bool = False,
    sort_keys: bool = True,
) -> Path:
    """Save ``data`` to ``<run_dir>/artifacts/<path>`` with lab-notebook metadata.

    Parameters
    ----------
    path:
        Filename relative to ``<run_dir>/artifacts/`` (or an absolute path). The
        suffix selects the format when ``format="auto"``.
    data:
        The payload. Interpreted per ``format`` (list[dict] for jsonl/csv, dict
        for json/yaml, str for text, bytes for bytes, ndarray for numpy, dict of
        arrays for npz, list[dict] for parquet).
    title, description:
        Required human label. ``description`` should be ~two sentences: what the
        data is, and how it was made / what it measures.
    manifest:
        Run manifest to register the artifact into (enables registry discovery).
        When omitted, the sidecar label is still written but no FileRecord is
        created — this lets standalone scripts (no plugin) still label data.
    run_dir:
        Run directory; defaults to the manifest's run dir, then ``"."``.
    name:
        Stable slug key (defaults to the file stem). Used as the artifact's
        ``name`` and for checkpoint labels.
    format:
        ``auto`` (default) or one of json|jsonl|csv|tsv|text|yaml|numpy|npz|
        parquet|bytes. Checkpoints are not supported here — use
        :func:`mlfactory.core.artifacts.save_checkpoint`.
    tags, caveats, sensitivity, schema:
        Optional lab metadata. ``caveats`` holds known-issue warnings
        (e.g. "do not use for training"); ``sensitivity`` is public|internal|
        restricted; ``schema`` describes columns/keys/units so the data is
        loadable without guessing.
    append:
        For jsonl: append instead of overwriting.
    sort_keys:
        For json/jsonl/yaml: sort object keys.

    Returns
    -------
    Path
        The absolute path of the data file written (the sidecar is alongside).
    """
    title = (title or "").strip()
    description = (description or "").strip()
    if not title:
        raise ValueError("datasave: title is required (a short human name for the data)")
    if not description:
        raise ValueError(
            "datasave: description is required (~2 sentences: what it is + "
            "how it was made / what it measures)"
        )

    run_dir = _resolve_run_dir(manifest, run_dir)
    dest = _resolve_dest(path, run_dir)
    fmt = _detect_format(dest, data, format)
    if fmt == "checkpoint":
        raise ValueError(
            "datasave: format='checkpoint' is not supported here — use "
            "mlfactory.core.artifacts.save_checkpoint (it accepts the same "
            "title/description)"
        )
    dest = _coerce_suffix(dest, fmt)
    dest.parent.mkdir(parents=True, exist_ok=True)

    _write_payload(dest, data, fmt, append=append, sort_keys=sort_keys)
    sha = sha256_file(dest)
    size = dest.stat().st_size
    name = name or dest.stem
    meta = _build_meta(
        dest, sha, size,
        name=name, title=title, description=description, fmt=fmt,
        tags=tags, caveats=caveats, sensitivity=sensitivity, schema=schema,
        manifest=manifest,
    )
    _write_meta_sidecar(dest, meta)

    if manifest is not None:
        artifacts_dir = run_dir / "artifacts"
        try:
            role_rel = dest.relative_to(artifacts_dir).as_posix()
        except ValueError:
            role_rel = dest.name
        _register_artifact(manifest, dest, sha, size, meta, role_rel)
        manifest.write(run_dir / "manifest.json")

    return dest.resolve()


class DataSaver:
    """Convenience binder of (run_dir, manifest) for repeated ``datasave`` calls.

    Equivalent to calling :func:`datasave` with the same ``manifest``/``run_dir``
    each time; ``finalize()`` replaces the plugin ``finalize()`` rglob boilerplate.
    """

    def __init__(
        self,
        run_dir: str | Path | None = None,
        manifest: RunManifest | None = None,
    ) -> None:
        self.run_dir = _resolve_run_dir(manifest, run_dir)
        self.manifest = manifest

    def save(self, path: str | Path, data: Any, *, title: str, description: str, **kw: Any) -> Path:
        kw.pop("manifest", None)
        kw.pop("run_dir", None)
        return datasave(
            path, data, title=title, description=description,
            manifest=self.manifest, run_dir=self.run_dir, **kw,
        )

    def finalize(self, include_logs: bool = True) -> None:
        finalize_artifacts(self.manifest, self.run_dir, include_logs=include_logs)


def _register_tree(
    root: Path,
    records: list[FileRecord],
    role_prefix: str,
) -> None:
    """Append FileRecords for files under ``root`` not already in ``records``.

    Skips sidecar ``.meta.json`` label cards (they are labels, not data) and
    de-duplicates by resolved path so artifacts already registered by
    :func:`datasave` keep their metadata instead of being overwritten by a bare
    record.
    """
    if not root.exists():
        return
    existing = {Path(r.path).resolve().as_posix() for r in records if r.path}
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        if p.name.endswith(META_SUFFIX):
            continue
        rp = p.resolve()
        if rp.as_posix() in existing:
            continue
        rel = p.relative_to(root)
        records.append(
            FileRecord(
                path=str(rp),
                sha256=sha256_file(p),
                role=f"{role_prefix}{rel}",
                size_bytes=p.stat().st_size,
            )
        )


def finalize_artifacts(
    manifest: RunManifest | None,
    run_dir: str | Path | None = None,
    *,
    include_logs: bool = True,
) -> None:
    """Hash any artifacts/logs not already registered and persist the manifest.

    Replaces the duplicated ``rglob`` + ``sha256_file`` + ``FileRecord`` block
    found in every plugin's ``finalize()``. Files already registered by
    :func:`datasave` (which carry metadata) are kept as-is; only stragglers
    (e.g. files written by code that has not migrated, or log files) get bare
    records. Sidecar ``.meta.json`` label cards are never registered.
    """
    run_dir = _resolve_run_dir(manifest, run_dir)
    _register_tree(run_dir / "artifacts", manifest.artifacts if manifest else [], "artifact:")
    if include_logs and manifest is not None:
        _register_tree(run_dir / "logs", manifest.logs, "log:")
    if manifest is not None:
        manifest.write(run_dir / "manifest.json")


def register_checkpoint_dir(
    manifest: RunManifest,
    run_dir: str | Path,
    ckpt_dir: str | Path,
    *,
    title: str,
    description: str,
    name: str | None = None,
    tags: list[str] | None = None,
    caveats: str | None = None,
    sensitivity: str | None = None,
    data_schema: dict[str, Any] | None = None,
    fmt: str = "checkpoint",
) -> Path:
    """Label an existing checkpoint directory and register its files into the manifest.

    For checkpoints written by ``model.save_pretrained()`` or any other writer
    that produces a directory of files. Each file gets a :class:`FileRecord`
    stamped with the checkpoint's lab metadata — one label for the whole
    checkpoint, mirroring :func:`mlfactory.core.artifacts.save_checkpoint`. A
    sidecar ``<ckpt_dir>.meta.json`` label card is written next to the directory.
    Existing records for the same files are upserted (re-saving a checkpoint
    updates rather than duplicates).
    """
    run_dir = _resolve_run_dir(manifest, run_dir)
    ckpt_dir = Path(ckpt_dir)
    name = name or ckpt_dir.name
    created_at = datetime.now(timezone.utc).isoformat()

    existing = {
        Path(a.path).resolve().as_posix(): i
        for i, a in enumerate(manifest.artifacts)
    }
    for p in sorted(ckpt_dir.rglob("*")):
        if not p.is_file():
            continue
        rec = FileRecord(
            path=str(p.resolve()),
            sha256=sha256_file(p),
            role=f"artifact:{name}/{p.relative_to(ckpt_dir)}",
            size_bytes=p.stat().st_size,
            name=name,
            title=title,
            description=description,
            format=fmt,
            tags=tags or [],
            caveats=caveats,
            sensitivity=sensitivity,
            data_schema=data_schema,
            created_at=created_at,
        )
        key = Path(rec.path).resolve().as_posix()
        if key in existing:
            manifest.artifacts[existing[key]] = rec
        else:
            existing[key] = len(manifest.artifacts)
            manifest.artifacts.append(rec)

    _write_meta_sidecar(ckpt_dir, {
        "name": name,
        "title": title,
        "description": description,
        "format": fmt,
        "tags": tags or [],
        "caveats": caveats,
        "sensitivity": sensitivity,
        "data_schema": data_schema,
        "path": str(ckpt_dir.resolve()),
        "sha256": None,
        "size_bytes": None,
        "created_at": created_at,
        "run_id": manifest.run_id,
        "stage": manifest.stage,
        "git_commit": manifest.git.commit,
        "parent_runs": list(manifest.parent_runs),
    })
    manifest.write(run_dir / "manifest.json")
    return ckpt_dir


def read_catalog(run_dir: str | Path) -> list[FileRecord]:
    """Return the labeled artifacts (those with a title) for a run directory.

    Reads the run's ``manifest.json``; useful for browsing what a run produced
    without scanning the registry. Cross-run discovery uses
    :meth:`mlfactory.core.registry.Registry.find_artifacts`.
    """
    mp = Path(run_dir) / "manifest.json"
    if not mp.exists():
        return []
    m = RunManifest.read(mp)
    return [a for a in m.artifacts if a.title]
