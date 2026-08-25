# Saving data with `datasave`

> Update when: `core/datasave.py`'s public API or semantics change. Source of
> truth: `mlfactory/core/datasave.py` (read its module docstring). This doc
> carries the *why* and *when*; the module carries the *what*.

## The rule (non-negotiable)

Every datum an experiment writes to disk goes through `datasave()` (or
`save_checkpoint()` for model checkpoints). Do not hand-write data with
`open()` + `json.dump`, `np.save`, pandas `to_csv`, etc. Files written
around `datasave` are invisible to the registry and to `find_artifacts`.

## Why: the lab-notebook model

A laboratory records two kinds of metadata for a generated datum:

- **Provenance** — who/what ran it, when, on what hardware, with what code,
  derived from what. This does not vary per artifact, so the run manifest
  captures it once (git commit, environment, hardware, parent runs, spec).
  `datasave` pulls it automatically; the caller never re-enters it.
- **Meaning** — what a scientist writes on the sample label: a title, a
  short description, the format/schema, tags, caveats, sensitivity. This
  varies per artifact and only the caller knows it.

So `datasave` requires only `title` and `description` — the two things the
code cannot derive. Everything else is optional.

## Required arguments

- `title` — short human name (e.g. `"Chunked corpus"`, `"DFT policy checkpoint"`).
- `description` — ~two sentences: what the data is, and how it was made / what it measures.

## Optional lab metadata

- `tags: list[str]` — findability keywords (queried via `Registry.find_artifacts(tag=...)`).
- `caveats: str` — known-issue warnings (e.g. `"batch 3 had a calibration drift — do not use for training"`).
- `sensitivity: str` — `public` | `internal` | `restricted` (privacy / human-subjects).
- `schema: dict` — how to load it: columns/keys, dtypes, units.
- `name: str` — stable slug key (defaults to the file stem).
- `format: str` — `auto` (default; inferred from suffix) or one of `json|jsonl|csv|tsv|text|yaml|numpy|npz|parquet|bytes`. Checkpoints use `save_checkpoint`, not this.
- `append: bool` — for jsonl, append instead of overwriting (metadata of the existing file is inherited; `created_at` preserved).
- `manifest: RunManifest | None` — register the artifact (enables `find_artifacts`); omit only for standalone scripts with no manifest.
- `run_dir` — defaults to the manifest's run dir, then `.`.

## What `datasave` does

Writes the file to `<run_dir>/artifacts/<path>`, drops a sidecar
`<file>.meta.json` label card next to it (so a scientist browsing the folder
can read what each file is), and registers a hashed `FileRecord` carrying
the metadata into the manifest — so `Registry.find_artifacts()` can
discover it across all runs.

## Forms

```python
from mlfactory.core.datasave import datasave, DataSaver, finalize_artifacts

# 1. one-off, with manifest
datasave("x.jsonl", rows, title="...", description="...", tags=[...], manifest=m)

# 2. plugin-bound (repeated saves with the same run_dir/manifest)
saver = DataSaver(self.run_dir, self.manifest)
saver.save("chunks.jsonl", chunk_records, title="Chunked corpus",
           description="...", tags=["corpus"], format="jsonl")
# the run summary is a normal datum too — mirror it onto manifest.summary
self.manifest.summary = summary
saver.save("summary.json", summary, title="Transform summary",
           description="...", format="json")
# in finalize(): one line replaces rglob + sha256 + FileRecord boilerplate
saver.finalize()            # or finalize_artifacts(self.manifest, self.run_dir)

# 3. standalone script, no manifest — still writes the sidecar label
datasave(path, data, title="...", description="...")   # no FileRecord created
```

## Model checkpoints

`save_checkpoint` (in `core/artifacts.py`) handles checkpoints written by
`model.save_pretrained()` — it accepts the same lab metadata
(`title`/`description`/`tags`/`caveats`/`sensitivity`/`schema`), attaches it
to every file in the checkpoint dir, and writes a `<ckpt_dir>.meta.json`
label. Defaults are sensible so existing callers keep working, but passing
explicit `title`/`description` makes the checkpoint discoverable.

For a checkpoint directory you wrote yourself (e.g. a numpy model with no
`save_pretrained`), use `register_checkpoint_dir(manifest, run_dir,
ckpt_dir, title=..., description=...)` to label it after the fact.

## Discovering data across runs

```python
registry.find_artifacts(tag="corpus")              # by tag
registry.find_artifacts(title="statistics")        # title substring (case-insensitive)
registry.find_artifacts(stage="train", format="checkpoint")
```

Only artifacts carrying a `title` (from `datasave` or `save_checkpoint`)
are returned. Bare `finalize_artifacts` records (logs, stragglers) are
not in the catalog.

## Browsing one run's artifacts

```python
from mlfactory.core.datasave import read_catalog
for art in read_catalog(run_dir):
    print(art.title, art.path)
```

## Common mistakes

- Saving data without `title`/`description` — `datasave` raises; they are
  the only provenance the code cannot derive.
- Hand-rolling the `finalize()` rglob+sha256+`FileRecord` loop — use
  `finalize_artifacts(self.manifest, self.run_dir)`.
- Writing outputs outside `self.run_dir` — artifacts belong under
  `<run_dir>/artifacts/`.
