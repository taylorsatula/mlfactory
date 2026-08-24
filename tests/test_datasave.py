"""Tests for the lab-notebook datasave feature."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from mlfactory.core.datasave import (
    DataSaver,
    datasave,
    finalize_artifacts,
    read_catalog,
    register_checkpoint_dir,
)
from mlfactory.core.manifest import FileRecord, RunManifest
from mlfactory.core.registry import Registry


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

def _make_manifest(run_dir: Path, run_id: str = "t.transform.001") -> RunManifest:
    from mlfactory.core.manifest import (
        EnvironmentInfo,
        GitInfo,
        HardwareInfo,
        SourceArchive,
    )

    m = RunManifest(
        run_id=run_id,
        stage="transform",
        spec={},
        git=GitInfo(commit="abc123", dirty=False, branch="main"),
        source=SourceArchive(path=str(run_dir / "source.tar.gz"), sha256="0" * 64),
        env=EnvironmentInfo(),
        hardware=HardwareInfo(),
    )
    m.write(run_dir / "manifest.json")
    return m


@pytest.fixture()
def run(tmp_path: Path) -> tuple[Path, RunManifest]:
    run_dir = tmp_path / "runs" / "t.transform.001"
    (run_dir / "artifacts").mkdir(parents=True)
    (run_dir / "logs").mkdir(parents=True)
    return run_dir, _make_manifest(run_dir)


# ---------------------------------------------------------------------------
# core contract: title + description required, metadata attached
# ---------------------------------------------------------------------------

def test_title_and_description_required(run: tuple[Path, RunManifest]) -> None:
    run_dir, m = run
    with pytest.raises(ValueError, match="title is required"):
        datasave("x.json", {"a": 1}, title="", description="d", manifest=m, run_dir=run_dir)
    with pytest.raises(ValueError, match="description is required"):
        datasave("x.json", {"a": 1}, title="T", description="   ", manifest=m, run_dir=run_dir)


def test_jsonl_saved_with_metadata_and_sidecar(run: tuple[Path, RunManifest]) -> None:
    run_dir, m = run
    dest = datasave(
        "chunks.jsonl",
        [{"i": 1, "t": "hello"}, {"i": 2, "t": "world"}],
        title="Chunked corpus",
        description="Input text split into chunks. Each row is one chunk with stats.",
        tags=["corpus", "sample"],
        manifest=m,
        run_dir=run_dir,
    )
    assert dest == (run_dir / "artifacts" / "chunks.jsonl").resolve()
    lines = dest.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["i"] == 1

    # FileRecord registered with full metadata.
    rec = [a for a in m.artifacts if a.name == "chunks"][0]
    assert rec.title == "Chunked corpus"
    assert rec.format == "jsonl"
    assert rec.tags == ["corpus", "sample"]
    assert rec.sha256 and len(rec.sha256) == 64
    assert rec.role == "artifact:chunks.jsonl"

    # Sidecar label card exists next to the data and carries provenance.
    sidecar = dest.with_name(dest.name + ".meta.json")
    assert sidecar.exists()
    card = json.loads(sidecar.read_text(encoding="utf-8"))
    assert card["title"] == "Chunked corpus"
    assert card["run_id"] == m.run_id
    assert card["git_commit"] == "abc123"
    assert card["sha256"] == rec.sha256

    # The sidecar is NOT registered as an artifact.
    assert not any("meta.json" in a.path for a in m.artifacts)


# ---------------------------------------------------------------------------
# format dispatch
# ---------------------------------------------------------------------------

def test_format_autodetect_and_write(run: tuple[Path, RunManifest]) -> None:
    run_dir, m = run
    datasave("a.json", {"x": 1}, title="J", description="d. d.", manifest=m, run_dir=run_dir)
    datasave("b.jsonl", [{"x": 1}], title="L", description="d. d.", manifest=m, run_dir=run_dir)
    datasave("c.csv", [{"a": 1, "b": 2}], title="C", description="d. d.", manifest=m, run_dir=run_dir)
    datasave("d.npy", np.arange(3), title="N", description="d. d.", manifest=m, run_dir=run_dir)
    datasave("e.md", "# hi", title="T", description="d. d.", manifest=m, run_dir=run_dir)
    datasave("f.yaml", {"k": "v"}, title="Y", description="d. d.", manifest=m, run_dir=run_dir)
    by_name = {a.name: a.format for a in m.artifacts}
    assert by_name == {"a": "json", "b": "jsonl", "c": "csv", "d": "numpy", "e": "text", "f": "yaml"}
    # contents are correct
    assert json.loads((run_dir / "artifacts" / "a.json").read_text())["x"] == 1
    assert (run_dir / "artifacts" / "c.csv").read_text().strip() == "a,b\r\n1,2" or \
           (run_dir / "artifacts" / "c.csv").read_text().strip() == "a,b\n1,2"
    assert np.load(run_dir / "artifacts" / "d.npy").tolist() == [0, 1, 2]


def test_numpy_suffix_coercion(run: tuple[Path, RunManifest]) -> None:
    run_dir, m = run
    # passing a .bin path but format numpy must coerce to .npy so the registered
    # path matches what np.save actually wrote.
    dest = datasave("arr.bin", np.arange(2), format="numpy",
                    title="N", description="d. d.", manifest=m, run_dir=run_dir)
    assert dest.suffix == ".npy"
    rec = [a for a in m.artifacts if a.name == "arr"][0]
    assert Path(rec.path).exists()  # registered path points at the real file


def test_parquet_missing_pyarrow_raises_clearly(run: tuple[Path, RunManifest]) -> None:
    run_dir, m = run
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        with pytest.raises(RuntimeError, match="parquet output requires pyarrow"):
            datasave("t.parquet", [{"a": 1}], title="P", description="d. d.",
                     manifest=m, run_dir=run_dir)
    else:
        pytest.skip("pyarrow is installed; clear-error path not exercisable")


def test_checkpoint_format_rejected_with_pointer(run: tuple[Path, RunManifest]) -> None:
    run_dir, m = run
    with pytest.raises(ValueError, match="save_checkpoint"):
        datasave("ck", None, format="checkpoint", title="C", description="d. d.",
                 manifest=m, run_dir=run_dir)


# ---------------------------------------------------------------------------
# upsert / append semantics: metadata survives append
# ---------------------------------------------------------------------------

def test_append_preserves_metadata(run: tuple[Path, RunManifest]) -> None:
    run_dir, m = run
    datasave("chunks.jsonl", [{"i": 1}], title="Chunked corpus",
             description="Input text split into chunks. Each row is one chunk.",
             tags=["corpus"], manifest=m, run_dir=run_dir)
    # append without re-passing tags — tags must be inherited, not wiped
    datasave("chunks.jsonl", [{"i": 2}], title="Chunked corpus",
             description="Input text split into chunks. Each row is one chunk.",
             manifest=m, run_dir=run_dir, append=True)
    rec = [a for a in m.artifacts if a.name == "chunks"][0]
    assert rec.tags == ["corpus"]
    assert (run_dir / "artifacts" / "chunks.jsonl").read_text().count("\n") == 2
    # exactly one FileRecord for the path (upsert, not duplicate)
    assert sum(1 for a in m.artifacts if a.name == "chunks") == 1


def test_upsert_replaces_record(run: tuple[Path, RunManifest]) -> None:
    run_dir, m = run
    datasave("a.json", {"x": 1}, title="Old", description="d. d.", tags=["t"],
             manifest=m, run_dir=run_dir)
    datasave("a.json", {"x": 2}, title="New", description="d. d.", tags=["t"],
             manifest=m, run_dir=run_dir)
    recs = [a for a in m.artifacts if Path(a.path).name == "a.json"]
    assert len(recs) == 1
    assert recs[0].title == "New"


# ---------------------------------------------------------------------------
# finalize_artifacts: de-dup, skip sidecars, catch stragglers
# ---------------------------------------------------------------------------

def test_finalize_artifacts_dedups_and_skips_sidecars(run: tuple[Path, RunManifest]) -> None:
    run_dir, m = run
    datasave("chunks.jsonl", [{"i": 1}], title="Chunked corpus",
             description="Input text split into chunks. Each row is one chunk.",
             tags=["corpus"], manifest=m, run_dir=run_dir)
    # a straggler not saved via datasave (e.g. a log file written manually)
    (run_dir / "logs" / "train.log").write_text("training...\n", encoding="utf-8")
    # and a stray artifact file written without datasave
    (run_dir / "artifacts" / "manual.json").write_text('{"k": 1}', encoding="utf-8")

    finalize_artifacts(m, run_dir)

    chunks_recs = [a for a in m.artifacts if a.name == "chunks"]
    manual_recs = [a for a in m.artifacts if Path(a.path).name == "manual.json"]
    log_recs = [r for r in m.logs]
    # chunks keeps its metadata record (not replaced by a bare one)
    assert len(chunks_recs) == 1
    assert chunks_recs[0].title == "Chunked corpus"
    # manual.json caught as a bare record (no title)
    assert len(manual_recs) == 1
    assert manual_recs[0].title is None
    # log registered
    assert len(log_recs) == 1
    # no sidecar registered as an artifact
    assert not any("meta.json" in a.path for a in m.artifacts)


# ---------------------------------------------------------------------------
# checkpoint dir labeling
# ---------------------------------------------------------------------------

def test_register_checkpoint_dir_labels_all_files(run: tuple[Path, RunManifest]) -> None:
    run_dir, m = run
    ckpt = run_dir / "artifacts" / "checkpoint-final"
    ckpt.mkdir(parents=True)
    (ckpt / "weights.npy").write_bytes(b"\x93NUMPY")
    (ckpt / "config.json").write_text('{"labels": ["a", "b"]}')
    register_checkpoint_dir(
        m, run_dir, ckpt,
        title="Numpy softmax checkpoint",
        description="Lightweight numpy softmax classifier checkpoint. Holds weights and labels.",
        name="checkpoint-final",
        data_schema={"files": {"weights.npy": "float64 [vocab, labels]"}},
    )
    recs = [a for a in m.artifacts if a.name == "checkpoint-final"]
    assert len(recs) == 2  # one per file
    assert all(r.title == "Numpy softmax checkpoint" for r in recs)
    assert all(r.format == "checkpoint" for r in recs)
    assert {Path(r.path).name for r in recs} == {"weights.npy", "config.json"}
    assert (run_dir / "artifacts" / "checkpoint-final.meta.json").exists()


def test_save_checkpoint_attaches_metadata(run: tuple[Path, RunManifest]) -> None:
    run_dir, m = run

    class FakeModel:
        def save_pretrained(self, d: Path) -> None:
            (d / "adapter_model.safetensors").write_bytes(b"weights")

    from mlfactory.core.artifacts import save_checkpoint
    save_checkpoint(run_dir, step=10, model=FakeModel(), manifest=m,
                    title="DFT policy checkpoint",
                    description="LoRA policy adapter saved at step 10. Consumed by the eval stage.")
    recs = [a for a in m.artifacts if a.title == "DFT policy checkpoint"]
    assert len(recs) == 1
    assert recs[0].name == "checkpoint-10"
    assert recs[0].format == "checkpoint"


# ---------------------------------------------------------------------------
# registry cross-run discovery + read_catalog
# ---------------------------------------------------------------------------

def test_find_artifacts_and_read_catalog(run: tuple[Path, RunManifest]) -> None:
    run_dir, m = run
    datasave("chunks.jsonl", [{"i": 1}], title="Chunked corpus",
             description="Input text split into chunks. Each row is one chunk.",
             tags=["corpus", "sample"], manifest=m, run_dir=run_dir)
    datasave("stats.json", {"n": 1}, title="Aggregate statistics",
             description="Per-run aggregate counts. Computed from chunks.jsonl.",
             manifest=m, run_dir=run_dir)
    finalize_artifacts(m, run_dir)

    reg = Registry(run_dir.parent.parent / "registry.db")
    reg.register(m)

    by_tag = reg.find_artifacts(tag="corpus")
    assert [a["name"] for a in by_tag] == ["chunks"]
    by_title = reg.find_artifacts(title="statistics")
    assert [a["name"] for a in by_title] == ["stats"]
    assert reg.find_artifacts(tag="nonexistent") == []

    cat = read_catalog(run_dir)
    assert {c.name for c in cat} == {"chunks", "stats"}
    assert all(c.title for c in cat)


# ---------------------------------------------------------------------------
# standalone (no manifest) use: sidecar still written
# ---------------------------------------------------------------------------

def test_standalone_without_manifest_writes_sidecar(tmp_path: Path) -> None:
    dest = datasave(
        tmp_path / "out.json", {"a": 1},
        title="Standalone report",
        description="A report saved outside a plugin run. No manifest is attached.",
    )
    assert dest.exists()
    sidecar = dest.with_name(dest.name + ".meta.json")
    assert sidecar.exists()
    card = json.loads(sidecar.read_text())
    assert card["title"] == "Standalone report"
    assert "run_id" not in card  # no provenance without a manifest


# ---------------------------------------------------------------------------
# DataSaver convenience class
# ---------------------------------------------------------------------------

def test_data_saver_finalize(run: tuple[Path, RunManifest]) -> None:
    run_dir, m = run
    saver = DataSaver(run_dir, m)
    saver.save("chunks.jsonl", [{"i": 1}], title="Chunked corpus",
               description="Input text split into chunks. Each row is one chunk.",
               tags=["corpus"], format="jsonl")
    (run_dir / "logs" / "x.log").write_text("x\n")
    saver.finalize()
    assert [a.name for a in m.artifacts] == ["chunks"]
    assert m.logs and m.logs[0].role == "log:x.log"
