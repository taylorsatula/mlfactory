from __future__ import annotations

import json
from pathlib import Path

from mlfactory.core.dashboard import _load_metric_history, _plot_metric
from mlfactory.core.dashboard_config import ExperimentDashboardConfig


def test_metric_history_reads_scalar_and_combined_jsonl_rows(tmp_path: Path) -> None:
    path = tmp_path / "dashboard.jsonl"
    rows = [
        {"key": "loss", "value": 3.0, "step": 1},
        {"step": 1, "loss": 3.0, "grad_norm": 2.0},
        {"key": "loss", "value": 2.0, "step": 2},
        {"step": 2, "loss": 2.0, "grad_norm": 1.5},
        {"event": "checkpoint_saved", "detail": {"step": 2}},
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    history = _load_metric_history(path, ["loss", "grad_norm"], max_points=10)

    assert history["loss"] == [(1, 3.0), (2, 2.0)]
    assert history["grad_norm"] == [(1, 2.0), (2, 1.5)]


def test_ascii_plot_has_axes_and_data_marker() -> None:
    plot = _plot_metric([(1, 3.0), (2, 2.0), (3, 2.5)], width=20, height=4, marker="L")

    assert len(plot) == 6  # four rows, x-axis, and step labels
    assert any("L" in line for line in plot)
    assert any("+" in line for line in plot)


def test_voice_training_dashboard_config_loads() -> None:
    path = Path(__file__).parents[1] / "mlfactory/experiments/voice/dashboard_voice-train.json"
    config = ExperimentDashboardConfig.load(path)

    assert any(p.type == "training_chart" for p in config.panes)
    assert any(p.type == "runs_table" for p in config.panes)
    assert config.progress is not None
    assert config.get_probe("last_grad_norm").type == "jsonl_metric_last"


def test_stage_plugin_dashboard_default_is_none() -> None:
    """The base class opts out by default; subclasses override to declare."""
    from mlfactory.plugins.base import StagePlugin

    assert StagePlugin.dashboard() is None


def test_sample_plugins_declare_dashboards() -> None:
    """Each sample stage returns a config from its dashboard() classmethod."""
    from mlfactory.experiments.sample.transform_plugin import SampleTransformPlugin
    from mlfactory.experiments.sample.classify_plugin import SampleClassifyPlugin
    from mlfactory.experiments.sample.train_plugin import SampleTrainPlugin
    from mlfactory.experiments.sample.eval_plugin import SampleEvalPlugin

    for plugin_cls in (SampleTransformPlugin, SampleClassifyPlugin, SampleTrainPlugin, SampleEvalPlugin):
        cfg = plugin_cls.dashboard()
        assert cfg is not None, f"{plugin_cls.__name__} declares no dashboard"
        assert cfg.experiment == "sample"
        assert len(cfg.probes) > 0


def test_create_run_persists_dashboard_to_run_dir(tmp_path: Path) -> None:
    """create_run writes dashboard.json into the run dir so the viewer resolves
    the config from the run itself, not from the source tree at view time."""
    import os
    import subprocess

    from mlfactory.core.runner import create_run

    spec = Path(__file__).parents[1] / "mlfactory/experiments/sample/specs/sample_transform.yaml"
    # Use an absolute runs_dir so paths stay valid regardless of cwd.
    runs_dir = tmp_path / "runs"
    # create_run archives the source from repo_dir (cwd), so run from the repo
    # root where the git tree lives — no tmpdir chdir needed.
    cwd_before = Path.cwd()
    assert cwd_before.name == "mlfactory", "test must run from repo root"
    try:
        m = create_run(spec, runs_dir=runs_dir)
    finally:
        os.chdir(cwd_before)
    run_dir = Path(m.source.path).parent
    assert (run_dir / "dashboard.json").exists(), "dashboard.json not persisted to run dir"
    # The viewer reads it from the run dir.
    from mlfactory.core.dashboard import _load_experiment_config
    cfg = _load_experiment_config(m)
    assert cfg.experiment == "sample"
    assert any(p.id == "chunks_count" for p in cfg.probes)


def test_create_run_dashboard_opt_out(tmp_path: Path) -> None:
    """dashboard: none in the spec suppresses dashboard.json persistence."""
    import os
    import subprocess
    import yaml

    from mlfactory.core.runner import create_run

    src = Path(__file__).parents[1] / "mlfactory/experiments/sample/specs/sample_transform.yaml"
    runs_dir = tmp_path / "runs"
    cwd_before = Path.cwd()
    try:
        spec_dict = yaml.safe_load(src.read_text())
        spec_dict["dashboard"] = "none"
        spec_path = tmp_path / "spec_optout.yaml"
        spec_path.write_text(yaml.safe_dump(spec_dict))
        m = create_run(spec_path, runs_dir=runs_dir)
    finally:
        os.chdir(cwd_before)
    run_dir = Path(m.source.path).parent
    assert not (run_dir / "dashboard.json").exists(), "opt-out wrote dashboard.json"


def test_load_experiment_config_generic_fallback() -> None:
    """A manifest with no source/run_dir yields the generic config, not an error."""
    from mlfactory.core.dashboard import _load_experiment_config
    from mlfactory.core.manifest import RunManifest

    m = RunManifest(run_id="x", stage="collect", status="completed")
    cfg = _load_experiment_config(m)
    assert cfg.experiment == "generic"


def test_json_file_value_probe_reads_dotted_path(tmp_path: Path) -> None:
    """The json_file_value probe reads a dotted path from a JSON status file."""
    from mlfactory.core.dashboard import _run_probe
    from mlfactory.core.dashboard_config import Probe

    progress = tmp_path / "progress.json"
    progress.write_text(json.dumps({"eta": {"remaining_s": 300, "abs": "2026-08-24T05:00:00Z"}}), encoding="utf-8")
    probe = Probe(id="eta", type="json_file_value", file=str(progress), path="eta.remaining_s")
    result = _run_probe(probe, run_dir=tmp_path, manifest=None)
    assert result.value == 300
    assert "300" in result.display


def test_json_file_value_probe_missing_file_is_na(tmp_path: Path) -> None:
    """A missing file yields n/a (dim), not an error."""
    from mlfactory.core.dashboard import _run_probe
    from mlfactory.core.dashboard_config import Probe

    probe = Probe(id="eta", type="json_file_value", file=str(tmp_path / "nope.json"), path="eta")
    result = _run_probe(probe, run_dir=tmp_path, manifest=None)
    assert result.value is None
    assert result.display == "n/a"
    assert result.style == "dim"
