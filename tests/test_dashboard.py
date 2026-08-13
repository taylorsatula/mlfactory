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
