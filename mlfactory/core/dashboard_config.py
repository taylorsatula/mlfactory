"""Declarative dashboard configuration for mlfactory experiments.

Each experiment can expose a ``dashboard.json`` file that tells the reusable
dashboard which probes to run, how to interpret them, and how to lay out the
TUI. The goal is to support runs of any shape: generation jobs, classifiers,
training loops, eval harnesses, and future stages we have not thought of yet.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator


ProbeType = Literal[
    "const",
    "spec_value",
    "file_line_count",
    "regex_count",
    "regex_last_match",
    "jsonl_last_record",
    "json_file_value",
    "process_alive",
    "http_status",
    "http_json",
    "gpu_status",
    "disk_usage",
    "file_exists",
    "shell_command",
    "jsonl_metric_last",
]

PaneType = Literal[
    "overview",
    "metrics_table",
    "bars",
    "recent_log",
    "gpu_table",
    "run_info",
    "lineage",
    "text",
    "training_chart",
    "runs_table",
]


class Probe(BaseModel):
    """A single measurement the dashboard can poll.

    Probes are deliberately generic. An experiment declares *what* to measure,
    and the dashboard handles the *how*. This keeps the TUI reusable while
    letting each experiment surface the right signals.
    """

    id: str
    type: ProbeType
    enabled: bool = True
    label: str = ""
    # Polling interval in seconds. If None, poll every dashboard refresh.
    interval: float | None = None
    # Probe-specific arguments.
    params: dict[str, Any] = Field(default_factory=dict)
    # Optional shell command to run and capture stdout for parsing.
    command: str | None = None
    # For regex probes: the regex to apply.
    regex: str | None = None
    # For jsonl probes: a JMESPath-like dotted path into the last JSON record.
    path: str | None = None
    # For http probes: the URL.
    url: str | None = None
    # For file probes: the file path. May contain {run_dir} placeholder.
    file: str | None = None
    # For process probes: a file containing the PID, or an env var, or a fixed int.
    pid_file: str | None = None
    pid_env: str | None = None
    # For shell_command: how to parse stdout.
    parser: Literal["raw", "json", "float", "int"] = "raw"
    # Display formatting.
    unit: str = ""
    precision: int = 2
    # Thresholds for styling.
    warn_if_above: float | None = None
    danger_if_above: float | None = None
    warn_if_below: float | None = None
    danger_if_below: float | None = None

    @field_validator("type")
    @classmethod
    def _known_type(cls, v: str) -> str:
        allowed = set(ProbeType.__args__)  # type: ignore[attr-defined]
        if v not in allowed:
            raise ValueError(f"probe type must be one of {sorted(allowed)}")
        return v


class ProgressConfig(BaseModel):
    """Defines how to compute a progress percentage and ETA."""

    numerator_probe: str
    denominator_probe: str
    denominator_label: str = "total"
    # If true, show a progress bar and ETA.
    show_eta: bool = True


class Pane(BaseModel):
    """A dashboard pane."""

    title: str
    type: PaneType
    # Which probes feed this pane. Meaning depends on pane type.
    probes: list[str] = Field(default_factory=list)
    # For text/bars panes: extra display configuration.
    config: dict[str, Any] = Field(default_factory=dict)
    # Pane layout ratio/size.
    ratio: int = 1
    size: int | None = None


class ExperimentDashboardConfig(BaseModel):
    """Top-level dashboard config for one experiment domain."""

    experiment: str
    description: str = ""
    # Default refresh interval in seconds.
    refresh: float = 2.0
    # Probes available for this experiment.
    probes: list[Probe] = Field(default_factory=list)
    # Progress bar definition.
    progress: ProgressConfig | None = None
    # Panes to render, in order.
    panes: list[Pane] = Field(default_factory=list)
    # Default log file to tail for the "recent_log" pane.
    log_file: str | None = None

    @classmethod
    def load(cls, path: Path) -> "ExperimentDashboardConfig":
        text = path.read_text(encoding="utf-8")
        if path.suffix in {".yaml", ".yml"}:
            data = yaml.safe_load(text)
        else:
            import json

            data = json.loads(text)
        return cls.model_validate(data)

    def get_probe(self, probe_id: str) -> Probe | None:
        for p in self.probes:
            if p.id == probe_id:
                return p
        return None


# ---------------------------------------------------------------------------
# Built-in generic config for runs that do not provide one.
# ---------------------------------------------------------------------------

def generic_config() -> ExperimentDashboardConfig:
    """A minimal generic dashboard that works for any manifest."""
    return ExperimentDashboardConfig(
        experiment="generic",
        description="Generic run monitor",
        refresh=2.0,
        probes=[
            Probe(id="run_status", type="const", params={"value": "(no run selected)"}),
            Probe(id="gpu_status", type="gpu_status"),
            Probe(id="disk_usage", type="disk_usage", label="Disk"),
        ],
        panes=[
            Pane(title="Run Info", type="run_info", probes=["run_status"], size=12),
            Pane(title="GPU Status", type="gpu_table", probes=["gpu_status"], ratio=2),
            Pane(title="Disk", type="metrics_table", probes=["disk_usage"], ratio=1),
        ],
    )
