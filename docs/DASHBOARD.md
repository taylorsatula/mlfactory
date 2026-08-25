# Dashboard config

> Update when: `mlfactory/core/dashboard_config.py` schema changes. Source of
> truth: that module (`ExperimentDashboardConfig`, `Probe`, `Pane`). This doc
> is an index of probe/pane types; the module carries the field definitions.

## What a dashboard is

A dashboard is a read-only projection of a run's live state. The config
**travels with the run**: `create_run` resolves it from the plugin's
`dashboard()` classmethod (or a sibling static file the method loads) and
persists it to `run_dir/dashboard.json`. The viewer reads it from the run
dir, not from the source tree at view time — so the dashboard is
reproducible from the run alone, like every other artifact.

## Declaring a dashboard (pick one)

- **Programmatic** — override `StagePlugin.dashboard() -> ExperimentDashboardConfig | None`
  on the plugin. Most common: load a sibling static file via
  `ExperimentDashboardConfig.load(Path(__file__).with_name("dashboard_<stage>.json"))`.
  The sample experiment does this for all four stages.
- **Opt out** — `dashboard: none` in the spec, or `dashboard()` returning
  `None`, suppresses persistence; the viewer falls back to a generic
  status/GPU/disk view.

## Probe types

`ProbeType` is a `Literal` in `dashboard_config.py`:

```
const, spec_value, file_line_count, regex_count, regex_last_match,
jsonl_last_record, json_file_value, process_alive, http_status, http_json,
gpu_status, disk_usage, file_exists, shell_command, jsonl_metric_last
```

Probe fields that select behavior: `command` (shell_command), `regex`
(regex_count/regex_last_match), `path` (jsonl/json_file dotted path), `url`
(http), `file` (file probes; may contain `{run_dir}`), `pid_file`/`pid_env`
(process_alive), `parser` (`raw|json|float|int` for shell_command),
`warn_if_*`/`danger_if_*` thresholds for styling.

## Pane types

`PaneType` is a `Literal` in `dashboard_config.py`:

```
overview, metrics_table, bars, recent_log, gpu_table, run_info,
lineage, text, training_chart, runs_table
```

A pane references probe ids via its `probes` field; interpretation depends
on pane type. `ratio` controls layout width; `size` fixes a height.

## Top-level config

`ExperimentDashboardConfig` carries: `experiment`, `description`,
`refresh` (default 2.0s), `probes: list[Probe]`, `progress:
ProgressConfig | None`, `panes: list[Pane]`, `log_file` (default for the
recent_log pane). `ProgressConfig` defines a progress bar + ETA from a
numerator/denominator probe pair.

## Generic fallback

`generic_config()` returns a minimal dashboard (run info, GPU table,
disk) that works for any manifest. Used when a run has no persisted
dashboard.
