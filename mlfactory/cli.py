"""Command-line interface for mlfactory."""
from __future__ import annotations

from pathlib import Path

import click

from mlfactory.core.manifest import RunManifest
from mlfactory.core.registry import Registry
from mlfactory.core.runner import create_run, run_from_spec


@click.group()
@click.option("--registry", "-r", default=".mlfactory/registry.db", help="Path to SQLite registry.")
@click.pass_context
def main(ctx: click.Context, registry: str) -> None:
    ctx.ensure_object(dict)
    ctx.obj["registry"] = Registry(registry)


@main.command()
@click.argument("spec", type=click.Path(exists=True, path_type=Path))
@click.option("--stage", "-s", default=None, help="Override stage from spec.")
@click.option("--run-id", default=None, help="Override auto-generated run id.")
@click.option("--runs-dir", default="runs", type=click.Path(path_type=Path))
@click.pass_context
def run(ctx: click.Context, spec: Path, stage: str | None, run_id: str | None, runs_dir: Path) -> None:
    """Execute a run from a spec file."""
    registry: Registry = ctx.obj["registry"]
    manifest = run_from_spec(
        spec_path=spec,
        stage=stage,
        run_id=run_id,
        runs_dir=runs_dir,
        registry=registry,
    )
    click.echo(f"Run {manifest.run_id} finished with status {manifest.status}")


@main.command()
@click.argument("spec", type=click.Path(exists=True, path_type=Path))
@click.option("--stage", "-s", default=None)
@click.option("--run-id", default=None)
@click.option("--runs-dir", default="runs", type=click.Path(path_type=Path))
@click.pass_context
def init(ctx: click.Context, spec: Path, stage: str | None, run_id: str | None, runs_dir: Path) -> None:
    """Create a run directory and manifest without executing."""
    manifest = create_run(
        spec_path=spec,
        stage=stage,
        run_id=run_id,
        runs_dir=runs_dir,
    )
    click.echo(f"Created run {manifest.run_id} at runs/{manifest.run_id}")


@main.command()
@click.option("--stage", "-s", default=None)
@click.option("--status", default=None)
@click.option("--limit", "-n", default=20, type=int)
@click.pass_context
def ls(ctx: click.Context, stage: str | None, status: str | None, limit: int) -> None:
    """List runs in the registry."""
    registry: Registry = ctx.obj["registry"]
    runs = registry.find(stage=stage, status=status, limit=limit)
    if not runs:
        click.echo("No runs found.")
        return
    click.echo(f"{'run_id':<50} {'stage':<12} {'status':<12} {'created_at'}")
    for r in runs:
        click.echo(f"{r.run_id:<50} {r.stage:<12} {r.status:<12} {r.created_at}")


@main.command()
@click.argument("run_id")
@click.pass_context
def show(ctx: click.Context, run_id: str) -> None:
    """Show a manifest from the registry."""
    registry: Registry = ctx.obj["registry"]
    manifest = registry.get(run_id)
    if manifest is None:
        raise click.ClickException(f"run {run_id} not found")
    click.echo(manifest.model_dump_json(indent=2))


@main.command()
@click.argument("run_id")
@click.pass_context
def lineage(ctx: click.Context, run_id: str) -> None:
    """Show parent/child lineage for a run."""
    registry: Registry = ctx.obj["registry"]
    parents = registry.parents(run_id)
    children = registry.children(run_id)
    click.echo(f"Run: {run_id}")
    click.echo("Parents:")
    for pid, rel in parents:
        click.echo(f"  {pid} ({rel})")
    click.echo("Children:")
    for cid, rel in children:
        click.echo(f"  {cid} ({rel})")


@main.command()
@click.argument("manifest", type=click.Path(exists=True, path_type=Path))
@click.option("--parent", "-p", multiple=True, help="Parent run id(s).")
@click.pass_context
def ingest(ctx: click.Context, manifest: Path, parent: tuple[str, ...]) -> None:
    """Ingest an existing manifest.json into the registry."""
    registry: Registry = ctx.obj["registry"]
    m = RunManifest.read(manifest)
    registry.ingest_manifest(m, list(parent))
    click.echo(f"Ingested {m.run_id}")


@main.command("dashboard")
@click.option("--watch-run", default=None)
@click.option("--stage", default=None)
@click.option("--refresh", type=float, default=2.0)
@click.pass_context
def dashboard_cmd(ctx: click.Context, watch_run: str | None, stage: str | None, refresh: float) -> None:
    """Launch the read-only Rich Live dashboard."""
    import subprocess
    import sys

    cmd = [sys.executable, "-m", "mlfactory.core.dashboard", "--registry", str(ctx.obj["registry"].db_path)]
    if watch_run:
        cmd.extend(["--watch-run", watch_run])
    if stage:
        cmd.extend(["--stage", stage])
    cmd.extend(["--refresh", str(refresh)])
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
