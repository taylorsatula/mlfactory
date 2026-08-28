"""Command-line interface for mlfactory."""
from __future__ import annotations

import json
from pathlib import Path

import click

from mlfactory.core.manifest import RunManifest
from mlfactory.core.notes import (
    append_note,
    read_notes,
    search_notes,
)
from mlfactory.core.registry import Registry
from mlfactory.core.runner import create_run, run_from_spec
from mlfactory.core.secrets import SecretsStore
from mlfactory.remote.ssh_runner import SSHConfig
from mlfactory.remote.vast import VastRunner, load_api_key


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
    """Show a manifest from the registry, with any lab notes appended."""
    registry: Registry = ctx.obj["registry"]
    manifest = registry.get(run_id)
    if manifest is None:
        raise click.ClickException(f"run {run_id} not found")
    click.echo(manifest.model_dump_json(indent=2))
    notes = _load_notes_for_show(ctx, registry, run_id)
    if notes:
        click.echo("\n# lab notes")
        for n in notes:
            ts = n.get("ts", "")
            author = n.get("author", "")
            text = n.get("text", "")
            click.echo(f"- [{ts}] ({author}) {text}")


def _load_notes_for_show(ctx: click.Context, registry: Registry, run_id: str) -> list[dict]:
    from pathlib import Path

    from mlfactory.core.notes import _resolve_run_dir, read_notes as _read_notes

    try:
        run_dir = _resolve_run_dir(registry, run_id)
    except Exception:
        return []
    return _read_notes(run_dir)


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


@main.command("note")
@click.argument("run_id")
@click.argument("text", nargs=-1, required=True)
@click.option("--author", default=None, help="Override author (default: MLFACTORY_AUTHOR, git user.name, or login).")
@click.pass_context
def note_cmd(ctx: click.Context, run_id: str, text: tuple[str, ...], author: str | None) -> None:
    """Append one timestamped lab note to a run.

    \b
    Examples:
      mlfactory note 20260823-141022.train "lr 3e-5 diverges step 800, same as run X"
      mlfactory note 20260823-141022.train suspect init -- try 1e-5

    TEXT is joined with spaces, so quotes are optional for single arguments.
    One line, no structure required — cheaper than a sticky note. See the
    "Lab notes" section of AGENTS.md for when to write one.
    """
    registry: Registry = ctx.obj["registry"]
    body = " ".join(text)
    try:
        record = append_note(registry, run_id, body, author=author)
    except ValueError as exc:
        raise click.ClickException(str(exc))
    click.echo(f"Noted on {run_id}: {record['ts']}")


@main.command("notes")
@click.argument("run_id", required=False)
@click.option("--grep", "pattern", default=None, help="Regex search across all runs' notes (run_id optional then).")
@click.option("-i/--ignore-case", "ignore_case", default=True, help="Case-insensitive grep (default on).")
@click.pass_context
def notes_cmd(ctx: click.Context, run_id: str | None, pattern: str | None, ignore_case: bool) -> None:
    """Read notes for a run, or grep across all runs' notes.

    \b
    Examples:
      mlfactory notes 20260823-141022.train        # all notes on one run
      mlfactory notes --grep diverges              # which runs hit this?
      mlfactory notes --grep 'lr 3e-5' -i          # case-insensitive

    With a run_id: prints that run's notes oldest-first. With --grep and no
    run_id: searches every run under runs/ and prints "run_id  ts  text".
    """
    from pathlib import Path

    from mlfactory.core.notes import _resolve_run_dir

    if pattern is not None:
        hits = search_notes(pattern, runs_dir=Path("runs"), ignore_case=ignore_case)
        if not hits:
            click.echo("No notes matched.")
            return
        for rid, n in hits:
            ts = n.get("ts", "")
            text = n.get("text", "")
            click.echo(f"{rid}\t{ts}\t{text}")
        return

    if run_id is None:
        raise click.ClickException("provide a run_id, or use --grep to search across all runs.")
    registry: Registry = ctx.obj["registry"]
    run_dir = _resolve_run_dir(registry, run_id)
    notes = read_notes(run_dir)
    if not notes:
        click.echo(f"No notes for {run_id}.")
        return
    for n in notes:
        ts = n.get("ts", "")
        author = n.get("author", "")
        text = n.get("text", "")
        click.echo(f"[{ts}] ({author}) {text}")


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


@main.command("ask-human")
@click.argument("message")
@click.option(
    "--to", "target", default=None,
    help=(
        "hermes send target for the escalation chat. Defaults to "
        "$MLFACTORY_ESCALATION_TARGET. Use 'discord:<channel-id>' for a "
        "Discord DM/channel (recommended)."
    ),
)
@click.option("--timeout", default=86400.0, show_default=True,
              help="Max seconds to wait for a reply (0 = wait forever).")
@click.option("--poll-interval", default=5.0, show_default=True,
              help="Seconds between reply polls.")
@click.option("--state-db", default=None,
              help="Override Hermes state.db path.")
@click.option("--subject", default="[mlfactory escalation]",
              help="Subject line prepended to the message.")
@click.option("--json", "as_json", is_flag=True,
              help="Emit a JSON result on stdout instead of bare reply text.")
@click.pass_context
def ask_human(
    ctx: click.Context,
    message: str,
    target: str | None,
    timeout: float,
    poll_interval: float,
    state_db: str | None,
    subject: str | None,
    as_json: bool,
) -> None:
    """Send an escalation to a human via Hermes and block until the human replies.

    For autonomous overseers that hit a condition needing a human decision.
    The message is delivered to your Discord through `hermes send`; the
    command then blocks on the intake-only FIFO (with a state.db poll
    fallback) for the first reply on the same chat, and prints that reply on
    stdout. Wrap with `timeout <s>` as a hard kill switch (this command
    already exits 124 on its own timeout).

    One-time setup: see docs/HUMAN_ESCALATION.md (configure Discord, make the
    escalation chat intake-only, start the gateway, set
    MLFACTORY_ESCALATION_TARGET).
    """
    from mlfactory.core.human_escalation import (
        EscalationError,
        EscalationTimeout,
        escalate_to_human,
    )

    if message == "-":
        import sys
        message = sys.stdin.read()

    wait = float("inf") if float(timeout) <= 0 else float(timeout)

    def _emit(payload: dict, code: int) -> None:
        if as_json:
            click.echo(json.dumps(payload))
        else:
            for line in payload.get("stderr", []) or []:
                click.echo(line, err=True)
            if "reply" in payload:
                click.echo(payload["reply"])
        ctx.exit(code)

    try:
        reply = escalate_to_human(
            message,
            target=target,
            timeout=wait,
            poll_interval=poll_interval,
            state_db=state_db,
            subject=subject,
        )
    except EscalationTimeout as exc:
        _emit({"status": "timeout", "timeout": exc.timeout,
               "stderr": [f"mlfactory ask-human: timed out after {exc.timeout:g}s"]}, 124)
        return
    except EscalationError as exc:
        _emit({"status": "error", "error": str(exc),
               "stderr": [f"mlfactory ask-human: {exc}"]}, 1)
        return
    _emit({"status": "ok", "reply": reply}, 0)


@main.command("dashboard")
@click.option("--watch-run", default=None)
@click.option("--stage", default=None)
@click.option("--config", "config_path", default=None,
              help="Explicit dashboard config file (bypasses registry lookup).")
@click.option("--refresh", type=float, default=2.0)
@click.pass_context
def dashboard_cmd(ctx: click.Context, watch_run: str | None, stage: str | None, config_path: str | None, refresh: float) -> None:
    """Launch the read-only Rich Live dashboard."""
    import subprocess
    import sys

    cmd = [sys.executable, "-m", "mlfactory.core.dashboard", "--registry", str(ctx.obj["registry"].db_path)]
    if config_path:
        cmd.extend(["--config", config_path])
    if watch_run:
        cmd.extend(["--watch-run", watch_run])
    if stage:
        cmd.extend(["--stage", stage])
    cmd.extend(["--refresh", str(refresh)])
    raise SystemExit(subprocess.call(cmd))


# ---------------------------------------------------------------------------
# registry commands
# ---------------------------------------------------------------------------

@main.group()
def registry() -> None:
    """Registry management commands."""
    pass


@registry.command("merge")
@click.argument("source", type=click.Path(exists=True, path_type=Path))
@click.option("--strategy", "-s", type=click.Choice(["skip", "replace"]), default="skip")
@click.pass_context
def registry_merge(ctx: click.Context, source: Path, strategy: str) -> None:
    """Merge another SQLite registry into the current one."""
    reg: Registry = ctx.obj["registry"]
    counts = reg.merge_from(source, on_conflict=strategy)
    click.echo(
        f"Merged registry from {source}: "
        f"added {counts['runs_added']} runs, "
        f"replaced {counts['runs_replaced']} runs, "
        f"skipped {counts['runs_skipped']} runs, "
        f"{counts['lineage']} lineage edges, "
        f"{counts['metrics']} metrics."
    )


# ---------------------------------------------------------------------------
# secrets commands
# ---------------------------------------------------------------------------

@main.group()
def secrets() -> None:
    """Manage stored API keys and credentials."""
    pass


@secrets.command("set")
@click.argument("key")
@click.argument("value")
def secrets_set(key: str, value: str) -> None:
    """Store a secret value."""
    store = SecretsStore()
    store.set(key, value)
    click.echo(f"Stored secret {key}")


@secrets.command("get")
@click.argument("key")
def secrets_get(key: str) -> None:
    """Retrieve a secret value."""
    store = SecretsStore()
    value = store.get(key)
    if value is None:
        raise click.ClickException(f"secret {key} not found")
    click.echo(value)


@secrets.command("list")
def secrets_list() -> None:
    """List stored secret keys (values are not shown)."""
    store = SecretsStore()
    keys = sorted(store.list().keys())
    if not keys:
        click.echo("No secrets stored.")
        return
    for key in keys:
        click.echo(key)


@secrets.command("delete")
@click.argument("key")
def secrets_delete(key: str) -> None:
    """Delete a stored secret."""
    store = SecretsStore()
    if store.delete(key):
        click.echo(f"Deleted secret {key}")
    else:
        raise click.ClickException(f"secret {key} not found")


# ---------------------------------------------------------------------------
# remote commands
# ---------------------------------------------------------------------------

@main.group()
def remote() -> None:
    """Remote execution on Vast.ai or generic SSH hosts."""
    pass


@remote.command("run")
@click.option("--host", required=True, help="Remote SSH host.")
@click.option("--port", default=22, type=int, help="Remote SSH port.")
@click.option("--key", "-i", default=None, help="SSH private key.")
@click.option("--spec", required=True, type=click.Path(exists=True, path_type=Path), help="Spec file to run.")
@click.option("--run-id", default=None, help="Override run id.")
@click.option("--workdir", default="/workspace/mlfactory", help="Remote working directory.")
@click.option("--python", default="python3", help="Remote python interpreter.")
@click.option("--setup-script", default=None, help="Optional remote setup script to run before execution.")
def remote_run(
    host: str,
    port: int,
    key: str | None,
    spec: Path,
    run_id: str | None,
    workdir: str,
    python: str,
    setup_script: str | None,
) -> None:
    """Sync code and run a spec on a remote host."""
    config = SSHConfig(host=host, port=port, key=key, remote_workdir=workdir, python=python)
    runner = VastRunner(vast_config=None, ssh_config=config)  # type: ignore[arg-type]
    runner.sync_code()
    if setup_script:
        runner.setup(setup_script=setup_script)
    rid = runner.run_spec(str(spec), run_id=run_id)
    click.echo(f"Remote run {rid} completed; outputs pulled to runs/{rid}")


@remote.command("provision")
@click.option("--query", default=None, help="Vast search query (default: H100, 2 GPUs, 300GB disk).")
@click.option("--api-key", default=None, help="Vast API key (defaults to VAST_API_KEY env var).")
@click.option("--image", default="nvidia/cuda:12.9.0-devel-ubuntu26.04", help="Container image.")
@click.option("--disk", default=300.0, type=float, help="Disk size in GB.")
@click.option("--label", default="mlfactory", help="Instance label.")
def remote_provision(query: str | None, api_key: str | None, image: str, disk: float, label: str) -> None:
    """Provision a new Vast.ai instance."""
    key = api_key or load_api_key()
    if not key:
        raise click.ClickException("Vast API key not found. Set VAST_API_KEY or use --api-key.")
    runner = VastRunner.from_search(query=query, api_key=key, image=image, disk_gb=disk)
    info = runner.provision(query=query)
    click.echo(f"Provisioned instance {runner.instance_id} at {runner.config.host}:{runner.config.port}")
    click.echo(json.dumps(info, indent=2))


@remote.command("destroy")
@click.option("--instance-id", required=True, type=int, help="Vast instance id.")
@click.option("--api-key", default=None, help="Vast API key.")
def remote_destroy(instance_id: int, api_key: str | None) -> None:
    """Destroy a Vast.ai instance."""
    key = api_key or load_api_key()
    if not key:
        raise click.ClickException("Vast API key not found.")
    runner = VastRunner.from_instance_id(instance_id, api_key=key)
    runner.destroy()
    click.echo(f"Destroyed instance {instance_id}")


@remote.command("list")
@click.option("--api-key", default=None, help="Vast API key.")
def remote_list(api_key: str | None) -> None:
    """List your Vast.ai instances."""
    from mlfactory.remote.vast import list_instances

    key = api_key or load_api_key()
    if not key:
        raise click.ClickException("Vast API key not found.")
    instances = list_instances(api_key=key)
    if not instances:
        click.echo("No instances found.")
        return
    click.echo(f"{'ID':<10} {'Status':<12} {'Host':<30} {'Port':<8} {'Label'}")
    for inst in instances:
        ssh = inst.get("ssh_host", "")
        port = inst.get("ssh_port", "")
        click.echo(f"{inst.get('id'):<10} {inst.get('actual_status',''):<12} {ssh:<30} {port:<8} {inst.get('label','')}")


if __name__ == "__main__":
    main()
