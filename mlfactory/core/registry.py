"""SQLite-backed registry for run manifests and metrics.

The registry is the central index that lets dashboards, notebooks, and humans
find, compare, and reconstruct experiments without guessing directory names.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from .manifest import RunManifest


SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    stage TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    started_at TEXT,
    completed_at TEXT,
    manifest_json TEXT NOT NULL,
    manifest_sha256 TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS run_lineage (
    parent_run_id TEXT NOT NULL,
    child_run_id TEXT NOT NULL,
    relation TEXT NOT NULL DEFAULT 'input',
    PRIMARY KEY (parent_run_id, child_run_id),
    FOREIGN KEY (parent_run_id) REFERENCES runs(run_id) ON DELETE CASCADE,
    FOREIGN KEY (child_run_id) REFERENCES runs(run_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS metrics (
    run_id TEXT NOT NULL,
    step INTEGER,
    timestamp TEXT NOT NULL,
    key TEXT NOT NULL,
    value REAL,
    json_value TEXT,
    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_runs_stage ON runs(stage);
CREATE INDEX IF NOT EXISTS idx_runs_status ON runs(status);
CREATE INDEX IF NOT EXISTS idx_runs_created ON runs(created_at);
CREATE INDEX IF NOT EXISTS idx_metrics_run ON metrics(run_id);
CREATE INDEX IF NOT EXISTS idx_metrics_key ON metrics(run_id, key);
CREATE INDEX IF NOT EXISTS idx_lineage_parent ON run_lineage(parent_run_id);
CREATE INDEX IF NOT EXISTS idx_lineage_child ON run_lineage(child_run_id);
"""


class Registry:
    def __init__(self, db_path: str | Path = "data/registry.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(SCHEMA)
            conn.commit()

    # ------------------------------------------------------------------
    # run CRUD
    # ------------------------------------------------------------------
    def register(self, manifest: RunManifest) -> None:
        with self._connect() as conn:
            self._insert_run_row(conn, manifest)
            conn.commit()

    def get(self, run_id: str) -> RunManifest | None:
        with self._connect() as conn:
            row = conn.execute("SELECT manifest_json FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        if not row:
            return None
        return RunManifest.model_validate_json(row["manifest_json"])

    def find(
        self,
        stage: str | None = None,
        status: str | None = None,
        limit: int = 100,
        order_by: str = "created_at DESC",
    ) -> list[RunManifest]:
        query = "SELECT manifest_json FROM runs WHERE 1=1"
        params: list[Any] = []
        if stage:
            query += " AND stage = ?"
            params.append(stage)
        if status:
            query += " AND status = ?"
            params.append(status)
        query += f" ORDER BY {order_by} LIMIT ?"
        params.append(limit)
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [RunManifest.model_validate_json(r["manifest_json"]) for r in rows]

    def update_status(self, run_id: str, status: str) -> None:
        manifest = self.get(run_id)
        if manifest is None:
            raise KeyError(f"run {run_id} not found")
        manifest.status = status  # type: ignore[assignment]
        if status in {"completed", "failed", "guarded", "aborted"}:
            from datetime import datetime, timezone
            manifest.completed_at = datetime.now(timezone.utc).isoformat()
        self.register(manifest)

    # ------------------------------------------------------------------
    # lineage
    # ------------------------------------------------------------------
    def link_runs(self, parent_run_id: str, child_run_id: str, relation: str = "input") -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO run_lineage (parent_run_id, child_run_id, relation)
                VALUES (?, ?, ?)
                ON CONFLICT(parent_run_id, child_run_id) DO UPDATE SET relation=excluded.relation
                """,
                (parent_run_id, child_run_id, relation),
            )
            conn.commit()

    def parents(self, run_id: str) -> list[tuple[str, str]]:
        """Return (parent_run_id, relation) tuples."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT parent_run_id, relation FROM run_lineage WHERE child_run_id = ?",
                (run_id,),
            ).fetchall()
        return [(r["parent_run_id"], r["relation"]) for r in rows]

    def children(self, run_id: str) -> list[tuple[str, str]]:
        """Return (child_run_id, relation) tuples."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT child_run_id, relation FROM run_lineage WHERE parent_run_id = ?",
                (run_id,),
            ).fetchall()
        return [(r["child_run_id"], r["relation"]) for r in rows]

    # ------------------------------------------------------------------
    # metrics
    # ------------------------------------------------------------------
    def insert_metric(
        self,
        run_id: str,
        key: str,
        value: float | None = None,
        json_value: dict | list | None = None,
        step: int | None = None,
        timestamp: str | None = None,
    ) -> None:
        from datetime import datetime, timezone

        ts = timestamp or datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO metrics (run_id, step, timestamp, key, value, json_value)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    step,
                    ts,
                    key,
                    value,
                    json.dumps(json_value) if json_value is not None else None,
                ),
            )
            conn.commit()

    def metrics_for_run(self, run_id: str, key: str | None = None) -> list[dict]:
        query = "SELECT * FROM metrics WHERE run_id = ?"
        params: list[Any] = [run_id]
        if key:
            query += " AND key = ?"
            params.append(key)
        query += " ORDER BY timestamp"
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    def metric_series(self, run_id: str, key: str) -> list[tuple[int | None, float]]:
        """Return (step, value) tuples for a scalar metric, sorted by step/time."""
        rows = self.metrics_for_run(run_id, key)
        out = []
        for r in rows:
            if r["value"] is not None:
                out.append((r["step"], r["value"]))
        return out

    # ------------------------------------------------------------------
    # merge
    # ------------------------------------------------------------------
    def merge_from(
        self,
        source_db_path: str | Path,
        on_conflict: str = "skip",
    ) -> dict[str, int]:
        """Merge runs, lineage, and metrics from another registry database.

        Args:
            source_db_path: path to the source SQLite registry.
            on_conflict: ``skip`` keeps existing runs; ``replace`` overwrites
                them and replaces their lineage/metrics.

        Returns:
            A dict with counts for ``runs_added``, ``runs_replaced``,
            ``runs_skipped``, ``lineage``, and ``metrics``.
        """
        source_db_path = Path(source_db_path)
        if not source_db_path.exists():
            raise FileNotFoundError(f"source registry not found: {source_db_path}")
        if source_db_path.resolve() == self.db_path.resolve():
            raise ValueError("cannot merge a registry into itself")
        if on_conflict not in {"skip", "replace"}:
            raise ValueError(f"on_conflict must be 'skip' or 'replace', got {on_conflict!r}")

        counts = {"runs_added": 0, "runs_replaced": 0, "runs_skipped": 0, "lineage": 0, "metrics": 0}

        # Read everything from the source registry into memory.
        source = Registry(source_db_path)
        source_runs = source.find(limit=10_000_000)
        source_lineage: list[tuple[str, str, str]] = []
        source_metrics: list[dict] = []

        with source._connect() as src_conn:
            for parent_id, child_id, relation in src_conn.execute(
                "SELECT parent_run_id, child_run_id, relation FROM run_lineage"
            ).fetchall():
                source_lineage.append((parent_id, child_id, relation))
            for row in src_conn.execute("SELECT run_id, step, timestamp, key, value, json_value FROM metrics").fetchall():
                source_metrics.append(dict(row))

        # Determine which runs are new vs existing in the target registry.
        existing_run_ids = {r.run_id for r in self.find(limit=10_000_000)}
        runs_to_add: list[RunManifest] = []
        runs_to_replace: list[RunManifest] = []
        for manifest in source_runs:
            if manifest.run_id in existing_run_ids:
                if on_conflict == "replace":
                    runs_to_replace.append(manifest)
                else:
                    counts["runs_skipped"] += 1
            else:
                runs_to_add.append(manifest)

        imported_run_ids = {m.run_id for m in runs_to_add + runs_to_replace}

        with self._connect() as conn:
            # Insert new runs.
            for manifest in runs_to_add:
                self._insert_run_row(conn, manifest)
            counts["runs_added"] = len(runs_to_add)

            # Replace existing runs and clear their old lineage/metrics.
            for manifest in runs_to_replace:
                run_id = manifest.run_id
                conn.execute("DELETE FROM run_lineage WHERE parent_run_id = ? OR child_run_id = ?", (run_id, run_id))
                conn.execute("DELETE FROM metrics WHERE run_id = ?", (run_id,))
                self._insert_run_row(conn, manifest)
            counts["runs_replaced"] = len(runs_to_replace)

            # Copy lineage for imported runs (both ends must be imported).
            for parent_id, child_id, relation in source_lineage:
                if parent_id in imported_run_ids and child_id in imported_run_ids:
                    try:
                        conn.execute(
                            "INSERT INTO run_lineage (parent_run_id, child_run_id, relation) VALUES (?, ?, ?)",
                            (parent_id, child_id, relation),
                        )
                        counts["lineage"] += 1
                    except sqlite3.IntegrityError:
                        pass

            # Copy metrics for imported runs.
            for metric in source_metrics:
                if metric["run_id"] in imported_run_ids:
                    conn.execute(
                        "INSERT INTO metrics (run_id, step, timestamp, key, value, json_value) VALUES (?, ?, ?, ?, ?, ?)",
                        (
                            metric["run_id"],
                            metric["step"],
                            metric["timestamp"],
                            metric["key"],
                            metric["value"],
                            metric["json_value"],
                        ),
                    )
                    counts["metrics"] += 1

            conn.commit()

        return counts

    @staticmethod
    def _insert_run_row(conn: sqlite3.Connection, manifest: RunManifest) -> None:
        import hashlib

        manifest_json = manifest.model_dump_json()
        manifest_sha256 = hashlib.sha256(manifest_json.encode("utf-8")).hexdigest()
        conn.execute(
            """
            INSERT INTO runs (run_id, stage, status, created_at, started_at, completed_at, manifest_json, manifest_sha256)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id) DO UPDATE SET
                stage=excluded.stage,
                status=excluded.status,
                started_at=excluded.started_at,
                completed_at=excluded.completed_at,
                manifest_json=excluded.manifest_json,
                manifest_sha256=excluded.manifest_sha256
            """,
            (
                manifest.run_id,
                manifest.stage,
                manifest.status,
                manifest.created_at,
                manifest.started_at,
                manifest.completed_at,
                manifest_json,
                manifest_sha256,
            ),
        )

    # ------------------------------------------------------------------
    # migration helpers
    # ------------------------------------------------------------------
    def ingest_manifest(self, manifest: RunManifest, parent_run_ids: list[str] | None = None) -> None:
        self.register(manifest)
        for parent in parent_run_ids or manifest.parent_runs:
            self.link_runs(parent, manifest.run_id, "input")
