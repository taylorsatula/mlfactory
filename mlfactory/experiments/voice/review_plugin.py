"""mlfactory plugin for GLM corpus review and deterministic pseudonymization."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from mlfactory.core.manifest import FileRecord, sha256_file
from mlfactory.plugins.base import PLUGINS, StagePlugin


class VoiceReviewPlugin(StagePlugin):
    stage = "review"

    def __init__(self, manifest):
        super().__init__(manifest)
        self.spec = manifest.spec
        self.review_dir = self.run_dir / "artifacts" / "review"

    def _script(self, name: str) -> Path:
        return Path(__file__).parent / name

    def _env(self) -> dict[str, str]:
        env = dict(os.environ)
        env.update(self.spec.get("env", {}))
        env.setdefault("PYTHONUNBUFFERED", "1")
        return env

    def prepare(self) -> None:
        (self.run_dir / "artifacts").mkdir(parents=True, exist_ok=True)
        (self.run_dir / "logs").mkdir(parents=True, exist_ok=True)
        threads = Path(self.spec.get("threads", Path(__file__).parent / "data" / "threads")).resolve()
        if not threads.is_dir():
            raise FileNotFoundError(f"thread directory not found: {threads}")
        records = []
        aggregate = hashlib.sha256()
        for path in sorted(threads.glob("*.json")):
            digest = sha256_file(path)
            records.append({"file": path.name, "sha256": digest, "size_bytes": path.stat().st_size})
            aggregate.update(path.name.encode("utf-8") + b"\0" + digest.encode("ascii") + b"\n")
        manifest_path = self.run_dir / "artifacts" / "input_corpus_manifest.json"
        manifest_path.write_text(json.dumps({
            "thread_count": len(records),
            "aggregate_sha256": aggregate.hexdigest(),
            "files": records,
        }, indent=2) + "\n", encoding="utf-8")
        os.chmod(manifest_path, 0o600)

        # The voice domain may be uncommitted/ignored because it contains private
        # data, so capture the exact executable policy files independently of
        # git archive. Never copy the corpus or secrets here.
        snapshot_dir = self.run_dir / "artifacts" / "review_source_snapshot"
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        snapshot_files = [
            self._script("review_corpus.py"),
            self._script("transform_placeholders.py"),
            self._script("review_plugin.py"),
            Path(self.spec.get("prompt", Path(__file__).parent / "prompts" / "review_session.md")).resolve(),
        ]
        for source in snapshot_files:
            shutil.copy2(source, snapshot_dir / source.name)

    def execute(self) -> None:
        s = self.spec
        python = str(s.get("python", sys.executable))
        cmd = [
            python,
            str(self._script("review_corpus.py")),
            "--mode", "corpus",
            "--limit", str(s.get("limit", -1)),
            "--seed", str(s.get("seed", 42)),
            "--workers", str(s.get("workers", 10)),
            "--model", str(s.get("model", "glm-5.2-vision-ballast")),
            "--retries", str(s.get("retries", 2)),
            "--threads", str(Path(s.get("threads", Path(__file__).parent / "data" / "threads")).resolve()),
            "--prompt", str(Path(s.get("prompt", Path(__file__).parent / "prompts" / "review_session.md")).resolve()),
            "--output", str(self.review_dir),
        ]
        log_path = self.run_dir / "logs" / "review.log"
        err_path = self.run_dir / "logs" / "review.err"
        resume_passes = int(s.get("resume_passes", 2))
        rc = 1
        with log_path.open("a", encoding="utf-8") as log, err_path.open("a", encoding="utf-8") as err:
            for attempt in range(resume_passes + 1):
                attempt_cmd = cmd + (["--resume"] if attempt else [])
                log.write(f"REVIEW_PASS {attempt + 1}/{resume_passes + 1}\n")
                log.flush()
                rc = subprocess.call(attempt_cmd, env=self._env(), stdout=log, stderr=err)
                if rc == 0:
                    break
                log.write(f"REVIEW_PASS_FAILED rc={rc}; resuming incomplete sessions\n")
                log.flush()
            if rc != 0:
                raise RuntimeError(f"voice review incomplete after {resume_passes + 1} passes")

            transform_cmd = [
                python,
                str(self._script("transform_placeholders.py")),
                str(self.review_dir),
                "--version", str(s.get("transform_version", "v1")),
            ]
            log.write("TRANSFORM_PLACEHOLDERS\n")
            log.flush()
            rc = subprocess.call(transform_cmd, env=self._env(), stdout=log, stderr=err)
            if rc != 0:
                raise RuntimeError(f"placeholder transform exited with code {rc}")

    def finalize(self) -> None:
        artifacts_dir = self.run_dir / "artifacts"
        for path in sorted(artifacts_dir.rglob("*")):
            if path.is_file():
                self.manifest.artifacts.append(FileRecord(
                    path=str(path.resolve()),
                    sha256=sha256_file(path),
                    role=f"artifact:{path.relative_to(artifacts_dir)}",
                    size_bytes=path.stat().st_size,
                ))
        summary_path = self.review_dir / "summary.json"
        if summary_path.exists():
            self.manifest.summary = json.loads(summary_path.read_text(encoding="utf-8"))
            transform_path = self.review_dir / f"transform_summary-{self.spec.get('transform_version', 'v1')}.json"
            if transform_path.exists():
                self.manifest.summary["transform"] = json.loads(transform_path.read_text(encoding="utf-8"))
        self.manifest.write(self.run_dir / "manifest.json")


PLUGINS.register(VoiceReviewPlugin)
