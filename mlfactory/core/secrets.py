"""Local secrets store for mlfactory.

Secrets are kept in ``.mlfactory/secrets.yaml`` (or a configured path) and are
never written into run manifests. Specs reference them with
``${secrets.KEY}`` and resolution happens at execution time.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml


DEFAULT_SECRETS_PATH = Path(".mlfactory/secrets.yaml")


class SecretsStore:
    """Key-value store for API keys and credentials."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else DEFAULT_SECRETS_PATH
        self._data: dict[str, str] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            self._data = {}
            return
        try:
            with self.path.open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        except Exception:
            data = {}
        if not isinstance(data, dict):
            data = {}
        # Resolve values that reference environment variables, e.g. "$ENV_VAR".
        resolved: dict[str, str] = {}
        for key, value in data.items():
            if isinstance(value, str):
                resolved[key] = self._expand_env(value)
            else:
                resolved[key] = str(value)
        self._data = resolved

    @staticmethod
    def _expand_env(value: str) -> str:
        """Expand $VAR and ${VAR} references from the process environment."""
        value = re.sub(r"\$\{([^}]+)\}", lambda m: os.environ.get(m.group(1), ""), value)
        value = re.sub(r"\$([A-Za-z_][A-Za-z0-9_]*)", lambda m: os.environ.get(m.group(1), ""), value)
        return value

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(self._data, f, sort_keys=True)
        # Restrict permissions so only the owner can read.
        os.chmod(self.path, 0o600)

    def get(self, key: str, default: str | None = None) -> str | None:
        return self._data.get(key, default)

    def set(self, key: str, value: str) -> None:
        self._data[key] = value
        self._save()

    def delete(self, key: str) -> bool:
        if key in self._data:
            del self._data[key]
            self._save()
            return True
        return False

    def list(self) -> dict[str, str]:
        return dict(self._data)

    def __contains__(self, key: str) -> bool:
        return key in self._data


def expand_secrets(value: Any, store: SecretsStore | None = None) -> Any:
    """Recursively expand ``${secrets.KEY}`` references in a value."""
    if store is None:
        store = SecretsStore()
    if isinstance(value, str):

        def repl(match: re.Match[str]) -> str:
            key = match.group(1)
            resolved = store.get(key)
            if resolved is None:
                raise KeyError(f"secret not found: {key}")
            return resolved

        return re.sub(r"\$\{secrets\.([^}]+)\}", repl, value)
    if isinstance(value, dict):
        return {k: expand_secrets(v, store) for k, v in value.items()}
    if isinstance(value, list):
        return [expand_secrets(v, store) for v in value]
    return value
