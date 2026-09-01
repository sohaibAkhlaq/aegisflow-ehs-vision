"""Content-addressed disk cache for LLM responses.

Two reasons this exists rather than being an optimisation we skipped:

* **Free-tier rate limits.** A demo re-run must not burn quota on questions already asked.
* **Reproducibility.** An evaluation run that consults a model is only comparable to the
  previous one if identical inputs give identical answers. Caching by content hash makes a
  re-run deterministic.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from aegisflow.core.logging import get_logger

log = get_logger(__name__)


def content_key(*parts: str | bytes) -> str:
    """Stable cache key over arbitrary text/binary inputs."""
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part.encode("utf-8") if isinstance(part, str) else part)
        digest.update(b"\x00")  # domain separator, so ("ab","c") != ("a","bc")
    return digest.hexdigest()[:32]


class ResponseCache:
    """Flat JSON-file cache under ``data/processed/llm_cache/``."""

    def __init__(self, root: Path, enabled: bool = True) -> None:
        self.root = root
        self.enabled = enabled
        if enabled:
            self.root.mkdir(parents=True, exist_ok=True)

    def _file(self, key: str) -> Path:
        return self.root / f"{key}.json"

    def get(self, key: str) -> Any | None:
        if not self.enabled:
            return None
        path = self._file(key)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))["value"]
        except (OSError, ValueError, KeyError):
            # A corrupt cache entry is a cache miss, never an error.
            log.debug("discarding unreadable cache entry %s", path.name)
            return None

    def set(self, key: str, value: Any) -> None:
        if not self.enabled:
            return
        try:
            self._file(key).write_text(
                json.dumps({"key": key, "value": value}, default=str), encoding="utf-8"
            )
        except OSError as exc:  # pragma: no cover - disk-full style failure
            log.debug("could not write cache entry %s: %s", key, exc)

    def clear(self) -> int:
        """Delete every entry; returns how many were removed."""
        if not self.root.exists():
            return 0
        removed = 0
        for path in self.root.glob("*.json"):
            path.unlink(missing_ok=True)
            removed += 1
        return removed
