"""Logging setup: rich console for humans, JSON for anything that scrapes logs."""

from __future__ import annotations

import json
import logging
import sys
from typing import Any

from rich.console import Console
from rich.logging import RichHandler

_console = Console(stderr=True)
_configured = False


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        for key, value in record.__dict__.items():
            if key.startswith("ctx_"):
                payload[key[4:]] = value
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO", as_json: bool = False) -> None:
    """Install the root handler. Idempotent - safe to call from the CLI and the API."""
    global _configured
    if _configured:
        return

    handler: logging.Handler
    if as_json:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(_JsonFormatter())
    else:
        handler = RichHandler(
            console=_console,
            rich_tracebacks=True,
            show_path=False,
            markup=False,
        )
        handler.setFormatter(logging.Formatter("%(message)s", datefmt="%H:%M:%S"))

    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level.upper())

    # Ultralytics is chatty at INFO and prints a banner per inference call.
    logging.getLogger("ultralytics").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    _configured = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def console() -> Console:
    """Shared rich console, for CLI tables and progress bars."""
    return _console
