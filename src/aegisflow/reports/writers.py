"""Module 4 - Automated Report Generation.

Every detected violation produces an immutable compliance record, written automatically.
Three sinks, all append-only:

* ``JsonlReportWriter`` - one JSON object per line. The append-only JSON log the assignment
  describes; newline-delimited so appending never rewrites the file and a truncated write
  can only ever damage the last record.
* ``CsvReportWriter`` - append-only audit CSV with the mandated fields as columns.
* ``JsonFileReportWriter`` - one file per event, for reviewers who want to diff individual
  records.

All three are opened in append mode and never seek backwards. There is no update path, by
design: a correction is a new record.
"""

from __future__ import annotations

import asyncio
import csv
import json
from pathlib import Path

from aegisflow.core.logging import get_logger
from aegisflow.core.schemas import REPORT_FIELDS, ViolationEvent

log = get_logger(__name__)


class JsonlReportWriter:
    """Append-only newline-delimited JSON log."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()
        self.written = 0

    async def write(self, event: ViolationEvent) -> None:
        line = json.dumps(event.to_report_row(), ensure_ascii=False)
        async with self._lock:
            await asyncio.to_thread(self._append, line)
        self.written += 1

    def _append(self, line: str) -> None:
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(line + "\n")

    async def write_all(self, events: list[ViolationEvent]) -> None:
        for event in events:
            await self.write(event)


class CsvReportWriter:
    """Append-only audit CSV.

    The header is written once, on creation. Column order is the assignment's field order,
    with our three additions after the mandated nine.
    """

    def __init__(self, path: Path, fields: tuple[str, ...] = REPORT_FIELDS) -> None:
        self.path = path
        self.fields = fields
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()
        self.written = 0
        if not self.path.exists() or self.path.stat().st_size == 0:
            self._write_header()

    def _write_header(self) -> None:
        with self.path.open("w", encoding="utf-8", newline="") as handle:
            csv.writer(handle).writerow(self.fields)

    async def write(self, event: ViolationEvent) -> None:
        row = event.to_report_row()
        values = [row.get(field, "") for field in self.fields]
        async with self._lock:
            await asyncio.to_thread(self._append, values)
        self.written += 1

    def _append(self, values: list[object]) -> None:
        with self.path.open("a", encoding="utf-8", newline="") as handle:
            csv.writer(handle).writerow(values)

    async def write_all(self, events: list[ViolationEvent]) -> None:
        for event in events:
            await self.write(event)


class JsonFileReportWriter:
    """One JSON file per event, named by event id."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self.written = 0

    async def write(self, event: ViolationEvent) -> None:
        path = self.directory / f"{event.event_id}.json"
        payload = json.dumps(event.to_report_row(), indent=2, ensure_ascii=False)
        await asyncio.to_thread(path.write_text, payload, encoding="utf-8")
        self.written += 1

    async def write_all(self, events: list[ViolationEvent]) -> None:
        for event in events:
            await self.write(event)


class MultiReportWriter:
    """Fans one event out to several writers.

    A failure in one sink must not stop the others: the CSV landing is not contingent on
    the JSONL landing.
    """

    def __init__(self, *writers: object) -> None:
        self.writers = list(writers)

    async def write(self, event: ViolationEvent) -> None:
        for writer in self.writers:
            try:
                await writer.write(event)  # type: ignore[attr-defined]
            except Exception as exc:
                log.warning("report writer %s failed: %s", type(writer).__name__, exc)

    async def write_all(self, events: list[ViolationEvent]) -> None:
        for event in events:
            await self.write(event)


def default_writers(outputs_root: Path) -> MultiReportWriter:
    """The standard trio, rooted at ``outputs/reports/``."""
    reports = outputs_root / "reports"
    return MultiReportWriter(
        JsonlReportWriter(reports / "audit_log.jsonl"),
        CsvReportWriter(reports / "audit_log.csv"),
        JsonFileReportWriter(reports / "events"),
    )


def read_jsonl(path: Path) -> list[dict[str, object]]:
    """Read back a JSONL audit log, skipping any damaged trailing line."""
    if not path.exists():
        return []
    records: list[dict[str, object]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except ValueError:
            log.warning("%s line %d is not valid JSON; skipped", path.name, number)
    return records
