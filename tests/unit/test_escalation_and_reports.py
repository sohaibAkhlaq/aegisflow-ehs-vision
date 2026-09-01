"""Modules 3 and 4: routing rules and the audit trail.

The routing table is a graded requirement, so these are assertions about the assignment's
words, not about our implementation's convenience.
"""

from __future__ import annotations

import csv
import json

import pytest

from aegisflow.core.enums import BehaviorClass, EscalationAction, Severity
from aegisflow.core.schemas import REPORT_FIELDS, REQUIRED_REPORT_FIELDS
from aegisflow.db import crud
from aegisflow.escalation import (
    AlertBus,
    AlertMessage,
    EscalationRouter,
    NullEscalationSink,
)
from aegisflow.reports import (
    CsvReportWriter,
    JsonFileReportWriter,
    JsonlReportWriter,
    MultiReportWriter,
    read_jsonl,
)


class TestRoutingRules:
    """LOW/MED -> log only. HIGH/CRIT -> alert + log."""

    @pytest.mark.parametrize(
        ("severity", "expected"),
        [
            (Severity.LOW, EscalationAction.LOGGED),
            (Severity.MEDIUM, EscalationAction.LOGGED),
            (Severity.HIGH, EscalationAction.ALERTED),
            (Severity.CRITICAL, EscalationAction.ALERTED),
        ],
    )
    async def test_action_per_tier(self, make_event, severity, expected):
        sink = NullEscalationSink()
        routed = await sink.route(make_event(severity=severity))
        assert routed.escalation_action is expected

    async def test_only_high_and_critical_publish(self, db_session, make_event):
        bus = AlertBus()
        subscriber = bus.subscribe()
        router = EscalationRouter(db_session, bus)

        events = [
            make_event(severity=Severity.LOW),
            make_event(severity=Severity.MEDIUM),
            make_event(severity=Severity.HIGH),
            make_event(severity=Severity.CRITICAL),
        ]
        await router.route_all(events)

        published = []
        while not subscriber.queue.empty():
            published.append(await subscriber.get())

        assert len(published) == 2
        assert {m.payload["severity"] for m in published} == {"HIGH", "CRITICAL"}
        assert router.summary() == {"logged": 4, "alerted": 2, "alert_deliveries": 2}

    async def test_all_four_are_persisted_regardless_of_tier(self, db_session, make_event):
        router = EscalationRouter(db_session, AlertBus())
        await router.route_all([make_event(severity=s) for s in Severity])
        assert await crud.count_events(db_session) == 4


class TestMultiViolationClips:
    """A clip with several violations must produce several independent decisions."""

    async def test_each_event_routes_on_its_own_merit(self, db_session, make_event):
        bus = AlertBus()
        subscriber = bus.subscribe()
        router = EscalationRouter(db_session, bus)

        events = [
            make_event(
                behavior=BehaviorClass.OPENED_PANEL_COVER,
                severity=Severity.LOW,
                clip_id="multi.mp4",
            ),
            make_event(
                behavior=BehaviorClass.SAFE_WALKWAY_VIOLATION,
                severity=Severity.HIGH,
                clip_id="multi.mp4",
            ),
            make_event(
                behavior=BehaviorClass.CARRYING_OVERLOAD_WITH_FORKLIFT,
                severity=Severity.CRITICAL,
                clip_id="multi.mp4",
            ),
        ]
        routed = await router.route_all(events)

        # Three records, three distinct severities - never collapsed to the maximum.
        assert len(routed) == 3
        assert [e.severity for e in routed] == [Severity.LOW, Severity.HIGH, Severity.CRITICAL]
        assert routed[0].escalation_action is EscalationAction.LOGGED
        assert routed[1].escalation_action is EscalationAction.ALERTED

        published = []
        while not subscriber.queue.empty():
            published.append(await subscriber.get())
        assert len(published) == 2, "only the HIGH and CRITICAL should alert"

        stored = await crud.list_events(db_session, clip_id="multi.mp4", limit=10)
        assert len(stored) == 3

    async def test_concurrent_criticals_all_delivered(self, db_session, make_event):
        bus = AlertBus()
        subscriber = bus.subscribe()
        router = EscalationRouter(db_session, bus)
        await router.route_all(
            [make_event(severity=Severity.CRITICAL, clip_id=f"c{i}.mp4") for i in range(12)]
        )
        received = 0
        while not subscriber.queue.empty():
            await subscriber.get()
            received += 1
        assert received == 12


class TestAlertBus:
    async def test_publish_without_subscribers_is_harmless(self, make_event):
        bus = AlertBus()
        assert await bus.publish_event(make_event(severity=Severity.HIGH)) == 0

    async def test_slow_subscriber_drops_oldest_and_never_blocks(self, make_event):
        """A stalled dashboard must not back up the detection pipeline."""
        bus = AlertBus()
        subscriber = bus.subscribe(maxsize=3)
        for index in range(10):
            await bus.publish(AlertMessage.run_status(index=index))
        assert subscriber.queue.qsize() == 3
        assert subscriber.dropped == 7

    async def test_unsubscribe_stops_delivery(self, make_event):
        bus = AlertBus()
        subscriber = bus.subscribe()
        bus.unsubscribe(subscriber)
        assert await bus.publish_event(make_event(severity=Severity.CRITICAL)) == 0

    async def test_envelope_shape_matches_the_contract(self, make_event):
        message = AlertMessage.violation(make_event(severity=Severity.HIGH))
        payload = message.to_json()
        assert set(payload) == {"type", "sent_at", "payload"}
        assert payload["type"] == "violation"
        assert payload["sent_at"].endswith("Z")
        assert payload["payload"]["severity"] == "HIGH"


class TestReportWriters:
    async def test_jsonl_is_append_only(self, tmp_path, make_event):
        path = tmp_path / "audit.jsonl"
        writer = JsonlReportWriter(path)
        await writer.write(make_event(clip_id="a.mp4"))
        await writer.write(make_event(clip_id="b.mp4"))

        records = read_jsonl(path)
        assert [r["clip_id"] for r in records] == ["a.mp4", "b.mp4"]

        # A second writer appends rather than truncating.
        await JsonlReportWriter(path).write(make_event(clip_id="c.mp4"))
        assert len(read_jsonl(path)) == 3

    async def test_jsonl_records_carry_every_mandated_field(self, tmp_path, make_event):
        path = tmp_path / "audit.jsonl"
        await JsonlReportWriter(path).write(make_event())
        record = read_jsonl(path)[0]
        for field in REQUIRED_REPORT_FIELDS:
            assert field in record

    async def test_csv_writes_header_once(self, tmp_path, make_event):
        path = tmp_path / "audit.csv"
        writer = CsvReportWriter(path)
        await writer.write(make_event(clip_id="a.mp4"))
        await CsvReportWriter(path).write(make_event(clip_id="b.mp4"))

        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.reader(handle))
        assert rows[0] == list(REPORT_FIELDS)
        assert len(rows) == 3
        assert rows[1][2] == "a.mp4"

    async def test_one_json_file_per_event(self, tmp_path, make_event):
        writer = JsonFileReportWriter(tmp_path / "events")
        events = [make_event(clip_id=f"{i}.mp4") for i in range(3)]
        await writer.write_all(events)
        files = sorted((tmp_path / "events").glob("*.json"))
        assert len(files) == 3
        payload = json.loads(files[0].read_text(encoding="utf-8"))
        assert "event_id" in payload

    async def test_a_failing_sink_does_not_block_the_others(self, tmp_path, make_event):
        class Broken:
            async def write(self, event):
                raise OSError("disk on fire")

        path = tmp_path / "audit.jsonl"
        multi = MultiReportWriter(Broken(), JsonlReportWriter(path))
        await multi.write(make_event())
        assert len(read_jsonl(path)) == 1

    def test_corrupt_trailing_line_is_skipped(self, tmp_path):
        path = tmp_path / "audit.jsonl"
        path.write_text('{"event_id": "a"}\n{"broken": \n', encoding="utf-8")
        assert len(read_jsonl(path)) == 1


class TestAppendOnlyDiscipline:
    def test_crud_exposes_no_update_or_delete(self):
        """Immutability enforced by absence, not by convention."""
        forbidden = {"update_event", "delete_event", "remove_event", "edit_event"}
        assert not forbidden & set(dir(crud))
