"""
tests/test_reports.py — Unit tests for Module 4: Automated Report Generation.

Tests cover:
  - ViolationEvent ORM serialization (to_dict, to_csv_row)
  - ReportGenerator: record creation, CSV/JSON append, query, export, stats
"""

from __future__ import annotations

import asyncio
import csv
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest
import pytest_asyncio

from src.detection.video_processor import ClipViolation
from src.severity.classifier import SeverityClassifier, SeverityResult
from src.reports.models import (
    ViolationEvent,
    Base,
    make_async_engine,
    make_async_session_factory,
    init_db,
)
from src.reports.report_generator import ReportGenerator


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def event_loop():
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture(scope="module")
async def db_setup(tmp_path_factory):
    """Create an in-memory SQLite DB for testing."""
    tmp = tmp_path_factory.mktemp("db")
    db_path = tmp / "test_compliance.db"
    engine = make_async_engine(str(db_path))
    session_factory = make_async_session_factory(engine)
    await init_db(engine)
    yield engine, session_factory, tmp
    await engine.dispose()


@pytest_asyncio.fixture
async def report_gen(db_setup):
    engine, session_factory, tmp = db_setup
    return ReportGenerator(session_factory, tmp)


def make_violation(
    class_id: int = 0,
    name: str = "Safe Walkway Violation",
    conf: float = 0.87,
) -> ClipViolation:
    return ClipViolation(
        event_id=str(uuid.uuid4()),
        clip_id="test_clip_001",
        clip_path="data/test_clip_001.mp4",
        frame_index=150,
        timestamp_sec=6.0,
        behavior_class_id=class_id,
        behavior_name=name,
        confidence=conf,
        zone="Zone-Mid-Center",
        description="Test violation for automated testing.",
        policy_ref=f"Section {class_id + 3}.3.2",
        detector_source="test",
        thumbnail_b64=None,
    )


def make_severity(tier: str = "CRITICAL") -> SeverityResult:
    clf = SeverityClassifier()
    class_map = {"CRITICAL": 0, "HIGH": 1, "LOW": 2}
    return clf.classify(class_map.get(tier, 2), confidence=0.9)


# ─────────────────────────────────────────────────────────────────────────────
# ViolationEvent model
# ─────────────────────────────────────────────────────────────────────────────

class TestViolationEventModel:
    """Tests for the ViolationEvent ORM model serialization."""

    def _make_event(self, **overrides) -> ViolationEvent:
        defaults = dict(
            event_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc),
            clip_id="clip_001",
            zone="Zone-A",
            behavior_class="Safe Walkway Violation",
            behavior_class_id=0,
            policy_rule_ref="Section 3.3.2",
            event_description="Person outside walkway.",
            severity="CRITICAL",
            escalation_action="Real-time alert triggered",
            confidence=0.88,
            detector_source="yolo+heuristic",
            frame_index=100,
            timestamp_in_clip=4.0,
            alert_triggered=True,
            logged_at=datetime.now(timezone.utc),
        )
        defaults.update(overrides)
        return ViolationEvent(**defaults)

    def test_to_dict_has_required_fields(self):
        evt = self._make_event()
        d   = evt.to_dict()
        required = [
            "event_id", "timestamp", "clip_id", "zone", "behavior_class",
            "policy_rule_ref", "event_description", "severity", "escalation_action"
        ]
        for f in required:
            assert f in d, f"Missing field: {f}"

    def test_to_csv_row_excludes_thumbnail(self):
        evt = self._make_event(thumbnail_b64="fake_base64_data")
        row = evt.to_csv_row()
        assert "thumbnail_b64" not in row

    def test_to_dict_timestamp_has_z_suffix(self):
        evt = self._make_event()
        d   = evt.to_dict()
        assert d["timestamp"].endswith("Z")

    def test_confidence_rounded(self):
        evt = self._make_event(confidence=0.876543)
        d   = evt.to_dict()
        # Should be rounded to 4 decimal places
        assert abs(d["confidence"] - 0.8765) < 0.0001


# ─────────────────────────────────────────────────────────────────────────────
# Report Generator
# ─────────────────────────────────────────────────────────────────────────────

class TestReportGenerator:
    """Integration tests for ReportGenerator with a test database."""

    @pytest.mark.asyncio
    async def test_record_violation_returns_event(self, report_gen):
        v   = make_violation(class_id=0)
        sev = make_severity("CRITICAL")
        evt = await report_gen.record_violation(v, sev)
        assert evt.event_id == v.event_id
        assert evt.severity == "CRITICAL"
        assert evt.behavior_class == v.behavior_name

    @pytest.mark.asyncio
    async def test_record_creates_csv_row(self, report_gen):
        v   = make_violation(class_id=1, name="Unauthorized Intervention")
        sev = make_severity("HIGH")
        await report_gen.record_violation(v, sev)
        # CSV file should exist and have data
        assert report_gen._csv_path.exists()
        with open(report_gen._csv_path, "r") as f:
            content = f.read()
        assert v.event_id in content

    @pytest.mark.asyncio
    async def test_record_creates_json_line(self, report_gen):
        v   = make_violation(class_id=2, name="Opened Panel Cover")
        sev = make_severity("LOW")
        await report_gen.record_violation(v, sev)
        assert report_gen._json_path.exists()
        with open(report_gen._json_path, "r") as f:
            lines = f.readlines()
        event_ids = [json.loads(l)["event_id"] for l in lines if l.strip()]
        assert v.event_id in event_ids

    @pytest.mark.asyncio
    async def test_get_events_returns_records(self, report_gen):
        records, total = await report_gen.get_events(limit=100)
        assert isinstance(records, list)
        assert isinstance(total, int)
        assert total >= 0

    @pytest.mark.asyncio
    async def test_get_events_severity_filter(self, report_gen):
        # Record one CRITICAL event
        v = make_violation(class_id=0)
        await report_gen.record_violation(v, make_severity("CRITICAL"))
        records, total = await report_gen.get_events(severity="CRITICAL")
        for r in records:
            assert r["severity"] == "CRITICAL"

    @pytest.mark.asyncio
    async def test_export_csv_produces_valid_csv(self, report_gen):
        csv_str = await report_gen.export_csv()
        lines = csv_str.strip().split("\n")
        assert len(lines) >= 1  # At least header
        reader = csv.DictReader(csv_str.splitlines())
        rows = list(reader)
        assert isinstance(rows, list)

    @pytest.mark.asyncio
    async def test_export_json_produces_valid_json(self, report_gen):
        json_str = await report_gen.export_json()
        data = json.loads(json_str)
        assert isinstance(data, list)

    @pytest.mark.asyncio
    async def test_get_summary_stats(self, report_gen):
        stats = await report_gen.get_summary_stats()
        assert "total_events" in stats
        assert "by_severity" in stats
        assert "by_class" in stats
        assert "alerts_triggered" in stats
        assert stats["total_events"] >= 0

    @pytest.mark.asyncio
    async def test_no_thumbnails_in_default_query(self, report_gen):
        v = make_violation(class_id=3)
        v_with_thumb = ClipViolation(
            **{**v.__dict__, "thumbnail_b64": "fake_b64_thumbnail_data"}
        )
        await report_gen.record_violation(v_with_thumb, make_severity("HIGH"))
        records, _ = await report_gen.get_events(include_thumbnails=False, limit=5)
        for r in records:
            assert "thumbnail_b64" not in r or r["thumbnail_b64"] is None

    @pytest.mark.asyncio
    async def test_alert_triggered_set_for_high(self, report_gen):
        v   = make_violation(class_id=1)
        sev = make_severity("HIGH")
        evt = await report_gen.record_violation(v, sev)
        assert evt.alert_triggered is True

    @pytest.mark.asyncio
    async def test_alert_not_triggered_for_low(self, report_gen):
        v   = make_violation(class_id=2)
        sev = make_severity("LOW")
        evt = await report_gen.record_violation(v, sev)
        assert evt.alert_triggered is False
