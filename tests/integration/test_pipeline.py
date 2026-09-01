"""End-to-end integration: policy PDF in, routed and reported compliance records out.

These exercise the seams between all five modules. The heavier ones need the dataset and
model weights and are marked ``slow``, so a clean checkout still runs the suite green.
"""

from __future__ import annotations

import pytest

from aegisflow.core.enums import BehaviorClass, DetectionMethod, EscalationAction, Severity
from aegisflow.core.schemas import DetectionRecord, FrameContext
from aegisflow.db import crud
from aegisflow.escalation import AlertBus, EscalationRouter
from aegisflow.pipeline import CompliancePipeline, RunStats, worst_severity
from aegisflow.reports import default_writers, read_jsonl
from aegisflow.severity import SeverityMatrix

pytestmark = pytest.mark.integration


class TestPolicyToSeverity:
    """Module 2a -> 2b, against the real PDF."""

    async def test_real_policy_drives_the_matrix(self, policy_pdf):
        from aegisflow.policy import parse_policy

        rule_set = await parse_policy(strict=True, persist=False)
        matrix = SeverityMatrix(rule_set)

        expected_base = {
            BehaviorClass.OPENED_PANEL_COVER: Severity.LOW,
            BehaviorClass.SAFE_WALKWAY_VIOLATION: Severity.HIGH,
            BehaviorClass.UNAUTHORIZED_INTERVENTION: Severity.CRITICAL,
            BehaviorClass.CARRYING_OVERLOAD_WITH_FORKLIFT: Severity.CRITICAL,
        }
        for behavior, expected in expected_base.items():
            rule = rule_set.require_rule(behavior)
            base, signals = matrix.base_severity(rule)
            assert base is expected, f"{behavior.value} derived {base} not {expected}"
            assert signals, "every tier must record its derivation"

    async def test_no_severity_literals_outside_the_matrix(self, repo_root):
        """Guards the policy-grounding requirement against future edits."""
        offenders = []
        for path in (repo_root / "src" / "aegisflow").rglob("*.py"):
            if path.parts[-2:] in {("severity", "matrix.py")}:
                continue
            if path.name in {"enums.py", "schemas.py", "matrix.py"}:
                continue
            text = path.read_text(encoding="utf-8")
            for behavior in ("safe_walkway_violation", "opened_panel_cover"):
                # A mapping of behaviour -> tier anywhere else would be a hard-coded rule.
                if f'"{behavior}": Severity.' in text or f"'{behavior}': Severity." in text:
                    offenders.append(f"{path.name}: {behavior}")
        assert not offenders, f"hard-coded severity mapping found: {offenders}"


class TestPipelineWiring:
    """Modules 1-4 composed, with detection stubbed so the test is fast and deterministic."""

    @pytest.fixture
    def pipeline(self, rule_set, db_session, monkeypatch):
        bus = AlertBus()
        router = EscalationRouter(db_session, bus)
        pipeline = CompliancePipeline(rule_set, sink=router)
        pipeline._bus = bus  # for assertions
        return pipeline, router, bus

    async def test_one_event_per_violation_never_merged(self, pipeline, monkeypatch, tmp_path):
        pipe, _router, bus = pipeline
        subscriber = bus.subscribe()
        pipe.writer = default_writers(tmp_path)

        records = [
            DetectionRecord(
                clip_id="multi.mp4",
                behavior_class=BehaviorClass.OPENED_PANEL_COVER,
                confidence=0.8,
                detection_method=DetectionMethod.CONTOUR,
                first_frame_index=0,
                first_timestamp_s=0.0,
                frame_count=6,
                description="Open panel observed.",
                zone="Zone-2",
                context=FrameContext(max_person_count=0),
            ),
            DetectionRecord(
                clip_id="multi.mp4",
                behavior_class=BehaviorClass.CARRYING_OVERLOAD_WITH_FORKLIFT,
                confidence=0.8,
                detection_method=DetectionMethod.CONTOUR,
                first_frame_index=4,
                first_timestamp_s=1.0,
                frame_count=4,
                description="Overloaded forklift observed.",
                zone="Zone-1",
                context=FrameContext(forklift_present=True),
            ),
        ]

        class FakeInfo:
            duration_s = 7.0

        async def fake_analyse(clip_path):
            return records, FakeInfo(), 28

        monkeypatch.setattr(pipe.engine, "analyse", fake_analyse)

        result = await pipe.process_clip("multi.mp4")

        # Two independent records with different tiers and different routing.
        assert len(result.events) == 2
        tiers = {e.behavior_class: e.severity for e in result.events}
        assert tiers[BehaviorClass.OPENED_PANEL_COVER] is Severity.LOW
        assert tiers[BehaviorClass.CARRYING_OVERLOAD_WITH_FORKLIFT] is Severity.CRITICAL

        actions = {e.behavior_class: e.escalation_action for e in result.events}
        assert actions[BehaviorClass.OPENED_PANEL_COVER] is EscalationAction.LOGGED
        assert actions[BehaviorClass.CARRYING_OVERLOAD_WITH_FORKLIFT] is EscalationAction.ALERTED

        # Exactly one alert, from the CRITICAL only.
        published = []
        while not subscriber.queue.empty():
            published.append(await subscriber.get())
        assert len(published) == 1
        assert published[0].payload["severity"] == "CRITICAL"

        # Both records in the audit log.
        rows = read_jsonl(tmp_path / "reports" / "audit_log.jsonl")
        assert len(rows) == 2
        assert {r["clip_id"] for r in rows} == {"multi.mp4"}

    async def test_events_carry_the_policy_rationale(self, pipeline, monkeypatch):
        pipe, _, _ = pipeline

        record = DetectionRecord(
            clip_id="c.mp4",
            behavior_class=BehaviorClass.UNAUTHORIZED_INTERVENTION,
            confidence=0.85,
            detection_method=DetectionMethod.HSV,
            first_frame_index=0,
            first_timestamp_s=0.0,
            frame_count=4,
            description="No green vest at equipment.",
            zone="Zone-2",
            context=FrameContext(max_person_count=1),
        )

        class FakeInfo:
            duration_s = 5.0

        monkeypatch.setattr(pipe.engine, "analyse", lambda p: _coro(([record], FakeInfo(), 20)))
        result = await pipe.process_clip("c.mp4")
        event = result.events[0]
        assert event.policy_rule_ref == "Section 4.3.2"
        assert "Section 4.3.2" in event.severity_rationale
        assert event.detection_method is DetectionMethod.HSV

    async def test_a_clean_clip_produces_no_events(self, pipeline, monkeypatch):
        pipe, _, _ = pipeline

        class FakeInfo:
            duration_s = 5.0

        monkeypatch.setattr(pipe.engine, "analyse", lambda p: _coro(([], FakeInfo(), 20)))
        result = await pipe.process_clip("clean.mp4")
        assert result.compliant
        assert result.events == ()


async def _coro(value):
    return value


class TestRunStats:
    def test_aggregates_by_tier_and_behaviour(self, make_event):
        from aegisflow.core.schemas import ClipResult

        stats = RunStats()
        stats.record(
            ClipResult(
                clip_id="a.mp4",
                clip_path="a.mp4",
                zone="Zone-1",
                frames_analysed=20,
                duration_s=5.0,
                processing_s=1.5,
                events=(
                    make_event(severity=Severity.LOW),
                    make_event(severity=Severity.CRITICAL),
                ),
            )
        )
        assert stats.clips == 1
        assert stats.events == 2
        assert stats.alerts == 1
        assert stats.by_severity == {"LOW": 1, "CRITICAL": 1}

    def test_worst_severity_picks_the_maximum(self, make_event):
        events = [make_event(severity=s) for s in (Severity.LOW, Severity.HIGH, Severity.MEDIUM)]
        assert worst_severity(events) is Severity.HIGH
        assert worst_severity([]) is None


@pytest.mark.slow
class TestRealClip:
    """Needs the dataset and YOLO weights."""

    async def test_engine_produces_valid_records(self, sample_clip, policy_pdf):
        from aegisflow.detection import DetectionEngine
        from aegisflow.policy import ensure_rule_set

        rule_set = await ensure_rule_set()
        engine = DetectionEngine(rule_set)
        records, info, frames = await engine.analyse(str(sample_clip))

        assert frames > 0
        assert info.duration_s > 0
        for record in records:
            assert record.behavior_class.is_unsafe
            assert 0.0 <= record.confidence <= 1.0
            assert record.zone
            assert record.description
            # Every detection must be explainable against a parsed rule.
            rule_set.require_rule(record.behavior_class)

    async def test_full_pipeline_persists_and_routes(
        self, sample_clip, policy_pdf, db_session, tmp_path
    ):
        from aegisflow.policy import ensure_rule_set

        rule_set = await ensure_rule_set()
        bus = AlertBus()
        router = EscalationRouter(db_session, bus)
        pipeline = CompliancePipeline(rule_set, sink=router, writer=default_writers(tmp_path))
        result = await pipeline.process_clip(str(sample_clip))

        stored = await crud.list_events(db_session, limit=50)
        assert len(stored) == len(result.events)
        for event in stored:
            assert event.escalation_action is not EscalationAction.PENDING
