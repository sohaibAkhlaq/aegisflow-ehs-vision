"""API integration tests - the surface the dashboard is built on.

Every route the three views depend on is exercised against a real in-memory database, so a
change that breaks View C's filters or the export button fails here rather than in a demo.
"""

from __future__ import annotations

import csv
import io
import json
from datetime import UTC, datetime, timedelta

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from aegisflow.api.app import create_app
from aegisflow.core.enums import BehaviorClass, EscalationAction, Severity
from aegisflow.core.schemas import REQUIRED_REPORT_FIELDS
from aegisflow.db import crud
from aegisflow.db.models import Base
from aegisflow.db.session import get_db_session

pytestmark = pytest.mark.integration


@pytest_asyncio.fixture
async def client(make_event):
    """App wired to a throwaway database, pre-loaded with a spread of events."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", future=True)
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

    base = datetime(2026, 3, 1, 9, 0, tzinfo=UTC)
    seed = [
        (BehaviorClass.OPENED_PANEL_COVER, Severity.LOW, "Zone-2", 0),
        (BehaviorClass.SAFE_WALKWAY_VIOLATION, Severity.MEDIUM, "Zone-1", 1),
        (BehaviorClass.SAFE_WALKWAY_VIOLATION, Severity.HIGH, "Zone-1", 2),
        (BehaviorClass.UNAUTHORIZED_INTERVENTION, Severity.CRITICAL, "Zone-2", 3),
        (BehaviorClass.CARRYING_OVERLOAD_WITH_FORKLIFT, Severity.CRITICAL, "Zone-1", 4),
    ]
    async with factory() as session:
        for behavior, severity, zone, offset in seed:
            event = make_event(
                behavior=behavior,
                severity=severity,
                clip_id=f"clip_{offset}.mp4",
                zone=zone,
                timestamp=base + timedelta(days=offset),
                escalation_action=(
                    EscalationAction.ALERTED
                    if severity.requires_realtime_alert
                    else EscalationAction.LOGGED
                ),
            )
            await crud.insert_event(session, event)
        await session.commit()

    app = create_app()

    async def override():
        async with factory() as session:
            yield session

    app.dependency_overrides[get_db_session] = override
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as http:
        yield http
    await engine.dispose()


class TestHealth:
    async def test_reports_component_state(self, client):
        response = await client.get("/api/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["database_ready"] is True
        assert body["llm_provider"] in {"offline", "groq", "gemini"}


class TestStats:
    async def test_counts_by_severity_and_behaviour(self, client):
        body = (await client.get("/api/stats")).json()
        assert body["events_recorded"] == 5
        assert body["by_severity"] == {"LOW": 1, "MEDIUM": 1, "HIGH": 1, "CRITICAL": 2}
        assert body["alerts_total"] == 3
        assert body["by_zone"] == {"Zone-1": 3, "Zone-2": 2}


class TestEventHistory:
    """View C: filter by date range, severity tier and behaviour class."""

    async def test_lists_newest_first_with_a_total(self, client):
        body = (await client.get("/api/events", params={"limit": 3})).json()
        assert body["total"] == 5
        assert len(body["items"]) == 3
        stamps = [item["timestamp"] for item in body["items"]]
        assert stamps == sorted(stamps, reverse=True)

    async def test_filters_by_severity(self, client):
        body = (await client.get("/api/events?severity=CRITICAL")).json()
        assert body["total"] == 2
        assert {item["severity"] for item in body["items"]} == {"CRITICAL"}

    async def test_filters_by_several_severities(self, client):
        body = (await client.get("/api/events?severity=HIGH&severity=CRITICAL")).json()
        assert body["total"] == 3

    async def test_filters_by_behaviour_class(self, client):
        body = (await client.get("/api/events?behavior_class=safe_walkway_violation")).json()
        assert body["total"] == 2

    async def test_filters_by_date_range(self, client):
        # Seeded events land on 2026-03-01 .. 2026-03-05, one per day.
        body = (
            await client.get(
                "/api/events",
                params={"date_from": "2026-03-03T00:00:00", "date_to": "2026-03-06T00:00:00"},
            )
        ).json()
        assert body["total"] == 3

    async def test_date_range_boundaries_are_inclusive(self, client):
        body = (
            await client.get(
                "/api/events",
                params={"date_from": "2026-03-02T09:00:00", "date_to": "2026-03-03T09:00:00"},
            )
        ).json()
        assert body["total"] == 2

    async def test_combined_filters_intersect(self, client):
        body = (
            await client.get(
                "/api/events",
                params={"severity": "CRITICAL", "behavior_class": "unauthorized_intervention"},
            )
        ).json()
        assert body["total"] == 1

    async def test_pagination_does_not_overlap(self, client):
        first = (await client.get("/api/events?limit=2&offset=0")).json()
        second = (await client.get("/api/events?limit=2&offset=2")).json()
        ids = {i["event_id"] for i in first["items"]} & {i["event_id"] for i in second["items"]}
        assert not ids

    async def test_rejects_an_invalid_severity(self, client):
        assert (await client.get("/api/events?severity=EXTREME")).status_code == 422

    async def test_rejects_an_inverted_date_range(self, client):
        response = await client.get(
            "/api/events",
            params={"date_from": "2026-05-01T00:00:00", "date_to": "2026-01-01T00:00:00"},
        )
        assert response.status_code == 422

    async def test_single_event_lookup(self, client):
        listing = (await client.get("/api/events?limit=1")).json()
        event_id = listing["items"][0]["event_id"]
        body = (await client.get(f"/api/events/{event_id}")).json()
        assert body["event_id"] == event_id
        assert body["severity_rationale"]

    async def test_unknown_event_is_404(self, client):
        response = await client.get("/api/events/2f1c8b1e-0000-0000-0000-000000000000")
        assert response.status_code == 404


class TestExport:
    """View C's export button."""

    async def test_csv_export_has_the_mandated_columns(self, client):
        response = await client.get("/api/events/export?format=csv")
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/csv")
        assert "attachment" in response.headers["content-disposition"]

        rows = list(csv.reader(io.StringIO(response.text)))
        for field in REQUIRED_REPORT_FIELDS:
            assert field in rows[0]
        assert len(rows) == 6  # header + 5 events

    async def test_json_export_round_trips(self, client):
        response = await client.get("/api/events/export?format=json")
        payload = json.loads(response.text)
        assert len(payload) == 5
        assert all("policy_rule_ref" in row for row in payload)

    async def test_export_honours_the_filters(self, client):
        response = await client.get("/api/events/export?format=json&severity=CRITICAL")
        assert len(json.loads(response.text)) == 2

    async def test_export_route_is_not_shadowed_by_the_id_route(self, client):
        """'/api/events/export' must not be read as an event id."""
        assert (await client.get("/api/events/export")).status_code == 200

    async def test_rejects_an_unknown_format(self, client):
        assert (await client.get("/api/events/export?format=xlsx")).status_code == 422


class TestPolicyEndpoint:
    async def test_exposes_rules_and_their_derivation(self, client):
        response = await client.get("/api/policy")
        if response.status_code == 503:
            pytest.skip("rules.json not present; run 'aegisflow policy parse'")
        body = response.json()
        assert body["document_id"] == "KMP-OHS-POL-001"
        assert len(body["rules"]) >= 4
        for rule in body["rules"]:
            assert rule["section_ref"].startswith("Section ")
            assert rule["derivation"], "the dashboard shows how each tier was derived"


class TestClips:
    async def test_empty_clip_list_is_not_an_error(self, client):
        response = await client.get("/api/clips")
        assert response.status_code == 200
        assert response.json() == []

    async def test_unknown_clip_video_is_404(self, client):
        assert (await client.get("/api/clips/nope.mp4/video")).status_code == 404


class TestOpenApi:
    async def test_schema_documents_every_dashboard_route(self, client):
        schema = (await client.get("/openapi.json")).json()
        for path in (
            "/api/health",
            "/api/stats",
            "/api/events",
            "/api/events/export",
            "/api/policy",
            "/api/clips",
        ):
            assert path in schema["paths"], f"{path} missing from the OpenAPI schema"
