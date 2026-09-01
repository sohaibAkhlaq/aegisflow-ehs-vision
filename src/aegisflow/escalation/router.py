"""Module 3 - Escalation Pipeline.

The routing rules are mandatory and come straight from the assignment:

======================  =======================================================
Severity                Route
======================  =======================================================
``LOW`` / ``MEDIUM``    persistent database log only, no alert
``HIGH`` / ``CRITICAL`` real-time alert **and** persistent database log
======================  =======================================================

Three invariants this module holds to, each of which is a test:

1. **Persist before publish.** A subscriber that has gone away must never cost us an audit
   record. The DB write happens first; the alert is best-effort after it.
2. **Every event routes on its own merit.** A clip with a MEDIUM and a CRITICAL violation
   produces two records and exactly one alert. Severities are never merged or maxed.
3. **Concurrent events are all delivered.** Two CRITICAL events milliseconds apart both
   reach the bus.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

from sqlalchemy.ext.asyncio import AsyncSession

from aegisflow.core.enums import EscalationAction
from aegisflow.core.logging import get_logger
from aegisflow.core.schemas import ViolationEvent
from aegisflow.db import crud
from aegisflow.escalation.bus import AlertBus, get_alert_bus

log = get_logger(__name__)


class EscalationRouter:
    """Routes events by severity. Implements :class:`aegisflow.core.protocols.EscalationSink`."""

    def __init__(
        self,
        session: AsyncSession,
        bus: AlertBus | None = None,
        *,
        persist: bool = True,
    ) -> None:
        self._session = session
        self._bus = bus if bus is not None else get_alert_bus()
        self._persist = persist
        self.logged_count = 0
        self.alerted_count = 0
        self.alert_recipients = 0

    async def route(self, event: ViolationEvent) -> ViolationEvent:
        """Route one event; returns it with ``escalation_action`` recorded."""
        needs_alert = event.severity.requires_realtime_alert
        action = EscalationAction.ALERTED if needs_alert else EscalationAction.LOGGED
        routed = event.with_escalation(action)

        # Invariant 1: the audit record lands before anything transient is attempted.
        if self._persist:
            await crud.insert_event(self._session, routed)
        self.logged_count += 1

        if needs_alert:
            # A publish failure must not undo the log, so it is contained here.
            try:
                self.alert_recipients += await self._bus.publish_event(routed)
                self.alerted_count += 1
            except Exception as exc:
                log.warning(
                    "alert publish failed for %s (%s); the DB record stands",
                    routed.event_id,
                    exc,
                )

        log.debug(
            "routed %s %s -> %s",
            routed.behavior_class.value,
            routed.severity.value,
            action.value,
        )
        return routed

    async def route_all(self, events: Sequence[ViolationEvent]) -> list[ViolationEvent]:
        """Route several events from one clip.

        Sequential by design. Invariant 2 requires each event to be judged independently,
        and SQLAlchemy's ``AsyncSession`` is not safe for concurrent use on one session, so
        parallelism here would buy nothing and risk interleaved writes.
        """
        return [await self.route(event) for event in events]

    def summary(self) -> dict[str, int]:
        return {
            "logged": self.logged_count,
            "alerted": self.alerted_count,
            "alert_deliveries": self.alert_recipients,
        }


class NullEscalationSink:
    """Records the routing decision without persisting or alerting.

    Used by ``aegisflow detect`` for a dry run, and by tests that only care about which
    action a severity maps to.
    """

    def __init__(self) -> None:
        self.routed: list[ViolationEvent] = []

    async def route(self, event: ViolationEvent) -> ViolationEvent:
        action = (
            EscalationAction.ALERTED
            if event.severity.requires_realtime_alert
            else EscalationAction.LOGGED
        )
        routed = event.with_escalation(action)
        self.routed.append(routed)
        return routed

    async def route_all(self, events: Sequence[ViolationEvent]) -> list[ViolationEvent]:
        return [await self.route(event) for event in events]


async def drain_with_timeout(bus: AlertBus, subscriber, limit: int, timeout: float = 1.0) -> list:
    """Collect up to ``limit`` messages from a subscriber. Test helper."""
    out = []
    try:
        async with asyncio.timeout(timeout):
            while len(out) < limit:
                out.append(await subscriber.get())
    except TimeoutError:
        pass
    return out
