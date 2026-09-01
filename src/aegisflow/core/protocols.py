"""Structural interfaces between modules.

Protocols rather than base classes so implementations stay independent and tests can
substitute a fake without inheriting anything.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from aegisflow.core.schemas import ViolationEvent


@runtime_checkable
class EscalationSink(Protocol):
    """Module 3. Routes one event and returns it with ``escalation_action`` recorded.

    Contract:
        * ``LOW``/``MEDIUM``  -> persistent log only, no alert.
        * ``HIGH``/``CRITICAL`` -> real-time alert **and** persistent log.
        * Persist before publishing, so a dropped subscriber never loses an audit record.
        * Each event routes on its own merit. Never collapse several violations from one
          clip into a single decision.
    """

    async def route(self, event: ViolationEvent) -> ViolationEvent: ...


@runtime_checkable
class ReportWriter(Protocol):
    """Module 4. Appends an immutable compliance record. Never updates or deletes."""

    async def write(self, event: ViolationEvent) -> None: ...


@runtime_checkable
class AlertPublisher(Protocol):
    """Real-time fan-out for HIGH/CRITICAL events (WebSocket, SSE, in-process pub/sub)."""

    async def publish(self, event: ViolationEvent) -> int:
        """Deliver to every subscriber; returns how many received it."""
        ...
