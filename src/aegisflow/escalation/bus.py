"""In-process publish/subscribe for real-time alerts.

The assignment allows the HIGH/CRITICAL alert to be "a message pushed to a live notification
queue (e.g. WebSocket, SSE, or in-process pub/sub)". This is that queue. The API layer
subscribes on behalf of each WebSocket client, so the escalation pipeline stays unaware of
transports and is testable without one.

Two properties that matter:

* **A slow or dead subscriber never blocks the pipeline.** Each subscriber has a bounded
  queue; if it fills, the oldest message is dropped for *that subscriber only* and counted.
  An audit record is never at risk, because Module 3 persists before it publishes.
* **Publishing is non-blocking.** ``publish`` returns the number of subscribers reached.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from aegisflow.core.logging import get_logger
from aegisflow.core.schemas import ViolationEvent

log = get_logger(__name__)

DEFAULT_QUEUE_SIZE = 256


@dataclass
class AlertMessage:
    """Envelope pushed to subscribers. Mirrors the WebSocket contract in HANDOVER.md."""

    type: str
    payload: dict[str, Any] = field(default_factory=dict)
    sent_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_json(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "sent_at": self.sent_at.isoformat().replace("+00:00", "Z"),
            "payload": self.payload,
        }

    @classmethod
    def violation(cls, event: ViolationEvent) -> AlertMessage:
        return cls(type="violation", payload=event.to_report_row())

    @classmethod
    def heartbeat(cls) -> AlertMessage:
        return cls(type="heartbeat")

    @classmethod
    def run_status(cls, **fields: Any) -> AlertMessage:
        return cls(type="run_status", payload=dict(fields))


class Subscriber:
    """One consumer's bounded mailbox."""

    def __init__(self, maxsize: int = DEFAULT_QUEUE_SIZE) -> None:
        self.queue: asyncio.Queue[AlertMessage] = asyncio.Queue(maxsize=maxsize)
        self.dropped = 0

    def offer(self, message: AlertMessage) -> bool:
        """Enqueue without blocking. Drops the oldest message when full."""
        try:
            self.queue.put_nowait(message)
            return True
        except asyncio.QueueFull:
            try:
                self.queue.get_nowait()  # discard oldest
                self.queue.put_nowait(message)
            except (asyncio.QueueEmpty, asyncio.QueueFull):  # pragma: no cover - race
                pass
            self.dropped += 1
            return False

    async def get(self) -> AlertMessage:
        return await self.queue.get()


class AlertBus:
    """Fan-out hub for real-time alerts."""

    def __init__(self) -> None:
        self._subscribers: set[Subscriber] = set()
        self._published = 0

    @property
    def subscriber_count(self) -> int:
        return len(self._subscribers)

    @property
    def published_count(self) -> int:
        return self._published

    def subscribe(self, maxsize: int = DEFAULT_QUEUE_SIZE) -> Subscriber:
        subscriber = Subscriber(maxsize=maxsize)
        self._subscribers.add(subscriber)
        log.debug("alert subscriber added (%d total)", len(self._subscribers))
        return subscriber

    def unsubscribe(self, subscriber: Subscriber) -> None:
        self._subscribers.discard(subscriber)
        log.debug("alert subscriber removed (%d remaining)", len(self._subscribers))

    @asynccontextmanager
    async def subscription(self, maxsize: int = DEFAULT_QUEUE_SIZE) -> AsyncIterator[Subscriber]:
        """Scoped subscription that always unsubscribes."""
        subscriber = self.subscribe(maxsize)
        try:
            yield subscriber
        finally:
            self.unsubscribe(subscriber)

    async def publish(self, message: AlertMessage) -> int:
        """Deliver to every subscriber. Returns how many were reached."""
        self._published += 1
        if not self._subscribers:
            return 0
        for subscriber in list(self._subscribers):
            subscriber.offer(message)
        return len(self._subscribers)

    async def publish_event(self, event: ViolationEvent) -> int:
        return await self.publish(AlertMessage.violation(event))


_bus: AlertBus | None = None


def get_alert_bus() -> AlertBus:
    """Process-wide bus, shared by the pipeline and the API."""
    global _bus
    if _bus is None:
        _bus = AlertBus()
    return _bus


def reset_alert_bus() -> None:
    """Drop the shared bus. For tests."""
    global _bus
    _bus = None
