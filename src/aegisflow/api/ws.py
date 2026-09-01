"""WebSocket alert channel.

Carries the real-time alert the assignment requires for HIGH and CRITICAL violations. Each
connected client gets its own bounded subscription on the in-process alert bus, so one slow
browser tab cannot back up the detection pipeline.

Only HIGH and CRITICAL events are published here. LOW and MEDIUM are database-only by
policy, and the dashboard reads those from ``GET /api/events``.
"""

from __future__ import annotations

import asyncio
import contextlib

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from aegisflow.core.logging import get_logger
from aegisflow.escalation.bus import AlertMessage, get_alert_bus

log = get_logger(__name__)
router = APIRouter(tags=["alerts"])

HEARTBEAT_SECONDS = 25.0
"""Keeps intermediaries from closing an idle connection, and lets the client show a live
indicator without polling."""


@router.websocket("/ws/alerts")
async def alerts(websocket: WebSocket) -> None:
    """Stream alerts to one client until it disconnects."""
    await websocket.accept()
    bus = get_alert_bus()

    async with bus.subscription() as subscriber:
        await websocket.send_json(
            AlertMessage.run_status(
                connected=True,
                subscribers=bus.subscriber_count,
                published_total=bus.published_count,
            ).to_json()
        )

        pump = asyncio.create_task(_pump(websocket, subscriber), name="ws-alert-pump")
        reader = asyncio.create_task(_drain_client(websocket), name="ws-client-reader")
        try:
            # Whichever finishes first ends the connection: the pump on a send failure,
            # the reader on disconnect.
            done, pending = await asyncio.wait({pump, reader}, return_when=asyncio.FIRST_COMPLETED)
            for task in pending:
                task.cancel()
            for task in done:
                with contextlib.suppress(WebSocketDisconnect, asyncio.CancelledError):
                    task.result()
        finally:
            for task in (pump, reader):
                if not task.done():
                    task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await task

    log.debug("alert websocket closed (%d subscribers remain)", bus.subscriber_count)


async def _pump(websocket: WebSocket, subscriber) -> None:
    """Forward bus messages, emitting a heartbeat when idle."""
    while True:
        try:
            message = await asyncio.wait_for(subscriber.get(), timeout=HEARTBEAT_SECONDS)
        except TimeoutError:
            message = AlertMessage.heartbeat()
        await websocket.send_json(message.to_json())


async def _drain_client(websocket: WebSocket) -> None:
    """Read and discard client frames.

    The channel is server-push only, but a receive loop is still needed: without one, a
    client disconnect is not observed until the next send, which may be a heartbeat away.
    """
    while True:
        await websocket.receive_text()
