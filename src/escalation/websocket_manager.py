"""
websocket_manager.py — Manages WebSocket connections for Module 3.

Maintains a registry of all active dashboard connections and provides
broadcast_alert() to push real-time alert payloads to all of them.
Thread-safe using asyncio locks.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import asdict
from typing import Any

from fastapi import WebSocket

logger = logging.getLogger(__name__)


class WebSocketManager:
    """
    Manages a set of active WebSocket connections.

    Usage:
        manager = WebSocketManager()
        # In FastAPI endpoint:
        await manager.connect(websocket)
        # To broadcast:
        await manager.broadcast_alert(payload_dict)
    """

    def __init__(self) -> None:
        self._connections: list[WebSocket] = []
        self._lock = asyncio.Lock()

    async def connect(self, ws: WebSocket) -> None:
        """Accept a new WebSocket connection and register it."""
        await ws.accept()
        async with self._lock:
            self._connections.append(ws)
        logger.info(f"WebSocket connected — total: {len(self._connections)}")

    async def disconnect(self, ws: WebSocket) -> None:
        """Remove a disconnected WebSocket from the registry."""
        async with self._lock:
            if ws in self._connections:
                self._connections.remove(ws)
        logger.info(f"WebSocket disconnected — remaining: {len(self._connections)}")

    async def broadcast(self, payload: dict[str, Any]) -> None:
        """
        Send a JSON payload to ALL connected WebSocket clients.
        Dead connections are removed automatically.
        """
        if not self._connections:
            return

        message = json.dumps(payload, default=str)
        dead: list[WebSocket] = []

        async with self._lock:
            targets = list(self._connections)

        for ws in targets:
            try:
                await ws.send_text(message)
            except Exception:
                dead.append(ws)

        if dead:
            async with self._lock:
                for ws in dead:
                    if ws in self._connections:
                        self._connections.remove(ws)
            logger.debug(f"Removed {len(dead)} dead WebSocket connection(s)")

    async def broadcast_alert(self, alert_data: dict[str, Any]) -> None:
        """
        Broadcast a compliance alert event to all clients.
        Wraps payload in a standard envelope with type="alert".
        """
        payload = {"type": "alert", **alert_data}
        await self.broadcast(payload)
        logger.info(
            f"Alert broadcast — class={alert_data.get('behavior_name')} "
            f"severity={alert_data.get('severity')} "
            f"clients={len(self._connections)}"
        )

    async def broadcast_event(self, event_data: dict[str, Any]) -> None:
        """
        Broadcast a general compliance event (LOG level, non-alert).
        Wraps payload with type="event".
        """
        payload = {"type": "event", **event_data}
        await self.broadcast(payload)

    async def broadcast_status(self, status: str, details: dict | None = None) -> None:
        """Broadcast a system status update (e.g., processing started/complete)."""
        payload = {"type": "status", "status": status, **(details or {})}
        await self.broadcast(payload)

    @property
    def connection_count(self) -> int:
        return len(self._connections)


# Singleton instance — shared across the FastAPI app lifecycle
ws_manager = WebSocketManager()
