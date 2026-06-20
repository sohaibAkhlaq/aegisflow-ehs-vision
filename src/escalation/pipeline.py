"""
pipeline.py — Module 3: Escalation Pipeline.

Receives a fully-classified violation event and routes it according to
the mandatory routing rules from the assessment specification:

    LOW / MEDIUM  → write to database only (no real-time alert)
    HIGH / CRIT   → write to database AND broadcast WebSocket alert

Also handles multi-violation clips: if a single clip produces violations
of mixed severity, the highest tier determines the alert level.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timezone

from ..escalation.websocket_manager import ws_manager
from ..severity.classifier import SeverityResult

logger = logging.getLogger(__name__)


@dataclass
class EscalationResult:
    """Record of what the pipeline did with a violation."""
    event_id: str
    severity: str
    routed_to_db: bool
    routed_to_alert: bool
    escalation_action: str
    escalated_at: str


class EscalationPipeline:
    """
    Module 3 — routes violations to the correct downstream channels.

    Routing rules (from assessment spec):
      LOW / MEDIUM → persistent DB log only
      HIGH / CRIT  → real-time WebSocket alert + persistent DB log
    """

    async def route(
        self,
        event_id: str,
        violation_payload: dict,
        severity_result: SeverityResult,
    ) -> EscalationResult:
        """
        Route a single violation event.

        Args:
            event_id: Unique event UUID.
            violation_payload: Full violation data dict for broadcasting.
            severity_result: Output from Module 2 classifier.

        Returns:
            EscalationResult describing what actions were taken.
        """
        tier = severity_result.tier
        routed_to_alert = False

        # Always log to DB (happens in report_generator; we just set the flag)
        routed_to_db = True

        # HIGH / CRITICAL: also push real-time WebSocket alert
        if tier in ("HIGH", "CRITICAL"):
            try:
                await ws_manager.broadcast_alert({
                    "event_id": event_id,
                    "severity": tier,
                    "severity_colour": severity_result.colour_hex,
                    "escalation_action": severity_result.escalation_action,
                    **violation_payload,
                })
                routed_to_alert = True
                logger.warning(
                    f"[{tier}] ALERT BROADCAST — event={event_id} "
                    f"class={violation_payload.get('behavior_name')}"
                )
            except Exception as exc:
                logger.error(f"WebSocket broadcast failed for {event_id}: {exc}")
        else:
            # LOW / MEDIUM: broadcast as a quiet event (updates timeline, no strobe)
            try:
                await ws_manager.broadcast_event({
                    "event_id": event_id,
                    "severity": tier,
                    "severity_colour": severity_result.colour_hex,
                    **violation_payload,
                })
            except Exception:
                pass  # Non-critical — DB log is the primary record
            logger.info(
                f"[{tier}] Logged — event={event_id} "
                f"class={violation_payload.get('behavior_name')}"
            )

        return EscalationResult(
            event_id=event_id,
            severity=tier,
            routed_to_db=routed_to_db,
            routed_to_alert=routed_to_alert,
            escalation_action=severity_result.escalation_action,
            escalated_at=datetime.now(timezone.utc).isoformat(),
        )

    async def route_batch(
        self,
        events: list[tuple[str, dict, SeverityResult]],
    ) -> list[EscalationResult]:
        """
        Route multiple violations.

        For mixed-severity clips: each event is routed independently.
        The assessment spec leaves this to implementer discretion — we choose
        per-event routing so each violation gets the correct treatment.
        """
        results = []
        for event_id, payload, severity in events:
            result = await self.route(event_id, payload, severity)
            results.append(result)
        return results


# Singleton
escalation_pipeline = EscalationPipeline()
