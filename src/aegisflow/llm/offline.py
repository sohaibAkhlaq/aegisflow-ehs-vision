"""The default provider: deterministic, no network, always available.

This is not a stub that raises. It is a real implementation whose answers are "I have
nothing to add", expressed in a shape the callers already handle:

* ``complete_json`` returns an empty object. The policy parser then keeps only what the
  deterministic extractor found - which for KMP-OHS-POL-001 is the complete rule set.
* ``answer_about_image`` returns ``"unknown"``. The tie-break caller treats that as "keep
  the classical CV verdict".

Because both behaviours are the same ones used when a configured provider fails, the
offline path is exercised by every test run rather than being a rarely-trodden fallback.
"""

from __future__ import annotations

from typing import Any

from aegisflow.core.logging import get_logger
from aegisflow.llm.base import LLMProvider

log = get_logger(__name__)


class OfflineProvider(LLMProvider):
    """No-network provider. The default for the whole system."""

    name = "offline"

    @property
    def is_offline(self) -> bool:
        return True

    async def complete_json(
        self, prompt: str, schema: dict[str, Any], *, system: str = ""
    ) -> dict[str, Any]:
        """Contribute nothing, so the deterministic parser's output stands unchanged."""
        log.debug("offline provider: skipping LLM structuring pass")
        return {}

    async def answer_about_image(self, image_png: bytes, question: str) -> str:
        """Decline to answer, so the classical-CV verdict is kept."""
        log.debug("offline provider: skipping VLM tie-break")
        return "unknown"
