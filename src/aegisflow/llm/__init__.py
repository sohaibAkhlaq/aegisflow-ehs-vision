"""LLM provider selection.

One entry point - :func:`build_provider` - which is guaranteed to return a working provider.
If a configured backend cannot be constructed (missing key, missing package, bad model id)
it logs the reason and hands back :class:`OfflineProvider`. Callers therefore never need a
try/except around provider construction, and the system has no configuration that makes it
fail to start.
"""

from __future__ import annotations

from aegisflow.core.enums import LLMProviderName
from aegisflow.core.errors import LLMProviderError
from aegisflow.core.logging import get_logger
from aegisflow.core.settings import Settings, get_settings
from aegisflow.llm.base import LLMProvider, LLMUsage
from aegisflow.llm.cache import ResponseCache, content_key
from aegisflow.llm.offline import OfflineProvider

log = get_logger(__name__)

__all__ = [
    "LLMProvider",
    "LLMUsage",
    "OfflineProvider",
    "ResponseCache",
    "build_provider",
    "content_key",
]


def build_provider(settings: Settings | None = None) -> LLMProvider:
    """Construct the configured provider, falling back to offline on any problem."""
    settings = settings or get_settings()
    requested = settings.llm_provider

    if requested is LLMProviderName.OFFLINE:
        return OfflineProvider()

    cache = ResponseCache(
        root=settings.path(settings.cache_root) / "llm_cache",
        enabled=settings.tuning.vlm_tiebreak.cache,
    )

    try:
        if requested is LLMProviderName.GROQ:
            from aegisflow.llm.groq_provider import GroqProvider

            provider = GroqProvider(
                api_key=settings.groq_api_key,
                text_model=settings.groq_text_model,
                vision_model=settings.groq_vision_model,
                cache=cache,
            )
        elif requested is LLMProviderName.GEMINI:
            from aegisflow.llm.gemini_provider import GeminiProvider

            provider = GeminiProvider(
                api_key=settings.gemini_api_key,
                model=settings.gemini_model,
                cache=cache,
            )
        else:  # pragma: no cover - enum is exhaustive
            raise LLMProviderError(f"unknown provider {requested!r}")
    except LLMProviderError as exc:
        log.warning("provider %r unavailable (%s); using offline mode", requested.value, exc)
        return OfflineProvider()

    log.info("LLM provider: %s", provider.name)
    return provider
