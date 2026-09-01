"""Exception hierarchy.

One root so callers can catch everything from this package with a single except.
"""

from __future__ import annotations


class AegisFlowError(Exception):
    """Base for every error raised by AegisFlow."""


class ConfigError(AegisFlowError):
    """Configuration is missing, malformed, or internally inconsistent."""


class PolicyParseError(AegisFlowError):
    """The compliance policy PDF could not be parsed into a usable rule set."""


class PolicyValidationError(AegisFlowError):
    """Extracted rules failed the faithfulness gate against the source PDF."""


class DetectionError(AegisFlowError):
    """The detection engine could not process a clip."""


class VideoReadError(DetectionError):
    """A video file could not be opened or decoded."""


class LLMProviderError(AegisFlowError):
    """An LLM provider call failed.

    Never fatal on its own: every LLM call site must degrade to the offline path.
    """


class EscalationError(AegisFlowError):
    """An event could not be routed."""
