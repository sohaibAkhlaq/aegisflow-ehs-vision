"""Provider interface for the two places the system can consult a model.

Two call sites only:

1. **Policy structuring** (``complete_json``) - recover indicators phrased in prose the
   deterministic regexes miss. Output goes through the faithfulness gate in
   ``policy/validate.py`` before it is trusted.
2. **Detection tie-break** (``answer_about_image``) - resolve an ambiguous frame with one
   narrow, closed question.

Both call sites must degrade to the offline path. A provider raising
:class:`LLMProviderError` is a normal, handled outcome, never fatal.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any


@dataclass
class LLMUsage:
    """Call accounting, surfaced in the CLI summary and the evaluation report."""

    calls: int = 0
    cache_hits: int = 0
    failures: int = 0
    providers: set[str] = field(default_factory=set)

    def merge(self, other: LLMUsage) -> None:
        self.calls += other.calls
        self.cache_hits += other.cache_hits
        self.failures += other.failures
        self.providers |= other.providers


class LLMProvider(abc.ABC):
    """Base class for every provider."""

    name: str = "base"

    def __init__(self) -> None:
        self.usage = LLMUsage()

    @property
    def is_offline(self) -> bool:
        """True when this provider makes no network calls."""
        return False

    @abc.abstractmethod
    async def complete_json(
        self, prompt: str, schema: dict[str, Any], *, system: str = ""
    ) -> dict[str, Any]:
        """Return a JSON object conforming to ``schema``.

        Raises:
            LLMProviderError: on transport failure, rate limiting or unparseable output.
        """

    @abc.abstractmethod
    async def answer_about_image(self, image_png: bytes, question: str) -> str:
        """Answer one closed question about a single frame.

        Questions are deliberately narrow ("How many blocks are on the forks? Reply with a
        single integer.") so the answer is cheap to parse and hard to get creatively wrong.
        """

    async def aclose(self) -> None:
        """Release any transport resources."""
        return None
