"""Groq provider - the primary configured backend.

Model ids come from configuration, not from code, because provider catalogues change and
availability differs per account - the original ``llama-3.3-70b-versatile`` and
``llama-4-scout`` defaults both 404'd on the first real account this was pointed at, and
most Groq models reject images entirely. Run ``aegisflow info --check-llm`` to verify the
configured pair before relying on it.

Two constraints shape this client, both learned the hard way against the free tier:

* **Images must be JPEG and small.** A 640 px PNG frame bills at ~2,300 tokens against an
  8,000 tokens-per-minute budget. See :func:`aegisflow.detection.video.encode_for_vlm`.
* **429s carry the exact wait.** They are honoured rather than treated as failures, so a
  batch run pauses instead of dropping detections.

Every failure mode here raises :class:`LLMProviderError`, which the call sites treat as
"fall back to the deterministic path". Nothing in the pipeline breaks when Groq is
unreachable, rate-limited, or returns something unparseable.
"""

from __future__ import annotations

import asyncio
import base64
import json
import re
from typing import Any

from aegisflow.core.errors import LLMProviderError
from aegisflow.core.logging import get_logger
from aegisflow.llm.base import LLMProvider
from aegisflow.llm.cache import ResponseCache, content_key

log = get_logger(__name__)

_MAX_RETRIES = 3
_MAX_BACKOFF_SECONDS = 30.0

# Groq's 429 body carries the exact wait: "Please try again in 7.274999999s."
_RETRY_AFTER_RE = re.compile(r"try again in ([0-9.]+)s")


def _retry_after_seconds(exc: Exception) -> float:
    """Seconds Groq asked us to wait, or 0.0 when this was not a rate-limit error."""
    match = _RETRY_AFTER_RE.search(str(exc))
    return float(match.group(1)) + 0.5 if match else 0.0


class GroqProvider(LLMProvider):
    """Groq chat-completions backend with disk caching and bounded retries."""

    name = "groq"

    def __init__(
        self,
        api_key: str,
        text_model: str,
        vision_model: str,
        cache: ResponseCache | None = None,
        timeout: float = 45.0,
    ) -> None:
        super().__init__()
        if not api_key:
            raise LLMProviderError("GROQ_API_KEY is empty; select the 'offline' provider instead")
        try:
            from groq import AsyncGroq
        except ImportError as exc:  # pragma: no cover - dependency is pinned
            raise LLMProviderError("the 'groq' package is not installed") from exc

        self._client = AsyncGroq(api_key=api_key, timeout=timeout, max_retries=0)
        self.text_model = text_model
        self.vision_model = vision_model
        self._cache = cache

    # ------------------------------------------------------------------ text

    async def complete_json(
        self, prompt: str, schema: dict[str, Any], *, system: str = ""
    ) -> dict[str, Any]:
        schema_text = json.dumps(schema, sort_keys=True)
        key = content_key("groq.json", self.text_model, system, prompt, schema_text)

        if self._cache is not None and (hit := self._cache.get(key)) is not None:
            self.usage.cache_hits += 1
            return dict(hit)

        system_prompt = system or (
            "You extract structured data from regulatory documents. Reply with JSON only. "
            "Copy wording verbatim from the source text; never paraphrase, summarise or "
            "invent. If the source does not state something, omit the field."
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"{prompt}\n\nReturn JSON matching:\n{schema_text}"},
        ]

        raw = await self._chat(
            model=self.text_model,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.0,
        )

        try:
            parsed = json.loads(raw)
        except ValueError as exc:
            self.usage.failures += 1
            raise LLMProviderError(f"Groq returned non-JSON output: {raw[:200]!r}") from exc
        if not isinstance(parsed, dict):
            self.usage.failures += 1
            raise LLMProviderError("Groq returned JSON that is not an object")

        if self._cache is not None:
            self._cache.set(key, parsed)
        return parsed

    # ---------------------------------------------------------------- vision

    async def answer_about_image(self, image_png: bytes, question: str) -> str:
        key = content_key("groq.vision", self.vision_model, question, image_png)

        if self._cache is not None and (hit := self._cache.get(key)) is not None:
            self.usage.cache_hits += 1
            return str(hit)

        data_url = "data:image/jpeg;base64," + base64.b64encode(image_png).decode("ascii")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ]

        answer = (
            await self._chat(
                model=self.vision_model,
                messages=messages,
                temperature=0.0,
                # Generous on purpose. This started as a single-word tie-break, but the
                # indicator prompt now asks for a four-field JSON object, and some models
                # (qwen, for one) emit a <think> preamble before the JSON. At 64 tokens the
                # reply was being truncated mid-object and failing to parse on roughly half
                # of all calls - which looked like the model being incapable rather than the
                # budget being wrong.
                max_tokens=512,
            )
        ).strip()

        if self._cache is not None:
            self._cache.set(key, answer)
        return answer

    # ----------------------------------------------------------------- plumbing

    async def _chat(self, **kwargs: Any) -> str:
        """One chat completion with bounded retries. Raises LLMProviderError on give-up."""
        last: Exception | None = None
        for attempt in range(_MAX_RETRIES + 1):
            try:
                self.usage.calls += 1
                self.usage.providers.add(self.name)
                response = await self._client.chat.completions.create(**kwargs)
                content = response.choices[0].message.content
                if not content:
                    raise LLMProviderError("Groq returned an empty message")
                return str(content)
            except Exception as exc:
                last = exc
                if attempt < _MAX_RETRIES:
                    delay = _retry_after_seconds(exc)
                    if delay:
                        # The free tier is token-per-minute limited and the 429 body says
                        # exactly how long to wait. Honouring it turns a hard failure into
                        # a pause, which is what a batch run wants.
                        log.debug("Groq rate limited; waiting %.1fs", delay)
                        await asyncio.sleep(min(delay, _MAX_BACKOFF_SECONDS))
                    log.debug("Groq call failed (attempt %d), retrying: %s", attempt + 1, exc)
                    continue
        self.usage.failures += 1
        raise LLMProviderError(f"Groq call failed after {_MAX_RETRIES + 1} attempts: {last}")

    async def aclose(self) -> None:
        await self._client.close()
