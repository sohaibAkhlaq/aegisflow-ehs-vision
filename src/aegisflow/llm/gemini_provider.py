"""Google Gemini provider - the alternate backend.

Kept because the original project plan named Gemini Vision, and because having a second
real implementation is what proves the provider interface is actually an abstraction rather
than a Groq-shaped hole. Selected with ``AEGISFLOW_LLM_PROVIDER=gemini``.

The ``google-generativeai`` SDK is synchronous, so calls are pushed to a worker thread to
keep the async pipeline from blocking.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from aegisflow.core.errors import LLMProviderError
from aegisflow.core.logging import get_logger
from aegisflow.llm.base import LLMProvider
from aegisflow.llm.cache import ResponseCache, content_key

log = get_logger(__name__)


class GeminiProvider(LLMProvider):
    """Gemini backend. Same contract as :class:`GroqProvider`."""

    name = "gemini"

    def __init__(
        self,
        api_key: str,
        model: str,
        cache: ResponseCache | None = None,
    ) -> None:
        super().__init__()
        if not api_key:
            raise LLMProviderError("GEMINI_API_KEY is empty; select the 'offline' provider instead")
        try:
            import google.generativeai as genai
        except ImportError as exc:  # pragma: no cover - dependency is pinned
            raise LLMProviderError("the 'google-generativeai' package is not installed") from exc

        genai.configure(api_key=api_key)
        self._genai = genai
        self.model_name = model
        self._cache = cache

    async def complete_json(
        self, prompt: str, schema: dict[str, Any], *, system: str = ""
    ) -> dict[str, Any]:
        schema_text = json.dumps(schema, sort_keys=True)
        key = content_key("gemini.json", self.model_name, system, prompt, schema_text)

        if self._cache is not None and (hit := self._cache.get(key)) is not None:
            self.usage.cache_hits += 1
            return dict(hit)

        instruction = system or (
            "You extract structured data from regulatory documents. Reply with JSON only. "
            "Copy wording verbatim from the source text; never paraphrase or invent."
        )
        full_prompt = f"{instruction}\n\n{prompt}\n\nReturn JSON matching:\n{schema_text}"

        raw = await self._generate(full_prompt, mime_type="application/json")
        try:
            parsed = json.loads(raw)
        except ValueError as exc:
            self.usage.failures += 1
            raise LLMProviderError(f"Gemini returned non-JSON output: {raw[:200]!r}") from exc
        if not isinstance(parsed, dict):
            self.usage.failures += 1
            raise LLMProviderError("Gemini returned JSON that is not an object")

        if self._cache is not None:
            self._cache.set(key, parsed)
        return parsed

    async def answer_about_image(self, image_png: bytes, question: str) -> str:
        key = content_key("gemini.vision", self.model_name, question, image_png)

        if self._cache is not None and (hit := self._cache.get(key)) is not None:
            self.usage.cache_hits += 1
            return str(hit)

        def _call() -> str:
            model = self._genai.GenerativeModel(self.model_name)
            response = model.generate_content(
                [question, {"mime_type": "image/png", "data": image_png}]
            )
            return str(response.text or "")

        answer = (await self._run(_call)).strip()
        if self._cache is not None:
            self._cache.set(key, answer)
        return answer

    async def _generate(self, prompt: str, mime_type: str | None = None) -> str:
        def _call() -> str:
            kwargs: dict[str, Any] = {}
            if mime_type:
                kwargs["generation_config"] = {
                    "response_mime_type": mime_type,
                    "temperature": 0.0,
                }
            model = self._genai.GenerativeModel(self.model_name)
            response = model.generate_content(prompt, **kwargs)
            return str(response.text or "")

        return await self._run(_call)

    async def _run(self, fn: Any) -> str:
        self.usage.calls += 1
        self.usage.providers.add(self.name)
        try:
            result = await asyncio.to_thread(fn)
        except Exception as exc:
            self.usage.failures += 1
            raise LLMProviderError(f"Gemini call failed: {exc}") from exc
        if not result:
            self.usage.failures += 1
            raise LLMProviderError("Gemini returned an empty response")
        return result
