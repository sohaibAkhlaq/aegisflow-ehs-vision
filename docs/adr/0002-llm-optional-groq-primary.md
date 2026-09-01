# ADR 0002 — LLM usage is optional; Groq is the primary provider

- **Status:** Accepted
- **Date:** 2026-09-01
- **Deciders:** Sohaib

## Context

Two places in the system could use a language model:

1. **Policy structuring** — turning prose in `compliance_policy.pdf` into structured rules.
2. **Detection tie-break** — resolving genuinely ambiguous frames, notably the assignment's own
   example: *"a forklift carrying what might be two or three blocks."*

The original project plan named Google Gemini Vision. The available free key is a **Groq** key.
Meanwhile the product framing promises offline operation ("no cloud dependency", "runs on a
Raspberry Pi"), and a demo may happen without reliable internet.

There is also a correctness concern the assignment raises directly: *"If you use an LLM to
parse the policy, how will you verify that its extracted rules are faithful to the source
document?"*

## Decision

**1. Deterministic first, LLM second.** The policy parser's primary path is PyMuPDF layout
extraction plus a section/callout parser. It handles this document completely on its own. The
LLM adds a structuring pass for prose the regexes miss — it is never the only path.

**2. `offline` is the default provider.** `AEGISFLOW_LLM_PROVIDER=offline` runs the entire
pipeline with zero network calls, and the full test suite must pass in that mode. A missing key
is a normal configuration, not a degraded one.

**3. Groq is the primary configured provider**, behind a narrow interface:

```python
class LLMProvider(abc.ABC):
    async def complete_json(self, prompt: str, schema: dict, *, system: str = "") -> dict: ...
    async def answer_about_image(self, image_png: bytes, question: str) -> str: ...
```

Implementations: `groq_provider.py` (primary), `gemini_provider.py` (alternate),
`offline.py` (default). Selected by one env var. Adding a provider means adding a file.

**4. Every LLM output is validated before it is trusted.** Extracted rule fields must appear as
literal, whitespace-normalised substrings of the PDF text. Anything that fails is dropped and
logged — never silently corrected. Structural assertions (4 unsafe domains, 2 WARNING + 2
CRITICAL SAFETY NOTICE callouts, resolvable section refs) act as a self-test.

**5. VLM calls are capped and recorded.** At most two per clip, cached by frame hash, gated by
a confidence band, and always written into the event's `detection_method` so the audit trail
shows when a model was consulted.

## Consequences

**Good**

- The system never fails because a key is missing, a rate limit is hit, or the network is
  down — which is what makes the offline product story true rather than aspirational.
- Groq's free tier is fast and sufficient; the vision-capable Llama 4 models cover the
  tie-break.
- Swapping back to Gemini, or to any future provider, is one env var and one file.
- The faithfulness gate gives a concrete answer to a question the assignment asks explicitly.

**Bad**

- Two code paths (deterministic and LLM-assisted) to keep in agreement. Mitigated by a test
  asserting identical *structural* output in `offline` and `groq` modes.
- The literal-substring gate is strict, so a correctly-paraphrased rule can be rejected. That
  is the right direction to fail: a missing rule is visible and logged, a hallucinated rule is
  not.

## What running it against a real free-tier account actually taught us

Everything below was measured, not anticipated. It is recorded because each item cost real
debugging time and would cost it again.

**1. Model ids go stale, and availability is per-account.** Both original defaults
(`llama-3.3-70b-versatile`, `meta-llama/llama-4-scout-17b-16e-instruct`) returned 404. Of the
14 models on the account, **only two accept images at all** — `qwen/qwen3.8-27b` and
`qwen/qwen3.6-27b`. Every other model rejects image content outright with
*"messages[0].content must be a string"*.

This is why ids live in `.env` and why `aegisflow info --check-llm` exists: it lists the
account's models, checks the configured ids against that list, and does a real round-trip
including an image. Run it before a demo; a stale id otherwise surfaces as a 404 on the first
clip.

**2. The token budget is the binding constraint, and image format decides it.** The free tier
allows **8,000 tokens per minute**. A 640 px PNG frame bills at ~2,300 tokens — three requests
a minute. The same frame as JPEG at 512 px is 28 KB and roughly 15x cheaper. Sending PNG was
simply wrong: it is a lossless format for photographic content whose losses do not matter to
the questions being asked. See `detection.video.encode_for_vlm`.

**3. A 429 is an instruction, not a failure.** Groq's rate-limit body says *"Please try again
in 7.27s"*. Parsing that and sleeping turns a dropped detection into a pause, which is what a
batch run wants. Before this, half of all calls were being abandoned.

**4. Token budgets must fit the prompt, not its history.** `max_tokens=64` was correct when
this was a one-word tie-break. Once the prompt asked for a four-field JSON object — and once
the chosen model turned out to emit a `<think>` preamble — replies were truncated mid-object
and failed to parse. The symptom looked like an incapable model; the cause was our budget.

**5. Coverage beats cleverness.** Asked about a single mid-clip frame, the model answered the
forklift block count on only about a third of clips — but **every number it returned was
correct**. The misses were frames with no forklift in them. The fix was sampling more frames
across the clip, not a better prompt.

## Notes

Current configured pair, verified working: `openai/gpt-oss-120b` (text) and
`qwen/qwen3.8-27b` (vision).
