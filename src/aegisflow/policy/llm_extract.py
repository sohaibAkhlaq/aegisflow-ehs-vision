"""Optional LLM structuring pass over the policy sections.

This never replaces the deterministic parser - it *supplements* it. The pass runs only when
a provider is configured, and only for fields the deterministic layer left thin (a missing
observable indicator, an empty source quote). Whatever comes back goes through the same
faithfulness gate as everything else, so a model that invents text cannot get a rule in.

Design choice worth stating: we ask for *verbatim extraction*, never interpretation. The
prompt tells the model to copy spans out of the supplied text. That turns a generative task
into a span-selection task, which is both easier for the model and trivially checkable
against the source.
"""

from __future__ import annotations

from typing import Any

from aegisflow.core.errors import LLMProviderError
from aegisflow.core.logging import get_logger
from aegisflow.core.schemas import PolicyRule
from aegisflow.llm.base import LLMProvider
from aegisflow.policy.extract import PolicyDocument

log = get_logger(__name__)

_SYSTEM = (
    "You extract spans from regulatory documents. You never paraphrase, summarise, "
    "translate or invent. Every string you return must be copied character-for-character "
    "from the SOURCE TEXT you are given. If the source does not state something, return an "
    "empty string for that field. Reply with JSON only."
)

_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "observable_indicator": {
            "type": "string",
            "description": "Verbatim span naming the visually observable indicator of the "
            "unsafe behaviour (a colour, a count, a state, a position).",
        },
        "unsafe_description": {
            "type": "string",
            "description": "Verbatim sentence defining the unsafe behaviour.",
        },
        "source_quote": {
            "type": "string",
            "description": "Verbatim sentence that best evidences the rule.",
        },
    },
    "required": ["observable_indicator"],
}

_MAX_SECTION_CHARS = 6000


async def enrich_rules(
    rules: list[PolicyRule],
    doc: PolicyDocument,
    provider: LLMProvider,
) -> tuple[list[PolicyRule], list[str]]:
    """Fill thin fields on ``rules`` using ``provider``.

    Returns ``(rules, notes)``. Offline providers return the input unchanged, so callers do
    not need to branch on provider type.
    """
    if provider.is_offline:
        return rules, []

    enriched: list[PolicyRule] = []
    notes: list[str] = []

    for rule in rules:
        gaps = _gaps(rule)
        if not gaps:
            enriched.append(rule)
            continue

        section_number = rule.section_ref.removeprefix("Section ").strip()
        section = doc.section(section_number)
        if section is None:
            enriched.append(rule)
            continue

        prompt = (
            f"SOURCE TEXT (Section {section_number} - {section.title}):\n"
            f"{section.text[:_MAX_SECTION_CHARS]}\n\n"
            f"The unsafe behaviour defined in this section is: "
            f"{rule.behavior_class.display_name}.\n"
            f"Extract the fields below by copying spans verbatim from the SOURCE TEXT."
        )

        try:
            result = await provider.complete_json(prompt, _SCHEMA, system=_SYSTEM)
        except LLMProviderError as exc:
            # Expected outcome, not an error: keep the deterministic result.
            notes.append(f"{rule.behavior_class.value}: LLM pass skipped ({exc})")
            log.info("LLM enrichment skipped for %s: %s", rule.behavior_class.value, exc)
            enriched.append(rule)
            continue

        updates = {
            name: str(result[name]).strip()
            for name in gaps
            if isinstance(result.get(name), str) and str(result[name]).strip()
        }
        if not updates:
            enriched.append(rule)
            continue

        notes.append(
            f"{rule.behavior_class.value}: LLM supplied {', '.join(sorted(updates))} "
            "(subject to the faithfulness gate)"
        )
        enriched.append(
            rule.model_copy(
                update={
                    **updates,
                    "extraction_method": "deterministic+llm",
                    # Reset: anything the model touched must be re-validated.
                    "validated": False,
                }
            )
        )

    return enriched, notes


def _gaps(rule: PolicyRule) -> list[str]:
    """Fields thin enough to be worth asking about."""
    gaps: list[str] = []
    if len(rule.observable_indicator.strip()) < 10:
        gaps.append("observable_indicator")
    if len(rule.unsafe_description.strip()) < 20:
        gaps.append("unsafe_description")
    if len(rule.source_quote.strip()) < 20:
        gaps.append("source_quote")
    return gaps
