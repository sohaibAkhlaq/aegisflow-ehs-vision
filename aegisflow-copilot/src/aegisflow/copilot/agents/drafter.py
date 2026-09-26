"""Drafter agent: writes a human-readable incident summary for a CRITICAL/HIGH event,
citing the policy section it breached.

Deliberately narrow - one job, one prompt, one output shape - so the Reviewer agent
(``reviewer.py``) has one clear thing to check.
"""

from __future__ import annotations

from aegisflow.copilot.schemas import IncidentDraft

DRAFT_PROMPT = """You are drafting an incident summary for a factory safety compliance \
report. Given the event details below, write ONE short paragraph (2-4 sentences) \
describing what happened, in plain professional language suitable for a compliance \
audit record. End with the policy section reference in the form "(Section X.X.X)".

Event details:
- Behaviour: {behavior_class}
- Zone: {zone}
- Severity: {severity}
- Policy rule: {policy_rule_ref}
- Raw detection description: {event_description}
{feedback_block}
"""


async def draft_incident(event: dict, feedback: str = "") -> IncidentDraft:
    """Call the LLM to draft (or redraft, given reviewer feedback) an incident summary.

    Uses the core project's own provider abstraction (``aegisflow.llm.build_provider``)
    rather than a LangChain model here - this agent is deliberately the "plain LLM call"
    half of the drafter/reviewer pair, so the two agents in this workflow aren't both
    built the same way, which would understate the point of showing two different
    integration styles side by side.
    """
    from aegisflow.core.settings import get_settings
    from aegisflow.llm import build_provider

    feedback_block = f"\nThe previous draft was rejected with this feedback: {feedback}\n" if feedback else ""
    prompt = DRAFT_PROMPT.format(feedback_block=feedback_block, **event)

    schema = {
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "cited_section": {"type": "string"},
        },
        "required": ["summary", "cited_section"],
    }

    settings = get_settings()
    provider = build_provider(settings)
    if provider.is_offline:
        # Deterministic fallback: no LLM available, produce a template summary rather
        # than failing the graph. Mirrors the core pipeline's own "abstain, don't
        # guess" philosophy - except here abstaining still needs to produce SOME
        # report, so it falls back to a plain template instead of silence.
        summary = (
            f"A {event['severity']} severity {event['behavior_class']} event was "
            f"detected in {event['zone']}. {event['event_description']} "
            f"({event['policy_rule_ref']})"
        )
        return IncidentDraft(
            event_id=event["event_id"], summary=summary, cited_section=event["policy_rule_ref"]
        )

    result = await provider.complete_json(prompt=prompt, schema=schema, system="")
    return IncidentDraft(
        event_id=event["event_id"],
        summary=result.get("summary", ""),
        cited_section=result.get("cited_section", event["policy_rule_ref"]),
    )
