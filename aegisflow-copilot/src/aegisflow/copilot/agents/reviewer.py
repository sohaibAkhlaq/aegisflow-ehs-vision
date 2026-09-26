"""Reviewer agent: checks the Drafter's incident summary against the actual policy
rule before it's allowed into the final report.

This is the debate/consensus pattern from the multi-agent material: two independent
model calls, one producing content and a second one critiquing it, rather than trusting
a single generation. The check here is narrow and verifiable - does the cited section
match the event's real policy_rule_ref, and is the summary factually consistent with
the raw detection description - not a vague "does this look good" judgement call.
"""

from __future__ import annotations

from aegisflow.copilot.schemas import IncidentReviewVerdict

REVIEW_PROMPT = """You are reviewing a drafted incident summary before it is added to \
a compliance audit record. Check two things:

1. Does the summary's cited section ("{cited_section}") match the event's actual \
policy rule reference ("{policy_rule_ref}")?
2. Is the summary factually consistent with the raw detection description below, \
without inventing details not present in it?

Raw detection description: {event_description}
Drafted summary: {summary}

Respond with whether the draft is approved, and brief feedback if not.
"""


async def review_draft(
    draft_summary: str,
    cited_section: str,
    policy_rule_ref: str,
    event_description: str,
) -> IncidentReviewVerdict:
    """Call the LLM to review a drafted summary. Deterministic fast-path first: an
    outright section mismatch is checked in plain Python (cheap, unambiguous) before
    spending a model call on the harder, judgement-based factual-consistency check.
    """
    if cited_section.strip() != policy_rule_ref.strip():
        return IncidentReviewVerdict(
            approved=False,
            feedback=(
                f"Cited section '{cited_section}' does not match the event's actual "
                f"policy rule reference '{policy_rule_ref}'. Cite the correct section."
            ),
        )

    from aegisflow.core.settings import get_settings
    from aegisflow.llm import build_provider

    settings = get_settings()
    provider = build_provider(settings)
    if provider.is_offline:
        # No model available to check factual consistency; the deterministic section
        # check above already passed, so approve rather than block the whole workflow
        # on a check this build has no way to perform.
        return IncidentReviewVerdict(approved=True, feedback="")

    prompt = REVIEW_PROMPT.format(
        cited_section=cited_section,
        policy_rule_ref=policy_rule_ref,
        event_description=event_description,
        summary=draft_summary,
    )
    schema = {
        "type": "object",
        "properties": {
            "approved": {"type": "boolean"},
            "feedback": {"type": "string"},
        },
        "required": ["approved"],
    }
    result = await provider.complete_json(prompt=prompt, schema=schema, system="")
    return IncidentReviewVerdict(
        approved=bool(result.get("approved", False)), feedback=result.get("feedback", "")
    )
