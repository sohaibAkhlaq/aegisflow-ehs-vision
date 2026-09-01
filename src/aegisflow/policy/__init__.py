"""Module 2a - Policy parsing.

Three layers, composed here:

    extract.py   PDF -> clean text, section tree, callouts, tables
    parse.py     structured text -> PolicyRule objects (deterministic)
    llm_extract  optional span-extraction pass for thin fields
    validate.py  faithfulness gate: literal grounding, citations, structure

Public surface::

    rule_set = await parse_policy()          # parse, validate, persist
    rule_set = load_rule_set()               # read artifacts/policy/rules.json
    rule_set = await ensure_rule_set()       # load, or parse if absent/stale
"""

from __future__ import annotations

import json
from pathlib import Path

from aegisflow.core.errors import PolicyParseError
from aegisflow.core.logging import get_logger
from aegisflow.core.schemas import PolicyRuleSet
from aegisflow.core.settings import Settings, get_settings
from aegisflow.llm import build_provider
from aegisflow.llm.base import LLMProvider
from aegisflow.policy.extract import PolicyDocument, extract_document, file_sha256
from aegisflow.policy.llm_extract import enrich_rules
from aegisflow.policy.parse import build_rules
from aegisflow.policy.validate import ValidationReport, validate_rules

log = get_logger(__name__)

__all__ = [
    "PolicyDocument",
    "PolicyRuleSet",
    "ValidationReport",
    "ensure_rule_set",
    "extract_document",
    "load_rule_set",
    "parse_policy",
]


async def parse_policy(
    settings: Settings | None = None,
    *,
    provider: LLMProvider | None = None,
    strict: bool = False,
    persist: bool = True,
) -> PolicyRuleSet:
    """Parse the policy PDF into a validated, persisted rule set."""
    settings = settings or get_settings()
    pdf_path = settings.path(settings.policy_pdf)

    doc = extract_document(pdf_path)
    rules, parse_warnings = build_rules(doc)

    owns_provider = provider is None
    provider = provider or build_provider(settings)
    try:
        rules, llm_notes = await enrich_rules(rules, doc, provider)
    finally:
        if owns_provider:
            await provider.aclose()

    report = validate_rules(rules, doc, strict=strict)
    if not report.accepted:
        raise PolicyParseError(
            "no policy rules survived validation: " + "; ".join(report.warnings) or "unknown reason"
        )

    methods = {r.extraction_method for r in report.accepted}
    rule_set = PolicyRuleSet(
        source_path=(
            str(pdf_path.relative_to(settings.root))
            if pdf_path.is_relative_to(settings.root)
            else str(pdf_path)
        ),
        source_sha256=doc.sha256,
        extraction_method="+".join(sorted(methods)) if methods else "deterministic",
        rules=tuple(report.accepted),
        sections={num: sec.title for num, sec in doc.sections.items()},
        warnings=tuple([*parse_warnings, *llm_notes, *report.warnings]),
    )

    if persist:
        save_rule_set(rule_set, settings)
    return rule_set


def save_rule_set(rule_set: PolicyRuleSet, settings: Settings | None = None) -> Path:
    """Write the rule set to ``artifacts/policy/rules.json``."""
    settings = settings or get_settings()
    path = settings.path(settings.rules_json)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rule_set.model_dump_json(indent=2), encoding="utf-8")
    log.info("wrote %d rules to %s", len(rule_set.rules), path)
    return path


def load_rule_set(settings: Settings | None = None) -> PolicyRuleSet:
    """Read a previously parsed rule set.

    Raises:
        PolicyParseError: if it is missing or unreadable. Callers that can recover should
            use :func:`ensure_rule_set` instead.
    """
    settings = settings or get_settings()
    path = settings.path(settings.rules_json)
    if not path.exists():
        raise PolicyParseError(
            f"{path} not found - run 'aegisflow policy parse' to derive rules from the PDF"
        )
    try:
        return PolicyRuleSet.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise PolicyParseError(f"could not read {path}: {exc}") from exc


async def ensure_rule_set(
    settings: Settings | None = None,
    *,
    provider: LLMProvider | None = None,
) -> PolicyRuleSet:
    """Return a rule set, parsing the PDF if the cached one is missing or stale.

    Staleness is decided by comparing the PDF's SHA-256 against the digest recorded in the
    cached rule set - so editing the policy document automatically invalidates the rules
    derived from the old one.
    """
    settings = settings or get_settings()
    path = settings.path(settings.rules_json)
    pdf_path = settings.path(settings.policy_pdf)

    if path.exists():
        try:
            cached = PolicyRuleSet.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            log.warning("cached rule set unreadable (%s); re-parsing", exc)
        else:
            if pdf_path.exists() and cached.source_sha256 == file_sha256(pdf_path):
                log.debug("using cached rule set (%d rules)", len(cached.rules))
                return cached
            log.info("policy PDF changed since rules.json was written; re-parsing")

    return await parse_policy(settings, provider=provider)


def rule_set_as_table(rule_set: PolicyRuleSet) -> list[dict[str, str]]:
    """Flatten for CLI display and the dashboard's policy panel."""
    return [
        {
            "behavior_class": rule.behavior_class.value,
            "domain": rule.domain,
            "section": rule.section_ref,
            "callout": rule.callout.value,
            "indicator": rule.observable_indicator,
            "threshold": "" if rule.numeric_threshold is None else str(rule.numeric_threshold),
            "validated": "yes" if rule.validated else "no",
        }
        for rule in rule_set.rules
    ]


def rules_json_schema() -> dict[str, object]:
    """JSON Schema of the persisted artefact, for docs and the API."""
    return json.loads(json.dumps(PolicyRuleSet.model_json_schema()))
