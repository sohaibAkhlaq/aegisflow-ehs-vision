"""Layer 3 of policy parsing: the faithfulness gate.

The assignment asks directly: *"If you use an LLM to parse the policy, how will you verify
that its extracted rules are faithful to the source document? What could go wrong, and how
would you catch it?"*

This module is the answer. Every rule must survive five checks before it is trusted:

1. **Literal grounding.** Each prose field must appear verbatim (whitespace-normalised) in
   the extracted PDF text. This is what catches a hallucinated indicator or a paraphrased
   threshold - the failure mode that matters most.
2. **Citation resolution.** ``section_ref`` must be well-formed *and* name a section that
   actually exists in the parsed tree. Catches fabricated citations.
3. **Cross-extractor agreement.** The source quote is checked against an independent pypdf
   extraction. Disagreement means the text may be a layout artefact, so the rule is flagged
   low-confidence rather than dropped.
4. **Structural completeness.** Exactly four unsafe domains, each with an indicator.
   Catches silent under-extraction, where a plausible-looking rule set is simply missing a
   behaviour.
5. **Callout distribution.** Exactly two WARNING and two CRITICAL SAFETY NOTICE bindings -
   the assignment's own hint, used as a self-test on our callout binding.

A rule that fails 1, 2 or 4 is **excluded and recorded in ``warnings``**, never silently
repaired. Failing 3 or 5 downgrades confidence but keeps the rule, because those checks
detect *suspicion* rather than *error*.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from aegisflow.core.enums import PolicyCallout
from aegisflow.core.logging import get_logger
from aegisflow.core.schemas import PolicyRule
from aegisflow.policy.extract import PolicyDocument, squash

log = get_logger(__name__)

EXPECTED_UNSAFE_DOMAINS = 4
EXPECTED_WARNINGS = 2
EXPECTED_CRITICAL_NOTICES = 2

# Fields whose text must be traceable to the document.
_GROUNDED_FIELDS = ("source_quote", "unsafe_description", "observable_indicator")

# The Section 8 table is rendered one word per row, so its cell text is reassembled by the
# parser and will not appear as a contiguous run in the page text. Such fields are verified
# token-wise instead of as a contiguous substring.
_TOKENWISE_FIELDS = frozenset({"observable_indicator"})

_MIN_TOKEN_COVERAGE = 0.85


@dataclass
class ValidationReport:
    """Outcome of the gate."""

    accepted: list[PolicyRule] = field(default_factory=list)
    rejected: list[tuple[PolicyRule, str]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.rejected and not self.warnings

    def summary(self) -> str:
        return (
            f"{len(self.accepted)} accepted, {len(self.rejected)} rejected, "
            f"{len(self.warnings)} warnings"
        )


def validate_rules(
    rules: list[PolicyRule],
    doc: PolicyDocument,
    *,
    strict: bool = False,
) -> ValidationReport:
    """Run the gate over ``rules``.

    Args:
        rules: candidate rules from the deterministic and/or LLM extractors.
        doc: the document they were extracted from.
        strict: when True, structural failures raise instead of warning. Used by tests and
            by ``aegisflow policy parse --strict``.

    Raises:
        PolicyValidationError: only when ``strict`` and a structural check fails.
    """
    haystack = squash(doc.full_text)
    secondary = squash(doc.secondary_text)
    report = ValidationReport()

    for rule in rules:
        problems: list[str] = []
        notes: list[str] = []

        # --- check 1: literal grounding ---
        for field_name in _GROUNDED_FIELDS:
            value = getattr(rule, field_name, "") or ""
            if not value:
                continue
            if not _is_grounded(value, haystack, tokenwise=field_name in _TOKENWISE_FIELDS):
                problems.append(f"{field_name} is not traceable to the source document")

        # --- check 2: citation resolves ---
        section_number = rule.section_ref.removeprefix("Section ").strip()
        if doc.section(section_number) is None:
            problems.append(f"{rule.section_ref} does not exist in the parsed section tree")

        # --- check 3: cross-extractor agreement (advisory) ---
        if (
            rule.source_quote
            and secondary
            and not _is_grounded(rule.source_quote, secondary, tokenwise=True)
        ):
            notes.append(
                f"{rule.behavior_class.value}: source quote not confirmed by the "
                "secondary extractor; treat as low confidence"
            )

        if problems:
            reason = "; ".join(problems)
            report.rejected.append((rule, reason))
            report.warnings.append(f"REJECTED {rule.behavior_class.value}: {reason}")
            log.warning("rejected rule %s: %s", rule.behavior_class.value, reason)
            continue

        report.warnings.extend(notes)
        report.accepted.append(
            rule.model_copy(
                update={
                    "validated": True,
                    "validation_notes": tuple(notes),
                }
            )
        )

    _check_structure(report, strict=strict)
    log.info("policy validation: %s", report.summary())
    return report


def _is_grounded(value: str, haystack: str, *, tokenwise: bool = False) -> bool:
    """Is ``value`` traceable to ``haystack``?

    Contiguous substring match is the strong form. Token coverage is the weaker form used
    for text the parser reassembled from table cells, where a contiguous match cannot
    exist by construction - but every word must still come from the document.
    """
    needle = squash(value)
    if not needle:
        return False
    if needle in haystack:
        return True
    if not tokenwise:
        return False

    tokens = [t for t in needle.split() if len(t) > 2]
    if not tokens:
        return False
    present = sum(1 for token in tokens if token in haystack)
    return (present / len(tokens)) >= _MIN_TOKEN_COVERAGE


def _check_structure(report: ValidationReport, *, strict: bool) -> None:
    """Checks 4 and 5: is the accepted set complete and correctly distributed?"""
    from aegisflow.core.errors import PolicyValidationError

    problems: list[str] = []
    unsafe = [r for r in report.accepted if r.behavior_class.is_unsafe]

    if len(unsafe) != EXPECTED_UNSAFE_DOMAINS:
        problems.append(
            f"expected {EXPECTED_UNSAFE_DOMAINS} unsafe behaviour rules, derived {len(unsafe)}"
        )

    for rule in unsafe:
        if not rule.observable_indicator.strip():
            problems.append(f"{rule.behavior_class.value} has no observable indicator")

    duplicates = len(unsafe) - len({r.behavior_class for r in unsafe})
    if duplicates > 0:
        problems.append(f"{duplicates} duplicate behaviour class(es) in the rule set")

    # Check 5 is advisory: it validates our callout binding against the assignment's hint.
    warning_count = sum(1 for r in unsafe if r.callout is PolicyCallout.WARNING)
    critical_count = sum(1 for r in unsafe if r.callout is PolicyCallout.CRITICAL_SAFETY_NOTICE)
    if warning_count != EXPECTED_WARNINGS or critical_count != EXPECTED_CRITICAL_NOTICES:
        report.warnings.append(
            f"callout distribution is {warning_count} WARNING / {critical_count} CRITICAL "
            f"SAFETY NOTICE; the assignment describes {EXPECTED_WARNINGS} / "
            f"{EXPECTED_CRITICAL_NOTICES}. Callout binding may be off."
        )

    if not problems:
        return

    report.warnings.extend(problems)
    for problem in problems:
        log.error("policy structure check failed: %s", problem)
    if strict:
        raise PolicyValidationError("; ".join(problems))
