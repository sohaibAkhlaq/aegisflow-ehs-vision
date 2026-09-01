"""Layer 2 of policy parsing: structured text -> :class:`PolicyRule` objects.

Deterministic and offline. For KMP-OHS-POL-001 this layer alone recovers the complete rule
set; the LLM pass in ``llm_extract.py`` only adds robustness for prose the patterns miss.

Where each field comes from - all of it read out of the PDF, none of it authored here:

============================  =================================================
Field                         Source in the document
============================  =================================================
``behavior_class``            Section 8 quick-reference table, "Unsafe Behavior"
``domain``                    Section 8 table, "Behavior Domain"
``observable_indicator``      Section 8 table, "Observable Indicator of Unsafe"
``section_ref``               heading that defines the behaviour (e.g. 3.3.2)
``callout``                   callout box bound to that section
``unsafe_description``        first sentence of that section's body
``numeric_threshold``         Section 6.2 load-capacity table
``high_frequency`` etc.       severity-signal phrases mined from section prose
============================  =================================================
"""

from __future__ import annotations

import re

from aegisflow.core.enums import BehaviorClass, PolicyCallout, slugify
from aegisflow.core.errors import PolicyParseError
from aegisflow.core.logging import get_logger
from aegisflow.core.schemas import PolicyRule
from aegisflow.policy.extract import Callout, PolicyDocument, Section, squash

log = get_logger(__name__)

# --- severity-signal phrases -------------------------------------------------
# Each is a phrase the manual actually uses. Matching is on the section body plus its
# callout text, so a signal is always backed by a quotable sentence.

_HIGH_FREQUENCY_RE = re.compile(
    r"most frequently occurring|highest[- ]frequency|frequency of this behavior", re.I
)
_UNAMBIGUOUS_RE = re.compile(
    r"\bunambiguous\b|must be assumed|threshold is unambiguous|regardless of the person's", re.I
)
_STANDALONE_RE = re.compile(
    r"regardless of how long|whether personnel are in the immediate vicinity"
    r"|no concurrent personnel|regardless of whether",
    re.I,
)

# "two (2) or fewer", "2 blocks or fewer"
_THRESHOLD_RE = re.compile(r"(?:\((\d+)\)|\b(\d+)\b)\s*(?:blocks?\s*)?or fewer", re.I)

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def build_rules(doc: PolicyDocument) -> tuple[list[PolicyRule], list[str]]:
    """Derive the behavioural rule set. Returns ``(rules, warnings)``."""
    entries = _parse_summary_table(doc)
    if not entries:
        entries = _parse_behaviours_from_headings(doc)
    if not entries:
        raise PolicyParseError(
            "could not derive any behaviour classes from the policy document "
            "(neither the Section 8 summary table nor the behavioural headings parsed)"
        )

    threshold = _parse_load_threshold(doc)
    rules: list[PolicyRule] = []
    warnings: list[str] = []

    for entry in entries:
        try:
            behavior = BehaviorClass.from_policy_name(entry.unsafe_name)
        except KeyError as exc:
            # Loud, not silent: the registry and the document have diverged.
            warnings.append(str(exc))
            log.warning("skipping unrecognised behaviour from policy: %s", exc)
            continue

        section = _governing_section(doc, entry.unsafe_name)
        if section is None:
            warnings.append(f"no governing section found for {entry.unsafe_name!r}")
            log.warning("no governing section for %s", entry.unsafe_name)
            continue

        callout = _dominant_callout(doc, section)
        signal_text = f"{section.body}\n{callout.text if callout else ''}"

        rules.append(
            PolicyRule(
                behavior_class=behavior,
                domain=entry.domain,
                section_ref=section.ref,
                observable_indicator=entry.indicator or _fallback_indicator(section),
                unsafe_description=_first_sentence(section.body) or entry.unsafe_name,
                safe_description=_safe_description(doc, entry.safe_name),
                callout=callout.kind if callout else PolicyCallout.NONE,
                callout_text=callout.text if callout else "",
                source_quote=_source_quote(section, entry.unsafe_name),
                high_frequency=bool(_HIGH_FREQUENCY_RE.search(signal_text)),
                unambiguous=bool(_UNAMBIGUOUS_RE.search(signal_text)),
                standalone_condition=bool(_STANDALONE_RE.search(signal_text)),
                numeric_threshold=threshold if behavior is _FORKLIFT else None,
                extraction_method="deterministic",
            )
        )

    log.info("deterministic parser derived %d rules", len(rules))
    return rules, warnings


_FORKLIFT = BehaviorClass.CARRYING_OVERLOAD_WITH_FORKLIFT


# ---------------------------------------------------------------------------
# Section 8 quick-reference table
# ---------------------------------------------------------------------------


class _Entry:
    """One row of the Section 8 behaviour summary."""

    __slots__ = ("class_id", "domain", "indicator", "safe_name", "unsafe_name")

    def __init__(self, class_id: str, domain: str, unsafe_name: str, safe_name: str) -> None:
        # Table cells wrap across lines in the source PDF (the "Behavior Domain" cell
        # arrives as "Pedestrian" then "Movement"), so each cell is collapsed on the
        # way in to a single spaced line.
        self.class_id = class_id
        self.domain = _collapse(domain)
        self.unsafe_name = _collapse(unsafe_name)
        self.safe_name = _collapse(safe_name)
        self.indicator = ""


def _parse_summary_table(doc: PolicyDocument) -> list[_Entry]:
    """Read the Section 8 grid.

    The PDF renders the indicator column as one word per row, so a row beginning with a
    class id opens an entry and the rows that follow contribute their trailing cells to
    that entry's indicator until the next class id appears.
    """
    for table in doc.tables:
        entries = _entries_from_table(table)
        if len(entries) >= 4:
            log.debug("Section 8 summary table parsed: %d entries", len(entries))
            return entries
    return []


def _entries_from_table(table: list[list[str]]) -> list[_Entry]:
    entries: list[_Entry] = []
    current: _Entry | None = None

    for row in table:
        cells = [c.strip() for c in row]
        non_empty = [c for c in cells if c]
        if not non_empty:
            continue

        first = non_empty[0]
        if re.fullmatch(r"\d", first) and len(non_empty) >= 4:
            current = _Entry(
                class_id=first,
                domain=non_empty[1],
                unsafe_name=non_empty[2],
                safe_name=non_empty[3],
            )
            entries.append(current)
            if len(non_empty) >= 5:
                current.indicator = non_empty[4]
        elif current is not None:
            # Continuation row: the indicator column wrapped.
            current.indicator = f"{current.indicator} {' '.join(non_empty)}".strip()

    for entry in entries:
        entry.indicator = re.sub(r"\s+", " ", entry.indicator).strip()
    return [e for e in entries if e.unsafe_name and e.safe_name]


def _parse_behaviours_from_headings(doc: PolicyDocument) -> list[_Entry]:
    """Fallback when the summary table cannot be read.

    Behavioural headings follow a fixed shape: "Non-Compliant Behavior - <name>" and
    "Required Behavior - <name> (Compliant)". Pairing them by parent section recovers the
    same information as the table.
    """
    unsafe: dict[str, tuple[str, Section]] = {}
    safe: dict[str, str] = {}

    for section in doc.sections.values():
        title = section.title
        parent = section.number.rsplit(".", 1)[0]
        if match := re.search(r"Non-Compliant (?:Behavior|Condition)\s*-\s*(.+)", title, re.I):
            name = re.sub(r"\((?:unsafe|compliant)\)", "", match.group(1), flags=re.I).strip()
            unsafe[parent] = (name, section)
        elif match := re.search(r"Required Behavior\s*-\s*(.+)", title, re.I):
            name = re.sub(r"\(compliant\)", "", match.group(1), flags=re.I).strip()
            safe[parent] = name

    entries: list[_Entry] = []
    for parent, (name, section) in sorted(unsafe.items()):
        domain_section = doc.section(parent.split(".")[0])
        entries.append(
            _Entry(
                class_id=parent.split(".")[0],
                domain=(domain_section.title.title() if domain_section else parent),
                unsafe_name=name,
                safe_name=safe.get(parent, ""),
            )
        )
        entries[-1].indicator = _fallback_indicator(section)
    log.debug("summary table unavailable; recovered %d entries from headings", len(entries))
    return entries


# ---------------------------------------------------------------------------
# Field helpers
# ---------------------------------------------------------------------------


def _governing_section(doc: PolicyDocument, unsafe_name: str) -> Section | None:
    """The section whose heading defines this behaviour, e.g. 3.3.2."""
    target = squash(unsafe_name)
    # Prefer the explicit "Non-Compliant Behavior/Condition - <name>" heading.
    for section in doc.sections.values():
        match = re.search(r"Non-Compliant (?:Behavior|Condition)\s*-\s*(.+)", section.title, re.I)
        if match is None:
            continue
        heading_name = squash(re.sub(r"\((?:unsafe|compliant)\)", "", match.group(1), flags=re.I))
        if heading_name == target:
            return section

    if section := doc.find_section_by_title(unsafe_name):
        return section

    # Heading wording differs from the table wording: fall back to the deepest section
    # whose body opens by defining the behaviour.
    needle = squash(unsafe_name)
    candidates = [
        s
        for s in doc.sections.values()
        if needle in squash(s.body[:400]) and len(s.number.split(".")) >= 2
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda s: (len(s.number.split(".")), s.number))


def _dominant_callout(doc: PolicyDocument, section: Section) -> Callout | None:
    """Highest-severity callout attached to this section.

    Falls back to a callout on the parent section, since the manual sometimes places a
    box under the parent "Behavioral Standards" heading rather than the leaf.
    """
    candidates = doc.callouts_for(section.number)
    if not candidates:
        parent = section.number.rsplit(".", 1)[0]
        candidates = [c for c in doc.callouts_for(parent) if c.kind is not PolicyCallout.NOTE]
    if not candidates:
        return None
    return max(candidates, key=lambda c: _CALLOUT_RANK[c.kind])


_CALLOUT_RANK = {
    PolicyCallout.CRITICAL_SAFETY_NOTICE: 4,
    PolicyCallout.WARNING: 3,
    PolicyCallout.IMPORTANT: 2,
    PolicyCallout.NOTE: 1,
    PolicyCallout.NONE: 0,
}


def _parse_load_threshold(doc: PolicyDocument) -> int | None:
    """Safe block count from Section 6.2 / 6.3.1 ("two (2) or fewer")."""
    for number in ("6.2", "6.3.1", "6.3.2", "6"):
        section = doc.section(number)
        if section is None:
            continue
        if match := _THRESHOLD_RE.search(section.text):
            value = match.group(1) or match.group(2)
            log.debug("load threshold %s parsed from Section %s", value, number)
            return int(value)
    if match := _THRESHOLD_RE.search(doc.full_text):
        return int(match.group(1) or match.group(2))
    return None


def _first_sentence(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return ""
    return _SENTENCE_SPLIT_RE.split(cleaned)[0].strip()


def _source_quote(section: Section, unsafe_name: str) -> str:
    """The literal sentence that defines the behaviour - the audit anchor.

    Must survive the faithfulness gate, so it is returned exactly as extracted.
    """
    cleaned = re.sub(r"\s+", " ", section.body).strip()
    needle = unsafe_name.lower()
    for sentence in _SENTENCE_SPLIT_RE.split(cleaned):
        if needle in sentence.lower() and "defined as" in sentence.lower():
            return sentence.strip()
    for sentence in _SENTENCE_SPLIT_RE.split(cleaned):
        if needle in sentence.lower():
            return sentence.strip()
    return _first_sentence(cleaned)


def _fallback_indicator(section: Section) -> str:
    """Derive an indicator from the section body when the table is unavailable."""
    cleaned = re.sub(r"\s+", " ", section.body).strip()
    for sentence in _SENTENCE_SPLIT_RE.split(cleaned):
        if re.search(r"\bis detected when\b|\bobservable\b|\bindicator\b", sentence, re.I):
            return sentence.strip()
    return _first_sentence(cleaned)


def _safe_section(doc: PolicyDocument, safe_name: str) -> Section | None:
    """Locate the *compliant* section for a safe behaviour.

    A plain substring search is wrong here: "Safe Walkway" is contained in "Safe Walkway
    Violation", and "Authorized Intervention" in "Unauthorized Intervention", so a naive
    lookup returns the non-compliant section every time. Match the manual's actual heading
    shape instead - "Required Behavior - <name> (Compliant)" - and require the name to
    match the whole heading remainder, not just appear inside it.
    """
    if not safe_name:
        return None
    target = squash(safe_name)
    for section in doc.sections.values():
        match = re.search(r"Required Behavior\s*-\s*(.+)", section.title, re.I)
        if match is None:
            continue
        heading_name = squash(re.sub(r"\(compliant\)", "", match.group(1), flags=re.I))
        if heading_name == target:
            return section
    # No "Required Behavior" heading: fall back to a whole-title match that is explicitly
    # not the non-compliant section.
    for section in doc.sections.values():
        if "non-compliant" in section.title.lower():
            continue
        if target in squash(section.title):
            return section
    return None


def _safe_description(doc: PolicyDocument, safe_name: str) -> str:
    section = _safe_section(doc, safe_name)
    return _first_sentence(section.body) if section else ""


def _collapse(text: str) -> str:
    """Single-line form of a wrapped table cell."""
    return re.sub(r"\s+", " ", text).strip()


def behaviour_keys(rules: list[PolicyRule]) -> list[str]:
    """Slugs derived from the document - used by validation to prove derivation."""
    return [slugify(rule.behavior_class.value) for rule in rules]
