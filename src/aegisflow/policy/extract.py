"""Layer 1 of policy parsing: get clean, structured text out of the PDF.

This layer makes no compliance decisions. It produces:

* per-page text with running headers/footers removed and mojibake repaired
* a section tree keyed by number (``"3.3.2"`` -> heading + body)
* callout boxes (WARNING / CRITICAL SAFETY NOTICE / NOTE / IMPORTANT) bound to the section
  they follow
* extracted tables, used for the Section 8 quick-reference grid and the Section 6.2
  load-threshold grid
* a SHA-256 of the source file, so a rule set can always be traced to the exact document

Two extractors run: PyMuPDF (primary, with layout and table support) and pypdf (independent
second opinion). ``validate.py`` cross-checks them - agreement is evidence that a quoted
sentence is really in the document and not an extraction artefact.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

from aegisflow.core.enums import PolicyCallout
from aegisflow.core.errors import PolicyParseError
from aegisflow.core.logging import get_logger

log = get_logger(__name__)

# Running headers/footers repeated on every page - noise for a section parser.
_CHROME_PATTERNS = (
    re.compile(r"^KMP-OHS-POL-001\s*\|", re.I),
    re.compile(r"^CONTROLLED DOCUMENT", re.I),
    re.compile(r"^Page\s*\d*$", re.I),
    re.compile(r"^Confidential", re.I),
)

# A numbered heading: "3.3.2  Non-Compliant Behavior - Safe Walkway Violation".
# Guarded so body lines like "2 blocks or fewer" and "3 or more blocks" never match:
# the title must begin with a capital letter, and headings are short.
_HEADING_RE = re.compile(r"^(?P<num>\d{1,2}(?:\.\d{1,2}){0,3})\s+(?P<title>[A-Z][^\n]{2,88})$")

# "SECTION 3 - PEDESTRIAN WALKWAY REGULATIONS"
_SECTION_RE = re.compile(r"^SECTION\s+(?P<num>\d{1,2})\s*[-—–:]\s*(?P<title>.+)$", re.I)

# Callout keywords can be split across lines by the PDF's table layout
# ("CRITICAL" / "SAFETY" / "NOTICE"), hence \s+ between words.
_CALLOUT_RES: tuple[tuple[PolicyCallout, re.Pattern[str]], ...] = (
    (PolicyCallout.CRITICAL_SAFETY_NOTICE, re.compile(r"^CRITICAL\s+SAFETY\s+NOTICE\b", re.M)),
    (PolicyCallout.WARNING, re.compile(r"^WARNING\b", re.M)),
    (PolicyCallout.IMPORTANT, re.compile(r"^IMPORTANT\b", re.M)),
    (PolicyCallout.NOTE, re.compile(r"^NOTE\b", re.M)),
)

_MAX_SECTION_NUMBER = 20


@dataclass
class Section:
    """One numbered section of the manual."""

    number: str
    title: str
    body: str = ""
    page: int = 0

    @property
    def ref(self) -> str:
        """Citation form used in compliance records: ``'Section 3.3.2'``."""
        return f"Section {self.number}"

    @property
    def text(self) -> str:
        return f"{self.title}\n{self.body}".strip()

    @property
    def top_level(self) -> str:
        return self.number.split(".")[0]


@dataclass
class Callout:
    """A WARNING / CRITICAL SAFETY NOTICE / NOTE / IMPORTANT box."""

    kind: PolicyCallout
    text: str
    section_number: str
    page: int = 0


@dataclass
class PolicyDocument:
    """Everything layer 1 recovered from the PDF."""

    source_path: Path
    sha256: str
    pages: list[str] = field(default_factory=list)
    sections: dict[str, Section] = field(default_factory=dict)
    callouts: list[Callout] = field(default_factory=list)
    tables: list[list[list[str]]] = field(default_factory=list)
    secondary_text: str = ""

    @property
    def full_text(self) -> str:
        return "\n".join(self.pages)

    def section(self, number: str) -> Section | None:
        return self.sections.get(number)

    def callouts_for(self, section_number: str) -> list[Callout]:
        return [c for c in self.callouts if c.section_number == section_number]

    def find_section_by_title(self, needle: str) -> Section | None:
        """Locate the section whose *heading* names ``needle`` (case-insensitive).

        Prefers the deepest match, so "Safe Walkway Violation" resolves to 3.3.2 rather
        than to the broader 3.3 "Behavioral Standards".
        """
        needle_l = needle.lower()
        matches = [s for s in self.sections.values() if needle_l in s.title.lower()]
        if not matches:
            return None
        return max(matches, key=lambda s: (len(s.number.split(".")), s.number))


# ---------------------------------------------------------------------------
# Text hygiene
# ---------------------------------------------------------------------------


def normalise(text: str) -> str:
    """Repair PDF text artefacts without changing wording.

    The manual's em-dashes and typographic quotes come through as U+FFFD from the embedded
    font encoding. Left alone they would break the literal-substring faithfulness check,
    because a quote extracted one way would not match the same quote extracted another way.
    """
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("�", "—")  # replacement char -> em dash
    text = text.replace("’", "'").replace("‘", "'")
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("–", "-").replace("—", "-")
    text = text.replace("\xa0", " ").replace("•", "-")
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text)


def strip_chrome(page_text: str) -> str:
    """Drop running headers and footers."""
    kept = [
        line
        for line in page_text.splitlines()
        if not any(pattern.match(line.strip()) for pattern in _CHROME_PATTERNS)
    ]
    return "\n".join(kept)


def squash(text: str) -> str:
    """Collapse all whitespace - the comparison form for substring checks."""
    return re.sub(r"\s+", " ", text).strip().lower()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


def extract_document(pdf_path: str | Path) -> PolicyDocument:
    """Run layer 1 over the policy PDF."""
    path = Path(pdf_path)
    if not path.exists():
        raise PolicyParseError(f"policy PDF not found: {path}")

    try:
        import fitz  # PyMuPDF
    except ImportError as exc:  # pragma: no cover - dependency is pinned
        raise PolicyParseError("PyMuPDF is required to parse the policy PDF") from exc

    doc = PolicyDocument(source_path=path, sha256=file_sha256(path))

    with fitz.open(path) as pdf:
        if pdf.page_count == 0:
            raise PolicyParseError(f"{path} contains no pages")
        for page in pdf:
            doc.pages.append(normalise(strip_chrome(page.get_text("text"))))
            doc.tables.extend(_extract_tables(page))

    doc.secondary_text = _extract_with_pypdf(path)
    doc.sections = _build_section_tree(doc.pages)
    doc.callouts = _extract_callouts(doc.pages, doc.sections)

    if not doc.sections:
        raise PolicyParseError(f"no numbered sections found in {path}")

    log.info(
        "policy extracted: %d pages, %d sections, %d callouts, %d tables",
        len(doc.pages),
        len(doc.sections),
        len(doc.callouts),
        len(doc.tables),
    )
    return doc


def _extract_tables(page: object) -> list[list[list[str]]]:
    """Pull tables from one page, tolerating PyMuPDF's optional table finder."""
    try:
        finder = page.find_tables()  # type: ignore[attr-defined]
    except Exception as exc:
        log.debug("table extraction unavailable on a page: %s", exc)
        return []
    out: list[list[list[str]]] = []
    for table in getattr(finder, "tables", []):
        try:
            rows = [[normalise(cell or "") for cell in row] for row in table.extract()]
        except Exception:
            continue
        if rows:
            out.append(rows)
    return out


def _extract_with_pypdf(path: Path) -> str:
    """Independent second extraction, used only for cross-validation."""
    try:
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        return normalise("\n".join(page.extract_text() or "" for page in reader.pages))
    except Exception as exc:
        log.debug("pypdf cross-check extraction failed: %s", exc)
        return ""


def _build_section_tree(pages: list[str]) -> dict[str, Section]:
    """Walk the document line by line, splitting on numbered headings."""
    sections: dict[str, Section] = {}
    current: Section | None = None
    buffer: list[str] = []

    def flush() -> None:
        if current is not None:
            current.body = "\n".join(buffer).strip()
            sections[current.number] = current

    for page_no, page_text in enumerate(pages, start=1):
        for raw_line in page_text.splitlines():
            line = raw_line.strip()
            if not line:
                if buffer:
                    buffer.append("")
                continue

            heading = _match_heading(line)
            if heading is None:
                buffer.append(line)
                continue

            number, title = heading
            flush()
            current = Section(number=number, title=title, page=page_no)
            buffer = []

    flush()
    return sections


def _match_heading(line: str) -> tuple[str, str] | None:
    """Return ``(number, title)`` when the line is a section heading."""
    if section_match := _SECTION_RE.match(line):
        number = section_match.group("num")
        if int(number) <= _MAX_SECTION_NUMBER:
            return number, section_match.group("title").strip()

    heading_match = _HEADING_RE.match(line)
    if heading_match is None:
        return None

    number = heading_match.group("num")
    parts = number.split(".")
    # Reject numeric prose ("3 or more blocks..."): a real heading is dotted, or a bare
    # top-level number within range whose title reads as a heading.
    if int(parts[0]) > _MAX_SECTION_NUMBER:
        return None
    if len(parts) == 1:
        return None  # bare "SECTION n" is handled by _SECTION_RE
    return number, heading_match.group("title").strip()


def _extract_callouts(pages: list[str], sections: dict[str, Section]) -> list[Callout]:
    """Find callout boxes and bind each to the section it appears under.

    Binding is positional: a callout belongs to the last numbered heading that precedes
    it in reading order. That is exactly how the manual lays them out - the WARNING under
    3.3.2 sits immediately after the 3.3.2 body.
    """
    callouts: list[Callout] = []

    for page_no, page_text in enumerate(pages, start=1):
        # Track which section each character offset falls under.
        anchors: list[tuple[int, str]] = []
        offset = 0
        for raw_line in page_text.splitlines():
            heading = _match_heading(raw_line.strip())
            if heading is not None and heading[0] in sections:
                anchors.append((offset, heading[0]))
            offset += len(raw_line) + 1

        for kind, pattern in _CALLOUT_RES:
            for match in pattern.finditer(page_text):
                body = _callout_body(page_text, match.end())
                if not body:
                    continue
                section_number = _section_at(anchors, match.start(), sections, page_no)
                callouts.append(
                    Callout(
                        kind=kind,
                        text=body,
                        section_number=section_number,
                        page=page_no,
                    )
                )

    # Deepest-first, so a CRITICAL SAFETY NOTICE outranks a NOTE on the same section.
    callouts.sort(key=lambda c: (c.page, -_CALLOUT_ORDER[c.kind]))
    return callouts


_CALLOUT_ORDER = {
    PolicyCallout.CRITICAL_SAFETY_NOTICE: 4,
    PolicyCallout.WARNING: 3,
    PolicyCallout.IMPORTANT: 2,
    PolicyCallout.NOTE: 1,
    PolicyCallout.NONE: 0,
}


def _callout_body(page_text: str, start: int) -> str:
    """Text of a callout: everything up to the next heading or a paragraph break."""
    tail = page_text[start:].lstrip("\n \t")
    lines: list[str] = []
    blanks = 0
    for line in tail.splitlines():
        stripped = line.strip()
        if not stripped:
            blanks += 1
            if blanks >= 2 and lines:
                break
            continue
        if _match_heading(stripped) is not None:
            break
        if any(p.match(stripped) for p, _ in ((r, k) for k, r in _CALLOUT_RES)):
            break
        blanks = 0
        lines.append(stripped)
        if len(lines) > 14:  # callouts in this manual are short paragraphs
            break
    return " ".join(lines).strip()


def _section_at(
    anchors: list[tuple[int, str]],
    position: int,
    sections: dict[str, Section],
    page_no: int,
) -> str:
    """Section number covering ``position``, falling back to the page's own section."""
    candidate = ""
    for offset, number in anchors:
        if offset <= position:
            candidate = number
        else:
            break
    if candidate:
        return candidate
    # Callout appeared before any heading on this page: attribute it to the deepest
    # section that started on this page or the nearest earlier one.
    earlier = [s for s in sections.values() if s.page <= page_no]
    return max(earlier, key=lambda s: (s.page, s.number)).number if earlier else ""
