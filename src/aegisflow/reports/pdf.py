"""Executive PDF compliance report (ReportLab).

The policy says event reports "constitute the primary compliance record for the facility"
and are used by the occupational safety expert to assess behavioural trends (Section 7.3).
This produces that document: a summary by severity, behaviour and zone, then the event
register with the policy reference and severity rationale for each entry.

The rationale column is the point. It is what lets an auditor confirm a severity tier
against the manual without asking anyone.
"""

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from aegisflow.core.enums import Severity
from aegisflow.core.logging import get_logger
from aegisflow.core.schemas import PolicyRuleSet, ViolationEvent

log = get_logger(__name__)

SEVERITY_COLOURS: dict[str, colors.Color] = {
    "LOW": colors.HexColor("#2563EB"),
    "MEDIUM": colors.HexColor("#059669"),
    "HIGH": colors.HexColor("#D97706"),
    "CRITICAL": colors.HexColor("#DC2626"),
}

_INK = colors.HexColor("#0F172A")
_MUTED = colors.HexColor("#64748B")
_RULE = colors.HexColor("#CBD5E1")
_BAND = colors.HexColor("#F1F5F9")


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "AfTitle", parent=base["Title"], fontSize=20, textColor=_INK, spaceAfter=4
        ),
        "subtitle": ParagraphStyle(
            "AfSubtitle", parent=base["Normal"], fontSize=10, textColor=_MUTED, spaceAfter=14
        ),
        "h2": ParagraphStyle(
            "AfH2",
            parent=base["Heading2"],
            fontSize=12,
            textColor=_INK,
            spaceBefore=14,
            spaceAfter=6,
        ),
        "body": ParagraphStyle(
            "AfBody", parent=base["Normal"], fontSize=8.5, textColor=_INK, leading=11
        ),
        "small": ParagraphStyle(
            "AfSmall",
            parent=base["Normal"],
            fontSize=7,
            textColor=_INK,
            leading=8.6,
            alignment=TA_LEFT,
        ),
    }


def build_compliance_pdf(
    events: list[ViolationEvent],
    output_path: Path,
    rule_set: PolicyRuleSet | None = None,
    *,
    facility: str = "Kafaoglu Metal Plastik Makine San. ve Tic. A.S.",
    document_id: str = "KMP-OHS-POL-001",
    period: str | None = None,
) -> Path:
    """Render a compliance report. Returns the written path."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    styles = _styles()
    story: list[object] = []

    generated = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    story.append(Paragraph("Automated EHS Compliance Report", styles["title"]))
    story.append(
        Paragraph(
            f"{facility} &nbsp;|&nbsp; Policy {document_id} &nbsp;|&nbsp; "
            f"Generated {generated}" + (f" &nbsp;|&nbsp; Period {period}" if period else ""),
            styles["subtitle"],
        )
    )

    story.append(Paragraph("1. Summary", styles["h2"]))
    story.append(_summary_table(events))

    story.append(Paragraph("2. Violations by behaviour class", styles["h2"]))
    story.append(_breakdown_table(events, lambda e: e.behavior_class.display_name, "Behaviour"))

    story.append(Paragraph("3. Violations by zone", styles["h2"]))
    story.append(_breakdown_table(events, lambda e: e.zone, "Zone"))

    if rule_set is not None:
        story.append(Paragraph("4. Policy rules in force", styles["h2"]))
        story.append(_policy_table(rule_set, styles))

    story.append(PageBreak())
    story.append(Paragraph("Event register", styles["h2"]))
    story.append(
        Paragraph(
            "Every record below was generated automatically. The rationale column quotes "
            "the policy language that determined the severity tier.",
            styles["body"],
        )
    )
    story.append(Spacer(1, 5 * mm))
    story.append(_event_table(events, styles))

    document = SimpleDocTemplate(
        str(output_path),
        pagesize=landscape(A4),
        leftMargin=12 * mm,
        rightMargin=12 * mm,
        topMargin=12 * mm,
        bottomMargin=12 * mm,
        title="AegisFlow EHS Compliance Report",
        author="AegisFlow EHS",
    )
    document.build(story, onLaterPages=_footer, onFirstPage=_footer)
    log.info("compliance PDF written: %s (%d events)", output_path, len(events))
    return output_path


def _footer(canvas: object, document: object) -> None:
    canvas.saveState()  # type: ignore[attr-defined]
    canvas.setFont("Helvetica", 7)  # type: ignore[attr-defined]
    canvas.setFillColor(_MUTED)  # type: ignore[attr-defined]
    canvas.drawString(  # type: ignore[attr-defined]
        12 * mm, 7 * mm, "AegisFlow EHS - automated compliance record. Append-only audit trail."
    )
    canvas.drawRightString(  # type: ignore[attr-defined]
        landscape(A4)[0] - 12 * mm, 7 * mm, f"Page {document.page}"  # type: ignore[attr-defined]
    )
    canvas.restoreState()  # type: ignore[attr-defined]


def _summary_table(events: list[ViolationEvent]) -> Table:
    counts = Counter(e.severity.value for e in events)
    alerted = sum(1 for e in events if e.severity.requires_realtime_alert)
    clips = len({e.clip_id for e in events})

    header = ["Total events", "Clips affected", "Real-time alerts", *[s.value for s in Severity]]
    row = [
        str(len(events)),
        str(clips),
        str(alerted),
        *[str(counts.get(s.value, 0)) for s in Severity],
    ]

    table = Table([header, row], colWidths=[30 * mm] * 3 + [26 * mm] * 4)
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), _INK),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.4, _RULE),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]
    for index, severity in enumerate(Severity):
        column = 3 + index
        style.append(("TEXTCOLOR", (column, 1), (column, 1), SEVERITY_COLOURS[severity.value]))
        style.append(("FONTNAME", (column, 1), (column, 1), "Helvetica-Bold"))
    table.setStyle(TableStyle(style))
    return table


def _breakdown_table(events: list[ViolationEvent], key, label: str) -> Table:
    counts = Counter(key(e) for e in events)
    severities = list(Severity)
    header = [label, *[s.value for s in severities], "Total"]
    rows = [header]

    for name in sorted(counts):
        subset = [e for e in events if key(e) == name]
        per_severity = Counter(e.severity.value for e in subset)
        rows.append(
            [
                name,
                *[str(per_severity.get(s.value, 0)) for s in severities],
                str(len(subset)),
            ]
        )
    if len(rows) == 1:
        rows.append(["(none recorded)", *["0"] * len(severities), "0"])

    table = Table(rows, colWidths=[72 * mm] + [24 * mm] * len(severities) + [22 * mm])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), _INK),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("GRID", (0, 0), (-1, -1), 0.4, _RULE),
                ("ALIGN", (1, 0), (-1, -1), "CENTER"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, _BAND]),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return table


def _policy_table(rule_set: PolicyRuleSet, styles: dict[str, ParagraphStyle]) -> Table:
    rows: list[list[object]] = [
        ["Behaviour", "Section", "Callout", "Observable indicator (from policy)"]
    ]
    for rule in rule_set.unsafe_rules:
        rows.append(
            [
                Paragraph(rule.behavior_class.display_name, styles["small"]),
                Paragraph(rule.section_ref, styles["small"]),
                Paragraph(rule.callout.value, styles["small"]),
                Paragraph(rule.observable_indicator, styles["small"]),
            ]
        )
    table = Table(rows, colWidths=[52 * mm, 24 * mm, 42 * mm, 128 * mm], repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), _INK),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 8),
                ("GRID", (0, 0), (-1, -1), 0.4, _RULE),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, _BAND]),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return table


def _event_table(events: list[ViolationEvent], styles: dict[str, ParagraphStyle]) -> Table:
    header = [
        "Timestamp (UTC)",
        "Clip",
        "Zone",
        "Behaviour",
        "Policy ref",
        "Sev",
        "Escalation",
        "Severity rationale (policy-grounded)",
    ]
    rows: list[list[object]] = [header]
    ordered = sorted(events, key=lambda e: e.timestamp)

    for event in ordered:
        rows.append(
            [
                Paragraph(event.timestamp.strftime("%Y-%m-%d %H:%M:%S"), styles["small"]),
                Paragraph(event.clip_id, styles["small"]),
                Paragraph(event.zone, styles["small"]),
                Paragraph(event.behavior_class.display_name, styles["small"]),
                Paragraph(event.policy_rule_ref, styles["small"]),
                Paragraph(event.severity.value, styles["small"]),
                Paragraph(event.escalation_action.value, styles["small"]),
                Paragraph(event.severity_rationale or "-", styles["small"]),
            ]
        )
    if len(rows) == 1:
        rows.append([Paragraph("No violations recorded.", styles["small"])] + [""] * 7)

    table = Table(
        rows,
        colWidths=[26 * mm, 24 * mm, 17 * mm, 34 * mm, 19 * mm, 14 * mm, 30 * mm, 108 * mm],
        repeatRows=1,
    )
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), _INK),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 7.5),
        ("GRID", (0, 0), (-1, -1), 0.3, _RULE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, _BAND]),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]
    for index, event in enumerate(ordered, start=1):
        colour = SEVERITY_COLOURS.get(event.severity.value, _INK)
        style.append(("TEXTCOLOR", (5, index), (5, index), colour))
        style.append(("FONTNAME", (5, index), (5, index), "Helvetica-Bold"))
    table.setStyle(TableStyle(style))
    return table
