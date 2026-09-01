"""Module 4 - Automated Report Generation."""

from aegisflow.reports.pdf import build_compliance_pdf
from aegisflow.reports.writers import (
    CsvReportWriter,
    JsonFileReportWriter,
    JsonlReportWriter,
    MultiReportWriter,
    default_writers,
    read_jsonl,
)

__all__ = [
    "CsvReportWriter",
    "JsonFileReportWriter",
    "JsonlReportWriter",
    "MultiReportWriter",
    "build_compliance_pdf",
    "default_writers",
    "read_jsonl",
]
