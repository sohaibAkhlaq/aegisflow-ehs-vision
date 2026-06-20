"""
policy_parser.py — Module 1 helper: Parse the EHS compliance policy PDF and
extract structured compliance rules.

Two extraction modes:
  1. Gemini-based (preferred): Uses Gemini Pro to extract a structured JSON of
     rules from the raw PDF text — faithfully grounded in the source document.
  2. Deterministic fallback: Regex / keyword parser that works fully offline
     without any API key.

The output is a list of ComplianceRule objects that feed into the detector.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ComplianceRule:
    """A single compliance rule extracted from the policy document."""

    class_id: int
    behavior_name: str           # e.g. "Safe Walkway Violation"
    domain: str                  # e.g. "Pedestrian Movement"
    policy_ref: str              # e.g. "Section 3.3.2"
    unsafe_behavior: str         # human-readable description of the violation
    safe_behavior: str           # the compliant counterpart
    observable_indicator: str    # what the detector looks for
    severity: str                # LOW | MEDIUM | HIGH | CRITICAL
    hazard_context: str = ""     # extracted from Section X.1 of each domain
    keywords: list[str] = field(default_factory=list)  # detection keywords
    active: bool = True          # toggle active/inactive status
    confidence_threshold: float = 0.45  # minimum confidence required

    def to_dict(self) -> dict:
        return {
            "class_id": self.class_id,
            "behavior_name": self.behavior_name,
            "domain": self.domain,
            "policy_ref": self.policy_ref,
            "unsafe_behavior": self.unsafe_behavior,
            "safe_behavior": self.safe_behavior,
            "observable_indicator": self.observable_indicator,
            "severity": self.severity,
            "hazard_context": self.hazard_context,
            "keywords": self.keywords,
            "active": self.active,
            "confidence_threshold": self.confidence_threshold,
        }


# ─────────────────────────────────────────────────────────────────────────────
# PDF text extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_pdf_text(pdf_path: Path) -> str:
    """Extract full text from PDF using PyMuPDF (fitz)."""
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(str(pdf_path))
        pages = []
        for page in doc:
            pages.append(page.get_text("text"))
        doc.close()
        full_text = "\n".join(pages)
        logger.info(f"Extracted {len(full_text)} characters from {pdf_path.name}")
        return full_text
    except ImportError:
        logger.warning("PyMuPDF not installed — falling back to embedded rules")
        return ""
    except Exception as exc:
        logger.error(f"PDF extraction failed: {exc}")
        return ""


# ─────────────────────────────────────────────────────────────────────────────
# Gemini-based extraction
# ─────────────────────────────────────────────────────────────────────────────

GEMINI_EXTRACTION_PROMPT = """
You are a compliance-rules extraction engine. Below is the full text of an
Occupational Health & Safety Policy Manual for a manufacturing facility.

Extract exactly 4 compliance rules — one per behavioral domain — and return
them as a JSON array with this exact schema for each object:

{
  "class_id": <0|1|2|3>,
  "behavior_name": "<unsafe behavior name>",
  "domain": "<domain name>",
  "policy_ref": "<Section X.X.X>",
  "unsafe_behavior": "<one-sentence description of the unsafe behavior>",
  "safe_behavior": "<one-sentence description of the safe/compliant behavior>",
  "observable_indicator": "<what a camera system would observe to classify this as unsafe>",
  "severity": "<LOW|MEDIUM|HIGH|CRITICAL>",
  "hazard_context": "<one sentence summary of why this is dangerous>",
  "keywords": ["<3-6 keyword strings relevant to detection>"]
}

Severity assignment rules:
- The policy uses 'WARNING' callouts for moderate risks → HIGH
- The policy uses 'CRITICAL SAFETY NOTICE' callouts for the most severe → CRITICAL
- State-based findings with no immediate personnel proximity → LOW
- Behavioral deviation with personnel present but not in immediate danger → MEDIUM

Class ID assignment:
- 0: Pedestrian / Walkway domain
- 1: Equipment Intervention domain
- 2: Electrical Panel domain
- 3: Forklift Load domain

Return ONLY the JSON array, no markdown fences, no explanation.

POLICY TEXT:
{policy_text}
"""


def _extract_with_gemini(policy_text: str, api_key: str) -> list[ComplianceRule] | None:
    """Use Gemini Pro to extract structured rules from policy text."""
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")

        prompt = GEMINI_EXTRACTION_PROMPT.format(policy_text=policy_text[:12000])
        response = model.generate_content(prompt)
        raw = response.text.strip()

        # Strip any accidental markdown fences
        raw = re.sub(r"^```[a-z]*\n?", "", raw, flags=re.MULTILINE)
        raw = re.sub(r"```$", "", raw, flags=re.MULTILINE).strip()

        rules_data = json.loads(raw)
        rules = [ComplianceRule(**r) for r in rules_data]
        logger.info(f"Gemini extracted {len(rules)} compliance rules successfully")
        return rules
    except Exception as exc:
        logger.error(f"Gemini extraction failed: {exc}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Deterministic fallback — hardcoded from KMP-OHS-POL-001
# ─────────────────────────────────────────────────────────────────────────────

FALLBACK_RULES: list[dict] = [
    {
        "class_id": 0,
        "behavior_name": "Safe Walkway Violation",
        "domain": "Pedestrian Movement",
        "policy_ref": "Section 3.3.2",
        "unsafe_behavior": (
            "A person moving or present outside the boundaries of the green-marked "
            "Designated Safe Walkway on the production floor."
        ),
        "safe_behavior": (
            "All foot traffic remains within the green-marked Designated Safe Walkway."
        ),
        "observable_indicator": (
            "Person's position in the video frame is beyond the green walkway boundaries, "
            "into an area designated for machinery or vehicle operation."
        ),
        "severity": "CRITICAL",
        "hazard_context": (
            "The Safe Walkway Violation is the highest-frequency unsafe behavior at the facility; "
            "any deviation places personnel in proximity to forklift and machinery hazards."
        ),
        "keywords": [
            "walkway", "pedestrian", "green marking", "boundary", "outside zone", "foot traffic"
        ],
    },
    {
        "class_id": 1,
        "behavior_name": "Unauthorized Intervention",
        "domain": "Equipment Interaction",
        "policy_ref": "Section 4.3.2",
        "unsafe_behavior": (
            "A person interacting with or adjusting production equipment while not wearing "
            "the designated green safety vest or required safety equipment."
        ),
        "safe_behavior": (
            "Equipment interaction is performed only by personnel wearing the designated "
            "green authorization safety vest."
        ),
        "observable_indicator": (
            "Person is seen interacting with equipment while wearing a red-black vest "
            "or no green authorization vest."
        ),
        "severity": "HIGH",
        "hazard_context": (
            "Unauthorized equipment interaction creates risk of injury and production disruption; "
            "the green vest is the sole observable authorization indicator."
        ),
        "keywords": [
            "equipment", "intervention", "green vest", "red vest", "authorization", "machinery"
        ],
    },
    {
        "class_id": 2,
        "behavior_name": "Opened Panel Cover",
        "domain": "Electrical Safety",
        "policy_ref": "Section 5.2.2",
        "unsafe_behavior": (
            "An electrical panel connected to a production machine has been left in the "
            "open-cover state during production operations."
        ),
        "safe_behavior": (
            "All electrical panel covers are fully closed during production operations."
        ),
        "observable_indicator": (
            "Surveillance camera observes an electrical panel with its protective cover "
            "in the open position."
        ),
        "severity": "LOW",
        "hazard_context": (
            "Open panel covers expose electrical components to personnel contact; "
            "even temporary or inadvertent open covers are classified as unsafe."
        ),
        "keywords": [
            "panel", "electrical", "cover", "open", "cabinet", "enclosure"
        ],
    },
    {
        "class_id": 3,
        "behavior_name": "Carrying Overload with Forklift",
        "domain": "Forklift Load Management",
        "policy_ref": "Section 6.3.2",
        "unsafe_behavior": (
            "A forklift is operated while carrying three or more standardized blocks "
            "as a single load."
        ),
        "safe_behavior": (
            "Forklift carries two or fewer standardized blocks in a single load operation."
        ),
        "observable_indicator": (
            "Three or more standardized blocks are visible on the forklift forks "
            "during travel, maneuvering, or loading/unloading."
        ),
        "severity": "HIGH",
        "hazard_context": (
            "Carrying overloads creates vehicle instability risks; the block count "
            "threshold is unambiguous — 3 or more blocks always constitutes an overload."
        ),
        "keywords": [
            "forklift", "blocks", "overload", "forks", "carrying", "load"
        ],
    },
]


def _get_fallback_rules() -> list[ComplianceRule]:
    """Return the hardcoded rule set derived from KMP-OHS-POL-001."""
    return [ComplianceRule(**r) for r in FALLBACK_RULES]


# ─────────────────────────────────────────────────────────────────────────────
# Public interface
# ─────────────────────────────────────────────────────────────────────────────

def load_compliance_rules(
    pdf_path: Optional[Path] = None,
    api_key: str = "",
) -> list[ComplianceRule]:
    """
    Load compliance rules from the policy PDF.

    Tries in order:
      1. Gemini-based extraction (if api_key provided and PDF found)
      2. Deterministic fallback (always works)

    Returns a list of exactly 4 ComplianceRule objects.
    """
    rules: list[ComplianceRule] | None = None

    if pdf_path and pdf_path.exists() and api_key:
        policy_text = extract_pdf_text(pdf_path)
        if policy_text:
            rules = _extract_with_gemini(policy_text, api_key)
            if rules and len(rules) == 4:
                logger.info("Using Gemini-extracted compliance rules")
                rules = _apply_dynamic_overrides(rules)
                return rules
            else:
                logger.warning("Gemini extraction incomplete — using fallback rules")

    elif pdf_path and pdf_path.exists() and not api_key:
        logger.info("No API key provided — using deterministic rule fallback")

    rules = _get_fallback_rules()
    rules = _apply_dynamic_overrides(rules)
    logger.info(f"Loaded {len(rules)} compliance rules (overrides applied)")
    return rules


def _apply_dynamic_overrides(rules: list[ComplianceRule]) -> list[ComplianceRule]:
    """Load outputs/rules_config.json and apply severity, active status, and threshold overrides."""
    config_path = Path(__file__).resolve().parent.parent.parent / "outputs" / "rules_config.json"
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            for rule in rules:
                rule_override = config.get(str(rule.class_id))
                if rule_override:
                    rule.severity = rule_override.get("severity", rule.severity)
                    rule.active = rule_override.get("active", rule.active)
                    rule.confidence_threshold = rule_override.get("confidence_threshold", rule.confidence_threshold)
            logger.info("Applied dynamic overrides from rules_config.json")
        except Exception as exc:
            logger.error(f"Failed to apply rule config overrides: {exc}")
    return rules


def rules_to_lookup(rules: list[ComplianceRule]) -> dict[int, ComplianceRule]:
    """Convert rule list to a dict keyed by class_id for O(1) lookup."""
    return {r.class_id: r for r in rules}
