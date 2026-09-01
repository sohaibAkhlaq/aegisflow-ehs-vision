# ADR 0001 — Classical CV on top of YOLOv8n, not a fine-tuned detector

- **Status:** Accepted
- **Date:** 2026-09-01
- **Deciders:** Sohaib

## Context

Module 1 must detect four behaviours whose observable indicators are, per the policy manual:
vest **colour** (§4.2), block **count** (§6.2), panel **state** (§5.2), and walkway
**position** relative to green floor markings (§3.2).

Three things constrain the choice:

1. **The dataset has no bounding-box annotations.** Labels are folder-level only — 691 clips
   across 8 class directories. Fine-tuning an object detector would mean annotating a training
   set by hand first.
2. **No GPU.** `torch 2.1.2+cpu`, 12 cores, 8 GB RAM. Training is not practical; even
   inference must be budgeted carefully.
3. **None of the indicators is a COCO class.** "Green vest", "standardized block", "open panel
   cover" and "walkway boundary" are not things a pretrained detector knows.

## Options considered

**A. Fine-tune YOLOv8 on custom classes.** Best ceiling on accuracy. Requires annotating
thousands of frames and a GPU we do not have. Rejected on cost, not on merit.

**B. Video classification model (e.g. a 3D CNN / video transformer) on the 8 folder labels.**
Matches the label granularity we actually have, and would likely score well on the test split.
Rejected because it produces a *clip label*, not a localised violation with an observable
indicator. The assignment requires a detection record with a zone and a policy-traceable
indicator, and requires the indicators to be traceable to policy sections. A black-box clip
classifier cannot supply that, and it would make the severity matrix's frame-context signals
(is a forklift also present? is a person near the panel?) impossible to compute.

**C. Zero-shot VLM on every frame.** Strong accuracy, no training. Rejected as the primary
path: it needs network access for all 691 clips, is rate-limited on a free tier, and makes the
system unusable offline — which is a stated product requirement. Retained as a *tie-break* on
ambiguous frames only.

**D. Pretrained YOLOv8n for localisation + classical CV for the indicators.** Chosen.

## Decision

Use YOLOv8-nano for what it already does well — locating `person` and vehicle boxes — and read
the policy's indicators off those boxes with OpenCV: HSV masks for vest and floor-line colour,
contour geometry for panel state and block counting, point-in-polygon for walkway containment.
Add temporal persistence across sampled frames to suppress single-frame noise. Escalate
genuinely ambiguous cases to a VLM (option C, scoped down).

## Consequences

**Good**

- No annotation effort and no training; works today on this machine.
- Each decision is explainable in the policy's own vocabulary — "person's foot point fell
  outside the green polygon" maps directly onto §3.3.2. That is exactly the traceability the
  assignment grades.
- The intermediate signals (person count, forklift present, proximity) are available to the
  severity matrix for context escalation.
- Retargeting to a different facility means re-tuning HSV bands, not retraining a model.

**Bad**

- Sensitive to lighting and camera angle; HSV bands need calibration per installation
  (`scripts/calibrate.py` measures the per-class separation for this).
- Opened Panel Cover is the weakest detector — it is a subtle state change with no object
  class to anchor on. Expected to be the main source of error, and documented as such in
  `docs/eval-baseline.md`.
- "Interacting with equipment" is approximated by proximity plus low centroid motion, not by
  true action recognition. A person standing near a machine without touching it may be flagged.

**Mitigations**

- Per-class temporal persistence windows in `config/settings.yaml`.
- The safe-behaviour classes (`4_safe_walkway`, `5_authorized_intervention`,
  `6_closed_panel_cover`, `7_safe_carrying`) are used as the false-positive test set, so
  precision is measured, not assumed.
- Honest per-class metrics in `docs/eval-baseline.md`. The assignment explicitly does not
  require perfect accuracy but does require documented limitations.

## Revisit if

A GPU becomes available and someone annotates a few hundred frames — option A then dominates,
and the module boundary (`clip -> list[DetectionRecord]`) means it can be swapped in without
touching Modules 2-5.
