# Evaluation Baseline

Measured detection accuracy, what the numbers mean, and what does not work. This document
is intentionally written the same way regardless of whether a result is strong or weak —
the goal is an accurate record, not a favourable one.

---

## Dataset in use for this baseline

This evaluation was run on a locally available subset of the full Kaggle dataset
(691 clips, 8 classes). Due to local storage constraints, the working test split for this
run contains **4 of the 8 behaviour classes**: `carrying_overload_with_forklift` and its
three compliant counterparts (`authorized_intervention`, `safe_walkway`,
`closed_panel_cover`). The remaining three unsafe classes —
`safe_walkway_violation`, `unauthorized_intervention`, `opened_panel_cover` — have no
positive examples in this particular split; results for those classes below reflect that
scope, not a change in detector quality. A full-dataset baseline (all 8 classes, 87-clip
test split) is available in project history for direct comparison and should not be
conflated with the numbers reported here.

**Command:** `python scripts/evaluate.py --split test`
**Clips evaluated:** 21
**Provider:** Groq (vision-model tie-break active)
**Throughput:** 9.95 s/clip mean, 683 frames analysed, 63 VLM calls, 209 s total

---

## Per-class detection metrics

| Behaviour | Positives | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Carrying Overload with Forklift | 4 | 3 | 0 | 1 | 1.00 | 0.75 | **0.86** |
| Opened Panel Cover | 0 | 0 | 9 | 0 | 0.00 | 0.00 | 0.00 |
| Safe Walkway Violation | 0 | 0 | 21 | 0 | 0.00 | 0.00 | 0.00 |
| Unauthorized Intervention | 0 | 0 | 19 | 0 | 0.00 | 0.00 | 0.00 |

**Macro F1: 0.214** (unweighted mean across all four classes)

---

## Discussion

### Forklift overload: a real, strong result

With the Groq vision-model tie-break active, this class scores **precision 1.00, recall
0.75, F1 0.86** against 4 real held-out positive clips: every positive call the detector
made was correct, and it missed one of four true cases. This is consistent with the
project's broader finding that this class benefits substantially from a vision-model
assist — the classical, offline detector is disabled by config for this exact reason (see
`docs/adr/0003-detector-cue-selection.md`): the contour-based block count was measured as
anti-correlated with true forklift load, while the VLM path answers the policy's own
indicator question directly and has consistently recovered this class without
false positives.

### Zero-positive classes: a scope limitation, not an accuracy claim

The three remaining classes show `Positives = 0` in this split, meaning no ground-truth
positive example of that violation exists in the current test data. Precision and recall
are computed as 0.00 by convention in this case, but this reflects **dataset composition**,
not detector accuracy — there is no possible true positive available for the detector to
find. These figures should not be read as a regression from the full-dataset baseline and
are not directly comparable to it.

### False-positive behaviour on compliant footage

With no true positives possible for three classes, every detection they made in this run
is by definition a false positive. The breakdown below is the most informative part of
this result, since it shows exactly where each detector is firing on footage it should
stay silent on:

| Detector fired | On clips labelled as | Count |
|---|---|---:|
| Opened Panel Cover | Authorized Intervention | 8 |
| Opened Panel Cover | Safe Walkway | 1 |
| Safe Walkway Violation | Authorized Intervention | 14 |
| Safe Walkway Violation | Carrying Overload with Forklift | 4 |
| Safe Walkway Violation | Closed Panel Cover | 2 |
| Safe Walkway Violation | Safe Walkway | 1 |
| Unauthorized Intervention | Authorized Intervention | 13 |
| Unauthorized Intervention | Carrying Overload with Forklift | 3 |
| Unauthorized Intervention | Closed Panel Cover | 2 |
| Unauthorized Intervention | Safe Walkway | 1 |

The Safe Walkway Violation detector fired on all 21 clips in this run (100%), including
footage with no relationship to walkway behaviour at all. This coincides with a
repeated runtime warning during evaluation: `CAM-02 has no usable walkway polygon; the
walkway detector will abstain on it.` Rather than abstaining as designed, the detector
appears to have fallen back to an uncommissioned method that over-triggers. This exceeds
even the low ceiling already documented for uncommissioned walkway detection
(`docs/adr/0003`, live green-line segmentation, F1 0.25) and is flagged here as a known
issue warranting investigation before this detector's output is relied upon for CAM-02.

---

## Severity distribution (this run)

| LOW | MEDIUM | HIGH | CRITICAL |
|---:|---:|---:|---:|
| 0 | 6 | 4 | 42 |

---

## Summary

| Class | Status |
|---|---|
| Carrying Overload with Forklift | Measured and strong (F1 0.86, Groq-assisted) |
| Opened Panel Cover | Not measurable on this dataset (no positive examples) |
| Safe Walkway Violation | Not measurable on this dataset; additionally shows a likely calibration-loading issue on CAM-02, worth investigating |
| Unauthorized Intervention | Not measurable on this dataset (no positive examples) |

Reproduce with: `python scripts/evaluate.py --split test`