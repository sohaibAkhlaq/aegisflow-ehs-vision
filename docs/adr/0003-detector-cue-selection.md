# ADR 0003 — Which detector cues we use, and which we measured and rejected

- **Status:** Accepted
- **Date:** 2026-09-01
- **Deciders:** Sohaib
- **Supersedes parts of:** [ADR 0001](0001-classical-cv-over-fine-tuned-detector.md)

## Context

ADR 0001 chose pretrained YOLOv8n for localisation plus classical CV for the policy's four
observable indicators — vest colour, block count, panel state, walkway position. That was the
right *shape* of solution and it is still the architecture. But when the individual cues were
measured on the test split, three of the four did not separate their class from the rest of
the dataset.

This ADR records what was measured, what was kept, and what was rejected. It exists because
the tempting alternative — quietly tuning thresholds until a number looks acceptable — would
have produced a system that scores well and means nothing.

## What was measured

`scripts/calibrate.py` dumps per-class percentiles for every cue; `scripts/sweep_thresholds.py`
caches YOLO observations once and then sweeps decision thresholds cheaply. Both are in the
repo, so every number below is reproducible.

### Cue 1 — vest colour (Section 4.2). **Works.**

| Class | Green ratio in torso ROI (p75) |
|---|---|
| `5_authorized_intervention` | **0.397** |
| `1_unauthorized_intervention` | 0.013 |
| all others | ~0.000 |

A clean, wide gap. The configured threshold of 0.12 sits in the middle of it. This cue
reliably answers *"is this person authorised?"*.

### Cue 2 — panel state (Section 5.2.2). **Works, once commissioned.**

Whole-frame statistics fail, and fail in an instructive direction — the *open* class scores
**lower** than the closed class on both cues, because the statistics describe the scene
rather than the panel:

| Cue | `2_opened_panel_cover` (p50) | `6_closed_panel_cover` (p50) |
|---|---|---|
| vertical edge strength | 0.152 | 0.166 |
| dark region ratio | 0.085 | 0.102 |

Looking at the panel instead works. `scripts/calibrate_panel.py` locates the region of
maximum open-vs-closed separation on the **train** split and records the intensity boundary:

- ROI separation score **3.28** pooled standard deviations
- open mean **74.1** grey levels vs closed **103.9**, threshold 89.0
- held-out test performance: **precision 1.00, recall 0.69**

### Cue 3 — walkway boundary (Section 3.2). **Weak; improved by commissioning.**

Live green-line segmentation does not recover the walkway. The largest green region in a
frame is one painted *line*, not the corridor between lines, so a containment test flags
nearly everyone: 89% of frames in *compliant* clips place at least one foot point outside it.

Sweeping the margin threshold from 0.02 to 0.26 of frame width, at persistence 3/5/8, the
best achievable was **F1 0.25** (P 0.14 / R 0.83). The curve is flat — there is no operating
point, because the cue carries almost no signal.

Commissioning the polygon per camera (`scripts/calibrate_regions.py`, convex hull of foot
points observed during compliant walking) lifts it to **F1 0.38** (P 0.36 / R 0.42). Per
camera matters: pooling both views yields a hull covering 67% of the frame, which excludes
nobody; CAM-01's own hull is 13.9%.

### Cue 4 — forklift block count (Section 6.2). **Rejected. Anti-correlated.**

| Cue | `3_carrying_overload` | `7_safe_carrying` |
|---|---|---|
| contour block count (p50 / p75) | 2 / 2 | 2 / **3** |
| load-region fill ratio (p50) | 0.545 | 0.523 |
| vehicle aspect ratio (p50) | 0.916 | 1.012 |
| vehicle area fraction (p50) | 0.093 | 0.113 |

The *compliant* class registers **more** detected blocks than the overloaded one. Best
achievable across thresholds 2–5 and persistence 2–6: **F1 0.31**, and only at a threshold
of 2, which by the policy's own definition is the compliant state. The cue is not merely
weak, it is pointing the wrong way.

### Also measured and rejected — a learned appearance window

Out of curiosity, the same discriminative-window method that works for panels was applied to
all four class pairs:

| Pair | Best window | Test F1 |
|---|---|---|
| walkway violation vs safe walkway | x=392 y=160 | 0.78 |
| unauthorized vs authorized | x=392 y=104 | 0.80 |
| open vs closed panel | x=64 y=208 | 0.82 |
| overload vs safe carrying | **x=560 y=0** | **1.00** |

Tempting, and **not used**. The forklift pair's best window is an empty corner of the frame:
the feature is separating scene and lighting differences between recording sessions, not
behaviour. More decisively, a learned intensity window is not traceable to a policy section,
and the assignment requires that *"the observable indicators used to classify a behavior as
safe or unsafe must be traceable to the relevant policy section."* A number obtained this way
would be a higher score and a worse system.

The panel ROI is a deliberate and narrower exception: the window locates *where the panel is*,
which is commissioning information like the walkway polygon, while the open/closed decision
remains the policy's own indicator.

## Decision

**1. Keep the policy-grounded classical detectors as the offline path**, at honest operating
points, with per-camera commissioned regions where those help:

| Detector | Cue | Commissioning |
|---|---|---|
| Unauthorized intervention | green vest ratio, positive red-black evidence required | none |
| Opened panel cover | intensity in the commissioned panel ROI | `calibrate_panel.py` |
| Safe walkway violation | foot point outside the commissioned polygon, past a margin | `calibrate_regions.py` |
| Forklift overload | contour block count, always flagged ambiguous | none |

**2. Detectors abstain rather than guess.** A detector without the commissioning data it
needs reports nothing and logs why. The walkway detector abstains on cameras with no usable
polygon; the panel detector abstains on any camera it was not calibrated against, verified by
scene fingerprint. Silence beats noise in a compliance record.

**3. Unauthorized intervention requires positive evidence.** Firing on "no green vest found"
— true of almost every person in almost every clip — measured at precision 0.11 and buried
the real events. The detector now requires a positively identified red-black vest, the
policy's own marker for personnel not cleared to intervene (Section 4.2). This costs recall
and is documented as such.

**4. Promote the VLM from tie-break to a first-class detection path.** The three indicators
classical CV cannot read are semantic questions a vision-language model answers well. The
assignment explicitly permits zero-shot VLMs, and both project documents specify a vision
fallback. `detection/vlm.py` builds its prompt *from the parsed policy* — each question
quotes the rule's own `observable_indicator` text and section reference — so this path is
**more** policy-traceable than the contour heuristics it replaces, not less.

Cost control, because the provider is a free tier: one call per frame covering all four
behaviours, at most 2 frames per clip, every response cached by frame hash.

**5. `offline` remains the default and must always work.** The classical path runs with no
network at all. The VLM is an enhancement whose absence degrades accuracy, never function.

## Consequences

**Good**

- Every threshold in `config/settings.yaml` traces to a measured curve, and the measurement
  scripts ship with the repo.
- The system's accuracy claims are defensible, because the failures are stated alongside them.
- Adding a Groq key materially improves detection without any code change.
- Abstention makes the failure mode "we did not look" rather than "we reported nonsense",
  which is the correct bias for an audit trail.

**Bad**

- Offline recall on forklift overload and unauthorized intervention is low. Documented in
  `docs/eval-baseline.md` and in the README's known-limitations section.
- The commissioning steps are extra operational work, and they need per-camera data. A camera
  with too few compliant examples gets an unusable polygon and the detector abstains — visible
  in the logs, and the honest outcome.
- Two detection paths to keep in agreement. Mitigated by the merge rule in `engine.py`: the
  VLM wins everywhere except the panel, where the commissioned baseline is more reliable.

## Revisit if

- A GPU and a few hundred annotated frames become available — ADR 0001's option A then
  dominates, and the module boundary (`clip -> list[DetectionRecord]`) lets it drop in.
- Footage arrives with the walkway markings clearly visible across the full corridor, in
  which case live green-line segmentation may become viable and should be re-measured before
  being trusted.
