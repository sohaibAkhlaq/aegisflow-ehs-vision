# Architecture

How the five modules fit together, what crosses each boundary, and why.

---

## 1. Data flow

```
compliance_policy.pdf ──▶ [2a] PolicyParser ──▶ PolicyRuleSet ──┐
                                │                               │
                          validation:                           │
                          literal-substring                     │
                          check vs. PDF text                    │
                                                                │
data/raw/*.mp4 ──▶ [1] DetectionEngine ──▶ DetectionRecord[] ───┤
                          │                                     │
                    FrameStream (4 fps, 640 px)                 ▼
                    YOLOv8n ──▶ FrameObservation[]     [2b] SeverityMatrix
                    HSV / contour analysis                      │
                    temporal persistence                        │
                    VLM indicator reading                       ▼
                                                        SeverityAssessment
                                                                │
                                                                ▼
                                                        ViolationEvent  ◀── FROZEN SEAM
                                                                │
                                          ┌─────────────────────┤
                                          ▼                     ▼
                                  [3] EscalationSink     [4] ReportWriter
                                   LOW/MED → DB           JSON · CSV · PDF
                                   HIGH/CRIT → DB + WS    append-only
                                          │                     │
                                          └──────────┬──────────┘
                                                     ▼
                                          [5] Operations Dashboard
                                       Live Feed · Timeline · History
```

The **frozen seam** is `ViolationEvent`: Modules 1-2 produce it, Modules 3-5 consume it. It is
immutable by type, because compliance records are evidence.

---

## 2. Module contracts

| Module | Input | Output | Package |
|---|---|---|---|
| 2a Policy Parser | `compliance_policy.pdf` | `PolicyRuleSet` (persisted to `artifacts/policy/rules.json`) | `policy/` |
| 1 Detection Engine | clip path + `PolicyRuleSet` | `list[DetectionRecord]` | `detection/` |
| 2b Severity Matrix | `DetectionRecord` + `PolicyRule` + frame context | `SeverityAssessment` | `severity/` |
| 3 Escalation | `ViolationEvent` | same event with `escalation_action` set; side effects: DB write, WS publish | `escalation/` |
| 4 Reports | `ViolationEvent` | append-only JSON / CSV / PDF records | `reports/` |
| 5 Dashboard | REST + WebSocket | three browser views | `frontend/` |

Module 1 depends on Module 2a — the detector is told which behaviour classes and observable
indicators exist by the parsed policy, rather than defining them itself. That inversion is
what satisfies the assignment's policy-grounding requirement.

---

## 3. Policy grounding contract

The rule everything else follows:

> Behaviour classes, observable indicators, policy section references and severity tiers are
> **derived at runtime from the PDF**. They are never string literals in the source.

Three consequences worth internalising:

1. `config/settings.yaml` may contain `min_vest_pixel_ratio` (how confidently we see green).
   It may not contain "green vest means authorised" (what green *means*). The first is
   engineering, the second is policy.
2. The tables in `CONTEXT.md` are for humans reading the repo. No module imports them.
3. Pointing the parser at a different facility's manual should retarget the system without a
   code change. That is the test of whether the grounding is real.

### How faithfulness is verified

The assignment asks explicitly how we would catch an LLM inventing a rule. The answer is a
validation gate every rule must pass before it enters the `PolicyRuleSet`:

| Check | Catches |
|---|---|
| Every rule field is a literal substring of the extracted PDF text (whitespace-normalised) | Hallucinated indicators, paraphrased thresholds |
| PyMuPDF and pypdf extractions agree on the rule's source span | Extraction artefacts, layout misreads |
| Section ref matches `^Section \d+(\.\d+)*$` and resolves to a real section in the tree | Fabricated citations |
| Exactly 4 unsafe domains, each with at least one indicator | Silent under-extraction |
| Exactly 2 WARNING and 2 CRITICAL SAFETY NOTICE callouts | Callout mis-binding |

A rule failing any check is **excluded and logged**, never silently repaired. The rule set
records `source_sha256` of the PDF that produced it.

---

## 4. Detection approach

### Why classical CV on top of YOLO

The policy's observable indicators are vest **colour**, block **count**, panel **state** and
walkway **position**. None of those is a COCO class. YOLOv8n contributes what it is good at —
locating people and vehicles — and the indicators are read off those boxes:

| Behaviour | YOLO gives | Read by |
|---|---|---|
| Safe Walkway Violation | person boxes | foot-point-in-polygon against the **commissioned** walkway boundary, past a margin |
| Unauthorized Intervention | person boxes | torso ROI; HSV green vs red-black vest classification |
| Opened Panel Cover | — | mean intensity in the **commissioned** panel ROI vs a calibrated threshold |
| Forklift Overload | area-filtered vehicle box | contour block count — **disabled offline**, see below |

This needs no bounding-box annotations (the dataset has none — labels are folder-level), runs
on CPU, and keeps each decision explainable in the same vocabulary the policy uses.

### Commissioning, and why two detectors depend on it

Three artefacts are produced once per installation, from the **train** split only:

| Artefact | Script | What it supplies |
|---|---|---|
| `camera_registry.json` | `calibrate_cameras.py` | which fixed camera a frame came from |
| `panel_baseline.json` | `calibrate_panel.py` | where the electrical panel is, and the open/closed intensity boundary |
| `region_baseline.json` | `calibrate_regions.py` | the Designated Safe Walkway polygon, per camera |

These are installation parameters, not learned classifiers: they tell a detector *where* to
look, while *what counts as unsafe* remains the policy's indicator. The equivalent in a real
deployment is an installer drawing the walkway and the panel on screen.

**Camera identity is recovered from the image, never from the clip's folder name.** The folder
is the ground-truth label; using it at inference would leak it. `detection/cameras.py`
fingerprints the scene at 9x16 greyscale — measured separation is ~6 within a camera against
~30 between cameras — and a detector whose calibration belongs to a different camera abstains.

**Detectors abstain when uncommissioned.** No polygon for this camera, no walkway verdict. A
panel baseline from another view, no panel verdict. Silence is a better failure than noise in
an audit trail, and it shows up in the logs so it can be fixed.

### Which cues survived measurement

Every cue was measured before being trusted, and three of the four original ones did not
separate their class. The findings are in `docs/adr/0003-detector-cue-selection.md` and the
numbers in `docs/eval-baseline.md`. In short: the vest cue works; the panel cue works once the
region is commissioned; live green-line segmentation of the walkway caps at F1 0.25 and is
replaced by the commissioned polygon; the contour block count is *anti-correlated* with the
true forklift load and is disabled offline.

### Temporal handling

Frames are sampled at 4 fps and inferred at 640 px — roughly 40x cheaper than every frame at
full resolution, which is what makes CPU-only viable. A detection must then persist across N
consecutive sampled frames before it becomes an event (N per class, in `config/settings.yaml`).

State-based behaviours get a longer window than action-based ones: an open panel cover is a
*condition* that holds for the whole clip, so demanding six frames of agreement costs nothing
and removes most flicker. A walkway violation is a brief *event*, so three frames is the
ceiling before recall starts to suffer.

### The VLM path

When a detector's confidence margin falls inside `vlm_tiebreak.band`, one annotated frame goes
to a vision model with a narrow, closed question — *"How many blocks are on the forks? Answer
with a single integer."* This targets exactly the ambiguity the assignment raises (a forklift
carrying what might be two or three blocks).

Guard rails: at most two calls per clip, results cached by frame hash, and the consultation
recorded in the event's `detection_method` so the audit trail shows when a model was asked.
With `AEGISFLOW_LLM_PROVIDER=offline` the tie-break resolves deterministically instead and no
network call is made.

---

## 5. Severity derivation

Three signals combine into a tier. Two come from the policy; one comes from the frame.

**Signal 1 — callout keyword.** `CRITICAL SAFETY NOTICE` → HIGH base, `WARNING` → MEDIUM base,
`NOTE`/none → LOW base.

**Signal 2 — hazard-context language.** The parser scores each section for recurrence
(*"highest-frequency"*), unambiguity (*"threshold is unambiguous"*, *"must be assumed"*) and
standalone-condition language (*"regardless of ... whether personnel are in the immediate
vicinity"*). The first two raise the tier; the third lowers the base, because the policy is
saying personnel exposure is not a precondition.

**Signal 3 — frame context.** Person count, forklift presence, person-near-panel proximity,
detection confidence. This adjusts the derived base per clip.

| Behaviour | Callout | Base | Context escalation |
|---|---|---|---|
| Opened Panel Cover (5.2.2) | WARNING + standalone | LOW | → MEDIUM with a person in frame; → HIGH with a person at the panel |
| Safe Walkway Violation (3.3.2) | WARNING + high-frequency | MEDIUM | → HIGH with a forklift concurrently in frame |
| Unauthorized Intervention (4.3.2) | CRITICAL SAFETY NOTICE | HIGH | → CRITICAL with multiple unauthorised persons |
| Forklift Overload (6.3.2) | CRITICAL SAFETY NOTICE + unambiguous | CRITICAL | — |

Every assessment carries `severity_rationale`: the quoted policy sentence plus the adjustment
applied. It appears in the exported audit record and in the dashboard, so any tier can be
traced back to a line of the manual.

---

## 6. Multi-violation handling

A clip is not one event. The pipeline emits **one `ViolationEvent` per detected violation**,
each with its own severity and its own routing decision. A clip containing a walkway violation
(MEDIUM → DB only) and an open panel with a person nearby (HIGH → DB + alert) produces two
records and one alert. Severities are never merged, maxed, or deduplicated across classes.

---

## 7. Persistence

SQLite via async SQLAlchemy 2.0. Three tables: `violation_events`, `policy_rules`,
`clip_runs`. Indexed on `timestamp`, `severity` and `behavior_class` — the exact filters the
dashboard's Historical Log needs.

**Append-only by construction.** `ViolationEvent` is a frozen Pydantic model and `db/crud.py`
exposes no update or delete path. Compliance records are evidence; a correction is a new row.

Postgres is a connection-string change (`AEGISFLOW_DB_URL`) with no code change, which is what
the product framing's "enterprise multi-facility" tier assumes.

---

## 8. Deployment shape

Single process for the demo: FastAPI serves the API, the WebSocket alert channel and the built
frontend. The CV pipeline runs as a CLI batch job writing to the same SQLite file, so the
dashboard replays processed clips rather than doing live inference — which is what keeps it
responsive on a CPU-only machine.

For the product story (Raspberry Pi, on-site, no cloud), the same single process is the whole
deployment. Docker packaging is Phase 2, task A8.
