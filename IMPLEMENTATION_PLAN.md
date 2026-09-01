# AegisFlow EHS — Implementation Plan (Phase 1: Sohaib)

> **Scope of this document:** everything Sohaib builds, in order, with an explicit exit
> criterion for each step. Phase 2 (Aimen) is summarised at the end and detailed in
> `HANDOVER.md`.
>
> Read `CONTEXT.md` first for *what* and *why*. This file is *how* and *in what order*.

---

## 0. The sequencing decision

The original schedule PDF assumed two developers working in parallel with daily syncs. **That
is no longer the plan.** Sohaib completes his entire scope and hands over a working, tested,
documented foundation; Aimen then builds on top of it without needing Sohaib online.

This changes one thing structurally: **Sohaib must produce the interfaces Aimen will code
against, not just his own modules.** A parallel plan can defer the contract because both sides
negotiate it live. A sequential plan cannot — the contract *is* the handover. That is why
Step 1 below is schemas and the database, before any CV work.

### Ownership

| | Sohaib (Phase 1 - build) | Aimen (Phase 2 - deploy) |
|---|---|---|
| **Modules** | All five, end to end | none - the system is complete |
| **Also owns** | Repo layout, environment, config, DB schema, shared contracts, LLM abstraction, CLI, calibration tooling, evaluation harness, full test suite, dashboard, all docs | Docker, running it on the target machine, Groq key, demo video, submission QA |
| **Definition of done** | `aegisflow run` turns clips into validated, routed, reported `ViolationEvent` records; test suite green; measured metrics published | Assignment submission checklist fully ticked |

This is a change from the original schedule PDF, which split AI/CV from backend/dashboard
across two people working in parallel. Building it as one coherent system and handing over a
tested artefact removes the interface-negotiation risk entirely.

---

## 1. Milestones at a glance

| # | Milestone | Deliverable | Exit criterion |
|---|---|---|---|
| **S0** | Foundation | Repo, env, config, contracts, DB, CLI skeleton | `pytest` green on contract tests; `aegisflow --help` works |
| **S1** | Policy parsing | `policy/` package + `artifacts/policy/rules.json` | 4 domains + 4 indicators + 4 callouts extracted, each verified against literal PDF text |
| **S2** | Severity matrix | `severity/` package | Every tier traceable to a quoted policy sentence; unit tests cover all 4 classes x context permutations |
| **S3** | Detection core | `detection/` frame pipeline + YOLO adapter | YOLOv8n runs on a sample clip in < 3 s at 4 fps / 640 px on CPU |
| **S4** | Four detectors | walkway, vest, panel, forklift | Each of the 4 unsafe classes fires on its own clips and stays quiet on its safe counterpart |
| **S5** | VLM tie-break | `llm/` provider abstraction + Groq vision | Offline mode unchanged; Groq mode measurably improves the forklift 2-vs-3 boundary |
| **S6** | Pipeline + eval | `pipeline.py`, `scripts/evaluate.py` | Full 691-clip run completes; per-class precision/recall table produced |
| **S7** | Handover | `HANDOVER.md`, seeded DB, frontend scaffold, README sections | Aimen can start Module 3 from the docs alone |

---

## 2. Step-by-step

### S0 — Foundation *(done in this session)*

The pieces that everything else imports. Built first because the sequential handover depends
on them being stable.

**S0.1 Repo & environment** — *complete*
- Directory tree matching the assignment's recommended layout.
- Dataset flattened from the nested Kaggle path to `data/raw/{train,test}/<class>/`; the
  `2_opened_panel cover` folder renamed to remove its space.
- `requirements.txt` pinned to verified-working versions; `pyproject.toml` with package
  discovery, pytest config, ruff, black, mypy.
- `.env.example` documenting every knob; `.gitignore` excluding the 9.4 GB dataset, model
  weights, the SQLite file and secrets.
- `config/settings.yaml` (CV tuning only) and `config/zones.yaml` (camera/zone map).

**S0.2 Core contracts** — `src/aegisflow/core/`
```
core/
  enums.py      BehaviorClass, Severity, DetectionMethod, EscalationAction
  schemas.py    FrameObservation, DetectionRecord, SeverityAssessment,
                ViolationEvent, PolicyRule, PolicyRuleSet, ClipResult
  settings.py   pydantic-settings loader: .env + config/*.yaml -> typed Settings
  zoning.py     clip -> zone resolution (override -> class default -> unassigned)
  logging.py    rich console + optional JSON handler
  errors.py     AegisFlowError hierarchy
```
`ViolationEvent` carries all 9 assignment-mandated fields plus `confidence`,
`detection_method`, `severity_rationale`. It is `model_config = ConfigDict(frozen=True)` —
immutability is an assignment requirement for audit records, so enforce it in the type, not by
convention.

**S0.3 Database** — `src/aegisflow/db/`
- SQLAlchemy 2.0 async models: `violation_events`, `policy_rules`, `clip_runs`.
- Indexes on `(timestamp)`, `(severity)`, `(behavior_class)` — Module 5's Historical Log
  filters on exactly these three, so build for it now rather than making Aimen retrofit.
- Append-only discipline: no `UPDATE`/`DELETE` paths in the CRUD layer at all.
- `init_db()` + `scripts/seed_db.py`.

**S0.4 CLI skeleton** — `src/aegisflow/cli.py` (Typer)
```
aegisflow policy parse          # S1  PDF -> artifacts/policy/rules.json
aegisflow policy show           # pretty-print the extracted rule set
aegisflow detect <clip>         # S4  single clip -> detection records
aegisflow run --split test      # S6  batch pipeline
aegisflow evaluate --split test # S6  per-class metrics
aegisflow serve                 # FastAPI app (Aimen extends)
```

> **Exit:** `pytest tests/unit/test_schemas.py` green, `aegisflow --help` renders.

---

### S1 — Policy parsing (Module 2, part 1)

The graded requirement is blunt: behaviour classes *"must be derived from the policy document
through your parsing pipeline, not manually transcribed as hard-coded strings."* So the parser
is the foundation of the whole system, not a side quest.

**Approach — three layers, each a fallback for the one above:**

1. **Layout extraction** (`policy/extract.py`) — PyMuPDF pulls text with span metadata.
   Callout boxes (`WARNING`, `CRITICAL SAFETY NOTICE`, `NOTE`, `IMPORTANT`) are visually
   distinct in this PDF; they are recovered by span font/colour grouping, not by regex over
   flat text. `pypdf` provides a second independent extraction for cross-validation.
2. **Deterministic structuring** (`policy/parse.py`) — section-number regex
   (`^(\d+(?:\.\d+)*)\s+(.+)$`) builds a section tree; the Section 8 quick-reference table is
   parsed for the class-ID -> domain -> indicator mapping; each unsafe behaviour is bound to
   its governing section and its nearest callout.
3. **Optional LLM structuring** (`policy/llm_extract.py`) — when a provider is configured,
   the section text is sent with a strict JSON schema to recover indicators phrased in prose
   the regexes miss.

**Verification — the part that actually matters.** The assignment asks directly: *"how will
you verify that its extracted rules are faithful to the source document?"* Our answer, in
`policy/validate.py`:

- Every rule field must be a **literal substring** of the extracted PDF text (normalised for
  whitespace). Anything the LLM invents fails this check and is dropped.
- Cross-check PyMuPDF vs. pypdf extractions; disagreement on a rule marks it low-confidence.
- Structural assertions: exactly 4 unsafe domains, each with >= 1 observable indicator, each
  bound to a section reference matching `^Section \d+(\.\d+)*$`, and exactly 2 WARNING + 2
  CRITICAL SAFETY NOTICE callouts (the assignment's own hint, used as a self-test).
- A rule that fails validation is *excluded and logged*, never silently corrected.

**Output** — `artifacts/policy/rules.json`, a versioned `PolicyRuleSet` with a
`source_sha256` of the PDF so we can prove which document produced which rules.

> **Exit:** `aegisflow policy parse` produces 4 validated unsafe-behaviour rules with correct
> section refs and callouts, in **both** `offline` and `groq` provider modes, with identical
> structural output.

---

### S2 — Severity matrix (Module 2, part 2)

`severity/matrix.py` turns a `PolicyRule` plus per-clip context into a `SeverityAssessment`.

**Signal 1 — callout keyword (from policy text):**

| Callout found in the rule's section | Base tier |
|---|---|
| `CRITICAL SAFETY NOTICE` | HIGH |
| `WARNING` | MEDIUM |
| `NOTE` / none | LOW |

**Signal 2 — hazard-context language (from policy text):** the parser scores each section for
recurrence language (*"most frequently occurring"*, *"highest-frequency"*), unambiguity
language (*"threshold is unambiguous"*, *"must be assumed"*), and standalone-condition
language (*"regardless of ... whether personnel are in the immediate vicinity"*). Recurrence
and unambiguity push a tier up; standalone-condition pulls the *base* tier down because the
policy itself says personnel exposure is not required.

**Signal 3 — per-clip context (from Module 1, not from policy):** person count, forklift
present, person-near-panel proximity, detection confidence.

The resulting matrix, all four cells derived rather than typed in:

| Behaviour | Section | Callout | Derived base | Context escalation |
|---|---|---|---|---|
| Opened Panel Cover | 5.2.2 | WARNING | **LOW** | -> MEDIUM if a person is in frame; -> HIGH if a person is within the panel proximity radius |
| Safe Walkway Violation | 3.3.2 | WARNING (+ high-frequency) | **MEDIUM** | -> HIGH if a forklift is concurrently in frame |
| Unauthorized Intervention | 4.3.2 | CRITICAL SAFETY NOTICE | **HIGH** | -> CRITICAL on multiple unauthorised persons |
| Forklift Overload | 6.3.2 | CRITICAL SAFETY NOTICE (+ unambiguous) | **CRITICAL** | — |

Every assessment carries `severity_rationale`: the quoted policy sentence plus the context
adjustment applied. That string appears in the dashboard and in the exported audit record, so
a reviewer can trace any tier back to a line of the manual.

> **Exit:** unit tests assert the base tier for all 4 classes and every context permutation;
> no severity literal appears anywhere outside `severity/` and the parsed rule set.

---

### S3 — Detection core (Module 1, part 1)

`detection/` — the frame pipeline that all four detectors share.

```
detection/
  video.py       stream frames at sample_fps, downscale to imgsz, never load whole clip
  yolo.py        Ultralytics adapter -> list[FrameObservation]; lazy weight download
  geometry.py    IoU, containment, proximity, torso-ROI extraction from a person box
  temporal.py    persistence smoothing + event de-duplication across frames
  engine.py      orchestrator: clip -> list[DetectionRecord]
```

Design decisions worth stating up front:

- **Sample at 4 fps, infer at 640 px.** A 7 s clip becomes ~28 frames instead of 175, at
  1/9 the pixels. On 12 CPU cores that is seconds per clip instead of minutes. The behaviours
  we detect persist for well over 250 ms, so nothing is lost.
- **YOLOv8n gives us `person`, `truck`/`car` (forklift proxy) and little else.** The policy's
  actual indicators — vest colour, block count, panel state, walkway position — are *not* COCO
  classes. They are recovered by classical CV applied to YOLO's boxes and to the frame. This
  is deliberate and defensible: the indicators are colour and geometry properties, and it
  avoids needing an annotated training set we do not have.
- **Temporal persistence, not single frames.** A detection must hold for N consecutive sampled
  frames (per class, in `config/settings.yaml`) before it becomes an event. This is the single
  biggest false-positive reducer available to us.

> **Exit:** `aegisflow detect <clip>` runs YOLOv8n end to end on a sample clip in under ~3 s.

---

### S4 — The four detectors (Module 1, part 2)

Built in ascending order of difficulty, each with its safe-class counterpart as the negative
test set.

**S4.1 Forklift overload — Section 6.3.2** *(easiest: a hard numeric threshold)*
- Locate the forklift via YOLO (`truck`/`car`), take the region above/around the forks.
- Segment block-like rectangles by edge + contour analysis, filtered on area and aspect ratio
  from `config/settings.yaml`.
- Count stable across the persistence window. `>= 3` -> violation; `<= 2` -> safe.
- Counts of exactly 2 or 3 with a weak margin are flagged `ambiguous` -> handed to S5.
- Negatives: `7_safe_carrying`.

**S4.2 Unauthorized intervention — Section 4.3.2**
- YOLO person boxes; take the upper-torso ROI (geometry.py).
- HSV masks for green vs red-black; the dominant vest colour wins above `min_vest_pixel_ratio`.
- "Interacting with equipment" is approximated by proximity/overlap with a static equipment
  region plus low motion of the person centroid — a person standing at a machine, not walking
  past. Documented as an approximation in the README's known-limitations section.
- Green vest -> authorised, no event. Non-green -> violation.
- Negatives: `5_authorized_intervention`.

**S4.3 Safe walkway violation — Section 3.3.2**
- Segment the green floor lines by HSV, morphologically close them, and take the largest
  contour as the walkway polygon (falling back to `config/zones.yaml:walkway_polygons` when
  the lines are occluded).
- A person's **foot point** (bottom-centre of the box), not the box centre, is tested for
  containment. Feet are what is on the floor.
- Outside the polygon for N consecutive frames -> violation.
- Negatives: `4_safe_walkway`. This class has the most clips and the highest expected false
  positive rate — occlusion and perspective are real problems here.

**S4.4 Opened panel cover — Section 5.2.2** *(hardest: state-based, no COCO class)*
- Panels are static in a fixed camera. Build a per-camera background reference from the
  `6_closed_panel_cover` clips, then detect the open state by contour-geometry deviation in
  the panel ROI (a swung-open cover changes the rectangle's aspect and adds a strong vertical
  edge).
- Requires the longest persistence window (6 frames) because it is a *condition*, not an
  action — the policy explicitly says duration and personnel presence are irrelevant to
  whether it counts.
- Negatives: `6_closed_panel_cover`.

> **Exit:** each detector fires on its own class and stays quiet on its safe counterpart, on a
> 20-clip-per-class sample. Per-class precision/recall recorded in `docs/eval-baseline.md`.

---

### S5 — LLM abstraction & VLM tie-break

`llm/` — one interface, three implementations, selected by `AEGISFLOW_LLM_PROVIDER`.

```
llm/
  base.py             LLMProvider ABC: complete_json(), answer_about_image()
  groq_provider.py    Groq — primary. Model ids come from GROQ_TEXT_MODEL /
                      GROQ_VISION_MODEL, never from code; see .env.example.
  gemini_provider.py  Google Gemini — alternate
  offline.py          deterministic stub; no network. THE DEFAULT.
  cache.py            content-hash memoisation to disk
```

Two consumers:

1. **Policy structuring (S1)** — text completion with a strict JSON schema, output validated
   against literal PDF substrings.
2. **Detection tie-break (S4)** — when a detector's confidence margin falls inside
   `vlm_tiebreak.band`, one annotated frame is sent to the vision model with a narrow,
   closed question ("How many blocks are on the forks? Answer with a single integer."). Capped
   at 2 calls per clip, results cached by frame hash, and every use recorded in the event's
   `detection_method` so the audit trail shows when a model was consulted.

Non-negotiable: **`offline` is the default and the whole pipeline must pass its tests with no
network access.** The Groq key is an enhancement, never a dependency.

> **Exit:** identical structural output in `offline` and `groq` modes; tie-break measurably
> improves forklift 2-vs-3 accuracy on the ambiguous subset; `pytest -m "not llm"` green with
> networking disabled.

---

### S6 — Pipeline integration & evaluation

`pipeline.py` composes the phase into one callable:

```
clip -> DetectionEngine -> list[DetectionRecord]
     -> SeverityMatrix(rules) -> list[SeverityAssessment]
     -> ViolationEvent[]     (one per violation, never merged)
     -> EscalationSink        (Phase 1: a no-op sink that writes to the DB;
                               Phase 2: Aimen swaps in the real router)
```

The `EscalationSink` protocol is defined here and implemented trivially, so Aimen's Module 3
is a drop-in replacement rather than a refactor.

**Multi-violation handling** is designed in from the start: `pipeline` emits one
`ViolationEvent` per detected violation per clip, each with its own severity and routing
decision. Clips containing two different classes are used as fixtures.

`scripts/evaluate.py` runs the full 691 clips and emits:
- per-class precision / recall / F1 (clip-level, using folder labels as ground truth)
- a confusion matrix across the 8 folders
- false positives on the four safe classes, listed by filename for inspection
- latency per clip, and VLM call counts

> **Exit:** full-dataset run completes; `docs/eval-baseline.md` written with real numbers and
> an honest known-limitations list.

---

### S7 — Handover package

The deliverable that makes the sequential model work.

- **`HANDOVER.md`** — the frozen `ViolationEvent` schema, the `EscalationSink` protocol, the
  DB schema with its indexes, the seeded database, the exact API routes Aimen needs to add,
  the WebSocket message envelope, and a ranked task list for Modules 3-5.
- **Seeded database** — `scripts/seed_db.py` populates `aegisflow.db` from a real pipeline run,
  so Aimen has hundreds of genuine events to build the dashboard against on day one, with no
  need to run the CV pipeline himself.
- **Frontend scaffold** — `frontend/` with Vite + React + TS, design tokens (the severity
  colour scale from the assignment: LOW blue / MED green / HIGH amber / CRIT red), a typed API
  client generated from the FastAPI OpenAPI schema, a WebSocket hook, and three empty routed
  views. Aimen fills in the views; the visual language is already consistent.
- **README sections** owned by Sohaib: architecture overview, policy-parsing approach,
  severity-mapping rationale, model-selection rationale, known limitations.

> **Exit:** a developer who has never seen this repo can run `scripts/setup.ps1`, open
> `HANDOVER.md`, and start Module 3 without asking a question.

---

## 3. Test strategy

| Layer | What it covers | Marker |
|---|---|---|
| `tests/unit/` | Schema validation & immutability, zone resolution, severity matrix for every class x context permutation, geometry helpers, HSV classification on synthetic swatches, policy validators | — |
| `tests/integration/` | PDF -> rules.json, clip -> DetectionRecord, full pipeline -> ViolationEvent, DB round-trip, multi-violation clip | `integration` |
| Real clips | Tests that need video read from `data/raw/` and **skip** when it is absent, so a clean checkout stays green | `slow` |
| Offline guarantee | The whole suite passes with `AEGISFLOW_LLM_PROVIDER=offline` and no network | — |
| Provider tests | Groq text + vision, skipped automatically when no key is present | `llm` |

Target: the assignment's own bar is "functional, with documented limitations", not perfect
accuracy. So tests assert *contract correctness* strictly and *detection accuracy* against
recorded baselines that we are allowed to move — with the move justified in the commit.

---

## 4. Risk register

| Risk | Likelihood | Mitigation |
|---|---|---|
| Panel open/closed detection is unreliable (no COCO class, subtle visual difference) | **High** | Background-reference approach + longest persistence window; if precision stays under ~0.6, document it honestly as a known limitation and lean on the VLM tie-break. The assignment explicitly does not require perfect accuracy. |
| Walkway false positives from occlusion/perspective | **High** | Foot-point containment instead of box centre; per-frame green-line re-segmentation with a static polygon fallback; tune on the 50 `4_safe_walkway` negatives. |
| CPU-only inference too slow for a live demo | Medium | 4 fps / 640 px sampling; cache detections to `data/processed/`; the dashboard replays processed clips rather than inferring live. |
| LLM extracts a rule that is not in the PDF | Medium | Literal-substring validation drops it; deterministic parser is the default path and needs no LLM at all. |
| Groq free-tier rate limits during the demo | Medium | Disk cache + 2-calls-per-clip cap + `offline` fallback that requires no code change. |
| Forklift detected as `truck` inconsistently by YOLOv8n | Medium | Accept `truck`, `car` and `forklift`-shaped boxes; fall back to motion + block-region heuristics; document it. |
| Scope creep into Aimen's modules | Low | The `EscalationSink` no-op is the hard stop. Phase 1 writes to the DB and nothing more. |

---

## 5. Phase 2 preview (Aimen)

Detailed in `HANDOVER.md`. Summarised here so the whole shape is visible:

| Step | Module | Work |
|---|---|---|
| A1 | 3 | Replace the no-op `EscalationSink` with the real router: LOW/MED -> DB only; HIGH/CRIT -> DB + WebSocket publish. Concurrent multi-violation clips must not drop events. |
| A2 | 4 | Report writers: append-only JSON log, append-only CSV audit, ReportLab PDF export. All 9 mandated fields. |
| A3 | 5 | View A — Live Feed Monitor: processed clip playback with severity-coloured overlay + alert banner for HIGH/CRIT. |
| A4 | 5 | View B — Alert Timeline Stream: live WebSocket feed, chronological, visual strobe on HIGH/CRIT. |
| A5 | 5 | View C — Historical Log & Export: filter by date range / severity / behaviour class, export button. |
| A6 | — | Dockerfile + compose, final QA, demo video, submission checklist. |

---

## 6. Progress tracker

All build steps are complete. Deployment (Docker, demo video, submission QA) is Phase 2 -
see `HANDOVER.md`.

| Step | Status | Notes |
|---|---|---|
| S0.1 Repo & environment | **Done** | Tree created, dataset flattened to `data/raw/`, deps pinned & verified |
| S0.2 Core contracts | **Done** | Frozen `ViolationEvent` with all 9 mandated fields + 3 extra; 127 unit tests |
| S0.3 Database | **Done** | Async SQLAlchemy, append-only CRUD, indexed on the 3 View-C filters |
| S0.4 CLI | **Done** | `policy parse/show/matrix`, `detect`, `run`, `report`, `serve`, `info` |
| S1 Policy parsing | **Done** | 4/4 rules derived and validated from the PDF, strict mode passes with 0 warnings |
| S2 Severity matrix | **Done** | Tiers derived from callout + hazard language; every tier carries its derivation |
| S3 Detection core | **Done** | 4 fps / 640 px sampling, batched inference at 98 ms/frame (from 471) |
| S4.1 Forklift overload | **Done (offline disabled)** | Contour count measured anti-correlated; abstains offline, VLM covers it - ADR 0003 |
| S4.2 Unauthorized intervention | **Done** | Vest colour separates cleanly; requires positive red-black evidence |
| S4.3 Safe walkway violation | **Done** | Commissioned per-camera polygon; abstains on uncommissioned cameras |
| S4.4 Opened panel cover | **Done** | Commissioned ROI + camera fingerprint gate; best-performing detector |
| S5 LLM + VLM path | **Done** | Groq/Gemini/offline providers; VLM promoted to a first-class detection path |
| S6 Pipeline & evaluation | **Done** | `scripts/evaluate.py`, `calibrate*.py`, `sweep_thresholds.py`; both providers measured (macro F1 0.26 offline, 0.37 Groq) in `docs/eval-baseline.md` |
| Module 3 Escalation | **Done** | LOW/MED -> log, HIGH/CRIT -> log + WebSocket; multi-violation independence tested |
| Module 4 Reports | **Done** | Append-only JSONL + CSV + per-event JSON + ReportLab PDF |
| Module 5 Dashboard | **Done** | 3 required views + policy panel; zero build step |
| S7 Handover | **Done** | `HANDOVER.md` runbook, `seed_db.py`, all docs |

### Test suite

| Suite | Count | Covers |
|---|---|---|
| `tests/unit/` | 127 | Schemas & immutability, severity derivation, policy parsing & the faithfulness gate, geometry & colour, temporal persistence, camera fingerprints, escalation routing, report writers |
| `tests/integration/` | 30 | Full API surface (all 3 views' endpoints), pipeline wiring, multi-violation independence, policy -> severity end to end |

Run `pytest` for everything, or `pytest -m "not slow and not llm"` for the subset that needs
neither the dataset nor a provider key.
