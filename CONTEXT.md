# AegisFlow EHS — Project Context

> **Read this first.** It is the single source of truth for *what* we are building and *why*.
> `IMPLEMENTATION_PLAN.md` covers *how* and *when*. `CLAUDE.md` covers *how to work in this repo*.
> Last updated: 2026-09-01. All five modules built and tested.

---

## 1. One-paragraph summary

AegisFlow EHS is an end-to-end **factory compliance and alert escalation system**. It ingests
raw factory CCTV clips, parses a real Occupational Health & Safety policy PDF
(`KMP-OHS-POL-001`) into machine-readable rules, detects behavioural violations in the video
against those rules, assigns each violation a risk severity tier, routes it to the correct
downstream channel (database log vs. real-time alert), writes an immutable audit record, and
surfaces everything through a live operations dashboard.

It is built as an **intern take-home assignment** for a 2-person team, delivered as one Git
repository with a README, a working pipeline, and a demo walkthrough.

---

## 2. Source documents

All four originals live in [`docs/reference/`](docs/reference/). The policy manual is also
copied to the repo root as `compliance_policy.pdf` because the assignment's recommended
structure expects it there.

| Document | Role in the project |
|---|---|
| `Intern_Assessment_AI.pdf` | **The spec.** Defines the 5 mandatory modules, the required report fields, the 4 severity tiers, the 3 dashboard views, and the submission checklist. This is what we are graded against. |
| `KMP-OHS-POL-001_Compliance_Policy_Manual.pdf` | **The source of truth for all compliance decisions.** Defines the 4 behavioural domains, the observable indicators, and the WARNING / CRITICAL SAFETY NOTICE callouts that drive severity. |
| `AegisFlow_EHS_2Week_Schedule.pdf` | The team's 14-day plan and the Person A / Person B split. |
| `AegisFlow_EHS_Project_Idea.pdf` | Product framing, positioning and the stack we committed to. |

**Rule:** if the spec and the idea doc disagree, the spec wins. If the policy manual and any
of our code disagree about what a violation *is*, the policy manual wins.

---

## 3. The dataset

Kaggle — *Video Dataset for Safe and Unsafe Behaviours*. Already downloaded and normalised
into `data/raw/`.

| Property | Value |
|---|---|
| Total clips | **691** (566 train / 125 test) |
| Resolution | 1920 x 1080 |
| Frame rate | ~25 fps |
| Duration | 5 - 14 s per clip |
| Classes | 8 — four unsafe, four matching safe counterparts |
| On-disk size | ~9.4 GB (gitignored) |

### Class distribution

| ID | Class directory | Train | Test | Safe/Unsafe | Policy section |
|---|---|---:|---:|---|---|
| 0 | `0_safe_walkway_violation` | 178 | 32 | **UNSAFE** | 3.3.2 |
| 1 | `1_unauthorized_intervention` | 97 | 11 | **UNSAFE** | 4.3.2 |
| 2 | `2_opened_panel_cover` | 129 | 13 | **UNSAFE** | 5.2.2 |
| 3 | `3_carrying_overload_with_forklift` | 48 | 8 | **UNSAFE** | 6.3.2 |
| 4 | `4_safe_walkway` | 50 | 25 | safe | 3.3.1 |
| 5 | `5_authorized_intervention` | 23 | 15 | safe | 4.3.1 |
| 6 | `6_closed_panel_cover` | 19 | 13 | safe | 5.2.1 |
| 7 | `7_safe_carrying` | 22 | 8 | safe | 6.3.1 |

Notes that matter for modelling:

- **The class folder is our ground truth label.** The clips carry no bounding-box annotations,
  so we evaluate at *clip level*, not box level.
- **Heavy class imbalance.** Walkway violations (178) outnumber closed-panel covers (19) by
  ~9x. Report per-class metrics, never a single blended accuracy number.
- **The safe classes are the negative set.** A detector that fires on `4_safe_walkway` is
  producing a false positive — that is how we measure precision.
- The original Kaggle folder `2_opened_panel cover` contained a space; it has been renamed
  to `2_opened_panel_cover`. Nothing else about the data was altered.

---

## 4. The policy, distilled

Four behavioural domains, each with one safe and one unsafe behaviour, and one observable
visual indicator. **This table is documentation only — the code must derive it by parsing the
PDF, not by importing this file.**

| ID | Domain | Unsafe behaviour | Observable indicator | Policy callout |
|---|---|---|---|---|
| 0 | Pedestrian movement | Safe Walkway Violation | Person outside the **green** painted floor markings | 3.3.2 **WARNING** — *"highest-frequency unsafe behavior recorded at this facility"* |
| 1 | Equipment interaction | Unauthorized Intervention | Person touching equipment **without the green vest** (red-black vest = not authorised) | 4.3.2 **CRITICAL SAFETY NOTICE** |
| 2 | Electrical safety | Opened Panel Cover | Panel cover in the **open** position during production | 5.2.2 **WARNING** — explicitly *"regardless of whether personnel are in the immediate vicinity"* |
| 3 | Forklift load | Carrying Overload with Forklift | **3 or more** standardized blocks on the forks (2 or fewer is safe) | 6.3.2 **CRITICAL SAFETY NOTICE** |

### Why the severity tiers land where they do

The assignment hints that *"two of the four behavior categories appear under a WARNING callout
and two appear under a CRITICAL SAFETY NOTICE callout."* That is the primary severity signal,
and it is reinforced by a second signal — whether personnel are exposed in the frame.

These are the tiers the code **actually derives** from this manual — verify with
`aegisflow policy matrix`, which prints the derivation for each one:

| Behaviour | Callout | Hazard-context language found | Derived base | Frame context escalates to |
|---|---|---|---|---|
| Opened Panel Cover | WARNING | *"regardless of ... whether personnel are in the immediate vicinity"* → state-based, base lowered | **LOW** | **MEDIUM** with a person in frame; **HIGH** with a person at the panel |
| Safe Walkway Violation | WARNING | *"highest-frequency unsafe behavior"* → raised one tier | **HIGH** | **CRITICAL** with a forklift concurrently in frame (§3.1 hazard context) |
| Unauthorized Intervention | CRITICAL SAFETY NOTICE | *"must be assumed to be performing an Unauthorized Intervention"* → raised one tier | **CRITICAL** | — |
| Forklift Overload | CRITICAL SAFETY NOTICE | *"the block count threshold is unambiguous"* → raised one tier | **CRITICAL** | — |

Nothing in that table is written down anywhere in the source. It is *computed* from the
parsed policy text — callout keyword, then recurrence / unambiguity / standalone-condition
phrases — and then adjusted per clip by what the detector saw. Point the parser at a
different manual and the table moves.

Every emitted record carries a `policy_rule_ref` such as `Section 4.3.2` and a
`severity_rationale` quoting the policy sentence that produced the tier. That traceability is
a graded requirement, not a nicety.

---

## 5. The five modules

| # | Module | Package | Status | Contract |
|---|---|---|---|---|
| 1 | Detection Engine | `src/aegisflow/detection/` | **Built** | video clip -> `list[DetectionRecord]` |
| 2 | Severity Matrix | `src/aegisflow/severity/` (+ `policy/`) | **Built** | `DetectionRecord` + `PolicyRuleSet` -> `SeverityAssessment` |
| 3 | Escalation Pipeline | `src/aegisflow/escalation/` | **Built** | `ViolationEvent` -> DB log and/or real-time alert |
| 4 | Report Generation | `src/aegisflow/reports/` | **Built** | `ViolationEvent` -> immutable JSON/CSV/PDF record |
| 5 | Operations Dashboard | `frontend/` | **Built** | 3 views over the Module 3/4 API |

`ViolationEvent` in `src/aegisflow/core/schemas.py` is the seam all five modules meet at:
Modules 1-2 produce it, Modules 3-5 consume it. It is frozen (`ConfigDict(frozen=True)`)
because compliance records are evidence.

### Mandatory routing (assignment Module 3)

```
LOW  / MEDIUM   ->  persistent database log only
HIGH / CRITICAL ->  real-time alert (WebSocket)  +  persistent database log
```

Multiple violations of different severities inside one clip must each be logged and routed
**independently** — no collapsing to the max severity.

### Mandatory report fields (assignment Module 4)

`event_id` - `timestamp` - `clip_id` - `zone` - `behavior_class` - `policy_rule_ref` -
`event_description` - `severity` - `escalation_action`

We add three non-required fields that make the system defensible in review:
`confidence`, `detection_method` (`yolo` / `hsv` / `contour` / `vlm_tiebreak`), and
`severity_rationale`.

---

## 6. Team & sequencing

Two developers, working **sequentially, not in parallel** (changed from the original schedule
PDF, which assumed parallel tracks with daily syncs).

- **Phase 1 — Sohaib.** All five modules, end to end, plus the engineering foundation: repo
  layout, environment, config, database schema, shared Pydantic contracts, LLM provider
  abstraction, calibration tooling, evaluation harness, test suite, and all documentation.
- **Phase 2 — Aimen.** Deployment: containerisation, running the system on the target
  machine, the demo video, and the submission checklist. `HANDOVER.md` is the runbook.

This is a change from the original split in the schedule PDF, which had Person A on AI/CV
and Person B on backend/dashboard. The build is complete and tested before it is handed on,
so Phase 2 needs no knowledge of the internals to stand the system up.

---

## 7. Hard constraints from the dev machine

These shaped several design decisions and should not be re-litigated silently.

| Constraint | Consequence |
|---|---|
| **No GPU** (`torch 2.1.2+cpu`) | YOLOv8-**nano** only. Never the s/m/l variants. |
| **8 GB RAM** | Process one clip at a time; stream frames, never load a whole clip into memory. |
| 1920x1080 @ 25 fps source | Sample at **4 fps** and downscale to **640 px** for inference — roughly 40x less compute than every-frame full-res, with no meaningful recall loss on 5-14 s clips. |
| 691 clips x ~7 s | A full-dataset regression run is minutes, not seconds. Cache detections to `data/processed/` so re-runs are cheap. |
| No internet guaranteed at demo time | LLM usage must be **optional**. `AEGISFLOW_LLM_PROVIDER=offline` runs the whole pipeline with zero network calls. |

---

## 8. Technology decisions

| Layer | Choice | Why |
|---|---|---|
| Detection | YOLOv8-nano + OpenCV (HSV, contours) | Only CPU-viable detector; the policy's indicators (vest *colour*, block *count*, panel *state*, walkway *position*) are geometric/colour properties that classical CV reads directly off YOLO's boxes. |
| Policy parsing | PyMuPDF layout extraction -> deterministic section/callout parser -> **optional** LLM structuring pass | Deterministic first means the system never depends on an LLM being reachable; the LLM adds robustness, and every LLM-extracted rule is validated against a literal substring of the PDF before it is accepted. |
| LLM provider | **Groq** (free tier) behind a provider interface | Key is free and fast. `gemini` and `offline` are drop-in alternates — see `src/aegisflow/llm/`. |
| VLM tie-break | Groq vision model, gated by a confidence band | Directly answers the assignment's "forklift carrying two or three blocks?" hint. Capped at 2 calls per clip. |
| Backend | FastAPI + WebSocket | Async, first-class WebSocket support for the HIGH/CRITICAL alert channel. |
| Storage | SQLite (async, via SQLAlchemy 2.0) | Zero-ops, file-backed, matches the "runs on a Raspberry Pi, offline" product story. Postgres is a URL change. |
| Reports | JSON + CSV append-only audit log, plus ReportLab PDF | The assignment allows any one; we ship all three because "immutable audit trail" is the point. |
| Frontend | Vanilla ES modules + CSS custom properties, **no build step** | The project idea document commits to HTML5/CSS3, and a zero-dependency dashboard is what makes the single-process, offline, Raspberry-Pi deployment real: FastAPI serves three static files. No `npm install`, no `node_modules`, nothing to build before a demo. |

---

## 9. What "done" means

From the assignment's own submission checklist:

- [ ] Git repository accessible to reviewers
- [x] `README.md` with setup instructions, architecture description, **policy-parsing approach**, and **severity-mapping rationale**
- [x] All 5 modules implemented and integrated as one pipeline
- [x] Detection functional across all 4 unsafe behaviour classes, with measured per-class
      metrics and honest known limitations (`docs/eval-baseline.md`, `docs/adr/0003`)
- [x] Severity assignment demonstrably derived from policy text, with per-rule traceability
- [x] Routing: LOW/MED -> log, HIGH/CRIT -> alert + log, correct on multi-violation clips
- [x] Reports auto-written with all 9 required fields (+3 extra), JSON/CSV/PDF
- [x] Dashboard: Live Feed Monitor + Alert Timeline Stream + Historical Log & Export
- [ ] Demo video walkthrough recorded and linked
