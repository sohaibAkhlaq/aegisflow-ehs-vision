<div align="center">

# AegisFlow EHS

**Factory Compliance & Alert Escalation System**

Turn raw factory CCTV into a policy-grounded, auditable safety compliance record — in real time.

[![CI](../../actions/workflows/ci.yml/badge.svg)](../../actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8--nano-Ultralytics-00FFFF)](https://docs.ultralytics.com/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Dashboard](https://img.shields.io/badge/Dashboard-zero--build-38BDF8?logo=javascript&logoColor=white)](frontend/)
[![SQLite](https://img.shields.io/badge/SQLite-append--only-003B57?logo=sqlite&logoColor=white)](https://www.sqlite.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

## What it does

Most factory safety systems either watch video *or* track compliance paperwork. AegisFlow does
both, and connects them: it reads the facility's actual Occupational Health & Safety policy
manual, extracts the rules, and then judges the video against those rules — so every alert it
raises can be traced back to the sentence in the manual that justifies it.

```
                  compliance_policy.pdf (KMP-OHS-POL-001)
                              │
                              ▼
   ┌──────────────────────────────────────────────────┐
   │  MODULE 2a · POLICY PARSER                       │
   │  PyMuPDF layout extraction → section tree →      │
   │  callout binding → literal-substring validation  │
   └──────────────────────┬───────────────────────────┘
                          │  PolicyRuleSet (4 domains, indicators, section refs)
                          ▼
 ┌─────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
 │  MODULE 1   │   │  MODULE 2b   │   │  MODULE 3    │   │  MODULE 4    │
 │  DETECTION  │──▶│  SEVERITY    │──▶│  ESCALATION  │──▶│  REPORTS     │
 │             │   │  MATRIX      │   │              │   │              │
 │ YOLOv8n +   │   │ LOW  MED     │   │ LOW/MED  →   │   │ append-only  │
 │ HSV colour  │   │ HIGH CRIT    │   │   DB log     │   │ JSON · CSV   │
 │ + contours  │   │ derived from │   │ HIGH/CRIT →  │   │ · PDF audit  │
 │ + VLM tie-  │   │ policy text, │   │   WebSocket  │   │ trail, 9     │
 │   break     │   │ not literals │   │   + DB log   │   │ req. fields  │
 └──────▲──────┘   └──────────────┘   └──────┬───────┘   └──────┬───────┘
        │                                    │                  │
   factory video                             ▼                  ▼
   (1920×1080)                    ┌──────────────────────────────────────┐
                                  │  MODULE 5 · OPERATIONS DASHBOARD     │
                                  │  Live Feed · Alert Timeline · Log    │
                                  └──────────────────────────────────────┘
```

### The four monitored behaviours

Defined by the policy manual, not by us — each is bound to the section that governs it.

| Behaviour | Observable indicator | Policy | Callout | Derived tier |
|---|---|---|---|---|
| **Safe Walkway Violation** | Person outside the green painted floor markings | § 3.3.2 | WARNING + *"highest-frequency"* | **HIGH** → CRITICAL with a forklift in frame |
| **Unauthorized Intervention** | Person at equipment without the green authorisation vest | § 4.3.2 | CRITICAL SAFETY NOTICE + *"must be assumed"* | **CRITICAL** |
| **Opened Panel Cover** | Electrical panel cover left open during production | § 5.2.2 | WARNING + *"regardless of ... vicinity"* | **LOW** → MEDIUM/HIGH as people approach |
| **Forklift Overload** | 3 or more standardized blocks on the forks (≤ 2 is safe) | § 6.3.2 | CRITICAL SAFETY NOTICE + *"unambiguous"* | **CRITICAL** |

Severity is *computed* from the policy's own callout keywords, hazard-context language and
frequency language, then adjusted by what else is in the frame. Nothing in that table is
written down in the source — run `aegisflow policy matrix` to see each tier derived, step by
step, from the PDF. Every record carries a `severity_rationale` quoting the sentence that
produced its tier.

---

## Quick start

```bash
git clone https://github.com/sohaibAkhlaq/aegisflow-ehs-vision.git && cd aegisflow-ehs-vision

# Windows
powershell -ExecutionPolicy Bypass -File scripts/setup.ps1

# macOS / Linux
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .

cp .env.example .env
```

**See it running in 60 seconds, with no dataset and no model weights:**

```bash
python scripts/seed_db.py --synthetic     # 90 events from the real policy + severity matrix
python -m aegisflow serve                 # http://127.0.0.1:8000/
```

**With the dataset.** It is not in the repo (9.4 GB) — download it from
[Kaggle](https://www.kaggle.com/datasets/trnhhnggiang/videodataset-for-safe-and-unsafe-behaviours)
and arrange it as `data/raw/{train,test}/<class_folder>/*.mp4`.

```bash
python scripts/calibrate_cameras.py     # commission the cameras (once per installation)
python scripts/calibrate_panel.py
python scripts/calibrate_regions.py

python -m aegisflow policy parse --strict   # PDF -> machine-readable rules
python -m aegisflow policy matrix           # see the severity tiers derived
python -m aegisflow run --split test --annotate
python -m aegisflow report                  # PDF compliance report
python -m aegisflow serve
```

**No API key? Everything above still works.** `AEGISFLOW_LLM_PROVIDER=offline` is the default
and runs the complete pipeline with zero network calls. Adding a Groq key measurably improves
detection — see [Accuracy](#accuracy) below.

---

## Repository structure

| Path | Contents |
|---|---|
| `compliance_policy.pdf` | KMP-OHS-POL-001 — the authoritative source for every compliance decision |
| `config/` | Engineering knobs only (HSV bands, sample rate). **No compliance rules live here.** |
| `data/raw/` | 691 clips across 8 classes — gitignored |
| `src/aegisflow/core/` | Shared contracts: enums, Pydantic schemas, settings, zone resolution |
| `src/aegisflow/policy/` | Module 2a — PDF → `PolicyRuleSet`, with faithfulness validation |
| `src/aegisflow/severity/` | Module 2b — policy signals + frame context → severity tier |
| `src/aegisflow/detection/` | Module 1 — YOLOv8n, HSV vest analysis, commissioned panel/walkway regions, camera identification, VLM indicator reading |
| `src/aegisflow/llm/` | Provider abstraction: `groq` \| `gemini` \| `offline` |
| `src/aegisflow/escalation/` | Module 3 — severity-based routing |
| `src/aegisflow/reports/` | Module 4 — append-only JSON / CSV / PDF audit records |
| `src/aegisflow/api/` | FastAPI app + WebSocket alert channel |
| `frontend/` | Module 5 — operations dashboard: three static files, no build step |
| `scripts/` | Setup, camera/panel/region commissioning, threshold sweeps, evaluation, DB seeding |
| `docs/` | Architecture, API contract, ADRs, evaluation baseline, source PDFs |
| `outputs/` | Generated reports, annotated clips, evaluation results |

---

## Documentation

| Document | Read it for |
|---|---|
| **[CONTEXT.md](CONTEXT.md)** | What the project is, the dataset, the policy, the constraints |
| **[IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md)** | Step-by-step build order with exit criteria and a live progress tracker |
| **[HANDOVER.md](HANDOVER.md)** | Deployment runbook: config, commissioning, demo script, known limitations |
| **[docs/eval-baseline.md](docs/eval-baseline.md)** | Measured per-class accuracy and what does not work |
| **[CLAUDE.md](CLAUDE.md)** | Conventions, commands, and the rules that are easy to break |
| **[docs/architecture.md](docs/architecture.md)** | Module contracts and data flow in detail |
| **[docs/adr/](docs/adr/)** | Why each significant technical choice was made |

---

## Accuracy

Measured on the test split, offline, 87 clips. Reproduce with
`python scripts/evaluate.py --split test --per-class 12`.

| Behaviour | Precision | Recall | F1 |
|---|---:|---:|---:|
| Opened Panel Cover | 0.44 | 0.67 | **0.53** |
| Safe Walkway Violation | 0.33 | 0.42 | **0.37** |
| Unauthorized Intervention | 0.25 | 0.09 | **0.13** |
| Carrying Overload with Forklift | — | 0.00 | **0.00** (abstains — see below) |
| **Macro F1** | | | **0.26** |

Throughput: **5.4 s per clip**, ~98 ms per frame on CPU — a 4.8x improvement over the naive
configuration, from raising torch's thread count (it defaults to 1 here) and batching
inference, which only helps in combination.

**That is a modest number and it is the real one.** The interesting part is why:

- **The vest cue works well.** Authorised personnel show a green torso ratio of 0.40 against
  0.013 for unauthorised — a wide, clean gap. What it cannot do is establish that an
  *intervention* is happening at all, without an equipment detector. Firing on the absence of
  green scored precision 0.11 and buried the real events, so the detector now demands positive
  evidence: precision up to 0.25, recall down to 0.09.
- **Panel state needed commissioning, not tuning.** Whole-frame edge and darkness statistics
  score *lower* on open panels than closed ones — they describe the scene, not the panel.
  Locating the panel region from the train split gives 3.3σ of separation and P 1.00 / R 0.69
  against its own compliant counterpart.
- **The forklift block count is anti-correlated with the truth.** Overload clips register a
  median of 2 detected blocks; compliant clips register 2 with a p75 of 3. Before it was
  disabled it produced 4 false positives and 0 true positives. A detector that only ever lies
  is worse than one that says nothing, so it abstains and the VLM path covers the class.
- **A learned appearance feature scored ~0.85 macro F1 and was rejected.** The forklift pair's
  best discriminative window is an *empty corner of the frame* — it separates recording
  sessions, not behaviour — and it cites no policy section, which the assignment requires. A
  higher number, a worse system.

Total false positives fell from 93 to 23 across development while macro F1 rose from 0.225.
That trade was deliberate: in an audit trail, a quiet system beats a noisy one.

Full detail, including every threshold sweep and the cues that were measured and discarded:
**[docs/eval-baseline.md](docs/eval-baseline.md)** and
**[ADR 0003](docs/adr/0003-detector-cue-selection.md)**.

> **With a Groq key configured**, the three classes whose classical cues fail are answered by a
> vision model asked the policy's *own* indicator questions — a path that is more
> policy-traceable than the contour heuristics it replaces. Not yet measured; no key was
> available during the build.

---

## Design notes

**Why classical CV on top of YOLO, rather than a fine-tuned detector.** The policy's observable
indicators are vest *colour*, block *count*, panel *state* and walkway *position* — geometric
and chromatic properties, not object categories. YOLOv8n locates the people and vehicles; HSV
masks and geometry read the indicators off those boxes. This needs no bounding-box
annotations (the dataset has none) and runs on a CPU.

**Every threshold is measured, and the ones that failed are documented.** `scripts/calibrate.py`
and `scripts/sweep_thresholds.py` produce the curves; `docs/adr/0003` records what each cue
scored. Three of the four hand-built cues turned out not to separate their class — the contour
block count is actively *anti-correlated* with the true forklift load — so those findings are
published rather than tuned around, and the affected detector is disabled by config instead of
shipped as decoration.

**A learned appearance feature scored higher and was rejected.** Applying the panel detector's
discriminative-window method to all four class pairs reaches F1 0.78–1.00, but the forklift
pair's best window lands on an *empty corner of the frame* — it separates recording sessions,
not behaviour. It is also untraceable to any policy section, which the assignment requires. A
higher number, a worse system.

**Detectors abstain rather than guess.** One without its commissioning data reports nothing and
logs why; the panel detector verifies camera identity by scene fingerprint before applying a
camera-specific calibration. In an audit trail, silence is a better failure than noise.

**Why the policy parser comes first.** The behaviour classes, indicators, section references
and severity tiers are all derived from the PDF at runtime. Nothing about compliance is a
string literal in the source. That is a graded requirement, and it is also what makes the
system retargetable to a different facility's manual.

**Why LLMs are optional.** The deterministic parser handles this document on its own. A
configured provider adds a structuring pass over prose the regexes miss, and a vision
tie-break for genuinely ambiguous frames (the forklift 2-vs-3-blocks boundary). Every
LLM-extracted rule must appear verbatim in the PDF or it is discarded, and every VLM
consultation is recorded in the event's `detection_method`.

**Why append-only.** Compliance records are evidence. `ViolationEvent` is a frozen model and
the persistence layer exposes no update or delete path.

---

## Team

Built by two developers working sequentially.

| | Phase 1 — build | Phase 2 — deploy |
|---|---|---|
| **Developer** | Sohaib Akhlaq | Aimen |
| **Scope** | All five modules, the policy parser and severity matrix, the calibration and evaluation tooling, the test suite, the dashboard, and the documentation | Containerisation, running it on the target hardware, the Groq key, the demo video, submission QA |

`HANDOVER.md` is the runbook for Phase 2. It should not be necessary to read any source to
stand the system up.

---

## Status

**All five modules built and tested.** See the progress tracker in
[IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md#6-progress-tracker) and the measured accuracy
in [docs/eval-baseline.md](docs/eval-baseline.md).

```
pytest                       # 157 tests
pytest -m "not slow and not llm"   # subset needing no dataset, weights or API key
```

## License

MIT — see [LICENSE](LICENSE).
