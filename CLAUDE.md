# CLAUDE.md

Working guide for Claude Code in this repository.

**Orientation:** `CONTEXT.md` = what & why · `IMPLEMENTATION_PLAN.md` = how & when ·
`HANDOVER.md` = the deployment runbook · this file = how to work here.

---

## What this project is

AegisFlow EHS ingests factory CCTV clips, parses an OHS policy PDF into machine-readable
rules, detects behavioural violations against those rules, assigns a severity tier, routes
each event (log vs. real-time alert), writes an immutable audit record, and displays it all in
an operations dashboard. All five modules are built and tested. See `CONTEXT.md`.

---

## The five rules that are easy to break

**1. Never hard-code a compliance rule or a severity tier.**
The assignment grades on this directly: behaviour classes *"must be derived from the policy
document through your parsing pipeline, not manually transcribed as hard-coded strings."*
Behaviour classes, observable indicators, section references and severity tiers come from
`artifacts/policy/rules.json`, which is produced by parsing `compliance_policy.pdf`. The
tables in `CONTEXT.md` are documentation for humans — code must not import them.

`config/settings.yaml` holds *engineering* knobs only (HSV bands, sample rate, contour areas).
If you find yourself adding a key named after a behaviour or a severity, stop: it belongs in
the policy parser.

**2. `offline` is the default and must always work.**
`AEGISFLOW_LLM_PROVIDER=offline` has to run the entire pipeline with zero network calls. The
Groq key is an enhancement. Never make an LLM call on a path that has no offline fallback, and
never let a test fail because a key is absent (mark it `@pytest.mark.llm`).

**3. Compliance records are immutable.**
`ViolationEvent` is a frozen Pydantic model and the DB layer is append-only. Do not add
`UPDATE` or `DELETE` paths to `db/crud.py`. Corrections are new rows, not edits. There is a
test asserting `crud` exposes no such function.

**4. Accuracy claims need a measurement, and detectors abstain rather than guess.**
Every threshold in `config/settings.yaml` traces to a curve from `scripts/calibrate.py` or
`scripts/sweep_thresholds.py`. If you change a detector, re-run
`python scripts/evaluate.py --split test --per-class 12` and update `docs/eval-baseline.md`
with the new numbers — including the ones that got worse. Three of the four original cues
were measured as non-separating and are documented as such in `docs/adr/0003` rather than
tuned around; the forklift cue is disabled by config because it produced only false
positives. A detector missing its commissioning data reports nothing and logs why. In an
audit trail, silence beats noise.

**5. Never resolve the camera or the zone from the clip's folder name at inference time.**
The folder is the ground-truth label. `detection/cameras.py` recovers camera identity from
the image by scene fingerprint; `core/zoning.py`'s path-based helper is for evaluation and
seeding only.

---

## Repository map

```
aegisflow-ehs/
├── CONTEXT.md, IMPLEMENTATION_PLAN.md, HANDOVER.md, CLAUDE.md, README.md
├── compliance_policy.pdf        # KMP-OHS-POL-001, the source of truth
├── config/
│   ├── settings.yaml            # CV/engineering knobs ONLY
│   └── zones.yaml               # camera + zone map, walkway polygons
├── data/
│   ├── raw/{train,test}/<class>/*.mp4   # 691 clips, ~9.4 GB, GITIGNORED
│   └── processed/               # detection + LLM response caches (gitignored)
├── docs/
│   ├── reference/               # the 4 source PDFs
│   ├── architecture.md, api-contract.md, eval-baseline.md
│   └── adr/                     # architecture decision records
├── src/aegisflow/
│   ├── core/        # enums, schemas, settings, zoning  <- shared contracts
│   ├── policy/      # Module 2a: PDF -> PolicyRuleSet (+ faithfulness gate)
│   ├── severity/    # Module 2b: rule + context -> tier
│   ├── detection/   # Module 1: clip -> DetectionRecord[]
│   │                 #   cameras.py  camera identity + commissioned regions
│   │                 #   vlm.py      VLM reading of the policy's indicators
│   ├── llm/         # provider abstraction: groq|gemini|offline
│   ├── escalation/  # Module 3: severity routing + in-process alert bus
│   ├── reports/     # Module 4: append-only JSONL/CSV/JSON + ReportLab PDF
│   ├── db/          # async SQLAlchemy models + append-only CRUD
│   ├── api/         # FastAPI app + WebSocket
│   ├── pipeline.py  # composes Modules 1-4
│   └── cli.py       # Typer entrypoint
├── frontend/        # Module 5: index.html + assets/{app.css,app.js}, no build step
├── tests/{unit,integration}/
├── scripts/         # setup, calibrate_{cameras,panel,regions}, calibrate,
│                    #   sweep_thresholds, evaluate, seed_db
├── artifacts/       # models/ (weights), policy/ (rules.json)
└── outputs/         # generated reports, annotated clips, eval results
```

**All five modules are built and tested.** Phase 2 (Aimen) is deployment only — Docker,
running it on the target hardware, the Groq key, the demo video. See `HANDOVER.md`.

---

## Commands

```bash
# Setup (Windows PowerShell)
powershell -ExecutionPolicy Bypass -File scripts/setup.ps1
cp .env.example .env            # then fill GROQ_API_KEY if you have one

# Commissioning (once per installation; needs the dataset)
python scripts/calibrate_cameras.py    # which camera is which, from the image
python scripts/calibrate_panel.py      # panel ROI + intensity baseline
python scripts/calibrate_regions.py    # walkway polygon per camera

# Policy (Module 2a)
python -m aegisflow policy parse          # PDF -> artifacts/policy/rules.json
python -m aegisflow policy show           # pretty-print the rule set

# Detection (Module 1)
python -m aegisflow detect "data/raw/test/0_safe_walkway_violation/0_te1.mp4" --evidence

# Pipeline & evaluation
python -m aegisflow run --split test
python scripts/evaluate.py --split test --per-class 12
python scripts/calibrate.py --split test        # per-class cue percentiles
python scripts/sweep_thresholds.py --cache      # threshold curves
python scripts/seed_db.py --synthetic     # 90 events, no dataset or weights needed
python scripts/seed_db.py --from-clips --annotate   # real pipeline + playback clips

# API + dashboard
python -m aegisflow serve                 # http://127.0.0.1:8000/docs

# Quality gate — run all of these before saying a step is done
pytest                                    # full suite
pytest -m "not slow and not llm"          # fast loop
ruff check src tests && black --check src tests && mypy
```

The console script `aegisflow` is also installed by `pip install -e .`; `python -m aegisflow`
always works regardless.

---

## Environment facts

Verified on this machine — do not assume otherwise:

- **Python 3.11.9**, Windows 11, **12 cores, 8 GB RAM, no GPU** (`torch 2.1.2+cpu`).
- All runtime dependencies are **already installed** on the system interpreter (ultralytics
  8.1.34, opencv 4.9.0, groq 0.37.1, fastapi 0.110, sqlalchemy 2.0.28, PyMuPDF 1.24.3,
  reportlab 4.1.0). `requirements.txt` pins these exact versions.
- `ffmpeg` and `nvidia-smi` are **not** on PATH. OpenCV's bundled decoder handles the clips;
  do not add an ffmpeg dependency.
- YOLOv8n weights download to `artifacts/models/` on first use (~6 MB). That is the only
  network call the offline path ever makes, and it happens once.
- **torch defaults to 1 thread here** (`OMP_NUM_THREADS=1` is set in the environment). The
  YOLO adapter raises it deliberately — that plus batching is worth 4.8x, and honouring the
  env var would silently cost it. Cap it with `detection.torch_threads` if you must.

Consequences: nano models only, one clip in memory at a time, sample at 4 fps and infer at
640 px. See `CONTEXT.md` section 7.

---

## Conventions

- **Typed everywhere.** Pydantic v2 for data crossing a module boundary; plain dataclasses are
  fine inside a module. `mypy` must pass.
- **Structured logging** via `core/logging.py` (rich console). No bare `print()` outside
  `cli.py`.
- **Config through `core/settings.py`.** Never read `os.environ` or open a YAML file directly
  in a module.
- **Paths through `pathlib`**, always relative to the repo root resolved by `core/settings.py`.
  This is a Windows machine; `/` in a string literal path will bite you.
- Line length 100, ruff + black, `snake_case` modules, `PascalCase` models.
- Docstrings on every public function: one line on *why*, not a restatement of the signature.
  Where a behaviour comes from the policy, cite the section (`# Policy 6.2: <=2 blocks safe`).

---

## Working style for this repo

- **Update the progress tracker** at the bottom of `IMPLEMENTATION_PLAN.md` when a step lands.
  It is how the other developer sees where things stand.
- **Detection accuracy claims need numbers.** If you change a detector, re-run
  `python scripts/evaluate.py --split test --per-class 12
python scripts/calibrate.py --split test        # per-class cue percentiles
python scripts/sweep_thresholds.py --cache      # threshold curves` and update `docs/eval-baseline.md`. Do not
  describe a detector as "working" without a per-class precision/recall figure.
- **Never commit** `data/raw/`, `*.db`, `.env`, or model weights — `.gitignore` covers these,
  keep it that way.
- **Honest limitations beat inflated metrics.** The assignment explicitly does not require
  perfect accuracy but does require documented model-selection rationale and known
  limitations. A README that admits panel detection is weak reads better than one that claims
  95% and cannot show it.
- When a design choice is non-obvious (why classical CV over a fine-tuned model, why SQLite,
  why React), write a short ADR in `docs/adr/` rather than burying it in a commit message.
