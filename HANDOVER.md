# HANDOVER — Sohaib (build) → Aimen (deployment)

> **The system is complete, tested and running.** All five modules are built, 157 tests pass,
> and the dashboard works. This is a deployment runbook, not a to-do list. You should not need
> to read any source code to stand it up.

---

## 0. Status at a glance

### Done — nothing here needs your attention

| | Built | Evidence |
|---|---|---|
| **Module 1** Detection Engine | YOLOv8n + 4 behaviour detectors + camera identification + 3 commissioning scripts | `aegisflow detect <clip>` |
| **Module 2** Policy parser + severity matrix | All 4 rules derived from the PDF; strict validation passes with 0 warnings | `aegisflow policy show` / `policy matrix` |
| **Module 3** Escalation | LOW/MED → DB log; HIGH/CRIT → DB log + WebSocket | 30 integration tests |
| **Module 4** Reports | Append-only JSONL + CSV + per-event JSON + ReportLab PDF | `outputs/reports/` |
| **Module 5** Dashboard | 3 required views + a policy panel, zero build step | `aegisflow serve` |
| Test suite | **157 tests**, green | `pytest` |
| Code quality | `ruff` and `black` clean | `ruff check src tests scripts` |
| Docs | README, CONTEXT, architecture, API contract, 3 ADRs, evaluation baseline | `docs/` |
| Commissioning | Camera registry, panel baseline, walkway polygons all calibrated | `artifacts/models/` |

### Left — yours

| # | Task | Effort | Blocked by |
|---|---|---|---|
| **D1** | Get your own Groq API key and put it in `.env` | 5 min | — **do this first**, see §4 |
| **D2** | Dockerfile + compose | ~1 h | — |
| **D3** | Run the full dataset and seed the demo database | ~35 min unattended | D1 |
| **D4** | Record the demo video | ~30 min | D3 |
| **D5** | Push to GitHub, verify reviewer access, tick the submission checklist | 20 min | all |

Nothing on that list requires changing application code.

---

## 1. Ten-minute start

```bash
git clone https://github.com/sohaibAkhlaq/aegisflow-ehs-vision.git && cd aegisflow-ehs-vision

# Windows
powershell -ExecutionPolicy Bypass -File scripts/setup.ps1

# macOS / Linux
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .

cp .env.example .env
python -m aegisflow info                  # confirm the toolchain
python scripts/seed_db.py --synthetic     # 90 events, no video needed
python -m aegisflow serve                 # http://127.0.0.1:8000/
```

`pip install -e .` matters — without it `python -m aegisflow` reports *"No module named
aegisflow"*. `setup.ps1` does it for you.

`--synthetic` seeds a realistic spread across all four severity tiers using the **real** parsed
policy and the **real** severity matrix; only the detections are fabricated. Enough to demo
every dashboard view on a machine with no dataset and no model weights.

---

## 2. What runs where

One process serves everything: REST API, WebSocket alert channel, and the dashboard. There is
no separate frontend server and no build step — the dashboard is three static files
(`frontend/index.html`, `assets/app.css`, `assets/app.js`) that FastAPI serves directly. No
`npm install`, nothing to compile.

| Surface | URL |
|---|---|
| Operations dashboard | `http://HOST:PORT/` |
| API documentation | `http://HOST:PORT/docs` |
| OpenAPI schema | `http://HOST:PORT/openapi.json` |
| Live alert channel | `ws://HOST:PORT/ws/alerts` |

Bind address comes from `.env` (`AEGISFLOW_API_HOST`, `AEGISFLOW_API_PORT`), or
`aegisflow serve --host 0.0.0.0 --port 8000`.

---

## 3. Configuration

Everything is in `.env` (copy from `.env.example`). The settings that actually matter:

| Variable | Default | Notes |
|---|---|---|
| `AEGISFLOW_LLM_PROVIDER` | `offline` | `offline` \| `groq` \| `gemini` |
| `GROQ_API_KEY` | empty | **See section 4** — this materially improves detection |
| `AEGISFLOW_DB_URL` | `sqlite+aiosqlite:///./aegisflow.db` | Postgres is a URL change, no code change |
| `AEGISFLOW_API_HOST` / `_PORT` | `127.0.0.1` / `8000` | use `0.0.0.0` to expose on a LAN |
| `AEGISFLOW_DEVICE` | `cpu` | `cuda:0` if a GPU is available |
| `AEGISFLOW_SAMPLE_FPS` | `4` | frames analysed per second of video |

A missing or invalid key is not an error: `build_provider()` logs the reason and falls back to
`offline`. The system has no configuration that prevents it from starting.

---

## 4. The Groq key — get your own

**Do not ask Sohaib for his key, and do not look for one in the repo.** `.env` is gitignored
and holds nothing you will receive. Secrets are never shared through a repository; each
developer uses their own.

Get a free one at **https://console.groq.com/keys**, then:

```bash
# .env  (this file is gitignored - never commit it)
AEGISFLOW_LLM_PROVIDER=groq
GROQ_API_KEY=<your own key>
```

Verify before you rely on it:

```bash
python -m aegisflow info --check-llm
```

That probes both models and does a real round-trip, including sending an image. Expect:

```
| text model     | openai/gpt-oss-120b yes |
| text call      | works                   |
| vision model   | qwen/qwen3.8-27b yes    |
| vision call    | works (saw 'Green')     |
```

### ⚠ Model ids go stale — this bit you already

Groq's catalogue changes and **availability differs per account**. The original defaults
(`llama-3.3-70b-versatile`, `meta-llama/llama-4-scout-...`) returned 404 on this account.
Worse, **most Groq models cannot accept images at all** — of the 14 models available here,
only `qwen/qwen3.8-27b` and `qwen/qwen3.6-27b` are vision-capable.

That is why model ids live in `.env` and not in code, and why `--check-llm` exists. **Run it
before the demo.** A stale id surfaces as a 404 on the first clip, mid-presentation.

If the vision model is unavailable on your account, the system does not break — it logs the
failure and falls back to the classical detectors. You simply lose the accuracy gain.

---

## 5. Processing real footage

```bash
# 1. Commission the cameras (once per installation, needs the dataset)
python scripts/calibrate_cameras.py          # -> artifacts/models/camera_registry.json
python scripts/calibrate_panel.py            # -> artifacts/models/panel_baseline.json
python scripts/calibrate_regions.py          # -> artifacts/models/region_baseline.json

# 2. Parse the policy (once per policy document)
python -m aegisflow policy parse --strict
python -m aegisflow policy matrix            # see the derived severity tiers

# 3. Run the pipeline
python -m aegisflow run --split test --annotate
python -m aegisflow report                   # PDF compliance report
```

**The commissioning step is not optional.** Detectors that lack their commissioning data
abstain and log why, rather than guessing — so skipping it produces a quiet system, not a
wrong one. `aegisflow info` shows what is commissioned.

`--annotate` renders playback clips with severity overlays into `outputs/annotated/` and
registers them for the dashboard's Live Feed Monitor. Without it, View A has nothing to play.

---

## 6. Verifying a deployment

```bash
pytest                                  # full suite
pytest -m "not slow and not llm"        # fast subset, no dataset or weights needed
ruff check src tests && black --check src tests
python -m aegisflow info
curl -s localhost:8000/api/health | python -m json.tool
```

`/api/health` reports provider, policy and database state in one object — the right thing for
a container healthcheck.

---

## 7. Your task list

| # | Task | Notes |
|---|---|---|
| D1 | Add **your own** `GROQ_API_KEY` to `.env`, run `info --check-llm`, re-run the evaluation | Section 4. Record the numbers in `docs/eval-baseline.md`. |
| D2 | Dockerfile + compose | Single service. Mount `data/`, `outputs/`, `artifacts/` and the SQLite file as volumes. `python -m aegisflow serve --host 0.0.0.0`. |
| D3 | Run the full dataset once and seed the demo database | `aegisflow run --split test --annotate` |
| D4 | Record the demo video | Section 8 has a suggested script. |
| D5 | Final submission QA | Checklist in `CONTEXT.md` section 9. |

Nothing on this list requires changing application code.

---

## 8. Suggested demo script

Roughly six minutes, and it hits every graded requirement in order:

1. **Policy parsing** — `aegisflow policy parse --strict`, then `aegisflow policy show`.
   Point out that the four behaviour classes, their indicators and their section references
   were all extracted from the PDF, and that each was checked against the source text.
2. **Severity derivation** — `aegisflow policy matrix`. Show the derivation column: callout
   keyword, then the hazard-context phrase that moved the tier. Nothing hard-coded.
3. **Detection on one clip** — `aegisflow detect <clip> --evidence`. Show the raw cues.
4. **The pipeline** — `aegisflow run --split test --per-class 3 --annotate`.
5. **The dashboard** — `aegisflow serve`, then walk the three views:
   - *Live Feed Monitor*: pick a CRITICAL clip, show the severity border and the alert banner.
   - *Alert Timeline*: leave it open while a run is in progress so events stream in live.
   - *Historical Log*: filter by severity and behaviour, then export CSV.
   - *Policy panel*: the differentiator — every rule, its source quote, and how its tier was derived.
6. **The audit trail** — `outputs/reports/audit_log.csv` and the generated PDF.

Be straightforward about the accuracy numbers. `docs/eval-baseline.md` states them plainly,
along with what does not work and why; the assignment asks for documented limitations, and a
candid account of a measured failure reads far better than a number nobody can reproduce.

---

## 9. Interfaces, if you do need to change something

### `ViolationEvent` — the seam (`src/aegisflow/core/schemas.py`)

Frozen. The nine assignment-mandated fields plus `confidence`, `detection_method` and
`severity_rationale`. To change a field on a copy, use `model_copy(update={...})`; the
escalation router does exactly that to record `escalation_action`.

### Swapping a module

Each module sits behind a protocol in `src/aegisflow/core/protocols.py`:

| Protocol | Implemented by | Swap it to |
|---|---|---|
| `EscalationSink` | `escalation/router.py` | route to SMS, email, a message queue |
| `ReportWriter` | `reports/writers.py` | write to S3, a document store |
| `AlertPublisher` | `escalation/bus.py` | Redis pub/sub for multi-process deployments |

`pipeline.py` takes `sink` and `writer` as constructor arguments, so a replacement is an
injection, not a refactor.

### Database

Append-only by construction: `db/crud.py` exposes no update or delete path for compliance
records, and there is a test asserting that. Adding one would break the audit-trail guarantee.

---

## 10. Known limitations you should be able to speak to

Full detail in [`docs/eval-baseline.md`](docs/eval-baseline.md) and
[`docs/adr/0003`](docs/adr/0003-detector-cue-selection.md).

- **Forklift overload does not work offline.** Contour block counting is anti-correlated with
  the true load on this footage, so it is disabled by config and the class relies on the VLM.
- **Unauthorized intervention has low offline recall.** The vest colour cue is excellent at
  telling authorised from unauthorised; what it cannot do is tell that an *intervention* is
  happening at all, without an equipment detector.
- **Walkway detection needs a commissioned polygon per camera** and abstains without one.
- **Panel detection is camera-specific** and verifies camera identity by scene fingerprint
  before applying its calibration.
- Class folders are the ground truth, so all evaluation is clip-level, not box-level.
