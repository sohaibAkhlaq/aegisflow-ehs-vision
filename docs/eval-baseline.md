# Evaluation Baseline

Measured, reproducible, and honest about what does not work. Every figure below comes from
a script in this repo; nothing here is estimated.

```bash
python scripts/evaluate.py --split test --per-class 12 --json outputs/eval/offline_final.json
```

**Headline:** on the test split, offline, the system detects the four unsafe behaviours at a
**macro F1 of 0.26**. That is a modest number and it is the real one. The rest of this
document explains where it comes from, which parts work, and why the parts that do not work
are disabled rather than tuned until they look better.

---

## Method

- **Ground truth is the class folder name.** The dataset carries no bounding-box annotations,
  so evaluation is at **clip level**, not box level.
- A clip in behaviour *X*'s folder is a positive for *X*. A clip in **any other** folder that
  triggers *X*'s detector is a false positive.
- The four safe folders are the most informative negatives, because each is the compliant
  counterpart of an unsafe class filmed on the same camera — exactly the confusion that
  matters operationally.
- Reported **per class, never blended**. The classes are ~9:1 imbalanced, so a single accuracy
  figure would be dominated by walkway violations and hide everything else.
- Commissioning artefacts are built from the **train** split only. The test split is untouched
  by calibration.

---

## Results — offline provider, test split, 87 clips

| Behaviour | Positives | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Opened Panel Cover | 12 | 8 | 10 | 4 | 0.44 | 0.67 | **0.53** |
| Safe Walkway Violation | 12 | 5 | 10 | 7 | 0.33 | 0.42 | **0.37** |
| Unauthorized Intervention | 11 | 1 | 3 | 10 | 0.25 | 0.09 | **0.13** |
| Carrying Overload with Forklift | 8 | 0 | 0 | 8 | — | 0.00 | **0.00** |
| **Macro average** | | | **23** | | | | **0.26** |

### Where the false positives come from

| Detector fired | On clips actually labelled | Count |
|---|---|---:|
| Opened Panel Cover | `authorized_intervention` (7), `unauthorized_intervention` (3) | 10 |
| Safe Walkway Violation | `unauthorized_intervention` (5), `carrying_overload` (2), `safe_carrying` (2), `authorized_intervention` (1) | 10 |
| Unauthorized Intervention | `opened_panel_cover` (2), `carrying_overload` (1) | 3 |

Note the pattern: the panel detector's false positives are almost all on
`authorized_intervention` clips, which are filmed on the **same camera** as the panel clips.
Those are frames where a panel genuinely may be open while an authorised intervention is in
progress — some fraction of them are arguably correct detections that the clip-level label
cannot credit.

### Throughput

| Metric | Value |
|---|---|
| Mean per clip | **5.4 s** (87 clips, 3014 frames analysed) |
| Per frame | ~98 ms (CPU, 12 threads, batch 8, 640 px) |
| Full 691-clip run | ~35 min, estimated from the above |
| VLM calls | 0 (offline) |

Inference throughput improved **4.8x** during development, from 471 ms/frame to 98 ms/frame.
Two changes were responsible: torch defaults to a single thread in this environment
(`OMP_NUM_THREADS=1`), and batching only pays off once that is raised. Neither helps alone.

---

## Progress across iterations

| Run | Macro F1 | Total FPs | What changed |
|---|---:|---:|---|
| Baseline v1 | 0.225 | 93 | First working detectors, hand-picked thresholds |
| v2 | 0.259 | 27 | Camera identification, commissioned panel ROI and walkway polygon, positive-evidence-only vest rule |
| **Final** | **0.259** | **23** | Forklift offline cue disabled after measuring it as anti-correlated |

The headline number barely moved between v2 and final; the false-positive count fell by 75%
from the baseline. That is the trade that was deliberately made — see below.

---

## What works, and what does not

### Vest colour (Section 4.2) — the cue works

| Class | Green ratio in torso ROI (p75) |
|---|---|
| `5_authorized_intervention` | **0.397** |
| `1_unauthorized_intervention` | 0.013 |
| every other class | ~0.000 |

A wide, clean gap; the configured threshold of 0.12 sits in the middle of it. This cue
answers *"is this person authorised?"* very reliably.

What it cannot answer is *"is an intervention happening at all?"* — there is no equipment
detector, and no cue for human-machine contact. Firing on the absence of green measured at
precision **0.11**, so the detector now requires a positively identified red-black vest.
That lifted precision to 0.25 and dropped recall to 0.09. **This is the single largest
accuracy limitation in the offline system**, and it is the one the VLM path most directly
fixes.

### Panel state (Section 5.2.2) — works, once commissioned

Whole-frame statistics fail, and fail *backwards*:

| Cue (frame-level) | `2_opened_panel_cover` p50 | `6_closed_panel_cover` p50 |
|---|---|---|
| vertical edge strength | 0.152 | 0.166 |
| dark region ratio | 0.085 | 0.102 |

The open class scores **lower** on both, because those statistics describe the scene, not the
panel. Looking at the panel itself works: `scripts/calibrate_panel.py` locates the region of
maximum open-vs-closed separation on the train split.

| Property | Value |
|---|---|
| ROI separation | **3.28** pooled standard deviations |
| Open / closed mean intensity | 74.1 / 103.9 grey levels |
| Threshold | 89.0 (open is *darker* — the cover exposes an unlit cavity) |
| Isolated pair test (open vs closed only) | **P 1.00 / R 0.69** |
| Full 1-vs-rest test | P 0.44 / R 0.67 |

### Walkway boundary (Section 3.2) — weak, improved by commissioning

Live green-line segmentation does not recover the walkway: the largest green region in a
frame is one painted *line*, not the corridor between lines. 89% of frames in *compliant*
clips place at least one foot point outside it.

| Approach | Best F1 |
|---|---|
| Live green-line segmentation, margin swept 0.02–0.26 × frame width | **0.25** (flat — no operating point) |
| Per-camera commissioned polygon (`calibrate_regions.py`) | **0.38** |

Per-camera matters: pooling both views yields a hull covering 67% of the frame, which excludes
nobody. CAM-01's own hull is 13.9%. The detector **abstains** on cameras without a usable
polygon rather than falling back to the segmentation cue.

### Forklift block count (Section 6.2) — rejected, anti-correlated

| Cue | `3_carrying_overload` | `7_safe_carrying` |
|---|---|---|
| contour block count (p50 / p75) | 2 / 2 | 2 / **3** |
| load-region fill ratio (p50) | 0.545 | 0.523 |
| vehicle aspect ratio (p50) | 0.916 | 1.012 |
| vehicle area fraction (p50) | 0.093 | 0.113 |

The *compliant* class registers **more** detected blocks than the overloaded one. Best across
thresholds 2–5 and persistence 2–6: F1 0.31, and only at a threshold of 2 — which by the
policy's own definition is the compliant state.

The detector is therefore **disabled offline** (`forklift.offline_detection_enabled: false`).
Before that change it contributed 4 false positives and 0 true positives: a detector that only
ever lies. Abstaining is strictly better, and the VLM path covers the class properly.

---

## The tempting shortcut we did not take

Applying the panel detector's discriminative-window method to all four class pairs:

| Pair | Best window | Test F1 |
|---|---|---|
| walkway violation vs safe walkway | x=392 y=160 | 0.78 |
| unauthorized vs authorized | x=392 y=104 | 0.80 |
| open vs closed panel | x=64 y=208 | 0.82 |
| overload vs safe carrying | **x=560 y=0** | **1.00** |

A macro F1 around 0.85 was available here — more than three times the number reported above.
It is not used, for two reasons:

1. **It is measuring the wrong thing.** The forklift pair's best window is an *empty corner of
   the frame*. It separates recording sessions and lighting, not behaviour. The perfect score
   is the tell, not the achievement.
2. **It is not policy-traceable.** The assignment requires that *"the observable indicators
   used to classify a behavior as safe or unsafe must be traceable to the relevant policy
   section."* A learned intensity window cites no section.

The panel ROI is a narrower, deliberate exception: the window locates *where the panel is* —
commissioning information, like the walkway polygon — while the open/closed decision remains
the policy's own indicator.

---

## Known limitations

1. **Forklift overload is not detected offline.** Disabled by config after measurement. Needs
   the VLM path, or footage where block boundaries are separable.
2. **Unauthorized intervention has low recall (0.09) offline.** The vest cue is excellent at
   authorised-vs-unauthorised but cannot establish that an intervention is occurring.
3. **Walkway detection needs a commissioned polygon per camera** and abstains without one.
   CAM-02's polygon is currently too permissive (63% of frame) because few compliant clips
   were filmed there; the detector abstains on that camera and says so in the logs.
4. **Panel detection is camera-specific.** It verifies camera identity by scene fingerprint
   before applying its calibration, and abstains on unknown views.
5. **Clip-level evaluation only.** No bounding-box ground truth exists, so localisation
   quality is unmeasured.
6. **Some "false positives" may be correct.** A clip labelled `authorized_intervention` can
   legitimately contain an open panel; the single-label ground truth cannot represent that.
7. **YOLOv8n reports large static machinery as `truck`/`car`.** Handled by an area filter
   (real forklifts occupy 0.04–0.17 of the frame, static machines 0.24–0.30), but the class
   assignment remains unreliable.

---

## Reproducing all of this

```bash
python scripts/calibrate.py --split test --per-class 6      # per-class cue percentiles
python scripts/sweep_thresholds.py --cache --per-class 12   # threshold curves
python scripts/calibrate_cameras.py                         # camera clustering quality
python scripts/calibrate_panel.py                           # panel ROI separation score
python scripts/calibrate_regions.py                         # walkway polygon coverage
python scripts/evaluate.py --split test --per-class 12
```

## Results — Groq provider, test split, 87 clips

Same clips, same commissioning artefacts, same thresholds. The only change is that the VLM
becomes available as a detection path, so ambiguous frames get read instead of abstained on.

```bash
AEGISFLOW_LLM_PROVIDER=groq GROQ_API_KEY=... \
  python scripts/evaluate.py --split test --per-class 12 --json outputs/eval/groq_final.json
```

| Behaviour | Positives | TP | FP | FN | Precision | Recall | F1 | vs. offline |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Opened Panel Cover | 12 | 8 | 10 | 4 | 0.44 | 0.67 | **0.53** | — |
| Safe Walkway Violation | 12 | 7 | 12 | 5 | 0.37 | 0.58 | **0.45** | +0.08 |
| Carrying Overload with Forklift | 8 | 2 | 0 | 6 | 1.00 | 0.25 | **0.40** | +0.40 |
| Unauthorized Intervention | 11 | 1 | 9 | 10 | 0.10 | 0.09 | **0.10** | −0.04 |
| **Macro average** | | | **31** | | | | **0.37** | **+0.11** |

**Macro F1 rises from 0.26 to 0.37** — a 43% relative gain, at the cost of 261 VLM calls and
~47% more wall-clock (7.9 s/clip vs 5.4 s). Read it per class, because the average hides the
interesting part:

- **Forklift overload goes from 0.00 to 0.40, at precision 1.00.** This is the whole argument
  for the VLM path. The contour cue was measured anti-correlated and is disabled offline
  (ADR 0003), so the class scores exactly zero without a provider. The VLM recovers 2 of 8
  positives and — importantly — fires on nothing else. It abstains rather than guesses.
- **Walkway improves modestly** (0.37 → 0.45): +2 TP for +2 FP.
- **Unauthorized intervention gets slightly worse** (0.13 → 0.10). The VLM trades 2 FP for
  6 more, without recovering a single extra positive. The vest cue is the weakest detector
  either way and the VLM does not rescue it.

So the honest summary is: the provider path is worth having for one class, neutral-to-helpful
for a second, and mildly harmful for a third. It is not a general uplift.

Both numbers are kept rather than one replacing the other, because offline is what runs when
the network is down.
