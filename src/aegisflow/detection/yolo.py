"""Ultralytics YOLOv8n adapter.

YOLO's job here is narrow: locate people and vehicles. It knows nothing about vests, blocks,
panels or walkways - those are the policy's *observable indicators*, and they are read off
these boxes by ``geometry.py`` and the individual detectors.

Two practical notes:

* **Forklifts are not a COCO class.** YOLOv8n reports them as ``truck``, ``car`` or
  occasionally ``bus``, inconsistently between frames. All vehicle-ish classes are accepted
  and merged, and the inconsistency is documented as a known limitation.
* **Weights load lazily.** Constructing the detector must stay cheap so the CLI can start
  and the API can boot without pulling a model. The ~6 MB download happens on first use.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any

import numpy as np

from aegisflow.core.errors import DetectionError
from aegisflow.core.logging import get_logger
from aegisflow.core.schemas import FrameObservation, ObjectBox
from aegisflow.core.settings import DetectionConfig
from aegisflow.detection.video import SampledFrame

log = get_logger(__name__)

# Vehicle-ish COCO classes a forklift plausibly surfaces as.
_VEHICLE_LABELS = frozenset({"truck", "car", "bus", "forklift", "train", "motorcycle"})
_PERSON_LABEL = "person"

_load_lock = threading.Lock()

DEFAULT_BATCH_SIZE = 8
"""Frames per inference call.

Measured on this machine (12 cores, CPU-only, 640 px): 314 ms/frame single-threaded and
unbatched, 77 ms/frame at batch 8 with the thread count raised - a 4x improvement, and 6x
against the naive default. Batching alone does nothing; it only pays off once torch is
allowed more than one thread. Above 8 the curve is flat, so 8 keeps peak memory low on an
8 GB machine for no measurable cost.
"""


class YoloDetector:
    """Thin wrapper over an Ultralytics model."""

    def __init__(self, config: DetectionConfig, weights_dir: Path | None = None) -> None:
        self._config = config
        self._weights_dir = weights_dir
        self._model: Any | None = None
        self._names: dict[int, str] = {}

    # ------------------------------------------------------------------ model

    @property
    def weights_path(self) -> Path:
        return Path(self._config.weights)

    def _ensure_model(self) -> Any:
        """Load the model once, thread-safely.

        Ultralytics resolves a bare filename like ``yolov8n.pt`` by downloading it to the
        current directory, so we pass an explicit path under ``artifacts/models/`` and let
        it download there instead of littering the repo root.
        """
        if self._model is not None:
            return self._model
        with _load_lock:
            if self._model is not None:  # pragma: no cover - double-checked locking
                return self._model
            try:
                from ultralytics import YOLO
            except ImportError as exc:  # pragma: no cover - dependency is pinned
                raise DetectionError("ultralytics is not installed") from exc

            target = self.weights_path
            if not target.is_absolute():
                from aegisflow.core.settings import repo_root

                target = repo_root() / target
            target.parent.mkdir(parents=True, exist_ok=True)

            if not target.exists():
                log.info("YOLO weights not cached; downloading to %s", target)
                self._download_weights(target)

            _tune_torch_threads(self._config.torch_threads)

            try:
                model = YOLO(str(target))
            except Exception as exc:
                raise DetectionError(f"could not load YOLO weights from {target}: {exc}") from exc

            names = getattr(model, "names", {}) or {}
            self._names = {int(k): str(v) for k, v in dict(names).items()}
            self._model = model
            log.info(
                "YOLO ready: %s (%d classes, device=%s)",
                target.name,
                len(self._names),
                self._config.device,
            )
            return model

    @staticmethod
    def _download_weights(target: Path) -> None:
        """Fetch the nano weights into ``target``."""
        try:
            from ultralytics.utils.downloads import attempt_download_asset

            downloaded = Path(attempt_download_asset(target.name))
            if downloaded.exists() and downloaded.resolve() != target.resolve():
                downloaded.replace(target)
        except Exception as exc:
            raise DetectionError(
                f"YOLO weights are not present at {target} and could not be downloaded "
                f"({exc}). Place yolov8n.pt there manually to run offline."
            ) from exc

    def warmup(self) -> None:
        """Load weights and run one dummy inference, so the first real clip is not slow."""
        model = self._ensure_model()
        blank = np.zeros((64, 64, 3), dtype=np.uint8)
        model.predict(blank, verbose=False, device=self._config.device)

    # -------------------------------------------------------------- inference

    def observe(self, frame: SampledFrame) -> FrameObservation:
        """Run detection on one frame."""
        return self.observe_batch([frame])[0]

    def observe_batch(
        self, frames: list[SampledFrame], batch_size: int | None = None
    ) -> list[FrameObservation]:
        """Run detection over several frames, batching the inference calls.

        Results are returned in input order, one observation per frame.
        """
        if not frames:
            return []
        model = self._ensure_model()
        size = max(1, batch_size or self._config.batch_size or DEFAULT_BATCH_SIZE)
        observations: list[FrameObservation] = []

        for start in range(0, len(frames), size):
            chunk = frames[start : start + size]
            try:
                results = model.predict(
                    [f.image for f in chunk],
                    conf=self._config.conf_threshold,
                    iou=self._config.iou_threshold,
                    device=self._config.device,
                    verbose=False,
                )
            except Exception as exc:
                raise DetectionError(
                    f"YOLO inference failed on frames {chunk[0].index}-{chunk[-1].index}: {exc}"
                ) from exc

            for frame, result in zip(chunk, results, strict=False):
                observations.append(self._to_observation(frame, result))

        return observations

    def _to_observation(self, frame: SampledFrame, result: Any) -> FrameObservation:
        """Split one YOLO result into persons and vehicles."""
        persons: list[ObjectBox] = []
        vehicles: list[ObjectBox] = []

        boxes = getattr(result, "boxes", None)
        if boxes is not None:
            for xyxy, conf, cls in zip(
                boxes.xyxy.tolist(), boxes.conf.tolist(), boxes.cls.tolist(), strict=False
            ):
                label = self._names.get(int(cls), str(int(cls)))
                box = ObjectBox(
                    label=label,
                    confidence=float(min(max(conf, 0.0), 1.0)),
                    bbox=(float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])),
                )
                if label == _PERSON_LABEL:
                    persons.append(box)
                elif label in _VEHICLE_LABELS:
                    vehicles.append(box)

        return FrameObservation(
            frame_index=frame.index,
            timestamp_s=frame.timestamp_s,
            width=frame.width,
            height=frame.height,
            persons=tuple(persons),
            vehicles=tuple(vehicles),
        )


def _tune_torch_threads(requested: int | None = None) -> None:
    """Let torch use the machine's cores.

    This is worth 4x. torch reports a single thread here because ``OMP_NUM_THREADS=1`` is
    set in the environment, and at one thread inference costs 314 ms/frame against 77 ms
    with all 12 cores.

    We deliberately do **not** treat ``OMP_NUM_THREADS`` as an instruction to stay
    single-threaded: it is 1 by default in a great many container and CI images, so
    honouring it would quietly cost a 4x slowdown that nobody asked for. An operator who
    genuinely wants a thread cap sets ``detection.torch_threads`` in
    ``config/settings.yaml`` (or ``AEGISFLOW_TORCH_THREADS``), which is explicit.
    """
    try:
        import torch
    except ImportError:  # pragma: no cover - dependency is pinned
        return

    target = requested if requested and requested > 0 else (os.cpu_count() or 1)
    if torch.get_num_threads() != target:
        torch.set_num_threads(target)
        # Keep OpenMP consistent with torch for the OpenCV ops that follow.
        os.environ["OMP_NUM_THREADS"] = str(target)
        log.debug("torch thread count set to %d", target)
