"""Populate the database so the dashboard has something to show.

Two modes:

``--from-clips`` (default when the dataset is present)
    Runs the real pipeline over a balanced sample and stores what it finds. Slower, and the
    only mode that produces genuine detections.

``--synthetic``
    Generates a plausible spread of events from the *parsed policy* without touching video.
    Every record still carries a real section reference and a real severity rationale,
    because the severity matrix produced them - only the detections are fabricated. Useful
    for demoing the dashboard, developing the frontend, or a deployment smoke test on a
    machine with no dataset.

    python scripts/seed_db.py                       # real pipeline if data/raw exists
    python scripts/seed_db.py --synthetic --events 120
    python scripts/seed_db.py --from-clips --per-class 4 --annotate --reset
"""

from __future__ import annotations

import argparse
import asyncio
import random
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from rich.console import Console  # noqa: E402
from sqlalchemy import delete  # noqa: E402

from aegisflow.core.enums import UNSAFE_BEHAVIORS, BehaviorClass  # noqa: E402
from aegisflow.core.logging import configure_logging  # noqa: E402
from aegisflow.core.schemas import DetectionRecord, FrameContext, ViolationEvent  # noqa: E402
from aegisflow.core.settings import get_settings  # noqa: E402
from aegisflow.core.zoning import behavior_from_clip_path  # noqa: E402
from aegisflow.db import crud, init_db, session_scope  # noqa: E402
from aegisflow.db.models import AnnotatedClipRow, ClipRunRow, ViolationEventRow  # noqa: E402
from aegisflow.detection.temporal import DetectionMethod  # noqa: E402
from aegisflow.escalation import EscalationRouter, get_alert_bus  # noqa: E402
from aegisflow.llm import build_provider  # noqa: E402
from aegisflow.pipeline import CompliancePipeline, discover_clips, worst_severity  # noqa: E402
from aegisflow.policy import ensure_rule_set  # noqa: E402
from aegisflow.reports import default_writers  # noqa: E402
from aegisflow.severity import SeverityMatrix  # noqa: E402

console = Console()

# Contexts that exercise every branch of the severity matrix, so a seeded database shows
# all four tiers rather than a wall of one colour.
_CONTEXTS: dict[BehaviorClass, list[dict[str, object]]] = {
    BehaviorClass.OPENED_PANEL_COVER: [
        {},
        {"max_person_count": 1},
        {"max_person_count": 2, "person_near_panel": True},
    ],
    BehaviorClass.SAFE_WALKWAY_VIOLATION: [
        {"max_person_count": 1},
        {"max_person_count": 2, "forklift_present": True},
    ],
    BehaviorClass.UNAUTHORIZED_INTERVENTION: [
        {"max_person_count": 1},
        {"max_person_count": 3, "multiple_unauthorized_persons": True},
    ],
    BehaviorClass.CARRYING_OVERLOAD_WITH_FORKLIFT: [{"forklift_present": True}],
}


async def reset(settings) -> None:
    async with session_scope(settings) as session:
        for model in (ViolationEventRow, ClipRunRow, AnnotatedClipRow):
            await session.execute(delete(model))
    console.print("[yellow]cleared existing events, clip runs and annotated clips[/yellow]")


async def seed_synthetic(args) -> int:
    settings = get_settings()
    rule_set = await ensure_rule_set(settings)
    matrix = SeverityMatrix(rule_set)
    rng = random.Random(args.seed)

    await init_db(settings)
    if args.reset:
        await reset(settings)

    writer = default_writers(settings.path(settings.outputs_root))
    start = datetime.now(UTC) - timedelta(days=args.days)
    behaviors = [b for b in UNSAFE_BEHAVIORS if rule_set.rule_for(b) is not None]

    async with session_scope(settings) as session:
        await crud.upsert_policy_rules(session, rule_set)
        router = EscalationRouter(session, get_alert_bus())

        for index in range(args.events):
            behavior = rng.choice(behaviors)
            context = rng.choice(_CONTEXTS.get(behavior, [{}]))
            clip_id = f"seed_{behavior.value[:4]}_{index:04d}.mp4"

            detection = DetectionRecord(
                clip_id=clip_id,
                behavior_class=behavior,
                confidence=round(rng.uniform(0.45, 0.95), 3),
                detection_method=rng.choice(
                    [DetectionMethod.HSV, DetectionMethod.CONTOUR, DetectionMethod.GEOMETRY]
                ),
                first_frame_index=rng.randrange(0, 40),
                first_timestamp_s=round(rng.uniform(0.0, 9.0), 2),
                frame_count=rng.randrange(3, 12),
                description=(
                    f"{behavior.display_name} observed on the production floor. "
                    f"Policy indicator: {rule_set.require_rule(behavior).observable_indicator}."
                ),
                zone="Zone-2" if behavior is BehaviorClass.OPENED_PANEL_COVER else "Zone-1",
                context=FrameContext(**context),  # type: ignore[arg-type]
            )
            assessment = matrix.assess(detection)

            event = ViolationEvent(
                timestamp=start + timedelta(minutes=rng.randrange(0, args.days * 24 * 60)),
                clip_id=clip_id,
                zone=detection.zone,
                behavior_class=behavior,
                policy_rule_ref=assessment.policy_rule_ref,
                event_description=detection.description,
                severity=assessment.severity,
                confidence=detection.confidence,
                detection_method=detection.detection_method,
                severity_rationale=assessment.rationale,
                clip_timestamp_s=detection.first_timestamp_s,
                frame_index=detection.first_frame_index,
            )
            routed = await router.route(event)
            await writer.write(routed)

        console.print(
            f"[green]seeded {args.events} synthetic events[/green] "
            f"({router.summary()['alerted']} would alert)"
        )
    return 0


async def seed_from_clips(args) -> int:
    settings = get_settings()
    clips = discover_clips(settings, split=args.split, per_class=args.per_class)
    if not clips:
        console.print("[red]no clips under data/raw/ - use --synthetic instead[/red]")
        return 1

    provider = build_provider(settings)
    rule_set = await ensure_rule_set(settings, provider=provider)
    await init_db(settings)
    if args.reset:
        await reset(settings)

    writer = default_writers(settings.path(settings.outputs_root))

    async with session_scope(settings) as session:
        await crud.upsert_policy_rules(session, rule_set)
        router = EscalationRouter(session, get_alert_bus())
        pipeline = CompliancePipeline(
            rule_set, settings, provider=provider, sink=router, writer=writer
        )
        with console.status("loading model..."):
            pipeline.warmup()

        for index, clip in enumerate(clips, start=1):
            result = await pipeline.process_clip(clip)
            await crud.insert_clip_run(
                session,
                result,
                ground_truth=behavior_from_clip_path(clip),
                policy_sha256=rule_set.source_sha256,
            )
            if args.annotate:
                from aegisflow.detection.annotate import render_annotated_clip

                path = render_annotated_clip(clip, result, settings)
                if path is not None:
                    await crud.record_annotated_clip(
                        session,
                        result.clip_id,
                        str(path),
                        result.zone,
                        worst_severity(result.events),
                        len(result.events),
                    )
            if index % 10 == 0 or index == len(clips):
                console.print(
                    f"  [dim]{index}/{len(clips)} clips, {pipeline.stats.events} events[/dim]"
                )

    await provider.aclose()
    console.print(
        f"[green]seeded from {pipeline.stats.clips} clips[/green]: "
        f"{pipeline.stats.events} events, {pipeline.stats.alerts} alerts"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--synthetic", action="store_true", help="no video required")
    mode.add_argument("--from-clips", action="store_true", help="run the real pipeline")

    parser.add_argument("--events", type=int, default=90, help="synthetic events to create")
    parser.add_argument("--days", type=int, default=14, help="spread events over N days")
    parser.add_argument("--split", default="test", choices=("train", "test"))
    parser.add_argument("--per-class", type=int, default=3)
    parser.add_argument("--annotate", action="store_true", help="render clips for View A")
    parser.add_argument("--reset", action="store_true", help="clear existing rows first")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    configure_logging("INFO")
    settings = get_settings()

    if not args.synthetic and not args.from_clips:
        # Default to whichever mode the machine can actually support.
        has_data = any(settings.path(settings.data_root).rglob("*.mp4"))
        args.from_clips = has_data
        args.synthetic = not has_data
        console.print(f"[dim]auto mode: {'from-clips' if has_data else 'synthetic'}[/dim]")

    runner = seed_from_clips if args.from_clips else seed_synthetic
    return asyncio.run(runner(args))


if __name__ == "__main__":
    raise SystemExit(main())
