"""Clip-level evaluation of the detection engine.

The dataset has no bounding-box annotations - labels are folder names - so evaluation is at
clip level. For each of the four unsafe behaviours:

* a clip in that behaviour's folder is a **positive**; detecting it is a TP, missing it a FN
* a clip in **any other** folder that triggers the detector is a **FP**

The four safe folders are the most informative negatives, because each is the compliant
counterpart of an unsafe class filmed on the same camera - which is exactly the confusion
that matters.

Reported per class, never blended: the classes are ~9:1 imbalanced, so a single accuracy
figure would be dominated by walkway violations and would hide everything else.

    python scripts/evaluate.py --split test
    python scripts/evaluate.py --split test --per-class 8 --json outputs/eval/run.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from rich.console import Console  # noqa: E402
from rich.table import Table  # noqa: E402

from aegisflow.core.enums import UNSAFE_BEHAVIORS, BehaviorClass, Severity  # noqa: E402
from aegisflow.core.logging import configure_logging  # noqa: E402
from aegisflow.core.settings import get_settings  # noqa: E402
from aegisflow.core.zoning import behavior_from_clip_path  # noqa: E402
from aegisflow.llm import build_provider  # noqa: E402
from aegisflow.pipeline import CompliancePipeline, discover_clips  # noqa: E402
from aegisflow.policy import ensure_rule_set  # noqa: E402

console = Console()


@dataclass
class ClassMetrics:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0
    fp_by_source: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    fp_clips: list[str] = field(default_factory=list)
    fn_clips: list[str] = field(default_factory=list)

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    def as_dict(self) -> dict[str, object]:
        return {
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "tn": self.tn,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "fp_by_source_class": dict(self.fp_by_source),
            "example_fp_clips": self.fp_clips[:8],
            "example_fn_clips": self.fn_clips[:8],
        }


async def run(args: argparse.Namespace) -> int:
    configure_logging("WARNING")
    settings = get_settings()

    clips = discover_clips(settings, split=args.split, per_class=args.per_class, limit=args.limit)
    if not clips:
        console.print(
            f"[red]no clips found under {settings.path(settings.data_root) / args.split}[/red]"
        )
        return 1

    provider = build_provider(settings)
    rule_set = await ensure_rule_set(settings, provider=provider)
    pipeline = CompliancePipeline(rule_set, settings, provider=provider)

    console.print(
        f"[bold]Evaluating[/bold] {len(clips)} clips | split={args.split} | "
        f"provider={provider.name}"
    )
    pipeline.warmup()

    metrics: dict[BehaviorClass, ClassMetrics] = {b: ClassMetrics() for b in UNSAFE_BEHAVIORS}
    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    severity_counts: dict[str, int] = defaultdict(int)
    latencies: list[float] = []
    started = time.perf_counter()

    for index, clip in enumerate(clips, start=1):
        truth = behavior_from_clip_path(clip)
        result = await pipeline.process_clip(clip)
        latencies.append(result.processing_s)

        detected = {record.behavior_class for record in result.detections}
        truth_key = truth.value if truth else "unknown"

        for event in result.events:
            severity_counts[event.severity.value] += 1
            confusion[truth_key][event.behavior_class.value] += 1
        if not result.events:
            confusion[truth_key]["(none)"] += 1

        for behavior in UNSAFE_BEHAVIORS:
            metric = metrics[behavior]
            fired = behavior in detected
            is_positive = truth is behavior
            if is_positive and fired:
                metric.tp += 1
            elif is_positive and not fired:
                metric.fn += 1
                metric.fn_clips.append(clip.name)
            elif not is_positive and fired:
                metric.fp += 1
                metric.fp_by_source[truth_key] += 1
                metric.fp_clips.append(f"{truth_key}/{clip.name}")
            else:
                metric.tn += 1

        if index % 25 == 0 or index == len(clips):
            console.print(
                f"  [dim]{index}/{len(clips)} clips | "
                f"{pipeline.stats.events} events | "
                f"{time.perf_counter() - started:.0f}s[/dim]"
            )

    await provider.aclose()
    _report(metrics, confusion, severity_counts, latencies, pipeline, args)
    if args.json:
        _write_json(args, metrics, confusion, severity_counts, latencies, pipeline, clips)
    return 0


def _report(metrics, confusion, severity_counts, latencies, pipeline, args) -> None:
    table = Table(title=f"Per-class detection metrics ({args.split} split)", title_style="bold")
    for column in ("Behaviour", "Pos", "TP", "FP", "FN", "Precision", "Recall", "F1"):
        table.add_column(column, justify="right" if column != "Behaviour" else "left")

    for behavior in sorted(UNSAFE_BEHAVIORS, key=lambda b: b.value):
        m = metrics[behavior]
        colour = "green" if m.f1 >= 0.7 else "yellow" if m.f1 >= 0.45 else "red"
        table.add_row(
            behavior.value,
            str(m.tp + m.fn),
            str(m.tp),
            str(m.fp),
            str(m.fn),
            f"{m.precision:.2f}",
            f"{m.recall:.2f}",
            f"[{colour}]{m.f1:.2f}[/{colour}]",
        )
    console.print(table)

    macro_f1 = sum(metrics[b].f1 for b in UNSAFE_BEHAVIORS) / len(UNSAFE_BEHAVIORS)
    console.print(f"[bold]Macro F1:[/bold] {macro_f1:.3f}  [dim](unweighted class mean)[/dim]")

    fp_table = Table(title="False positives by source class", title_style="bold")
    fp_table.add_column("Detector fired")
    fp_table.add_column("On clips actually labelled")
    fp_table.add_column("Count", justify="right")
    for behavior in sorted(UNSAFE_BEHAVIORS, key=lambda b: b.value):
        for source, count in sorted(metrics[behavior].fp_by_source.items(), key=lambda kv: -kv[1]):
            fp_table.add_row(behavior.value, source, str(count))
    if fp_table.row_count:
        console.print(fp_table)

    sev_table = Table(title="Severity distribution", title_style="bold")
    for severity in Severity:
        sev_table.add_column(severity.value, justify="right")
    sev_table.add_row(*[str(severity_counts.get(s.value, 0)) for s in Severity])
    console.print(sev_table)

    mean_latency = sum(latencies) / len(latencies) if latencies else 0.0
    console.print(
        f"\n[bold]Throughput[/bold]  clips={pipeline.stats.clips}  "
        f"frames={pipeline.stats.frames}  "
        f"mean={mean_latency:.2f}s/clip  "
        f"total={sum(latencies):.0f}s  "
        f"vlm_calls={pipeline.stats.vlm_calls}"
    )
    if pipeline.stats.failures:
        console.print(f"[yellow]{len(pipeline.stats.failures)} clip(s) failed to process[/yellow]")


def _write_json(args, metrics, confusion, severity_counts, latencies, pipeline, clips) -> None:
    payload = {
        "split": args.split,
        "clips_evaluated": len(clips),
        "per_class": {b.value: metrics[b].as_dict() for b in UNSAFE_BEHAVIORS},
        "macro_f1": round(sum(metrics[b].f1 for b in UNSAFE_BEHAVIORS) / len(UNSAFE_BEHAVIORS), 4),
        "confusion": {k: dict(v) for k, v in confusion.items()},
        "severity_distribution": dict(severity_counts),
        "throughput": {
            "mean_clip_seconds": round(sum(latencies) / len(latencies) if latencies else 0.0, 3),
            "total_seconds": round(sum(latencies), 1),
            "frames": pipeline.stats.frames,
            "vlm_calls": pipeline.stats.vlm_calls,
        },
        "failures": pipeline.stats.failures,
    }
    path = Path(args.json)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    console.print(f"[dim]wrote {path}[/dim]")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="test", choices=("train", "test"))
    parser.add_argument("--per-class", type=int, default=None, help="clips per class")
    parser.add_argument("--limit", type=int, default=None, help="overall clip cap")
    parser.add_argument("--json", default=None, help="write machine-readable results here")
    args = parser.parse_args()
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
