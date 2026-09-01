"""Command-line interface.

    aegisflow policy parse            PDF -> artifacts/policy/rules.json
    aegisflow policy show             pretty-print the parsed rules
    aegisflow policy matrix           show the derived severity matrix
    aegisflow detect <clip>           run one clip, print what was found
    aegisflow run --split test        full pipeline over a split, into the DB and reports
    aegisflow report                  render the PDF compliance report
    aegisflow serve                   FastAPI + dashboard
    aegisflow info                    environment and configuration summary

``python -m aegisflow ...`` works identically.
"""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
from rich.table import Table

from aegisflow import __version__
from aegisflow.core.enums import BehaviorClass, Severity
from aegisflow.core.errors import AegisFlowError
from aegisflow.core.logging import configure_logging
from aegisflow.core.settings import get_settings
from aegisflow.core.zoning import behavior_from_clip_path

console = Console()

app = typer.Typer(
    name="aegisflow",
    help="AegisFlow EHS - policy-grounded factory compliance and alert escalation.",
    no_args_is_help=True,
    add_completion=False,
)
policy_app = typer.Typer(help="Parse and inspect the compliance policy.", no_args_is_help=True)
app.add_typer(policy_app, name="policy")

SEVERITY_STYLE = {
    "LOW": "blue",
    "MEDIUM": "green",
    "HIGH": "yellow",
    "CRITICAL": "bold red",
}


def _severity_text(value: str) -> str:
    return f"[{SEVERITY_STYLE.get(value, 'white')}]{value}[/]"


@app.callback()
def main(
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="DEBUG logging")] = False,
    quiet: Annotated[bool, typer.Option("--quiet", "-q", help="warnings only")] = False,
) -> None:
    level = "DEBUG" if verbose else "WARNING" if quiet else "INFO"
    configure_logging(level)


# ---------------------------------------------------------------------------
# policy
# ---------------------------------------------------------------------------


@policy_app.command("parse")
def policy_parse(
    strict: Annotated[bool, typer.Option(help="fail if a structural check fails")] = False,
) -> None:
    """Derive machine-readable rules from the compliance PDF."""
    from aegisflow.policy import parse_policy

    async def run() -> None:
        settings = get_settings()
        rule_set = await parse_policy(settings, strict=strict)
        console.print(
            Panel(
                f"[bold]{rule_set.document_id}[/bold]\n"
                f"source     {rule_set.source_path}\n"
                f"sha256     {rule_set.source_sha256[:32]}...\n"
                f"method     {rule_set.extraction_method}\n"
                f"rules      {len(rule_set.rules)} ({rule_set.unsafe_rule_count} unsafe)\n"
                f"sections   {len(rule_set.sections)}",
                title="Policy parsed",
                border_style="green",
            )
        )
        if rule_set.warnings:
            console.print("[yellow]Validation notes:[/yellow]")
            for warning in rule_set.warnings:
                console.print(f"  - {warning}")
        console.print(f"[dim]wrote {settings.path(settings.rules_json)}[/dim]")

    _run(run())


@policy_app.command("show")
def policy_show() -> None:
    """Print the parsed rule set."""
    from aegisflow.policy import load_rule_set

    rule_set = _load_policy_or_exit(load_rule_set)
    table = Table(title=f"{rule_set.document_id} - parsed rules", title_style="bold")
    table.add_column("Behaviour")
    table.add_column("Section")
    table.add_column("Callout")
    table.add_column("Observable indicator (from PDF)", max_width=52)
    table.add_column("Thr", justify="right")
    table.add_column("OK", justify="center")

    for rule in rule_set.rules:
        table.add_row(
            rule.behavior_class.value,
            rule.section_ref,
            rule.callout.value,
            rule.observable_indicator,
            "" if rule.numeric_threshold is None else str(rule.numeric_threshold),
            "[green]yes[/]" if rule.validated else "[red]no[/]",
        )
    console.print(table)


@policy_app.command("matrix")
def policy_matrix() -> None:
    """Show the severity matrix derived from the policy text."""
    from aegisflow.policy import load_rule_set
    from aegisflow.severity import describe_matrix

    rule_set = _load_policy_or_exit(load_rule_set)
    table = Table(
        title="Derived severity matrix (base tiers, before frame context)", title_style="bold"
    )
    table.add_column("Behaviour")
    table.add_column("Section")
    table.add_column("Callout")
    table.add_column("Base")
    table.add_column("Derivation", max_width=62)

    for row in describe_matrix(rule_set):
        table.add_row(
            row["behavior_class"],
            row["section"],
            row["callout"],
            _severity_text(row["base_severity"]),
            row["derivation"],
        )
    console.print(table)
    console.print(
        "[dim]Frame context (personnel present, forklift in frame, person at panel) "
        "adjusts these per clip.[/dim]"
    )


# ---------------------------------------------------------------------------
# detect
# ---------------------------------------------------------------------------


@app.command()
def detect(
    clip: Annotated[Path, typer.Argument(help="path to a video clip")],
    show_evidence: Annotated[bool, typer.Option("--evidence", help="print raw cues")] = False,
) -> None:
    """Analyse one clip without writing to the database."""
    from aegisflow.escalation import NullEscalationSink
    from aegisflow.llm import build_provider
    from aegisflow.pipeline import CompliancePipeline
    from aegisflow.policy import ensure_rule_set

    async def run() -> None:
        settings = get_settings()
        provider = build_provider(settings)
        rule_set = await ensure_rule_set(settings, provider=provider)
        pipeline = CompliancePipeline(
            rule_set, settings, provider=provider, sink=NullEscalationSink()
        )

        with console.status("loading model..."):
            pipeline.warmup()
        started = time.perf_counter()
        result = await pipeline.process_clip(clip)
        await provider.aclose()

        truth = behavior_from_clip_path(clip)
        console.print(
            Panel(
                f"clip       {result.clip_id}\n"
                f"zone       {result.zone}\n"
                f"duration   {result.duration_s:.1f}s ({result.frames_analysed} frames analysed)\n"
                f"processed  {time.perf_counter() - started:.2f}s\n"
                f"dataset    {truth.value if truth else 'unknown'}\n"
                f"vlm calls  {result.vlm_calls}",
                title="Clip",
                border_style="cyan",
            )
        )

        if not result.events:
            console.print("[green]No violation detected - compliant.[/green]")
            return

        table = Table(title=f"{len(result.events)} violation(s)", title_style="bold")
        table.add_column("Behaviour")
        table.add_column("Sev")
        table.add_column("Policy")
        table.add_column("Conf", justify="right")
        table.add_column("Method")
        table.add_column("Escalation")
        for event in result.events:
            table.add_row(
                event.behavior_class.value,
                _severity_text(event.severity.value),
                event.policy_rule_ref,
                f"{event.confidence:.2f}",
                event.detection_method.value,
                event.escalation_action.value,
            )
        console.print(table)

        for event in result.events:
            console.print(f"\n[bold]{event.behavior_class.display_name}[/bold]")
            console.print(f"  {event.event_description}")
            console.print(f"  [dim]{event.severity_rationale}[/dim]")

        if show_evidence:
            for record in result.detections:
                console.print(f"\n[cyan]{record.behavior_class.value}[/cyan] evidence:")
                for key, value in sorted(record.evidence.items()):
                    console.print(f"  {key} = {value}")

    _run(run())


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


@app.command()
def run(
    split: Annotated[str, typer.Option(help="train | test | all")] = "test",
    per_class: Annotated[int | None, typer.Option(help="sample N clips per class")] = None,
    limit: Annotated[int | None, typer.Option(help="overall clip cap")] = None,
    behavior: Annotated[str | None, typer.Option(help="only this behaviour class")] = None,
    annotate: Annotated[
        bool, typer.Option(help="render annotated clips for the dashboard")
    ] = False,
    reset: Annotated[bool, typer.Option(help="delete existing events first")] = False,
) -> None:
    """Run the full pipeline over a dataset split, persisting events and reports."""
    from aegisflow.db import crud, init_db, session_scope
    from aegisflow.escalation import EscalationRouter
    from aegisflow.llm import build_provider
    from aegisflow.pipeline import CompliancePipeline, discover_clips, worst_severity
    from aegisflow.policy import ensure_rule_set
    from aegisflow.reports import default_writers

    async def go() -> None:
        settings = get_settings()
        target = None if split == "all" else split
        behavior_filter = BehaviorClass(behavior) if behavior else None

        clips = discover_clips(
            settings, split=target, behavior=behavior_filter, per_class=per_class, limit=limit
        )
        if not clips:
            console.print("[red]no clips found - is the dataset in data/raw/?[/red]")
            raise typer.Exit(1)

        provider = build_provider(settings)
        rule_set = await ensure_rule_set(settings, provider=provider)
        await init_db(settings)

        if reset:
            await _reset_events(settings)

        writer = default_writers(settings.path(settings.outputs_root))

        async with session_scope(settings) as session:
            await crud.upsert_policy_rules(session, rule_set)
            router = EscalationRouter(session)
            pipeline = CompliancePipeline(
                rule_set, settings, provider=provider, sink=router, writer=writer
            )
            with console.status("loading model..."):
                pipeline.warmup()

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TextColumn("{task.fields[events]} events"),
                console=console,
            ) as progress:
                task = progress.add_task(
                    f"processing {len(clips)} clips", total=len(clips), events=0
                )

                for clip in clips:
                    try:
                        result = await pipeline.process_clip(clip)
                    except AegisFlowError as exc:
                        pipeline.stats.failures.append((str(clip), str(exc)))
                        progress.advance(task)
                        continue

                    await crud.insert_clip_run(
                        session,
                        result,
                        ground_truth=behavior_from_clip_path(clip),
                        policy_sha256=rule_set.source_sha256,
                    )
                    if annotate:
                        from aegisflow.detection.annotate import render_annotated_clip

                        out = render_annotated_clip(clip, result, settings)
                        if out is not None:
                            await crud.record_annotated_clip(
                                session,
                                result.clip_id,
                                str(out),
                                result.zone,
                                worst_severity(result.events),
                                len(result.events),
                            )
                    progress.update(task, events=pipeline.stats.events)
                    progress.advance(task)

        await provider.aclose()
        _print_run_summary(pipeline.stats, router.summary())

    _run(go())


@app.command()
def report(
    output: Annotated[Path | None, typer.Option(help="output PDF path")] = None,
    limit: Annotated[int, typer.Option(help="max events to include")] = 500,
) -> None:
    """Render the executive PDF compliance report from stored events."""
    from aegisflow.db import crud, init_db, session_scope
    from aegisflow.policy import load_rule_set
    from aegisflow.reports import build_compliance_pdf

    async def go() -> None:
        settings = get_settings()
        await init_db(settings)
        async with session_scope(settings) as session:
            events = await crud.list_events(session, limit=limit, newest_first=False)
        if not events:
            console.print("[yellow]no events stored - run 'aegisflow run' first[/yellow]")
            raise typer.Exit(1)

        try:
            rule_set = load_rule_set(settings)
        except AegisFlowError:
            rule_set = None

        path = output or settings.path(settings.outputs_root) / "reports" / "compliance_report.pdf"
        build_compliance_pdf(events, path, rule_set)
        console.print(f"[green]wrote[/green] {path} [dim]({len(events)} events)[/dim]")

    _run(go())


@app.command()
def serve(
    host: Annotated[str | None, typer.Option(help="bind host")] = None,
    port: Annotated[int | None, typer.Option(help="bind port")] = None,
) -> None:
    """Start the API and dashboard."""
    import uvicorn

    settings = get_settings()
    bind_host = host or settings.api_host
    bind_port = port or settings.api_port
    console.print(
        Panel(
            f"dashboard  http://{bind_host}:{bind_port}/\n"
            f"api docs   http://{bind_host}:{bind_port}/docs\n"
            f"alerts     ws://{bind_host}:{bind_port}/ws/alerts",
            title="AegisFlow EHS",
            border_style="cyan",
        )
    )
    from aegisflow.api.app import APP_FACTORY

    uvicorn.run(
        APP_FACTORY,
        host=bind_host,
        port=bind_port,
        log_level=settings.log_level.lower(),
        factory=True,
    )


@app.command()
def info(
    check_llm: Annotated[
        bool, typer.Option("--check-llm", help="probe the configured provider's models")
    ] = False,
) -> None:
    """Print environment and configuration."""
    import torch

    settings = get_settings()
    tuning = settings.tuning
    weights = settings.path(tuning.detection.weights)
    panel_baseline = settings.path(tuning.panel.baseline)
    rules = settings.path(settings.rules_json)

    def tick(ok: bool) -> str:
        return "[green]yes[/]" if ok else "[red]no[/]"

    table = Table(title=f"AegisFlow EHS {__version__}", title_style="bold", show_header=False)
    table.add_column("Setting", style="dim")
    table.add_column("Value")
    rows = [
        ("python / torch", f"{torch.__version__} (cuda={torch.cuda.is_available()})"),
        ("device", tuning.detection.device),
        # The count torch will actually use once the model loads - the detector raises it
        # then, so reading it before that would report a misleading 1.
        (
            "torch threads",
            str(tuning.detection.torch_threads or os.cpu_count() or 1)
            + (" (configured)" if tuning.detection.torch_threads else " (all cores)"),
        ),
        (
            "llm provider",
            f"{settings.llm_provider.value} (key present: {tick(bool(settings.llm_key))})",
        ),
        (
            "sampling",
            f"{tuning.video.sample_fps} fps @ {tuning.video.infer_imgsz}px, "
            f"batch {tuning.detection.batch_size}",
        ),
        ("yolo weights", f"{weights.name} {tick(weights.exists())}"),
        ("panel baseline", f"commissioned {tick(panel_baseline.exists())}"),
        ("policy rules", f"parsed {tick(rules.exists())}"),
        ("database", settings.db_url),
        (
            "dataset",
            f"{settings.path(settings.data_root)} "
            f"({len(list(settings.path(settings.data_root).rglob('*.mp4')))} clips)",
        ),
    ]
    for name, value in rows:
        table.add_row(name, value)
    console.print(table)

    if check_llm:
        _check_llm(settings)


def _check_llm(settings) -> None:
    """Probe the configured provider: does the key work, and are the models reachable?

    Model catalogues change and availability differs per account, so a stale id in `.env`
    is a live failure mode - it surfaces as a 404 on the first real clip, mid-demo. This
    checks both models before that happens.
    """
    from aegisflow.core.enums import LLMProviderName

    if settings.llm_provider is LLMProviderName.OFFLINE:
        console.print("\n[dim]provider is 'offline' - nothing to probe.[/dim]")
        return
    if not settings.llm_key:
        console.print(
            f"\n[yellow]provider is '{settings.llm_provider.value}' but no API key is "
            "set; the system will run offline.[/yellow]"
        )
        return

    table = Table(title="LLM provider check", title_style="bold", show_header=False)
    table.add_column("Check", style="dim")
    table.add_column("Result")

    if settings.llm_provider is LLMProviderName.GROQ:
        _probe_groq(settings, table)
    else:
        table.add_row("provider", f"{settings.llm_provider.value} (no probe implemented)")
    console.print(table)


def _probe_groq(settings, table: Table) -> None:
    import base64

    import numpy as np
    from groq import Groq

    client = Groq(api_key=settings.groq_api_key)

    try:
        available = sorted(m.id for m in client.models.list().data)
        table.add_row("models visible", str(len(available)))
    except Exception as exc:
        table.add_row("models visible", f"[red]could not list ({str(exc)[:60]})[/red]")
        available = []

    text_model = settings.groq_text_model
    mark = "[green]yes[/]" if not available or text_model in available else "[red]NOT LISTED[/]"
    table.add_row("text model", f"{text_model} {mark}")
    try:
        client.chat.completions.create(
            model=text_model, max_tokens=5, messages=[{"role": "user", "content": "ok"}]
        )
        table.add_row("text call", "[green]works[/]")
    except Exception as exc:
        table.add_row("text call", f"[red]{str(exc)[:70]}[/red]")

    vision_model = settings.groq_vision_model
    mark = "[green]yes[/]" if not available or vision_model in available else "[red]NOT LISTED[/]"
    table.add_row("vision model", f"{vision_model} {mark}")
    try:
        import cv2

        swatch = np.zeros((64, 64, 3), np.uint8)
        swatch[:, :] = (40, 180, 40)
        png = base64.b64encode(cv2.imencode(".png", swatch)[1].tobytes()).decode()
        reply = client.chat.completions.create(
            model=vision_model,
            max_tokens=20,
            temperature=0,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What colour fills this image? One word."},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{png}"},
                        },
                    ],
                }
            ],
        )
        answer = (reply.choices[0].message.content or "").strip()[:30]
        good = "green" in answer.lower()
        table.add_row(
            "vision call",
            f"[green]works[/] (saw {answer!r})" if good else f"[yellow]replied {answer!r}[/yellow]",
        )
    except Exception as exc:
        table.add_row(
            "vision call",
            f"[red]{str(exc)[:70]}[/red]\n"
            "[dim]most Groq text models reject images; pick a vision-capable id[/dim]",
        )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _run(coro) -> None:
    try:
        asyncio.run(coro)
    except AegisFlowError as exc:
        console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(1) from exc
    except KeyboardInterrupt:  # pragma: no cover - interactive
        console.print("\n[yellow]interrupted[/yellow]")
        raise typer.Exit(130) from None


def _load_policy_or_exit(loader):
    try:
        return loader(get_settings())
    except AegisFlowError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc


async def _reset_events(settings) -> None:
    """Drop stored events. Explicit opt-in only - the audit trail is append-only."""
    from sqlalchemy import delete

    from aegisflow.db import session_scope
    from aegisflow.db.models import AnnotatedClipRow, ClipRunRow, ViolationEventRow

    async with session_scope(settings) as session:
        for model in (ViolationEventRow, ClipRunRow, AnnotatedClipRow):
            await session.execute(delete(model))
    console.print("[yellow]cleared stored events, clip runs and annotated clips[/yellow]")


def _print_run_summary(stats, routing: dict[str, int]) -> None:
    table = Table(title="Run summary", title_style="bold", show_header=False)
    table.add_column("Metric", style="dim")
    table.add_column("Value")
    table.add_row("clips processed", str(stats.clips))
    table.add_row("clips with violations", str(stats.clips_with_violations))
    table.add_row("events recorded", str(stats.events))
    table.add_row("real-time alerts", str(stats.alerts))
    table.add_row("frames analysed", str(stats.frames))
    table.add_row("vlm tie-breaks", str(stats.vlm_calls))
    table.add_row("mean per clip", f"{stats.mean_clip_seconds:.2f}s")
    table.add_row("total time", f"{stats.seconds:.0f}s")
    if stats.failures:
        table.add_row("failed clips", f"[yellow]{len(stats.failures)}[/yellow]")
    console.print(table)

    if stats.by_severity:
        severity_table = Table(title="By severity", title_style="bold")
        for tier in Severity:
            severity_table.add_column(tier.value, justify="right")
        severity_table.add_row(*[str(stats.by_severity.get(t.value, 0)) for t in Severity])
        console.print(severity_table)

    if stats.by_behavior:
        behavior_table = Table(title="By behaviour", title_style="bold")
        behavior_table.add_column("Behaviour")
        behavior_table.add_column("Events", justify="right")
        for name, count in sorted(stats.by_behavior.items(), key=lambda kv: -kv[1]):
            behavior_table.add_row(name, str(count))
        console.print(behavior_table)

    console.print(
        f"[dim]routing: {routing['logged']} logged, {routing['alerted']} alerted "
        f"({routing['alert_deliveries']} deliveries)[/dim]"
    )


if __name__ == "__main__":
    app()
