"""Click CLI entry point for RegScope."""

import logging
import sys

import click
from rich.console import Console

from regscope.config import load_config, setup_logging

console = Console()
logger = logging.getLogger(__name__)


def get_api_key(api_key: str | None, config: dict) -> str:
    """Resolve API key from CLI flag, config, or environment."""
    if api_key:
        return api_key
    key = config.get("api", {}).get("api_key", "")
    if key:
        return key
    console.print("[red]Error:[/red] No API key provided. Use --api-key, set REGSCOPE_API_KEY, or add to config.toml")
    sys.exit(1)


@click.group()
@click.option("--config", "config_path", default=None, type=click.Path(), help="Path to config.toml")
@click.pass_context
def cli(ctx: click.Context, config_path: str | None) -> None:
    """RegScope — Federal Rulemaking Public Comment Analyzer.

    Download, structure, and analyze public comments on federal rulemakings
    from the Regulations.gov API.
    """
    ctx.ensure_object(dict)
    config = load_config(config_path)
    setup_logging(config)
    ctx.obj["config"] = config


@cli.command()
@click.argument("docket_id")
@click.option("--api-key", default=None, help="Regulations.gov API key")
@click.pass_context
def analyze(ctx: click.Context, docket_id: str, api_key: str | None) -> None:
    """Download and run full analysis pipeline on a docket."""
    config = ctx.obj["config"]
    api_key = get_api_key(api_key, config)

    from regscope.pipeline.ingest import run_ingest
    from regscope.pipeline.extract import run_extract
    from regscope.pipeline.dedup import run_dedup
    from regscope.pipeline.embed import run_embed
    from regscope.pipeline.topics import run_topics
    from regscope.pipeline.classify import run_classify
    from regscope.pipeline.report import run_report
    from regscope.db import get_db, log_pipeline_run

    db = get_db(docket_id, config)
    try:
        console.print(f"\n[bold]RegScope — Analyzing docket {docket_id}[/bold]\n")

        stages = [
            ("ingest", lambda: run_ingest(db, docket_id, api_key, config)),
            ("extract", lambda: run_extract(db, docket_id, config)),
            ("dedup", lambda: run_dedup(db, docket_id, config)),
            ("embed", lambda: run_embed(db, docket_id, config)),
            ("topics", lambda: run_topics(db, docket_id, config)),
            ("classify", lambda: run_classify(db, docket_id, config)),
            ("report", lambda: run_report(db, docket_id, config)),
        ]

        for stage_name, stage_fn in stages:
            console.print(f"[cyan]Running stage:[/cyan] {stage_name}")
            try:
                stage_fn()
                log_pipeline_run(db, docket_id, stage_name, "completed")
                console.print(f"  [green]Completed:[/green] {stage_name}")
            except Exception:
                logger.exception("Stage %s failed", stage_name)
                log_pipeline_run(db, docket_id, stage_name, "failed")
                console.print(f"  [red]Failed:[/red] {stage_name}")
                raise

        console.print(f"\n[bold green]Analysis complete for {docket_id}[/bold green]")
    finally:
        db.close()


@cli.command()
@click.argument("docket_id")
@click.option("--api-key", default=None, help="Regulations.gov API key")
@click.pass_context
def ingest(ctx: click.Context, docket_id: str, api_key: str | None) -> None:
    """Download comments for a docket (no analysis)."""
    config = ctx.obj["config"]
    api_key = get_api_key(api_key, config)

    from regscope.pipeline.ingest import run_ingest
    from regscope.db import get_db, log_pipeline_run

    db = get_db(docket_id, config)
    try:
        console.print(f"\n[bold]Ingesting docket {docket_id}[/bold]\n")
        run_ingest(db, docket_id, api_key, config)
        log_pipeline_run(db, docket_id, "ingest", "completed")
        console.print(f"\n[bold green]Ingest complete for {docket_id}[/bold green]")
    finally:
        db.close()


@cli.command()
@click.argument("docket_id")
@click.pass_context
def process(ctx: click.Context, docket_id: str) -> None:
    """Run analysis on already-downloaded data."""
    config = ctx.obj["config"]

    from regscope.pipeline.extract import run_extract
    from regscope.pipeline.dedup import run_dedup
    from regscope.pipeline.embed import run_embed
    from regscope.pipeline.topics import run_topics
    from regscope.pipeline.classify import run_classify
    from regscope.pipeline.report import run_report
    from regscope.db import get_db, log_pipeline_run

    db = get_db(docket_id, config)
    try:
        console.print(f"\n[bold]Processing docket {docket_id}[/bold]\n")

        stages = [
            ("extract", lambda: run_extract(db, docket_id, config)),
            ("dedup", lambda: run_dedup(db, docket_id, config)),
            ("embed", lambda: run_embed(db, docket_id, config)),
            ("topics", lambda: run_topics(db, docket_id, config)),
            ("classify", lambda: run_classify(db, docket_id, config)),
            ("report", lambda: run_report(db, docket_id, config)),
        ]

        for stage_name, stage_fn in stages:
            console.print(f"[cyan]Running stage:[/cyan] {stage_name}")
            try:
                stage_fn()
                log_pipeline_run(db, docket_id, stage_name, "completed")
                console.print(f"  [green]Completed:[/green] {stage_name}")
            except Exception:
                logger.exception("Stage %s failed", stage_name)
                log_pipeline_run(db, docket_id, stage_name, "failed")
                console.print(f"  [red]Failed:[/red] {stage_name}")
                raise

        console.print(f"\n[bold green]Processing complete for {docket_id}[/bold green]")
    finally:
        db.close()


@cli.command("run-stage")
@click.argument("docket_id")
@click.option("--stage", required=True, type=click.Choice(["extract", "dedup", "embed", "topics", "classify", "report"]))
@click.option("--api-key", default=None, help="Regulations.gov API key (only needed for ingest)")
@click.pass_context
def run_stage(ctx: click.Context, docket_id: str, stage: str, api_key: str | None) -> None:
    """Run a specific pipeline stage on a docket."""
    config = ctx.obj["config"]

    from regscope.db import get_db, log_pipeline_run

    stage_map = {
        "extract": lambda: __import__("regscope.pipeline.extract", fromlist=["run_extract"]).run_extract(db, docket_id, config),
        "dedup": lambda: __import__("regscope.pipeline.dedup", fromlist=["run_dedup"]).run_dedup(db, docket_id, config),
        "embed": lambda: __import__("regscope.pipeline.embed", fromlist=["run_embed"]).run_embed(db, docket_id, config),
        "topics": lambda: __import__("regscope.pipeline.topics", fromlist=["run_topics"]).run_topics(db, docket_id, config),
        "classify": lambda: __import__("regscope.pipeline.classify", fromlist=["run_classify"]).run_classify(db, docket_id, config),
        "report": lambda: __import__("regscope.pipeline.report", fromlist=["run_report"]).run_report(db, docket_id, config),
    }

    db = get_db(docket_id, config)
    try:
        console.print(f"\n[bold]Running stage '{stage}' on {docket_id}[/bold]\n")
        stage_map[stage]()
        log_pipeline_run(db, docket_id, stage, "completed")
        console.print(f"\n[bold green]Stage '{stage}' complete[/bold green]")
    finally:
        db.close()


@cli.command()
@click.argument("docket_id")
@click.option("--format", "fmt", default="csv", type=click.Choice(["csv", "json", "excel"]))
@click.option("--output", "output_dir", default="./output/", help="Output directory")
@click.pass_context
def export(ctx: click.Context, docket_id: str, fmt: str, output_dir: str) -> None:
    """Export analysis results to CSV, JSON, or Excel."""
    config = ctx.obj["config"]

    from regscope.pipeline.report import run_export
    from regscope.db import get_db

    db = get_db(docket_id, config)
    try:
        console.print(f"\n[bold]Exporting {docket_id} as {fmt}[/bold]\n")
        run_export(db, docket_id, fmt, output_dir, config)
        console.print(f"\n[bold green]Export complete → {output_dir}[/bold green]")
    finally:
        db.close()


@cli.command()
@click.argument("docket_id")
@click.option("--output", "output_path", default=None, help="Output file path for report")
@click.pass_context
def report(ctx: click.Context, docket_id: str, output_path: str | None) -> None:
    """Generate a Markdown analysis report."""
    config = ctx.obj["config"]

    from regscope.pipeline.report import run_report
    from regscope.db import get_db

    db = get_db(docket_id, config)
    try:
        if output_path is None:
            output_path = f"./output/{docket_id}_report.md"
        console.print(f"\n[bold]Generating report for {docket_id}[/bold]\n")
        run_report(db, docket_id, config, output_path=output_path)
        console.print(f"\n[bold green]Report saved → {output_path}[/bold green]")
    finally:
        db.close()


@cli.command()
@click.argument("docket_id")
@click.pass_context
def status(ctx: click.Context, docket_id: str) -> None:
    """Show pipeline status for a docket."""
    config = ctx.obj["config"]

    from regscope.db import get_db, get_pipeline_status
    from rich.table import Table

    db = get_db(docket_id, config)
    try:
        stats = get_pipeline_status(db, docket_id)

        table = Table(title=f"Pipeline Status: {docket_id}")
        table.add_column("Stage", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Last Run", style="yellow")
        table.add_column("Details")

        for stage_info in stats:
            table.add_row(
                stage_info["stage"],
                stage_info["status"],
                stage_info["last_run"],
                stage_info["details"],
            )

        console.print(table)
    finally:
        db.close()


@cli.command("list")
@click.pass_context
def list_dockets(ctx: click.Context) -> None:
    """List all downloaded dockets."""
    config = ctx.obj["config"]

    from regscope.db import list_all_dockets
    from rich.table import Table

    dockets = list_all_dockets(config)

    if not dockets:
        console.print("[yellow]No dockets found.[/yellow]")
        return

    table = Table(title="Downloaded Dockets")
    table.add_column("Docket ID", style="cyan")
    table.add_column("Title")
    table.add_column("Comments", justify="right")
    table.add_column("Last Updated", style="yellow")

    for d in dockets:
        table.add_row(d["docket_id"], d["title"], str(d["comment_count"]), d["last_updated"])

    console.print(table)
