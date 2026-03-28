from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer
from rich.console import Console

from src.assets.pipeline import build_assets_step
from src.common.config import load_config
from src.common.errors import PipelineError, RenderPipelineError, ValidationPipelineError
from src.common.models import Article, Asset, Scene
from src.common.retry import run_with_retry
from src.common.validation import validate_payload
from src.ingest.stage0 import ingest_article
from src.planner.engine import plan_storyboard
from src.postprocess.exporter import build_export_package
from src.renderer.pipeline import build_render_stage
from src.storage.repository import JobRepository

app = typer.Typer(help="Deterministic news-to-video pipeline (Step 1 baseline)")
console = Console()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _run_stage(
    repo: JobRepository,
    job_id: str,
    stage: str,
    retries: int,
    fn,
):
    job = repo.load_render_job(job_id)
    job = repo.mark_stage_started(job, stage)
    logger = repo.get_stage_logger(job_id, stage)
    logger.emit("info", "stage_started", job_id=job_id, stage=stage)

    try:
        result = run_with_retry(fn, retries=retries)
        job = repo.mark_stage_completed(job, stage)
        logger.emit("info", "stage_completed", job_id=job_id, stage=stage)
        return result, job
    except PipelineError as exc:
        repo.update_stage_status(job_id, stage, "failed", error=str(exc))
        logger.emit("error", "stage_failed", job_id=job_id, stage=stage, error=str(exc))
        raise


@app.command()
def ingest(
    file_or_url: str = typer.Argument(..., help="Raw text, file path, or URL."),
    job_id: str | None = typer.Option(None, help="Existing job id. Leave empty to create new."),
    title: str = typer.Option("", help="Optional article title."),
    dry_run: bool = typer.Option(False, help="Run without paid provider calls."),
) -> None:
    """Stage 0 ingestion: normalize text and persist Article payload."""
    cfg = load_config()
    repo = JobRepository(cfg.jobs_root)

    if job_id:
        job = repo.load_render_job(job_id)
    else:
        job = repo.create_job(dry_run=dry_run)
        job_id = job.job_id

    def _work() -> Article:
        article = ingest_article(file_or_url, title=title)
        payload = validate_payload(Article, article.model_dump())
        job_dir = repo.job_path(job_id)
        _write_json(job_dir / "parsed" / "article.json", payload.model_dump())
        (job_dir / "input" / "source.txt").write_text(file_or_url, encoding="utf-8")
        repo.add_artifact(job_id, "article", "parsed/article.json")
        return payload

    try:
        article, updated_job = _run_stage(repo, job_id, "ingest", cfg.retry_limits["ingest"], _work)
        updated_job = updated_job.model_copy(update={"article_id": article.article_id})
        repo.save_render_job(updated_job)
        console.print(f"[green]Ingest complete[/green] job_id={job_id} article_id={article.article_id}")
    except PipelineError as exc:
        console.print(f"[red]Ingest failed:[/red] {exc}")
        raise typer.Exit(code=1)


@app.command()
def plan(
    job_id: str = typer.Argument(..., help="Job id created by ingest stage."),
    dry_run: bool = typer.Option(False, help="Run without paid provider calls."),
) -> None:
    """LLM storyboard planning stage with deterministic fallback and timing policy."""
    cfg = load_config()
    repo = JobRepository(cfg.jobs_root)

    def _work() -> list[Scene]:
        job_dir = repo.job_path(job_id)
        article_payload = _read_json(job_dir / "parsed" / "article.json")
        article = validate_payload(Article, article_payload)
        llm_enabled = bool(cfg.gemini_api_key) and not dry_run
        _, scenes = plan_storyboard(article, cfg, job_dir=job_dir, llm_enabled=llm_enabled)
        for scene in scenes:
            validate_payload(Scene, scene.model_dump())
        _write_json(job_dir / "storyboard" / "scenes.json", [s.model_dump() for s in scenes])
        repo.add_artifact(job_id, "scenes", "storyboard/scenes.json")
        repo.add_artifact(job_id, "article_understanding", "storyboard/article_understanding.json")
        repo.add_artifact(job_id, "storyboard", "storyboard/storyboard.json")
        repo.add_artifact(job_id, "planner_prompt", "storyboard/planner_prompt.txt")
        repo.add_artifact(job_id, "planner_output_raw", "storyboard/planner_output_raw.txt")
        return scenes

    try:
        _run_stage(repo, job_id, "plan", cfg.retry_limits["plan"], _work)
        console.print(f"[green]Plan complete[/green] job_id={job_id} dry_run={dry_run}")
    except PipelineError as exc:
        console.print(f"[red]Plan failed:[/red] {exc}")
        raise typer.Exit(code=1)


@app.command()
def assets(
    job_id: str = typer.Argument(..., help="Job id created by ingest stage."),
    dry_run: bool = typer.Option(False, help="Run without paid provider calls."),
) -> None:
    """Step 3: resolve visuals with provider fallbacks, then build voiceover + subtitles."""
    cfg = load_config()
    repo = JobRepository(cfg.jobs_root)

    def _work() -> list[Asset]:
        job_dir = repo.job_path(job_id)
        article_payload = _read_json(job_dir / "parsed" / "article.json")
        article = validate_payload(Article, article_payload)
        scenes_payload = _read_json(job_dir / "storyboard" / "scenes.json")
        scenes = [validate_payload(Scene, row) for row in scenes_payload]
        logger = repo.get_stage_logger(job_id, "assets")
        step3 = build_assets_step(
            scenes,
            cfg,
            job_dir,
            article_language=article.language,
            dry_run=dry_run,
            logger=logger,
        )
        built_assets = step3.assets
        for item in built_assets:
            validate_payload(Asset, item.model_dump())

        # Compatibility with step contracts and downstream pipeline consumers.
        _write_json(job_dir / "assets" / "assets.json", [a.model_dump() for a in built_assets])
        repo.add_artifact(job_id, "assets", "assets/assets.json")
        repo.add_artifact(job_id, "assets_registry", step3.assets_registry_rel_path)
        repo.add_artifact(job_id, "visual_plan", "assets/visual_plan.json")
        repo.add_artifact(job_id, "visual_plan_prompt", "assets/visual_plan_prompt.txt")
        repo.add_artifact(job_id, "visual_plan_output_raw", "assets/visual_plan_output_raw.txt")
        repo.add_artifact(job_id, "voiceover", step3.audio.voiceover_rel_path)
        repo.add_artifact(job_id, "subtitles_srt", step3.audio.subtitles_srt_rel_path)
        repo.add_artifact(job_id, "subtitles_vtt", step3.audio.subtitles_vtt_rel_path)
        repo.add_artifact(job_id, "audio_manifest", step3.audio.audio_manifest_rel_path)
        return built_assets

    try:
        _run_stage(repo, job_id, "assets", cfg.retry_limits["assets"], _work)
        console.print(f"[green]Assets complete[/green] job_id={job_id} dry_run={dry_run}")
    except PipelineError as exc:
        console.print(f"[red]Assets failed:[/red] {exc}")
        raise typer.Exit(code=1)


@app.command()
def render(
    job_id: str = typer.Argument(..., help="Job id created by ingest stage."),
    dry_run: bool = typer.Option(False, help="Run without paid provider calls."),
) -> None:
    """Step 4 render stage: compose scenes with FFmpeg and produce final outputs."""
    cfg = load_config()
    repo = JobRepository(cfg.jobs_root)

    def _work() -> dict[str, Any]:
        job_dir = repo.job_path(job_id)
        logger = repo.get_stage_logger(job_id, "render")
        render_payload = build_render_stage(job_id, cfg, job_dir=job_dir, logger=logger)

        if "job_id" not in render_payload:
            raise RenderPipelineError("Render payload missing job_id")

        _write_json(job_dir / "renders" / "render_job.json", render_payload)
        repo.add_artifact(job_id, "render_job", "renders/render_job.json")
        repo.add_artifact(job_id, "scene_manifest", "renders/scene_manifest.json")
        repo.add_artifact(job_id, "intermediate_raw", "renders/intermediate_raw.mp4")
        repo.add_artifact(job_id, "intermediate_with_audio", "renders/intermediate_with_audio.mp4")
        return render_payload

    try:
        _run_stage(repo, job_id, "render", cfg.retry_limits["render"], _work)
        console.print(f"[green]Render complete[/green] job_id={job_id} dry_run={dry_run}")
    except PipelineError as exc:
        console.print(f"[red]Render failed:[/red] {exc}")
        raise typer.Exit(code=1)


@app.command(name="export")
def export_job(
    job_id: str = typer.Argument(..., help="Job id created by ingest stage."),
    dry_run: bool = typer.Option(False, help="Run without paid provider calls."),
) -> None:
    """Export stage: ffmpeg packaging and final report generation."""
    cfg = load_config()
    repo = JobRepository(cfg.jobs_root)

    def _work() -> dict[str, Any]:
        job_dir = repo.job_path(job_id)
        logger = repo.get_stage_logger(job_id, "export")
        export_payload = build_export_package(job_id, cfg, job_dir=job_dir, logger=logger)
        _write_json(job_dir / "renders" / "export.json", export_payload)
        repo.add_artifact(job_id, "export", "renders/export.json")
        repo.add_artifact(job_id, "final", export_payload["final"])
        repo.add_artifact(job_id, "preview", export_payload["preview"])
        repo.add_artifact(job_id, "thumbnail", export_payload["thumbnail"])
        repo.add_artifact(job_id, "render_report", export_payload["render_report"])
        manifest = repo.load_manifest(job_id)
        _write_json(job_dir / "renders" / "final_manifest.json", manifest.model_dump())
        return export_payload

    try:
        _run_stage(repo, job_id, "export", cfg.retry_limits["export"], _work)
        console.print(f"[green]Export complete[/green] job_id={job_id} dry_run={dry_run}")
    except ValidationPipelineError as exc:
        console.print(f"[red]Export validation failed:[/red] {exc}")
        raise typer.Exit(code=1)
    except PipelineError as exc:
        console.print(f"[red]Export failed:[/red] {exc}")
        raise typer.Exit(code=1)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
