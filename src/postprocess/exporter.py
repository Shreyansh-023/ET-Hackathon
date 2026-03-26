from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any

from src.common.config import PipelineConfig
from src.common.errors import RenderPipelineError, ValidationPipelineError


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def _run_command(cmd: list[str], *, cwd: Path) -> None:
    try:
        completed = subprocess.run(
            cmd,
            cwd=str(cwd),
            check=False,
            text=True,
            capture_output=True,
        )
    except OSError as exc:
        raise RenderPipelineError(f"Failed running command: {' '.join(cmd)} ({exc})") from exc

    if completed.returncode != 0:
        stderr = (completed.stderr or "").strip()
        stdout = (completed.stdout or "").strip()
        details = stderr or stdout or "Unknown command failure"
        raise RenderPipelineError(f"Command failed ({completed.returncode}): {' '.join(cmd)} :: {details[:1200]}")


def _probe_duration_seconds(path: Path, *, cwd: Path) -> float | None:
    if shutil.which("ffprobe") is None:
        return None
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    try:
        completed = subprocess.run(
            cmd,
            cwd=str(cwd),
            check=False,
            text=True,
            capture_output=True,
        )
    except OSError:
        return None
    if completed.returncode != 0:
        return None

    raw = (completed.stdout or "").strip()
    try:
        return round(float(raw), 3)
    except ValueError:
        return None


def _resolve_intermediate_video(job_dir: Path) -> Path:
    candidates = [
        job_dir / "renders" / "intermediate_with_audio.mp4",
        job_dir / "renders" / "intermediate_raw.mp4",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise ValidationPipelineError("No intermediate render output found. Run render stage first.")


def _build_subtitle_filter(subtitles_path: Path) -> str:
    resolved = subtitles_path.resolve().as_posix()
    if len(resolved) >= 2 and resolved[1] == ":":
        # ffmpeg filter parser needs escaped drive-letter colon on Windows.
        resolved = f"{resolved[0]}\\:{resolved[2:]}"
    path_value = resolved.replace("'", "\\'")
    return f"subtitles='{path_value}'"


def _render_final(
    *,
    job_dir: Path,
    input_video: Path,
    subtitles_path: Path | None,
    burn_subtitles: bool,
) -> Path:
    final_path = job_dir / "renders" / "final.mp4"
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_video),
    ]
    if burn_subtitles and subtitles_path and subtitles_path.exists():
        cmd.extend(["-vf", _build_subtitle_filter(subtitles_path)])

    cmd.extend(
        [
            "-c:v",
            "libx264",
            "-preset",
            os.getenv("FFMPEG_PRESET", "medium"),
            "-crf",
            os.getenv("FFMPEG_CRF", "22"),
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-af",
            "loudnorm=I=-16:TP=-1.5:LRA=11",
            "-movflags",
            "+faststart",
            str(final_path),
        ]
    )
    _run_command(cmd, cwd=job_dir)
    return final_path


def _render_preview(*, job_dir: Path, final_path: Path) -> Path:
    preview_path = job_dir / "renders" / "preview.mp4"
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(final_path),
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "30",
        "-maxrate",
        "700k",
        "-bufsize",
        "1400k",
        "-c:a",
        "aac",
        "-b:a",
        "96k",
        "-movflags",
        "+faststart",
        str(preview_path),
    ]
    _run_command(cmd, cwd=job_dir)
    return preview_path


def _render_thumbnail(*, job_dir: Path, final_path: Path) -> Path:
    thumbnail_path = job_dir / "renders" / "thumbnail.jpg"
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(final_path),
        "-ss",
        "00:00:01",
        "-vframes",
        "1",
        str(thumbnail_path),
    ]
    _run_command(cmd, cwd=job_dir)
    return thumbnail_path


def build_export_package(
    job_id: str,
    cfg: PipelineConfig,
    *,
    job_dir: Path,
    logger=None,
    force: bool = False,
) -> dict[str, Any]:
    job_dir = job_dir.resolve()
    if shutil.which("ffmpeg") is None:
        raise RenderPipelineError("ffmpeg is required for export stage but was not found in PATH")

    final_path = job_dir / "renders" / "final.mp4"
    preview_path = job_dir / "renders" / "preview.mp4"
    thumbnail_path = job_dir / "renders" / "thumbnail.jpg"
    report_path = job_dir / "renders" / "render_report.json"

    warnings: list[str] = []
    failures: list[str] = []
    subtitles_path = job_dir / "audio" / "subtitles.srt"
    burn_subtitles_raw = os.getenv("BURN_SUBTITLES", "auto").strip().lower() or "auto"
    if burn_subtitles_raw in {"0", "false", "no", "off"}:
        burn_subtitles = False
    elif burn_subtitles_raw in {"1", "true", "yes", "on"}:
        burn_subtitles = True
    else:
        burn_subtitles = subtitles_path.exists()

    try:
        if force or not final_path.exists():
            input_video = _resolve_intermediate_video(job_dir)
            final_path = _render_final(
                job_dir=job_dir,
                input_video=input_video,
                subtitles_path=subtitles_path,
                burn_subtitles=burn_subtitles,
            )
        if force or not preview_path.exists():
            preview_path = _render_preview(job_dir=job_dir, final_path=final_path)
        if force or not thumbnail_path.exists():
            thumbnail_path = _render_thumbnail(job_dir=job_dir, final_path=final_path)
    except RenderPipelineError as exc:
        failures.append(str(exc))
        report = {
            "job_id": job_id,
            "status": "failed",
            "settings": {
                "aspect_ratio": cfg.aspect_ratio,
                "duration_target_seconds": cfg.duration_seconds,
                "burn_subtitles": burn_subtitles,
            },
            "warnings": warnings,
            "failures": failures,
        }
        _write_json(report_path, report)
        raise

    final_duration = _probe_duration_seconds(final_path, cwd=job_dir)
    if final_duration is not None and abs(final_duration - cfg.duration_seconds) > 5.0:
        warnings.append(
            f"Final duration {final_duration}s differs from target {cfg.duration_seconds}s by more than tolerance"
        )
    if burn_subtitles and not subtitles_path.exists():
        warnings.append("Subtitle burn-in requested but subtitles.srt was not found")

    report = {
        "job_id": job_id,
        "status": "completed",
        "settings": {
            "aspect_ratio": cfg.aspect_ratio,
            "duration_target_seconds": cfg.duration_seconds,
            "burn_subtitles": burn_subtitles,
            "video_codec": "libx264",
            "audio_codec": "aac",
        },
        "durations": {
            "final_seconds": final_duration,
        },
        "outputs": {
            "final": str(final_path.relative_to(job_dir)),
            "preview": str(preview_path.relative_to(job_dir)),
            "thumbnail": str(thumbnail_path.relative_to(job_dir)),
        },
        "warnings": warnings,
        "failures": failures,
    }
    _write_json(report_path, report)

    payload = {
        "job_id": job_id,
        "status": "exported",
        "final": str(final_path.relative_to(job_dir)),
        "preview": str(preview_path.relative_to(job_dir)),
        "thumbnail": str(thumbnail_path.relative_to(job_dir)),
        "render_report": str(report_path.relative_to(job_dir)),
    }
    if logger:
        logger.emit(
            "info",
            "export_outputs_written",
            final=payload["final"],
            preview=payload["preview"],
            thumbnail=payload["thumbnail"],
        )
    return payload


def build_export_stub(job_id: str) -> dict[str, Any]:
    # Backward-compatible helper retained for older callers.
    return {"job_id": job_id, "status": "exported_stub", "output": f"renders/{job_id}.mp4"}
