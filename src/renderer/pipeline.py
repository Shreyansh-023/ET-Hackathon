from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any

from src.common.config import PipelineConfig
from src.common.errors import RenderPipelineError, ValidationPipelineError
from src.common.models import Scene
from src.common.validation import validate_payload


SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".mpeg", ".mpg"}
SCENE_TEMPLATES = {
    "headline_card": "headline_card",
    "image_text_split": "image_plus_text_split",
    "quote_card": "quote_card",
    "bullet_scene": "bullet_scene",
    "outro_card": "outro_card",
}


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def _run_command(cmd: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> None:
    try:
        completed = subprocess.run(
            cmd,
            cwd=str(cwd),
            env=env,
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


def _aspect_to_dimensions(aspect_ratio: str) -> tuple[int, int]:
    normalized = aspect_ratio.strip()
    if normalized == "16:9":
        return 1920, 1080
    if normalized == "1:1":
        return 1080, 1080
    # Default mobile-first profile.
    return 1080, 1920


def _load_scenes(job_dir: Path) -> list[Scene]:
    scenes_path = job_dir / "storyboard" / "scenes.json"
    storyboard_path = job_dir / "storyboard" / "storyboard.json"

    payload: Any
    if scenes_path.exists():
        payload = _read_json(scenes_path)
        if not isinstance(payload, list):
            raise ValidationPipelineError("storyboard/scenes.json must be a JSON array")
    elif storyboard_path.exists():
        blob = _read_json(storyboard_path)
        payload = blob.get("scenes") if isinstance(blob, dict) else None
        if not isinstance(payload, list):
            raise ValidationPipelineError("storyboard/storyboard.json must include a scenes array")
    else:
        raise ValidationPipelineError(f"Missing storyboard scenes file: {scenes_path} or {storyboard_path}")

    return [validate_payload(Scene, row) for row in payload]


def _load_assets_by_scene(job_dir: Path) -> dict[str, dict[str, Any]]:
    registry_path = job_dir / "assets" / "assets_registry.json"
    if not registry_path.exists():
        raise ValidationPipelineError(f"Missing assets registry: {registry_path}")

    payload = _read_json(registry_path)
    rows = payload.get("assets") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise ValidationPipelineError("assets_registry.json must contain an assets array")

    by_scene: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        scene_id = str(row.get("scene_id", "")).strip()
        if not scene_id:
            continue
        by_scene[scene_id] = row
    return by_scene


def _choose_template_name(scene: Scene) -> str:
    if scene.type == "hook":
        return SCENE_TEMPLATES["headline_card"]
    if scene.type == "closing":
        return SCENE_TEMPLATES["outro_card"]

    lower_text = scene.on_screen_text.lower()
    if "\n-" in scene.on_screen_text or lower_text.count("; ") >= 2:
        return SCENE_TEMPLATES["bullet_scene"]
    if '"' in scene.narration or "quote" in lower_text:
        return SCENE_TEMPLATES["quote_card"]
    return SCENE_TEMPLATES["image_text_split"]


def _build_scene_manifest(
    job_id: str,
    scenes: list[Scene],
    assets_by_scene: dict[str, dict[str, Any]],
    *,
    job_dir: Path,
    fps: int,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    warnings: list[str] = []

    for scene in scenes:
        asset = assets_by_scene.get(scene.scene_id)
        if not asset:
            raise ValidationPipelineError(f"No visual asset for scene {scene.scene_id}")

        rel_image_path = str(asset.get("path", "")).strip()
        if not rel_image_path:
            raise ValidationPipelineError(f"Scene {scene.scene_id} has empty asset path")

        abs_image_path = job_dir / rel_image_path
        if not abs_image_path.exists():
            raise ValidationPipelineError(f"Scene {scene.scene_id} asset does not exist: {abs_image_path}")

        start_frame = int(round(scene.start * fps))
        end_frame = int(round(scene.end * fps))
        if end_frame <= start_frame:
            end_frame = start_frame + 1

        if abs_image_path.suffix.lower() not in (SUPPORTED_IMAGE_EXTENSIONS | SUPPORTED_VIDEO_EXTENSIONS):
            warnings.append(
                f"Scene {scene.scene_id} uses unsupported asset format ({abs_image_path.suffix}); FFmpeg color fallback will be used"
            )

        rows.append(
            {
                "scene_id": scene.scene_id,
                "template_name": _choose_template_name(scene),
                "start_frame": start_frame,
                "end_frame": end_frame,
                "image_path": str(abs_image_path),
                "text_blocks": [scene.on_screen_text, scene.narration],
                "transition_type": scene.transition_hint or "clean_cut",
                "motion_parameters": {
                    "profile": scene.motion_hint or "slow_push_in",
                    "safe_zone": {"x": 0.08, "y": 0.1, "width": 0.84, "height": 0.8},
                },
            }
        )

    return {
        "job_id": job_id,
        "fps": fps,
        "scene_count": len(rows),
        "templates": list(SCENE_TEMPLATES.values()),
        "scenes": rows,
        "warnings": warnings,
    }


def _transition_mix_frames(transition_hint: str, *, fps: int, left_frames: int, right_frames: int) -> int:
    hint = transition_hint.strip().lower()
    # Only skip transition if explicitly set to "none"
    if hint == "none":
        return 0

    # News-style: short, punchy transitions (~0.4s)
    if any(token in hint for token in ("fade", "dissolve", "cross", "mix", "blend")):
        base = max(1, int(round(fps * 0.4)))
    else:
        # Default for clean_cut and everything else: brief transition
        base = max(1, int(round(fps * 0.35)))

    return min(base, max(1, left_frames // 3), max(1, right_frames // 3))


def _ffmpeg_timestamp(seconds: float) -> str:
    return f"{max(0.0, seconds):.3f}"


def _transition_to_xfade(transition_hint: str, *, scene_index: int = 0) -> str:
    """Map transition hints — kept for fallback but primary approach is red bumper clips."""
    return "fade"


def _generate_red_transition_clip(
    output_path: Path,
    width: int,
    height: int,
    fps: int,
    duration_frames: int = 8,
) -> Path:
    """Generate a short red diagonal-stripe transition clip (news bumper style).

    Creates an animated red/white diagonal stripe wipe effect.
    """
    duration_s = duration_frames / float(fps)

    # Build an animated diagonal red stripe transition using geq + overlay
    # Phase 1 (first half): red stripes sweep across screen left-to-right
    # Phase 2 (second half): stripes continue and clear to reveal next scene
    #
    # We use multiple angled drawbox passes to create the diagonal stripe look,
    # animated with the 't' (time) variable for motion.
    stripe_w = width // 3  # each stripe width

    # Create the transition as: fade-in red → hold → fade-out
    # With diagonal stripes simulated via multiple colored bands
    filter_graph = (
        f"color=c=0xCC0000:s={width}x{height}:r={fps}:d={duration_s:.3f}[base];"
        # Lighter red diagonal stripe
        f"color=c=0xE83030:s={width}x{height}:r={fps}:d={duration_s:.3f}[s1];"
        # Pink/white highlight stripe
        f"color=c=0xFF6666:s={width}x{height}:r={fps}:d={duration_s:.3f}[s2];"
        # White flash stripe
        f"color=c=0xFFAAAA:s={width}x{height}:r={fps}:d={duration_s:.3f}[s3];"
        # Crop stripes to diagonal bands and overlay them
        # Stripe 1: offset animates with time
        f"[s1]crop=w={stripe_w}:h={height}:x='mod(t/{duration_s:.3f}*{width}+{stripe_w*0},{width})':y=0[c1];"
        f"[base][c1]overlay=x='mod(t/{duration_s:.3f}*{width*2}+0,{width})-{stripe_w//2}':y=0:shortest=1[o1];"
        # Stripe 2
        f"[s2]crop=w={stripe_w//2}:h={height}:x=0:y=0[c2];"
        f"[o1][c2]overlay=x='mod(t/{duration_s:.3f}*{width*2}+{stripe_w},{width})-{stripe_w//3}':y=0:shortest=1[o2];"
        # Stripe 3 (white flash)
        f"[s3]crop=w={stripe_w//3}:h={height}:x=0:y=0[c3];"
        f"[o2][c3]overlay=x='mod(t/{duration_s:.3f}*{width*2}+{stripe_w*2},{width})-{stripe_w//4}':y=0:shortest=1[o3];"
        # Add fade in/out for smooth entry and exit
        f"[o3]fade=t=in:st=0:d={duration_s*0.3:.3f},fade=t=out:st={duration_s*0.6:.3f}:d={duration_s*0.4:.3f}[out]"
    )

    cmd = [
        "ffmpeg", "-y",
        "-filter_complex", filter_graph,
        "-map", "[out]",
        "-an",
        "-c:v", "libx264",
        "-preset", os.getenv("FFMPEG_PRESET", "medium"),
        "-crf", os.getenv("FFMPEG_CRF", "22"),
        "-pix_fmt", "yuv420p",
        "-t", f"{duration_s:.3f}",
        str(output_path),
    ]

    try:
        _run_command(cmd, cwd=output_path.parent)
    except RenderPipelineError:
        # Fallback: simple solid red with fade if complex filter fails
        simple_cmd = [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i",
            f"color=c=0xCC0000:s={width}x{height}:r={fps}:d={duration_s:.3f}",
            "-vf", f"fade=t=in:st=0:d={duration_s*0.4:.3f},fade=t=out:st={duration_s*0.5:.3f}:d={duration_s*0.5:.3f}",
            "-an",
            "-c:v", "libx264",
            "-preset", os.getenv("FFMPEG_PRESET", "medium"),
            "-crf", os.getenv("FFMPEG_CRF", "22"),
            "-pix_fmt", "yuv420p",
            str(output_path),
        ]
        _run_command(simple_cmd, cwd=output_path.parent)

    return output_path


def _build_visual_filter(width: int, height: int, fps: int) -> str:
    return (
        f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,"
        "setsar=1,"
        f"fps={fps},"
        "format=yuv420p"
    )


def _run_ffmpeg_scene_render(
    *,
    source_path: Path | None,
    output_path: Path,
    duration_seconds: float,
    width: int,
    height: int,
    fps: int,
) -> str:
    filter_chain = _build_visual_filter(width, height, fps)

    if source_path is None:
        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"color=c=black:s={width}x{height}:r={fps}:d={_ffmpeg_timestamp(duration_seconds)}",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            os.getenv("FFMPEG_PRESET", "medium"),
            "-crf",
            os.getenv("FFMPEG_CRF", "22"),
            "-pix_fmt",
            "yuv420p",
            str(output_path),
        ]
        _run_command(cmd, cwd=output_path.parent)
        return "color:black"

    suffix = source_path.suffix.lower()
    if suffix in SUPPORTED_IMAGE_EXTENSIONS:
        cmd = [
            "ffmpeg",
            "-y",
            "-loop",
            "1",
            "-t",
            _ffmpeg_timestamp(duration_seconds),
            "-i",
            str(source_path),
            "-vf",
            filter_chain,
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            os.getenv("FFMPEG_PRESET", "medium"),
            "-crf",
            os.getenv("FFMPEG_CRF", "22"),
            "-pix_fmt",
            "yuv420p",
            str(output_path),
        ]
        _run_command(cmd, cwd=output_path.parent)
        return str(source_path)

    cmd = [
        "ffmpeg",
        "-y",
        "-stream_loop",
        "-1",
        "-i",
        str(source_path),
        "-t",
        _ffmpeg_timestamp(duration_seconds),
        "-vf",
        filter_chain,
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        os.getenv("FFMPEG_PRESET", "medium"),
        "-crf",
        os.getenv("FFMPEG_CRF", "22"),
        "-pix_fmt",
        "yuv420p",
        str(output_path),
    ]
    _run_command(cmd, cwd=output_path.parent)
    return str(source_path)


def _attempt_ffmpeg_run(
    *,
    job_dir: Path,
    manifest: dict[str, Any],
    width: int,
    height: int,
    fps: int,
) -> dict[str, Any]:
    if shutil.which("ffmpeg") is None:
        return {"status": "skipped", "reason": "ffmpeg_binary_not_found"}

    scenes = manifest.get("scenes")
    if not isinstance(scenes, list) or not scenes:
        return {"status": "skipped", "reason": "ffmpeg_manifest_has_no_scenes"}

    scene_renders_dir = job_dir / "renders" / "scenes"
    scene_renders_dir.mkdir(parents=True, exist_ok=True)
    intermediate_raw = job_dir / "renders" / "intermediate_raw.mp4"
    concat_list_path = job_dir / "renders" / "scene_concat.txt"

    segment_paths: list[Path] = []
    scene_attempts: list[dict[str, Any]] = []
    transitions: list[dict[str, Any]] = []

    for index, row in enumerate(scenes, start=1):
        if not isinstance(row, dict):
            continue

        scene_id = str(row.get("scene_id", f"scene-{index:03d}"))
        start_frame = int(row.get("start_frame", 0))
        end_frame = int(row.get("end_frame", start_frame + 1))
        duration = max(1, end_frame - start_frame)

        source_path = Path(str(row.get("image_path", "")))
        if source_path.exists() and source_path.suffix.lower() in (SUPPORTED_IMAGE_EXTENSIONS | SUPPORTED_VIDEO_EXTENSIONS):
            normalized_source: Path | None = source_path
        else:
            normalized_source = None

        duration_seconds = duration / float(fps)
        segment_path = scene_renders_dir / f"scene_{index:03d}.mp4"
        segment_reference = _run_ffmpeg_scene_render(
            source_path=normalized_source,
            output_path=segment_path,
            duration_seconds=duration_seconds,
            width=width,
            height=height,
            fps=fps,
        )
        segment_paths.append(segment_path)

        if index > 1:
            previous_duration_frames = int(scene_attempts[-1]["duration_frames"])
            mix_frames = _transition_mix_frames(
                str(scene_attempts[-1].get("transition_hint", "")),
                fps=fps,
                left_frames=previous_duration_frames,
                right_frames=duration,
            )
            transition_hint = str(scene_attempts[-1].get("transition_hint", ""))
            transitions.append(
                {
                    "from_scene": scene_attempts[-1]["scene_id"],
                    "to_scene": scene_id,
                    "hint": transition_hint,
                    "xfade": _transition_to_xfade(transition_hint, scene_index=index),
                    "duration_frames": mix_frames,
                }
            )

        scene_attempts.append(
            {
                "scene_id": scene_id,
                "attempts": 1,
                "segment": str(segment_path),
                "segment_source": segment_reference,
                "duration_frames": duration,
                "duration_seconds": round(duration_seconds, 3),
                "transition_hint": str(row.get("transition_type", "")).strip(),
            }
        )

    if not segment_paths:
        return {"status": "skipped", "reason": "ffmpeg_no_renderable_scenes"}

    # Generate a red transition bumper clip
    red_transition_path = scene_renders_dir / "red_transition.mp4"
    transition_frames = 8  # ~0.27s at 30fps
    _generate_red_transition_clip(
        red_transition_path, width, height, fps, duration_frames=transition_frames,
    )

    concat_mode = "red_bumper"
    concat_list_value = ""

    # Build concat list: scene1 → red → scene2 → red → scene3 → ...
    concat_segments: list[Path] = []
    for i, seg_path in enumerate(segment_paths):
        concat_segments.append(seg_path)
        if i < len(segment_paths) - 1 and red_transition_path.exists():
            concat_segments.append(red_transition_path)

    concat_file_payload = "\n".join(f"file '{path.resolve().as_posix()}'" for path in concat_segments)
    concat_list_path.write_text(f"{concat_file_payload}\n", encoding="utf-8")
    concat_list_value = str(concat_list_path)

    # Try concat with stream copy first, fall back to re-encode
    concat_copy_cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", str(concat_list_path),
        "-c", "copy",
        str(intermediate_raw),
    ]
    try:
        _run_command(concat_copy_cmd, cwd=job_dir)
    except RenderPipelineError:
        concat_reencode_cmd = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", str(concat_list_path),
            "-c:v", "libx264",
            "-preset", os.getenv("FFMPEG_PRESET", "medium"),
            "-crf", os.getenv("FFMPEG_CRF", "22"),
            "-pix_fmt", "yuv420p",
            str(intermediate_raw),
        ]
        _run_command(concat_reencode_cmd, cwd=job_dir)
        concat_mode = "reencode"

    if not intermediate_raw.exists():
        return {"status": "skipped", "reason": "ffmpeg_output_not_produced"}

    return {
        "status": "completed",
        "output": str(intermediate_raw),
        "segments": [str(path) for path in segment_paths],
        "concat_list": concat_list_value,
        "concat_mode": concat_mode,
        "transitions": transitions,
        "fps": fps,
        "resolution": f"{width}x{height}",
        "scene_attempts": scene_attempts,
    }


def _add_voiceover_and_optional_music(
    *,
    job_dir: Path,
    input_video: Path,
    voiceover_path: Path,
) -> Path:
    output_path = job_dir / "renders" / "intermediate_with_audio.mp4"
    music_bed = job_dir / "audio" / "music_bed.wav"

    if not voiceover_path.exists():
        shutil.copyfile(input_video, output_path)
        return output_path

    if music_bed.exists():
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(input_video),
            "-i",
            str(voiceover_path),
            "-i",
            str(music_bed),
            "-filter_complex",
            (
                "[2:a]volume=0.22[music];"
                "[music][1:a]sidechaincompress=threshold=0.08:ratio=10:attack=15:release=300[ducked];"
                "[1:a][ducked]amix=inputs=2:weights='1 0.55':normalize=0[aout]"
            ),
            "-map",
            "0:v:0",
            "-map",
            "[aout]",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-shortest",
            str(output_path),
        ]
    else:
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(input_video),
            "-i",
            str(voiceover_path),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-shortest",
            str(output_path),
        ]

    _run_command(cmd, cwd=job_dir)
    return output_path


def build_render_stage(job_id: str, cfg: PipelineConfig, *, job_dir: Path, logger=None) -> dict[str, Any]:
    job_dir = job_dir.resolve()
    width, height = _aspect_to_dimensions(cfg.aspect_ratio)
    fps = int(os.getenv("RENDER_FPS", "30"))
    report_path = job_dir / "renders" / "render_report.json"

    try:
        scenes = _load_scenes(job_dir)
        assets_by_scene = _load_assets_by_scene(job_dir)
        manifest = _build_scene_manifest(
            job_id,
            scenes,
            assets_by_scene,
            job_dir=job_dir,
            fps=fps,
        )

        manifest_path = job_dir / "renders" / "scene_manifest.json"
        _write_json(manifest_path, manifest)

        requested_engine = os.getenv("RENDER_ENGINE", "auto").strip().lower() or "auto"
        if requested_engine not in {"auto", "ffmpeg"}:
            raise ValidationPipelineError(
                f"Unsupported RENDER_ENGINE='{requested_engine}'. This pipeline now supports FFmpeg only."
            )

        ffmpeg_result = _attempt_ffmpeg_run(
            job_dir=job_dir,
            manifest=manifest,
            width=width,
            height=height,
            fps=fps,
        )
        if ffmpeg_result.get("status") != "completed":
            reason = str(ffmpeg_result.get("reason", "ffmpeg_render_failed"))
            raise RenderPipelineError(f"FFmpeg render failed: {reason}")

        intermediate_raw = job_dir / "renders" / "intermediate_raw.mp4"
        scene_attempts = list(ffmpeg_result.get("scene_attempts", []))

        voiceover_path = job_dir / "audio" / "voiceover.wav"
        intermediate_with_audio = _add_voiceover_and_optional_music(
            job_dir=job_dir,
            input_video=intermediate_raw,
            voiceover_path=voiceover_path,
        )

        payload = {
            "job_id": job_id,
            "status": "rendered",
            "engine": "ffmpeg",
            "scene_manifest": str(manifest_path.relative_to(job_dir)),
            "intermediate_raw": str(intermediate_raw.relative_to(job_dir)),
            "intermediate_with_audio": str(intermediate_with_audio.relative_to(job_dir)),
            "ffmpeg": ffmpeg_result,
            "scene_attempts": scene_attempts,
            "fps": fps,
            "resolution": f"{width}x{height}",
            "warnings": manifest.get("warnings", []),
        }

        if logger:
            logger.emit(
                "info",
                "render_stage_outputs",
                manifest=payload["scene_manifest"],
                intermediate=payload["intermediate_with_audio"],
            )
        return payload
    except Exception as exc:
        _write_json(
            report_path,
            {
                "job_id": job_id,
                "status": "failed",
                "settings": {
                    "aspect_ratio": cfg.aspect_ratio,
                    "fps": fps,
                    "resolution": f"{width}x{height}",
                },
                "warnings": [],
                "failures": [str(exc)],
            },
        )
        if logger:
            logger.emit("error", "render_stage_failed", error=str(exc))
        raise
