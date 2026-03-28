from __future__ import annotations

import json
import os
import re
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
    if os.getenv("DEBUG_FFMPEG_CMD", "").strip().lower() in {"1", "true", "yes", "on"}:
        print("FFMPEG CMD:", " ".join(cmd))
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


def _normalize_input_video(
    input_video: Path,
    *,
    job_dir: Path,
    fps: int,
    force: bool,
) -> Path:
    """Normalize input video to a stable CFR stream for export compositing."""
    normalized_path = job_dir / "renders" / "intermediate_normalized.mp4"
    if normalized_path.exists() and not force:
        return normalized_path

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_video),
        "-vf",
        f"fps={fps},format=yuv420p",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        os.getenv("FFMPEG_PRESET", "medium"),
        "-crf",
        os.getenv("FFMPEG_CRF", "22"),
        str(normalized_path),
    ]
    _run_command(cmd, cwd=job_dir)
    return normalized_path


def _parse_srt(srt_path: Path) -> list[dict]:
    """Parse an SRT file into a list of {start, end, text} dicts (times in seconds)."""
    content = srt_path.read_text(encoding="utf-8")
    blocks = re.split(r"\n\s*\n", content.strip())
    entries = []
    for block in blocks:
        lines = block.strip().splitlines()
        if len(lines) < 3:
            continue
        time_match = re.match(
            r"(\d{2}):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})[,.](\d{3})",
            lines[1],
        )
        if not time_match:
            continue
        g = [int(x) for x in time_match.groups()]
        start = g[0] * 3600 + g[1] * 60 + g[2] + g[3] / 1000
        end = g[4] * 3600 + g[5] * 60 + g[6] + g[7] / 1000
        text = " ".join(lines[2:]).strip()
        if text:
            entries.append({"start": start, "end": end, "text": text})
    return entries


def _escape_drawtext(text: str) -> str:
    """Escape special characters for ffmpeg drawtext filter script syntax."""
    text = text.replace("\\", "\\\\")
    text = text.replace("'", "\u2019")  # replace apostrophes with unicode right single quote
    text = text.replace(":", "\\:")
    text = text.replace(";", "\\;")
    text = text.replace("%", "%%")
    text = text.replace("[", "\\[")
    text = text.replace("]", "\\]")
    return text


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def _escape_ffmpeg_path(path: Path) -> str:
    """Escape a path for ffmpeg filter values."""
    resolved = path.resolve().as_posix()
    if len(resolved) >= 2 and resolved[1] == ":":
        resolved = f"{resolved[0]}\\:{resolved[2:]}"
    return resolved.replace("'", "\\'")


def _resolve_drawtext_fontfile() -> Path | None:
    """Resolve a font that can render Devanagari text for drawtext."""
    env_candidates = [
        os.getenv("FFMPEG_FONT_FILE", "").strip(),
        os.getenv("HINDI_FONT_FILE", "").strip(),
    ]
    candidates = [Path(p) for p in env_candidates if p]

    root = _project_root()
    candidates.extend(
        [
            root / "assets" / "fonts" / "NotoSansDevanagari-Regular.ttf",
            root / "assets" / "fonts" / "Nirmala.ttc",
            root / "assets" / "fonts" / "Nirmala.ttf",
            Path("C:/Windows/Fonts/NotoSansDevanagari-Regular.ttf"),
            Path("C:/Windows/Fonts/Nirmala.ttc"),
            Path("C:/Windows/Fonts/Nirmala.ttf"),
            Path("C:/Windows/Fonts/Mangal.ttf"),
        ]
    )

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _resolve_subtitle_font_name() -> str:
    """Resolve a libass font family name for subtitle rendering."""
    return (
        os.getenv("FFMPEG_FONT_NAME", "").strip()
        or os.getenv("HINDI_FONT_NAME", "").strip()
        or "Nirmala UI"
    )


def _build_drawtext_filter_script(
    srt_path: Path,
    job_dir: Path,
    *,
    fontfile: Path | None = None,
) -> Path:
    """Build a drawtext filter script file from SRT — works without libass.

    Returns the path to the filter script file.
    """
    entries = _parse_srt(srt_path)
    script_path = job_dir / "renders" / "subtitle_filter.txt"
    script_path.parent.mkdir(parents=True, exist_ok=True)

    if not entries:
        script_path.write_text("null", encoding="utf-8")
        return script_path

    filters = []
    for entry in entries:
        escaped = _escape_drawtext(entry["text"])
        start = entry["start"]
        end = entry["end"]
        fontfile_part = ""
        if fontfile is not None:
            fontfile_part = f":fontfile='{_escape_ffmpeg_path(fontfile)}'"
        # News-style subtitle: white text, semi-transparent dark box, bottom-center
        dt = (
            f"drawtext=text='{escaped}'"
            f"{fontfile_part}"
            f":fontsize=28"
            f":fontcolor=white"
            f":borderw=2"
            f":bordercolor=black"
            f":box=1"
            f":boxcolor=black@0.6"
            f":boxborderw=8"
            f":x=(w-text_w)/2"
            f":y=h-th-40"
            f":enable='between(t\\,{start:.3f}\\,{end:.3f})'"
        )
        filters.append(dt)

    script_path.write_text(",\n".join(filters), encoding="utf-8")
    return script_path


def _build_subtitle_filter(subtitles_path: Path, *, font_name: str | None = None) -> str:
    """Use libass subtitles filter if available."""
    path_value = _escape_ffmpeg_path(subtitles_path)
    if font_name:
        safe_font_name = font_name.replace("'", "\\'")
        return (
            f"subtitles='{path_value}'"
            f":force_style='FontName={safe_font_name},BorderStyle=1,Outline=1,Shadow=0'"
        )
    return f"subtitles='{path_value}'"


def _has_subtitle_filter() -> bool:
    """Check if ffmpeg was built with the subtitles (libass) filter."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-filters"],
            capture_output=True, text=True, check=False,
        )
        return "subtitles" in (result.stdout or "")
    except OSError:
        return False


MAX_DRAWTEXT_SUBTITLE_CUES = 40


def _asset_search_roots() -> list[Path]:
    project_root = Path(__file__).resolve().parent.parent.parent
    return [
        project_root,
        project_root / "assets",
        project_root / "assets" / "branding",
        project_root / "branding",
    ]


def _resolve_override_path(env_var: str) -> Path | None:
    raw = os.getenv(env_var, "").strip()
    if not raw:
        return None
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = _project_root() / raw
    return candidate if candidate.exists() else None


def _find_project_asset(patterns: list[str], keywords: list[str]) -> Path | None:
    """Find a file in project roots matching patterns and keywords."""
    for root in _asset_search_roots():
        if not root.exists():
            continue
        for pattern in patterns:
            for candidate in root.glob(pattern):
                name_lower = candidate.name.lower()
                if any(kw in name_lower for kw in keywords):
                    return candidate
    return None


def _find_background_music() -> Path | None:
    return _find_project_asset(
        ["*.mp3", "*.wav", "*.aac", "*.m4a"],
        ["music", "intro", "royalty"],
    )


def _find_background_video() -> Path | None:
    return _find_project_asset(
        ["Background.mp4", "background.mp4", "Background*.mp4", "background*.mp4"],
        ["background"],
    )


def _find_et_logo() -> Path | None:
    override = _resolve_override_path("ET_LOGO_PATH")
    if override is not None:
        return override
    return _find_project_asset(
        ["ET Logo.*", "et logo.*", "ET_Logo.*", "et_logo.*"],
        ["et", "logo"],
    )


def _find_header_image() -> Path | None:
    override = _resolve_override_path("HEADER_IMAGE_PATH")
    if override is not None:
        return override
    return _find_project_asset(
        ["Main header.*", "main header.*", "Main_header.*", "header.*"],
        ["header"],
    )


def _read_headline(job_dir: Path) -> str:
    """Read the news headline from the storyboard output."""
    understanding_path = job_dir / "storyboard" / "article_understanding.json"
    if understanding_path.exists():
        try:
            with understanding_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            return str(data.get("headline", "")).strip()
        except Exception:
            pass
    return ""


def _escape_drawtext_value(text: str) -> str:
    """Escape text for use inside ffmpeg drawtext filter value.

    Since the command is passed as a list to subprocess (no shell), we only need
    one level of escaping for ffmpeg's drawtext parser.
    """
    text = text.replace("\\", "\\\\")
    text = text.replace("'", "\u2019")
    text = text.replace(":", "\\:")
    text = text.replace(";", "\\;")
    text = text.replace("[", "\\[")
    text = text.replace("]", "\\]")
    text = text.replace("%", "%%")
    return text


def _wrap_headline(text: str, max_chars: int = 38) -> list[str]:
    """Word-wrap headline text into lines that fit the canvas width."""
    words = text.split()
    if not words:
        return []
    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        if len(current) + 1 + len(word) <= max_chars:
            current += " " + word
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def _render_final(
    *,
    job_dir: Path,
    input_video: Path,
    audio_path: Path | None,
    subtitles_path: Path | None,
    burn_subtitles: bool,
) -> Path:
    final_path = job_dir / "renders" / "final.mp4"
    can_burn = burn_subtitles and subtitles_path and subtitles_path.exists()
    bg_music = _find_background_music()
    bg_video = _find_background_video()
    header_img = _find_header_image()
    et_logo = _find_et_logo()
    headline = _read_headline(job_dir)
    if audio_path and audio_path.exists():
        target_duration = _probe_duration_seconds(audio_path, cwd=job_dir)
    else:
        target_duration = _probe_duration_seconds(input_video, cwd=job_dir)
    render_fps = max(1, int(os.getenv("RENDER_FPS", "30")))
    drawtext_fontfile = _resolve_drawtext_fontfile()
    drawtext_font_part = ""
    if drawtext_fontfile is not None:
        drawtext_font_part = f":fontfile='{_escape_ffmpeg_path(drawtext_fontfile)}'"

    # --- Build inputs ---
    cmd = ["ffmpeg", "-y"]

    # Input 0: pipeline video
    cmd.extend(["-i", str(input_video)])

    # Input 1: narration audio (optional)
    input_audio_idx = None
    next_idx = 1
    if audio_path and audio_path.exists():
        input_audio_idx = next_idx
        cmd.extend(["-i", str(audio_path)])
        next_idx += 1

    # Input 1: background video (looped, no audio)
    input_bg_idx = None
    if bg_video:
        input_bg_idx = next_idx
        cmd.extend(["-stream_loop", "-1", "-i", str(bg_video)])
        next_idx += 1

    # Input 2: header image
    input_header_idx = None
    if header_img:
        input_header_idx = next_idx
        cmd.extend(["-i", str(header_img)])
        next_idx += 1

    # Input 3: background music
    input_music_idx = None
    if bg_music:
        input_music_idx = next_idx
        cmd.extend(["-i", str(bg_music)])
        next_idx += 1

    # Input 4: ET logo
    input_logo_idx = None
    if et_logo:
        input_logo_idx = next_idx
        cmd.extend(["-i", str(et_logo)])
        next_idx += 1

    # --- Build filter_complex ---
    filters = []

    if bg_video or header_img or bg_music:
        # Output canvas: 1080x1920 (portrait)
        canvas_w, canvas_h = 1080, 1920

        # Layout measurements
        top_padding = 25                             # padding above ET header
        header_h = 161 if header_img else 0        # ET header height
        header_gap = 30 if header_img else 0       # gap between header and headline

        # Dynamic headline: wrap to fit inside the box with margins
        # At font size 52, approx char width ~30px. Box width = 1080 - 2*20(margin) - 2*20(inner pad) = 1000px
        # Safe max_chars ≈ 1000/30 ≈ 30 chars per line
        headline_font_size = 52
        headline_max_chars = 30
        headline_lines = _wrap_headline(headline, max_chars=headline_max_chars) if headline else []

        # If headline wraps to too many lines (>3), reduce font to fit
        if len(headline_lines) > 3:
            headline_font_size = 42
            headline_max_chars = 36
            headline_lines = _wrap_headline(headline, max_chars=headline_max_chars) if headline else []

        headline_line_h = headline_font_size + 14  # line height with spacing
        headline_bar_padding = 40                  # top+bottom inner padding
        headline_bar_h = (len(headline_lines) * headline_line_h + headline_bar_padding) if headline_lines else 0

        subtitle_area_h = 300                      # space for subtitles below video
        # Crop pipeline video to ~square (1:1) by trimming left/right, then scale to canvas width
        # Input is 1920x1080 landscape → crop center 1080x1080 → scale to canvas_w x canvas_w
        pipeline_h = canvas_w                       # 1080 (square after crop+scale)

        # Position pipeline video: leave space for header + gap + headline + gap
        top_section_h = top_padding + header_h + header_gap + headline_bar_h
        remaining_h = canvas_h - top_section_h - subtitle_area_h
        pipeline_y = top_section_h + max(0, (remaining_h - pipeline_h) // 2)

        # 1. Background video
        if bg_video:
            if target_duration is not None and target_duration > 0:
                filters.append(
                    f"[{input_bg_idx}:v]scale={canvas_w}:{canvas_h}:force_original_aspect_ratio=increase,"
                    f"crop={canvas_w}:{canvas_h},fps={render_fps},setsar=1,"
                    f"tpad=stop_mode=clone:stop_duration={target_duration:.3f}[bg_scaled]"
                )
            else:
                filters.append(
                    f"[{input_bg_idx}:v]scale={canvas_w}:{canvas_h}:force_original_aspect_ratio=increase,"
                    f"crop={canvas_w}:{canvas_h},fps={render_fps},setsar=1[bg_scaled]"
                )
        else:
            filters.append(f"color=c=black:s={canvas_w}x{canvas_h}:r={render_fps}[bg_scaled]")

        # 2. Crop pipeline video to square (center crop) then scale to canvas width
        filters.append(
            f"[0:v]crop=ih:ih:(iw-ih)/2:0,"
            f"scale={canvas_w}:{pipeline_h},fps={render_fps},setsar=1[pipeline_scaled]"
        )

        # 3. Header image
        if header_img:
            filters.append(f"[{input_header_idx}:v]scale={canvas_w}:-1[header_scaled]")

        # 4. Overlay pipeline on background
        filters.append(
            f"[bg_scaled][pipeline_scaled]overlay=0:{pipeline_y}:shortest=1[v1]"
        )
        current_v = "v1"

        # 5. Overlay header at top
        if header_img:
            filters.append(f"[{current_v}][header_scaled]overlay=0:{top_padding}[v2]")
            current_v = "v2"

        # 6. Draw headline bar below header with gap
        #    Style: black semi-transparent box with white thin border,
        #    first line golden yellow, remaining lines white.
        if headline_lines:
            # Center headline bar between header bottom and pipeline video top
            header_bottom = top_padding + header_h
            gap_between = pipeline_y - header_bottom
            bar_y = header_bottom + (gap_between - headline_bar_h) // 2

            box_margin = 20  # left/right margin for the box
            box_x = box_margin
            box_w = canvas_w - 2 * box_margin
            border_t = 2  # white border thickness
            inner_pad_x = 16  # horizontal inner padding

            # Semi-transparent black background box
            filters.append(
                f"[{current_v}]drawbox=x={box_x}:y={bar_y}:w={box_w}:h={headline_bar_h}"
                f":color=black@0.70:t=fill[v3a]"
            )
            # White thin border (top, bottom, left, right)
            filters.append(
                f"[v3a]drawbox=x={box_x}:y={bar_y}:w={box_w}:h={headline_bar_h}"
                f":color=white@0.9:t={border_t}[v3b]"
            )
            current_v = "v3b"

            # Render each headline line centered in the bar
            text_block_h = len(headline_lines) * headline_line_h
            text_start_y = bar_y + (headline_bar_h - text_block_h) // 2

            for i, line in enumerate(headline_lines):
                escaped_line = _escape_drawtext_value(line)
                line_y = text_start_y + i * headline_line_h
                tag = f"v3_line{i}"
                # First line: golden yellow, rest: white
                font_color = "0xFFD700" if i == 0 else "white"
                filters.append(
                    f"[{current_v}]drawtext=text='{escaped_line}'"
                    f"{drawtext_font_part}"
                    f":fontsize={headline_font_size}:fontcolor={font_color}"
                    f":borderw=2:bordercolor={font_color}"
                    f":x=(w-text_w)/2:y={line_y}[{tag}]"
                )
                current_v = tag

        # 7. News broadcast-style subtitles below the video
        #    Professional look: dark box, white text, red accent bar on left,
        #    word-wrapped across as many lines as needed within subtitle area.
        if can_burn and subtitles_path and subtitles_path.exists():
            srt_entries = _parse_srt(subtitles_path)
            use_subtitles_filter = _has_subtitle_filter() and len(srt_entries) > MAX_DRAWTEXT_SUBTITLE_CUES

            if use_subtitles_filter:
                subtitle_filter = _build_subtitle_filter(
                    subtitles_path,
                    font_name=_resolve_subtitle_font_name(),
                )
                filters.append(f"[{current_v}]{subtitle_filter}[v_sub_ass]")
                current_v = "v_sub_ass"
            else:
                sub_font_size = 42
                sub_line_h = sub_font_size + 14      # line height with spacing
                # Keep one fewer line than the max available to reduce strip height.
                available_sub_lines = max(1, ((subtitle_area_h - 40) - 24) // sub_line_h)
                max_sub_lines = max(1, available_sub_lines - 1)
                sub_box_h = max_sub_lines * sub_line_h + 24
                sub_box_y = pipeline_y + pipeline_h + 20  # 20px gap below video
                sub_box_x = 40                       # left margin
                sub_box_w = canvas_w - 80            # 40px margin each side
                accent_w = 6                         # red accent bar width
                sub_text_x = sub_box_x + accent_w + 16  # text starts after accent+padding
                max_sub_chars = 44                   # chars per line before wrapping

                # Persistent subtitle background box (always visible, like a lower-third)
                filters.append(
                    f"[{current_v}]drawbox=x={sub_box_x}:y={sub_box_y}"
                    f":w={sub_box_w}:h={sub_box_h}"
                    f":color=black@0.75:t=fill[v_sub_bg]"
                )
                # Red accent bar on left edge of box
                filters.append(
                    f"[v_sub_bg]drawbox=x={sub_box_x}:y={sub_box_y}"
                    f":w={accent_w}:h={sub_box_h}"
                    f":color=0xCC0000@1.0:t=fill[v_sub_accent]"
                )
                current_v = "v_sub_accent"

                for si, entry in enumerate(srt_entries):
                    # Word-wrap subtitle text and preserve existing line breaks.
                    words = entry["text"].replace("\n", " \n ").split()
                    lines: list[str] = []
                    cur_line = ""
                    for word in words:
                        if word == "\n":
                            if cur_line:
                                lines.append(cur_line)
                                cur_line = ""
                            continue
                        test = (cur_line + " " + word).strip()
                        if len(test) <= max_sub_chars:
                            cur_line = test
                        else:
                            if cur_line:
                                lines.append(cur_line)
                            cur_line = word
                    if cur_line:
                        lines.append(cur_line)
                    if not lines:
                        continue

                    # Keep all wrapped text as long as it fits in subtitle area.
                    lines = lines[:max_sub_lines]

                    start_t = entry["start"]
                    end_t = entry["end"]

                    for li, line_text in enumerate(lines):
                        escaped = _escape_drawtext_value(line_text)
                        line_y = sub_box_y + 12 + li * sub_line_h
                        tag = f"v_s{si}l{li}"
                        filters.append(
                            f"[{current_v}]drawtext=text='{escaped}'"
                            f"{drawtext_font_part}"
                            f":fontsize={sub_font_size}"
                            f":fontcolor=white"
                            f":borderw=1:bordercolor=black"
                            f":x={sub_text_x}"
                            f":y={line_y}"
                            f":enable='between(t\\,{start_t:.3f}\\,{end_t:.3f})'"
                            f"[{tag}]"
                        )
                        current_v = tag

        # 8. ET Logo watermark in lower-left corner
        if et_logo and input_logo_idx is not None:
            logo_size = 80  # width & height of the logo
            logo_margin = 30  # margin from edges
            logo_y = canvas_h - logo_size - logo_margin
            filters.append(
                f"[{input_logo_idx}:v]scale={logo_size}:{logo_size}[et_logo_scaled]"
            )
            next_tag = f"v_logo"
            filters.append(
                f"[{current_v}][et_logo_scaled]overlay={logo_margin}:{logo_y}[{next_tag}]"
            )
            current_v = next_tag

        # 8b. Clamp visual timeline to the pipeline video duration.
        if target_duration is not None and target_duration > 0:
            filters.append(
                f"[{current_v}]trim=duration={target_duration:.3f},setpts=PTS-STARTPTS[v_trim]"
            )
            current_v = "v_trim"

        # 9. Audio: mix narration + background music (increased bg volume)
        audio_out_tag = "aout"
        narration_idx = input_audio_idx if input_audio_idx is not None else 0
        if bg_music:
            filters.append(f"[{narration_idx}:a]volume=1.0[narration]")
            filters.append(
                f"[{input_music_idx}:a]volume=0.40,"
                f"afade=t=in:st=0:d=2,afade=t=out:st=0:d=3[bgm]"
            )
            filters.append(
                f"[narration][bgm]amix=inputs=2:duration=first:dropout_transition=3,"
                f"loudnorm=I=-16:TP=-1.5:LRA=11[aout]"
            )
        else:
            filters.append(f"[{narration_idx}:a]loudnorm=I=-16:TP=-1.5:LRA=11[aout]")

        if target_duration is not None and target_duration > 0:
            filters.append(
                f"[{audio_out_tag}]atrim=duration={target_duration:.3f},asetpts=N/SR/TB[aout_trim]"
            )
            audio_out_tag = "aout_trim"

        filter_str = ";\n".join(filters)
        cmd.extend(["-filter_complex", filter_str])
        cmd.extend(["-map", f"[{current_v}]", "-map", f"[{audio_out_tag}]"])

    else:
        # No background video/header/music — simple pass-through with subtitles
        if can_burn:
            if _has_subtitle_filter():
                cmd.extend(
                    [
                        "-vf",
                        _build_subtitle_filter(
                            subtitles_path,
                            font_name=_resolve_subtitle_font_name(),
                        ),
                    ]
                )
            else:
                script_path = _build_drawtext_filter_script(
                    subtitles_path,
                    job_dir,
                    fontfile=drawtext_fontfile,
                )
                cmd.extend(["-filter_script:v", str(script_path)])
        cmd.extend(["-af", "loudnorm=I=-16:TP=-1.5:LRA=11"])

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
    header_override_raw = os.getenv("HEADER_IMAGE_PATH", "").strip()
    logo_override_raw = os.getenv("ET_LOGO_PATH", "").strip()
    header_override = _resolve_override_path("HEADER_IMAGE_PATH") if header_override_raw else None
    logo_override = _resolve_override_path("ET_LOGO_PATH") if logo_override_raw else None
    header_img = _find_header_image()
    et_logo = _find_et_logo()
    bg_video = _find_background_video()
    bg_music = _find_background_music()

    if header_override_raw and header_override is None:
        warnings.append(f"HEADER_IMAGE_PATH set but not found: {header_override_raw}")
    if logo_override_raw and logo_override is None:
        warnings.append(f"ET_LOGO_PATH set but not found: {logo_override_raw}")
    if header_img is None:
        warnings.append("Header image not found; ET header overlay disabled")
    if et_logo is None:
        warnings.append("ET logo not found; watermark disabled")

    branding_diagnostics = {
        "header_image": str(header_img) if header_img else "",
        "header_source": "override"
        if header_override is not None
        else ("auto" if header_img else "missing"),
        "header_override": header_override_raw,
        "logo_image": str(et_logo) if et_logo else "",
        "logo_source": "override" if logo_override is not None else ("auto" if et_logo else "missing"),
        "logo_override": logo_override_raw,
        "background_video": str(bg_video) if bg_video else "",
        "background_music": str(bg_music) if bg_music else "",
    }
    burn_subtitles_raw = os.getenv("BURN_SUBTITLES", "auto").strip().lower() or "auto"
    if burn_subtitles_raw in {"0", "false", "no", "off"}:
        burn_subtitles = False
    elif burn_subtitles_raw in {"1", "true", "yes", "on"}:
        burn_subtitles = True
    else:
        burn_subtitles = subtitles_path.exists()

    final_rendered = False
    preview_rendered = False
    thumbnail_rendered = False

    try:
        if force or not final_path.exists():
            voiceover_path = job_dir / "audio" / "voiceover.wav"
            audio_path = voiceover_path if voiceover_path.exists() else None
            if audio_path is None:
                warnings.append("Voiceover audio not found; using intermediate audio track")

            render_fps = max(1, int(os.getenv("RENDER_FPS", "30")))
            if audio_path is not None:
                raw_path = job_dir / "renders" / "intermediate_raw.mp4"
                base_video = raw_path if raw_path.exists() else _resolve_intermediate_video(job_dir)
                input_video = _normalize_input_video(
                    base_video,
                    job_dir=job_dir,
                    fps=render_fps,
                    force=force,
                )
            else:
                input_video = _resolve_intermediate_video(job_dir)
            final_path = _render_final(
                job_dir=job_dir,
                input_video=input_video,
                audio_path=audio_path,
                subtitles_path=subtitles_path,
                burn_subtitles=burn_subtitles,
            )
            final_rendered = True
        if force or not preview_path.exists():
            preview_path = _render_preview(job_dir=job_dir, final_path=final_path)
            preview_rendered = True
        if force or not thumbnail_path.exists():
            thumbnail_path = _render_thumbnail(job_dir=job_dir, final_path=final_path)
            thumbnail_rendered = True
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
        "export_actions": {
            "force": force,
            "final_rendered": final_rendered,
            "preview_rendered": preview_rendered,
            "thumbnail_rendered": thumbnail_rendered,
        },
        "branding_assets": branding_diagnostics,
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
