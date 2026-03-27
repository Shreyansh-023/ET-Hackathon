from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
import io
import re
import wave
from typing import Any

from src.common.config import PipelineConfig
from src.common.errors import IOPipelineError, ProviderPipelineError
from src.common.models import Scene


SENTENCE_SPLIT = re.compile(r"(?<=[.!?।॥])\s+")
DEVANAGARI_CHAR = re.compile(r"[\u0900-\u097F]")
ARABIC_CHAR = re.compile(r"[\u0600-\u06FF]")


@dataclass(frozen=True)
class SentenceTiming:
    text: str
    start: float
    end: float


@dataclass(frozen=True)
class AudioBuildResult:
    voiceover_rel_path: str
    subtitles_srt_rel_path: str
    subtitles_vtt_rel_path: str
    audio_manifest_rel_path: str
    narration_text: str
    sentence_timings: list[SentenceTiming]
    duration_seconds: float


def _has_script(lines: list[SentenceTiming], pattern: re.Pattern[str]) -> bool:
    for line in lines:
        if pattern.search(line.text):
            return True
    return False


def _read_wav_bytes(wav_bytes: bytes) -> tuple[int, int, bytes]:
    with wave.open(io.BytesIO(wav_bytes), "rb") as wav_reader:
        sample_rate = wav_reader.getframerate()
        sample_width = wav_reader.getsampwidth()
        channels = wav_reader.getnchannels()
        if channels != 1:
            raise ProviderPipelineError("Expected mono audio from TTS provider")
        frames = wav_reader.readframes(wav_reader.getnframes())
    return sample_rate, sample_width, frames


def _write_wav(path: Path, sample_rate: int, sample_width: int, frames: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav_writer:
        wav_writer.setnchannels(1)
        wav_writer.setsampwidth(sample_width)
        wav_writer.setframerate(sample_rate)
        wav_writer.writeframes(frames)


def _concat_pcm_chunks(chunks: list[bytes]) -> bytes:
    if not chunks:
        raise ProviderPipelineError("No sentence audio chunks available for concatenation")
    merged = bytearray()
    for chunk in chunks:
        merged.extend(chunk)
    return bytes(merged)


def _sentences_from_scenes(scenes: list[Scene]) -> list[str]:
    merged = " ".join(scene.narration.strip() for scene in scenes if scene.narration.strip())
    if not merged:
        return []
    parts = [row.strip() for row in SENTENCE_SPLIT.split(merged) if row.strip()]
    return parts or [merged]


def _estimate_sentence_duration(text: str) -> float:
    words = max(1, len(text.split()))
    # ~2.6 words/second is a natural news narration cadence.
    return max(0.8, round(words / 2.6, 3))


def _build_fallback_silence_audio(sentences: list[str]) -> tuple[int, int, bytes, list[SentenceTiming]]:
    sample_rate = 22050
    sample_width = 2
    timeline = 0.0
    timings: list[SentenceTiming] = []
    frames = bytearray()

    for sentence in sentences:
        duration = _estimate_sentence_duration(sentence)
        frame_count = int(duration * sample_rate)
        frames.extend(b"\x00\x00" * frame_count)
        timings.append(SentenceTiming(text=sentence, start=round(timeline, 3), end=round(timeline + duration, 3)))
        timeline += duration

    if not timings:
        frames.extend(b"\x00\x00" * sample_rate)
        timings.append(SentenceTiming(text="", start=0.0, end=1.0))

    return sample_rate, sample_width, bytes(frames), timings


def _make_riva_service(cfg: PipelineConfig):
    """Create a reusable Riva TTS service + auth object."""
    try:
        import riva.client  # type: ignore
    except Exception as exc:  # pragma: no cover - environment dependent
        raise ProviderPipelineError("nvidia-riva-client is not installed") from exc

    auth = riva.client.Auth(
        uri=cfg.nvidia_riva_uri,
        use_ssl=True,
        metadata_args=[
            ["function-id", cfg.nvidia_riva_function_id],
            ["authorization", f"Bearer {cfg.nvidia_riva_api_key}"],
        ],
    )
    return riva.client.SpeechSynthesisService(auth)


_RIVA_MAX_RETRIES = 5
_RIVA_BASE_DELAY = 2.0  # seconds


def _synthesize_riva_sentence(sentence: str, cfg: PipelineConfig, *, service=None) -> bytes:
    import riva.client  # type: ignore

    if service is None:
        service = _make_riva_service(cfg)

    # Chunk the sentence into pieces of 300 characters or less
    chunks = [sentence[i:i + 300] for i in range(0, len(sentence), 300)]
    audio_chunks = []

    for chunk in chunks:
        last_exc = None
        for attempt in range(_RIVA_MAX_RETRIES):
            try:
                response = service.synthesize(
                    text=chunk,
                    language_code=cfg.nvidia_riva_language,
                    voice_name=cfg.nvidia_riva_voice,
                    sample_rate_hz=22050,
                    encoding=riva.client.AudioEncoding.LINEAR_PCM,
                )
                break
            except Exception as exc:
                last_exc = exc
                err_str = str(exc).lower()
                if "rate limit" in err_str or "resource_exhausted" in err_str:
                    delay = _RIVA_BASE_DELAY * (2 ** attempt)
                    time.sleep(delay)
                else:
                    raise ProviderPipelineError(f"Riva TTS failed: {exc}") from exc
        else:
            raise ProviderPipelineError(
                f"Riva TTS rate limit exceeded after {_RIVA_MAX_RETRIES} retries"
            ) from last_exc

        audio = getattr(response, "audio", None)
        if not isinstance(audio, (bytes, bytearray)) or not audio:
            raise ProviderPipelineError("Riva TTS returned an empty audio payload for a chunk")
        audio_chunks.append(bytes(audio))

    return b"".join(audio_chunks)


def _fmt_srt_timestamp(seconds: float) -> str:
    millis = max(0, int(round(seconds * 1000)))
    h = millis // 3_600_000
    m = (millis % 3_600_000) // 60_000
    s = (millis % 60_000) // 1000
    ms = millis % 1000
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _fmt_vtt_timestamp(seconds: float) -> str:
    millis = max(0, int(round(seconds * 1000)))
    h = millis // 3_600_000
    m = (millis % 3_600_000) // 60_000
    s = (millis % 60_000) // 1000
    ms = millis % 1000
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def _line_wrap(text: str, max_chars: int = 42) -> list[str]:
    words = text.split()
    if not words:
        return [""]

    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        test = f"{current} {word}"
        if len(test) <= max_chars:
            current = test
            continue
        lines.append(current)
        current = word
    lines.append(current)
    return lines


def _timings_from_groq(
    audio_path: Path,
    script_text: str,
    cfg: PipelineConfig,
) -> list[SentenceTiming]:
    try:
        from groq import Groq  # type: ignore
    except Exception as exc:  # pragma: no cover - dependency/environment dependent
        raise ProviderPipelineError("groq package is not installed") from exc

    if not cfg.groq_api_key:
        raise ProviderPipelineError("GROQ_API_KEY is not configured")

    model = cfg.groq_stt_model or "whisper-large-v3"
    language = (cfg.nvidia_riva_language or "en-US").split("-")[0]

    client = Groq(api_key=cfg.groq_api_key)
    try:
        with audio_path.open("rb") as audio_file:
            response = client.audio.transcriptions.create(
                file=(audio_path.name, audio_file.read()),
                model=model,
                response_format="verbose_json",
                timestamp_granularities=["word", "segment"],
                language=language,
            )
    except Exception as exc:
        raise ProviderPipelineError(f"Groq STT request failed: {exc}") from exc

    payload: dict[str, Any]
    if hasattr(response, "model_dump"):
        payload = response.model_dump()
    elif isinstance(response, dict):
        payload = response
    else:
        payload = {}

    segments = payload.get("segments") if isinstance(payload, dict) else None
    if not isinstance(segments, list) or not segments:
        raise ProviderPipelineError("Groq STT returned no segment-level timestamps")

    lines: list[SentenceTiming] = []
    for seg in segments:
        if not isinstance(seg, dict):
            continue
        text = str(seg.get("text", "")).strip()
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", start + 0.5))
        if not text:
            continue
        if end <= start:
            end = start + 0.5
        lines.append(SentenceTiming(text=text, start=round(start, 3), end=round(end, 3)))

    if not lines:
        raise ProviderPipelineError("Groq STT segments could not be parsed")

    # Keep caption wording anchored to narration script when possible.
    if script_text and len(lines) == 1:
        lines[0] = SentenceTiming(text=script_text.strip(), start=lines[0].start, end=lines[0].end)
    return lines


def _timings_from_offline_whisper(
    audio_path: Path,
    script_text: str,
    cfg: PipelineConfig,
) -> list[SentenceTiming]:
    try:
        import whisper  # type: ignore
    except Exception as exc:  # pragma: no cover - dependency/environment dependent
        raise ProviderPipelineError("openai-whisper is not installed") from exc

    model_name = cfg.groq_stt_model or "base"
    try:
        model = whisper.load_model(model_name)
    except Exception:
        # Groq model ids are not always valid local Whisper model names.
        model = whisper.load_model("base")

    try:
        result: dict[str, Any] = model.transcribe(
            str(audio_path),
            language=(cfg.nvidia_riva_language or "en-US").split("-")[0],
            fp16=False,
            verbose=False,
        )
    except Exception as exc:
        raise ProviderPipelineError(f"Offline Whisper transcription failed: {exc}") from exc

    segments = result.get("segments") if isinstance(result, dict) else None
    if not isinstance(segments, list) or not segments:
        raise ProviderPipelineError("Offline Whisper returned no segment-level timestamps")

    lines: list[SentenceTiming] = []
    for seg in segments:
        if not isinstance(seg, dict):
            continue
        text = str(seg.get("text", "")).strip()
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", start + 0.5))
        if not text:
            continue
        if end <= start:
            end = start + 0.5
        lines.append(SentenceTiming(text=text, start=round(start, 3), end=round(end, 3)))

    if not lines:
        raise ProviderPipelineError("Offline Whisper segments could not be parsed")

    if script_text and len(lines) == 1:
        lines[0] = SentenceTiming(text=script_text.strip(), start=lines[0].start, end=lines[0].end)
    return lines


def _write_subtitle_files(
    srt_path: Path,
    vtt_path: Path,
    lines: list[SentenceTiming],
) -> None:
    srt_path.parent.mkdir(parents=True, exist_ok=True)
    vtt_path.parent.mkdir(parents=True, exist_ok=True)

    srt_rows: list[str] = []
    vtt_rows: list[str] = ["WEBVTT", ""]

    for idx, row in enumerate(lines, start=1):
        start = row.start
        end = max(row.end, start + 0.2)
        wrapped = "\n".join(_line_wrap(row.text))

        srt_rows.extend(
            [
                str(idx),
                f"{_fmt_srt_timestamp(start)} --> {_fmt_srt_timestamp(end)}",
                wrapped,
                "",
            ]
        )
        vtt_rows.extend(
            [
                f"{_fmt_vtt_timestamp(start)} --> {_fmt_vtt_timestamp(end)}",
                wrapped,
                "",
            ]
        )

    srt_path.write_text("\n".join(srt_rows).strip() + "\n", encoding="utf-8")
    vtt_path.write_text("\n".join(vtt_rows).strip() + "\n", encoding="utf-8")


def build_voiceover_and_subtitles(
    scenes: list[Scene],
    cfg: PipelineConfig,
    job_dir: Path,
    *,
    dry_run: bool,
    logger=None,
) -> AudioBuildResult:
    sentences = _sentences_from_scenes(scenes)
    narration_text = " ".join(sentences).strip()

    use_provider = bool(
        cfg.nvidia_riva_api_key and cfg.nvidia_riva_function_id and cfg.nvidia_riva_uri and not dry_run
    )

    if use_provider and sentences:
        pcm_chunks: list[bytes] = []
        sample_rate = 22050
        sample_width = 2
        timeline = 0.0
        sentence_timings: list[SentenceTiming] = []
        riva_service = _make_riva_service(cfg)
        for idx, sentence in enumerate(sentences):
            if idx > 0:
                time.sleep(0.5)  # pace requests to avoid rate limits
            pcm_bytes = _synthesize_riva_sentence(sentence, cfg, service=riva_service)
            duration = len(pcm_bytes) / (sample_rate * sample_width)
            sentence_timings.append(
                SentenceTiming(
                    text=sentence,
                    start=round(timeline, 3),
                    end=round(timeline + max(duration, 0.2), 3),
                )
            )
            timeline += max(duration, 0.2)
            pcm_chunks.append(pcm_bytes)

        merged_frames = _concat_pcm_chunks(pcm_chunks)
    else:
        sample_rate, sample_width, merged_frames, sentence_timings = _build_fallback_silence_audio(sentences)

    voiceover_rel_path = "audio/voiceover.wav"
    voiceover_path = job_dir / voiceover_rel_path
    _write_wav(voiceover_path, sample_rate, sample_width, merged_frames)
    duration_seconds = round(len(merged_frames) / (sample_rate * sample_width), 3)

    subtitle_lines: list[SentenceTiming]
    stt_provider = "sentence_timing_fallback"
    if not dry_run and cfg.groq_api_key:
        try:
            subtitle_lines = _timings_from_groq(voiceover_path, narration_text, cfg)
            stt_provider = "groq_whisper"
        except ProviderPipelineError:
            try:
                subtitle_lines = _timings_from_offline_whisper(voiceover_path, narration_text, cfg)
                stt_provider = "offline_whisper"
            except ProviderPipelineError:
                subtitle_lines = sentence_timings
    elif not dry_run:
        try:
            subtitle_lines = _timings_from_offline_whisper(voiceover_path, narration_text, cfg)
            stt_provider = "offline_whisper"
        except ProviderPipelineError:
            subtitle_lines = sentence_timings
    else:
        subtitle_lines = sentence_timings

    # Guardrail for Hindi: if STT output drifts to Arabic/Urdu script,
    # keep subtitle wording in source Devanagari script from narration text.
    if cfg.language == "hindi" and sentence_timings:
        stt_has_devanagari = _has_script(subtitle_lines, DEVANAGARI_CHAR)
        stt_has_arabic = _has_script(subtitle_lines, ARABIC_CHAR)
        source_has_devanagari = _has_script(sentence_timings, DEVANAGARI_CHAR)
        if source_has_devanagari and (stt_has_arabic or not stt_has_devanagari):
            subtitle_lines = sentence_timings
            stt_provider = f"{stt_provider}_script_fallback"

    # Ensure strict monotonic timing and no overlap in exported subtitle files.
    normalized_lines: list[SentenceTiming] = []
    cursor = 0.0
    for line in subtitle_lines:
        start = max(cursor, line.start)
        end = max(start + 0.2, line.end)
        normalized_lines.append(SentenceTiming(text=line.text, start=round(start, 3), end=round(end, 3)))
        cursor = end

    if normalized_lines:
        overflow = normalized_lines[-1].end - duration_seconds
        if overflow > 0.0:
            last = normalized_lines[-1]
            clamped_end = max(last.start + 0.1, last.end - overflow)
            normalized_lines[-1] = SentenceTiming(text=last.text, start=last.start, end=round(clamped_end, 3))

    subtitles_srt_rel_path = "audio/subtitles.srt"
    subtitles_vtt_rel_path = "audio/subtitles.vtt"
    _write_subtitle_files(job_dir / subtitles_srt_rel_path, job_dir / subtitles_vtt_rel_path, normalized_lines)

    audio_manifest_rel_path = "audio/audio_manifest.json"
    manifest_payload = {
        "voice_profile": cfg.nvidia_riva_voice,
        "language": cfg.nvidia_riva_language,
        "function_id": cfg.nvidia_riva_function_id,
        "tts_provider": "nvidia_riva" if use_provider else "fallback_silence",
        "stt_provider": stt_provider,
        "narration_text": narration_text,
        "duration_seconds": duration_seconds,
        "sentence_timings": [
            {"text": row.text, "start": row.start, "end": row.end} for row in sentence_timings
        ],
    }
    try:
        import json

        (job_dir / audio_manifest_rel_path).write_text(
            json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except OSError as exc:
        raise IOPipelineError(f"Failed writing audio manifest: {exc}") from exc

    if logger:
        logger.emit(
            "info",
            "audio_outputs_written",
            voiceover_path=voiceover_rel_path,
            subtitles_srt=subtitles_srt_rel_path,
            subtitles_vtt=subtitles_vtt_rel_path,
        )

    return AudioBuildResult(
        voiceover_rel_path=voiceover_rel_path,
        subtitles_srt_rel_path=subtitles_srt_rel_path,
        subtitles_vtt_rel_path=subtitles_vtt_rel_path,
        audio_manifest_rel_path=audio_manifest_rel_path,
        narration_text=narration_text,
        sentence_timings=sentence_timings,
        duration_seconds=duration_seconds,
    )


def build_audio_stub() -> dict[str, str]:
    return {"status": "not_started", "notes": "Reserved for Step 3"}
