import json
import wave
from pathlib import Path
from src.common.models import Scene
from src.common.validation import validate_payload

job_id = "20260324-064615-b73827a5"
job_dir = Path("jobs") / job_id

# Load scenes
with open(job_dir / "storyboard" / "scenes.json") as f:
    scenes_data = json.load(f)

scenes = [validate_payload(Scene, row) for row in scenes_data]

# Create fallback voiceover with silence
sample_rate = 22050
sample_width = 2
audio_dir = job_dir / "audio"
audio_dir.mkdir(parents=True, exist_ok=True)

# Generate narration text and estimate timing
narration_parts = []
sentence_timings = []
timeline = 0.0

for scene in scenes:
    text = scene.narration.strip()
    if text:
        words = len(text.split())
        duration = max(0.8, round(words / 2.6, 3))
        narration_parts.append(text)
        sentence_timings.append({
            "text": text,
            "start": round(timeline, 3),
            "end": round(timeline + duration, 3)
        })
        timeline += duration

narration_text = " ".join(narration_parts)
total_duration = timeline

# Create silence audio
frame_count = int(total_duration * sample_rate)
frames = b"\x00\x00" * frame_count

# Write WAV file
voiceover_path = audio_dir / "voiceover.wav"
with wave.open(str(voiceover_path), "wb") as wav:
    wav.setnchannels(1)
    wav.setsampwidth(sample_width)
    wav.setframerate(sample_rate)
    wav.writeframes(frames)

# Write subtitles
srt_content = ""
for idx, timing in enumerate(sentence_timings, 1):
    start = timing["start"]
    end = max(timing["end"], start + 0.2)
    start_ts = f"{int(start//3600):02d}:{int((start%3600)//60):02d}:{int(start%60):02d},{int((start%1)*1000):03d}"
    end_ts = f"{int(end//3600):02d}:{int((end%3600)//60):02d}:{int(end%60):02d},{int((end%1)*1000):03d}"
    srt_content += f"{idx}\n{start_ts} --> {end_ts}\n{timing['text']}\n\n"

(audio_dir / "subtitles.srt").write_text(srt_content, encoding="utf-8")

# Write VTT
vtt_content = "WEBVTT\n\n"
for timing in sentence_timings:
    start = timing["start"]
    end = max(timing["end"], start + 0.2)
    start_ts = f"{int(start//3600):02d}:{int((start%3600)//60):02d}:{int(start%60):02d}.{int((start%1)*1000):03d}"
    end_ts = f"{int(end//3600):02d}:{int((end%3600)//60):02d}:{int(end%60):02d}.{int((end%1)*1000):03d}"
    vtt_content += f"{start_ts} --> {end_ts}\n{timing['text']}\n\n"

(audio_dir / "subtitles.vtt").write_text(vtt_content, encoding="utf-8")

# Write audio manifest
manifest = {
    "voice_profile": "Fallback-Silence",
    "language": "en-US",
    "function_id": "fallback",
    "tts_provider": "fallback_silence",
    "stt_provider": "fallback_timing",
    "narration_text": narration_text,
    "duration_seconds": round(total_duration, 3),
    "sentence_timings": sentence_timings,
}

(audio_dir / "audio_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

print(f"Fallback audio created: {total_duration:.2f}s")
print(f"Sentences: {len(sentence_timings)}")
print(f"Voiceover: {voiceover_path}")
