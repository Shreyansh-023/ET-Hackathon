from __future__ import annotations

from dataclasses import dataclass
import os

from dotenv import load_dotenv


@dataclass(frozen=True)
class PipelineConfig:
    language: str  # "english" or "hindi"
    duration_seconds: int
    aspect_ratio: str
    style_preset: str
    jobs_root: str
    cache_root: str
    nvidia_api_key: str
    gemini_api_key: str
    gemini_visual_api_key: str
    gemini_model: str
    groq_api_key: str
    groq_base_url: str
    groq_model: str
    pexels_api_key: str
    replicate_api_token: str
    clip_drop_api_key: str
    nvidia_riva_api_key: str
    nvidia_riva_function_id: str
    nvidia_riva_uri: str
    nvidia_riva_voice: str
    nvidia_riva_language: str
    groq_stt_model: str
    retry_limits: dict[str, int]



def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()



def load_config() -> PipelineConfig:
    load_dotenv(override=False)
    language = _env("LANGUAGE", "english").lower()
    if language not in ("english", "hindi"):
        language = "english"

    # Override TTS voice/language based on pipeline language
    if language == "hindi":
        riva_voice = _env("NVIDIA_RIVA_VOICE", "Magpie-Multilingual.HI-IN.Aria")
        riva_language = _env("NVIDIA_RIVA_LANGUAGE", "hi-IN")
    else:
        riva_voice = _env("NVIDIA_RIVA_VOICE", "Magpie-Multilingual.EN-US.Aria")
        riva_language = _env("NVIDIA_RIVA_LANGUAGE", "en-US")

    return PipelineConfig(
        language=language,
        duration_seconds=int(_env("DURATION_SECONDS", "90")),
        aspect_ratio=_env("ASPECT_RATIO", "9:16"),
        style_preset=_env("STYLE_PRESET", "documentary_clean"),
        jobs_root=_env("JOBS_ROOT", "jobs"),
        cache_root=_env("CACHE_ROOT", ".cache/news2video"),
        nvidia_api_key=_env("NVIDIA_API_KEY"),
        gemini_api_key=_env("GEMINI_API_KEY"),
        gemini_visual_api_key=_env("GEMINI_VISUAL_API_KEY"),
        gemini_model=_env("GEMINI_MODEL", "gemini-2.5-flash"),
        groq_api_key=_env("GROQ_API_KEY"),
        groq_base_url=_env("GROQ_BASE_URL", "https://api.groq.com/openai/v1"),
        groq_model=_env("GROQ_MODEL", "openai/gpt-oss-20b"),
        pexels_api_key=_env("PEXELS_API_KEY"),
        replicate_api_token=_env("REPLICATE_API_TOKEN"),
        clip_drop_api_key=_env("CLIP_DROP_API"),
        nvidia_riva_api_key=_env("NVIDIA_RIVA_API_KEY"),
        nvidia_riva_function_id=_env("NVIDIA_RIVA_FUNCTION_ID", "877104f7-e885-42b9-8de8-f6e4c6303969"),
        nvidia_riva_uri=_env("NVIDIA_RIVA_URI", "grpc.nvcf.nvidia.com:443"),
        nvidia_riva_voice=riva_voice,
        nvidia_riva_language=riva_language,
        groq_stt_model=_env("GROQ_STT_MODEL", "whisper-large-v3"),
        retry_limits={
            "ingest": 1,
            "plan": 2,
            "assets": 2,
            "render": 0,
            "export": 0,
        },
    )
