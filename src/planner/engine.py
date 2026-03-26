from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import traceback
from typing import Any

import requests

try:
    from google import genai
    from google.genai import types as genai_types
except Exception:  # pragma: no cover - optional dependency for plan stage only
    genai = None
    genai_types = None

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - optional dependency for plan stage only
    OpenAI = None

from src.common.config import PipelineConfig
from src.common.errors import ProviderPipelineError, ValidationPipelineError
from src.common.models import Article, ArticleUnderstanding, Scene
from src.common.validation import validate_payload


NVIDIA_CHAT_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
NVIDIA_PLANNER_MODEL = "deepseek-ai/deepseek-v3.1"
TARGET_DURATION_SECONDS = 90.0


def _call_gemini(cfg: PipelineConfig, prompt: str) -> str:
    if not cfg.gemini_api_key:
        raise ProviderPipelineError("GEMINI_API_KEY is not configured")
    if genai is None or genai_types is None:
        raise ProviderPipelineError("google-genai is not installed")

    try:
        client = genai.Client(api_key=cfg.gemini_api_key)
        response = client.models.generate_content(
            model=cfg.gemini_model,
            contents=prompt,
            config=genai_types.GenerateContentConfig(
                temperature=0.2,
                response_mime_type="application/json",
            ),
        )
    except Exception as exc:
        raise ProviderPipelineError(f"Gemini request failed: {exc}") from exc

    text = getattr(response, "text", None)
    if isinstance(text, str) and text.strip():
        return text.strip()

    raise ProviderPipelineError("Gemini returned an empty response")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def _extract_json_payload(raw: str) -> Any:
    text = raw.strip()
    if not text:
        raise ValidationPipelineError("Planner output is empty")

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    first_obj = text.find("{")
    first_arr = text.find("[")
    starts = [idx for idx in (first_obj, first_arr) if idx != -1]
    if not starts:
        raise ValidationPipelineError("Planner output did not contain JSON")

    start = min(starts)
    end_obj = text.rfind("}")
    end_arr = text.rfind("]")
    end = max(end_obj, end_arr)
    if end <= start:
        raise ValidationPipelineError("Planner output had malformed JSON boundaries")

    candidate = text[start : end + 1]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise ValidationPipelineError(f"Planner JSON parse failed: {exc}") from exc


def _read_streamed_completion(response: requests.Response) -> str:
    parts: list[str] = []
    for line in response.iter_lines(decode_unicode=True):
        if not line:
            continue
        if not line.startswith("data:"):
            continue

        chunk = line[5:].strip()
        if chunk == "[DONE]":
            break

        try:
            payload = json.loads(chunk)
        except json.JSONDecodeError:
            continue

        choices = payload.get("choices") or []
        if not choices:
            continue
        delta = choices[0].get("delta") or {}
        content = delta.get("content")
        if isinstance(content, str):
            parts.append(content)
    return "".join(parts).strip()


def _generate_storyboard_with_gemini(
    article: Article,
    cfg: PipelineConfig,
    *,
    llm_enabled: bool,
) -> tuple[list[dict[str, Any]], str, str, ArticleUnderstanding]:
    prompt = _build_gemini_storyboard_prompt(article)
    if not llm_enabled:
        # This is a stub for dry-run mode, will need a more robust fallback
        understanding = _article_understanding_from_text(article)
        scenes = [scene.model_dump() for scene in _fallback_template_plan(article, understanding)]
        return scenes, prompt, json.dumps({"scenes": scenes}, indent=2), understanding

    raw = _call_gemini(cfg, prompt)
    
    try:
        payload = _extract_json_payload(raw)
        
        # Create a simplified ArticleUnderstanding
        understanding = ArticleUnderstanding(
            headline=payload.get("headline", article.title or "Untitled"),
            summary=payload.get("summary", ""),
            key_points=[],
            entities=[],
            visual_hooks=[],
            tone="informative",
            topic="news"
        )

        rows = _normalize_storyboard_rows(payload)
    except Exception as exc:
        repair_prompt = _build_repair_prompt(raw, exc, "storyboard")
        repaired_raw = _call_gemini(cfg, repair_prompt)
        payload = _extract_json_payload(repaired_raw)
        
        understanding = ArticleUnderstanding(
            headline=payload.get("headline", article.title or "Untitled"),
            summary=payload.get("summary", ""),
            key_points=[],
            entities=[],
            visual_hooks=[],
            tone="informative",
            topic="news"
        )
        
        rows = _normalize_storyboard_rows(payload)
        return (
            rows,
            prompt + "\n\n--- REPAIR PROMPT ---\n" + repair_prompt,
            f"# Provider: Gemini\n{raw}\n\n{repaired_raw}",
            understanding,
        )

    return rows, prompt, f"# Provider: Gemini\n{raw}", understanding


def _build_gemini_storyboard_prompt(article: Article) -> str:
    return (
        "You are a video producer creating a 90-second news summary video. "
        "Analyze the provided article and generate a complete storyboard and visual plan in a single JSON object. "
        "Return ONLY the JSON object, with no additional text or markdown formatting.\n\n"
        "The JSON object should have this exact schema:\n"
        "{\n"
        '  "headline": "string",\n'
        '  "summary": "string",\n'
        '  "scenes": [\n'
        "    {\n"
        '      "id": "scene-001",\n'
        '      "narration": "string",\n'
        '      "on_screen_text": "string",\n'
        '      "visual_suggestions": {\n'
        '          "type": "photo|graph|chart|map",\n'
        '          "description": "A detailed description of the visual.",\n'
        '          "pexels_search_queries": ["query1", "query2"],\n'
        '          "replicate_prompt": "A prompt for an AI image generator if needed."\n'
        '      }\n'
        "    }\n"
        "  ]\n"
        "}\n\n"
        "Guidelines:\n"
        "- The total video duration should be approximately 90 seconds.\n"
        "- Create 8-12 scenes.\n"
        "- For `visual_suggestions.type`:\n"
        "  - Use 'photo' for general stock imagery.\n"
        "  - Use 'graph', 'chart', or 'map' for data visualizations or geographical context mentioned in the article. Only use these if the article explicitly contains data, figures, or locations that would benefit from a visual representation.\n"
        "- For `pexels_search_queries`, provide 2-3 diverse and specific queries for finding stock photos on Pexels.\n"
        "- For `replicate_prompt`, only provide a prompt if the `type` is 'graph', 'chart', or 'map'. The prompt should be detailed enough for an AI image generator (like Replicate with Flux 1.1 Pro) to create the specific visualization.\n"
        "- Ensure the narration flows logically and tells a coherent story based on the article.\n\n"
        f"Article Title: {article.title or 'Untitled'}\n"
        f"Article Text:\n{article.clean_article_text}"
    )


def _build_repair_prompt(raw: str, error: Exception, target: str) -> str:
    return (
        f"Repair the following {target} output and return ONLY valid JSON."
        " Do not add commentary or markdown.\n"
        f"Validation error: {error}\n"
        "Output to repair:\n"
        f"{raw}"
    )


def _normalize_storyboard_rows(raw: Any) -> list[dict[str, Any]]:
    if isinstance(raw, dict):
        scenes = raw.get("scenes")
    elif isinstance(raw, list):
        scenes = raw
    else:
        raise ValidationPipelineError("Storyboard output must be an object or array")

    if not isinstance(scenes, list) or not scenes:
        raise ValidationPipelineError("Storyboard must contain non-empty scenes")

    normalized: list[dict[str, Any]] = []
    for idx, row in enumerate(scenes, start=1):
        if not isinstance(row, dict):
            raise ValidationPipelineError(f"Scene {idx} is not a JSON object")

        scene_id = str(row.get("id") or row.get("scene_id") or f"scene-{idx:03d}")
        start = float(row.get("start", (idx - 1) * 8.0))
        end = float(row.get("end", start + 8.0))
        scene_type = str(row.get("type", "body")).lower()
        if scene_type not in {"hook", "body", "closing"}:
            scene_type = "body"

        visual_suggestions = row.get("visual_suggestions", {})
        visual_type = visual_suggestions.get("type", "photo")
        
        # Simple mapping from new 'type' to old 'visual_strategy'
        strategy_map = {
            "photo": "editorial_still",
            "graph": "data_visualization",
            "chart": "data_visualization",
            "map": "map_visualization",
        }
        visual_strategy = strategy_map.get(visual_type, "editorial_still")

        # Construct a visual prompt from the description
        visual_prompt = visual_suggestions.get("description", "Editorial still frame")

        normalized.append(
            {
                "scene_id": scene_id,
                "id": scene_id,
                "index": idx,
                "start": start,
                "end": end,
                "type": scene_type,
                "narration": str(row.get("narration", "")).strip(),
                "on_screen_text": str(row.get("on_screen_text", "")).strip(),
                "visual_strategy": visual_strategy,
                "visual_prompt": visual_prompt,
                "visual_suggestions": visual_suggestions, # Pass this through
                "transition_hint": str(row.get("transition_hint", "clean_cut")).strip(),
                "motion_hint": str(row.get("motion_hint", "slow_push_in")).strip(),
            }
        )
    return normalized


def _scale_bucket(rows: list[dict[str, Any]], target: float) -> None:
    if not rows:
        return

    durations = [max(float(row["end"]) - float(row["start"]), 1.0) for row in rows]
    current = sum(durations)
    if current <= 0:
        even = target / len(rows)
        durations = [even for _ in rows]
    else:
        factor = target / current
        durations = [max(1.0, dur * factor) for dur in durations]

    drift = target - sum(durations)
    durations[-1] += drift
    for row, dur in zip(rows, durations):
        row["_duration"] = max(0.5, round(dur, 3))


def _enforce_timing_policy(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        raise ValidationPipelineError("Storyboard has no scenes")

    has_hook = any(row["type"] == "hook" for row in rows)
    has_body = any(row["type"] == "body" for row in rows)
    has_closing = any(row["type"] == "closing" for row in rows)

    if not has_hook:
        rows[0]["type"] = "hook"
    if not has_closing:
        rows[-1]["type"] = "closing"
    if not has_body and len(rows) > 2:
        rows[1]["type"] = "body"

    hooks = [row for row in rows if row["type"] == "hook"]
    bodies = [row for row in rows if row["type"] == "body"]
    closings = [row for row in rows if row["type"] == "closing"]

    if not bodies:
        rows.insert(-1, dict(rows[-1]))
        rows[-2]["scene_id"] = f"{rows[-2]['scene_id']}-body"
        rows[-2]["id"] = rows[-2]["scene_id"]
        rows[-2]["type"] = "body"
        rows[-2]["narration"] = "Core context and key implications."
        rows[-2]["on_screen_text"] = "What matters most"
        rows[-2]["visual_strategy"] = "context_montage"
        rows[-2]["visual_prompt"] = "Cinematic montage of context visuals"
        rows[-2]["transition_hint"] = "match_cut"
        rows[-2]["motion_hint"] = "medium_pan"
        bodies = [rows[-2]]

    hook_target = 10.0
    closing_target = 10.0
    body_target = TARGET_DURATION_SECONDS - hook_target - closing_target
    body_target = max(60.0, min(70.0, body_target))

    _scale_bucket(hooks, hook_target)
    _scale_bucket(bodies, body_target)
    _scale_bucket(closings, closing_target)

    # Rebalance any drift into body first, then closing/hook.
    total = sum(float(row.get("_duration", 0.0)) for row in rows)
    drift = TARGET_DURATION_SECONDS - total
    if abs(drift) > 0.001:
        target_rows = bodies or closings or hooks
        target_rows[-1]["_duration"] = round(target_rows[-1]["_duration"] + drift, 3)

    timeline = 0.0
    for idx, row in enumerate(rows, start=1):
        duration = max(0.5, float(row.get("_duration", 5.0)))
        row["index"] = idx
        row["start"] = round(timeline, 3)
        timeline += duration
        row["end"] = round(timeline, 3)
        row["duration_seconds"] = round(duration, 3)
        row.pop("_duration", None)

    if rows:
        rows[-1]["end"] = TARGET_DURATION_SECONDS
        rows[-1]["duration_seconds"] = round(rows[-1]["end"] - rows[-1]["start"], 3)

    return rows


def _semantic_validate_scene_rows(rows: list[dict[str, Any]]) -> None:
    prev_start = -1.0
    total = 0.0
    for idx, row in enumerate(rows, start=1):
        start = float(row["start"])
        end = float(row["end"])
        if end <= start:
            raise ValidationPipelineError(f"Scene {idx} has non-positive duration")
        if start < prev_start:
            raise ValidationPipelineError(f"Scene {idx} start time is non-monotonic")
        if not str(row.get("visual_strategy", "")).strip():
            raise ValidationPipelineError(f"Scene {idx} missing visual_strategy")
        prev_start = start
        total += end - start

    if abs(total - TARGET_DURATION_SECONDS) > 1.0:
        raise ValidationPipelineError(
            f"Storyboard duration must be near {TARGET_DURATION_SECONDS}s, got {total:.2f}s"
        )


def _finalize_scene_rows(
    article: Article,
    understanding: ArticleUnderstanding,
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    key_points = [kp.text for kp in understanding.key_points]
    for row in rows:
        if not row["narration"]:
            row["narration"] = understanding.summary
        if not row["on_screen_text"]:
            row["on_screen_text"] = understanding.headline

        seed_source = f"{article.article_id}:{row['scene_id']}"
        row["visual_seed"] = int(sha256(seed_source.encode("utf-8")).hexdigest()[:8], 16)
        row["factual_risk"] = row.get("factual_risk", "medium")
        row["source_key_points"] = key_points[:2]
    return rows


def _validate_scenes(rows: list[dict[str, Any]]) -> list[Scene]:
    scenes: list[Scene] = []
    for row in rows:
        scenes.append(validate_payload(Scene, row))
    return scenes


def _fallback_template_plan(article: Article, understanding: ArticleUnderstanding) -> list[Scene]:
    hook_text = understanding.visual_hooks[0] if understanding.visual_hooks else understanding.headline
    key_points = [kp.text for kp in understanding.key_points]
    body_points = key_points[:4] if key_points else [understanding.summary]

    rows: list[dict[str, Any]] = [
        {
            "scene_id": "scene-001",
            "id": "scene-001",
            "index": 1,
            "start": 0.0,
            "end": 8.0,
            "type": "hook",
            "narration": hook_text,
            "on_screen_text": understanding.headline,
            "visual_strategy": "impact_opening",
            "visual_prompt": f"Dynamic opening visual for: {hook_text}",
            "transition_hint": "hard_cut",
            "motion_hint": "fast_push_in",
        }
    ]

    for idx, point in enumerate(body_points, start=2):
        scene_id = f"scene-{idx:03d}"
        rows.append(
            {
                "scene_id": scene_id,
                "id": scene_id,
                "index": idx,
                "start": 0.0,
                "end": 10.0,
                "type": "body",
                "narration": point,
                "on_screen_text": point,
                "visual_strategy": "evidence_montage",
                "visual_prompt": f"Editorial montage showing: {point}",
                "transition_hint": "match_cut",
                "motion_hint": "slow_pan",
            }
        )

    closing_id = f"scene-{len(rows) + 1:03d}"
    rows.append(
        {
            "scene_id": closing_id,
            "id": closing_id,
            "index": len(rows) + 1,
            "start": 0.0,
            "end": 8.0,
            "type": "closing",
            "narration": understanding.summary,
            "on_screen_text": "What to watch next",
            "visual_strategy": "resolution_close",
            "visual_prompt": f"Closing shot that reinforces topic: {understanding.topic}",
            "transition_hint": "fade_to_black",
            "motion_hint": "slow_pull_back",
        }
    )

    rows = _enforce_timing_policy(rows)
    rows = _finalize_scene_rows(article, understanding, rows)
    _semantic_validate_scene_rows(rows)
    return _validate_scenes(rows)


def _article_understanding_from_text(article: Article) -> ArticleUnderstanding:
    words = article.clean_article_text.split()
    headline = article.title.strip() if article.title.strip() else "Article Brief"
    summary = " ".join(words[:45])
    if len(words) > 45:
        summary += "..."

    parts = [p.strip() for p in article.clean_article_text.replace("\n", " ").split(".") if p.strip()]
    key_points = []
    for part in parts[:5]:
        key_points.append({"text": part, "importance": 3})
    if not key_points:
        key_points = [{"text": summary or "No article content provided.", "importance": 3}]

    payload = {
        "headline": headline,
        "summary": summary or "No summary available.",
        "key_points": key_points,
        "entities": [],
        "visual_hooks": [key_points[0]["text"]],
        "tone": "informative",
        "topic": "news",
    }
    return validate_payload(ArticleUnderstanding, payload)





def build_storyboard_stub(clean_text: str) -> list[Scene]:
    snippet = clean_text[:220].strip() or "No source text available."
    row = {
        "scene_id": "scene-001",
        "id": "scene-001",
        "index": 1,
        "start": 0.0,
        "end": 5.0,
        "type": "hook",
        "narration": snippet,
        "on_screen_text": "Top Story",
        "visual_strategy": "editorial_still",
        "visual_prompt": "Editorial still: newsroom background, cinematic lighting",
        "transition_hint": "hard_cut",
        "motion_hint": "static",
    }
    return [validate_payload(Scene, row)]


def plan_storyboard(
    article: Article,
    cfg: PipelineConfig,
    *,
    job_dir: Path,
    llm_enabled: bool,
) -> tuple[ArticleUnderstanding, list[Scene]]:
    story_dir = job_dir / "storyboard"
    prompt_fragments: list[str] = []
    raw_fragments: list[str] = []

    understanding: ArticleUnderstanding
    scenes: list[Scene]
    fallback_reason = ""
    try:
        rows, prompt, raw, understanding = _generate_storyboard_with_gemini(
            article,
            cfg,
            llm_enabled=llm_enabled,
        )
        prompt_fragments.append("# Prompt: Gemini Storyboard\n" + prompt)
        raw_fragments.append("# Output\n" + raw)

        rows = _enforce_timing_policy(rows)
        rows = _finalize_scene_rows(article, understanding, rows)
        _semantic_validate_scene_rows(rows)
        scenes = _validate_scenes(rows)
    except Exception as exc:
        # Deterministic fallback if LLM flow fails after bounded attempts.
        fallback_reason = (
            f"{type(exc).__name__}: {exc}\n\n"
            "Traceback:\n"
            f"{traceback.format_exc()}"
        )
        understanding = _article_understanding_from_text(article)
        scenes = _fallback_template_plan(article, understanding)
        prompt_fragments.append(
            "# Fallback Planner\n"
            "Deterministic template planner used due to upstream failure.\n\n"
            "## Fallback Reason\n"
            f"{type(exc).__name__}: {exc}"
        )
        raw_fragments.append(
            "# Fallback Reason\n"
            f"{fallback_reason}\n\n"
            "# Fallback Output\n"
            + json.dumps([s.model_dump() for s in scenes], indent=2)
        )

    _write_json(story_dir / "article_understanding.json", understanding.model_dump())
    _write_json(story_dir / "storyboard.json", {"scenes": [s.model_dump() for s in scenes]})
    _write_text(story_dir / "planner_prompt.txt", "\n\n".join(prompt_fragments).strip() + "\n")
    _write_text(story_dir / "planner_output_raw.txt", "\n\n".join(raw_fragments).strip() + "\n")

    return understanding, scenes
