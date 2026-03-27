from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import re
import time
from typing import Any

import requests

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - optional dependency for Groq visual planning
    OpenAI = None

from src.audio.pipeline import AudioBuildResult, build_voiceover_and_subtitles
from src.common.config import PipelineConfig
from src.common.errors import ProviderPipelineError
from src.common.models import Asset, Scene
from src.common.retry import run_with_retry


PEXELS_SEARCH_URL = "https://api.pexels.com/v1/search"
REPLICATE_PREDICT_URL = "https://api.replicate.com/v1/models/black-forest-labs/flux-1.1-pro/predictions"
TOKEN_SPLIT = re.compile(r"[^a-zA-Z0-9]+")
MAX_PEXELS_QUERY_COUNT = 12
MIN_PEXELS_QUERY_COUNT = 9


@dataclass(frozen=True)
class Step3Result:
    assets: list[Asset]
    assets_registry_rel_path: str
    audio: AudioBuildResult


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def _extract_json_payload(raw: str) -> Any:
    text = raw.strip()
    if not text:
        raise ProviderPipelineError("Visual planner returned an empty response")

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    first_obj = text.find("{")
    first_arr = text.find("[")
    starts = [idx for idx in (first_obj, first_arr) if idx != -1]
    if not starts:
        raise ProviderPipelineError("Visual planner response did not contain JSON")

    start = min(starts)
    end_obj = text.rfind("}")
    end_arr = text.rfind("]")
    end = max(end_obj, end_arr)
    if end <= start:
        raise ProviderPipelineError("Visual planner returned malformed JSON boundaries")

    candidate = text[start : end + 1]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise ProviderPipelineError(f"Visual planner JSON parse failed: {exc}") from exc


def _download_bytes(url: str, *, headers: dict[str, str] | None = None) -> bytes:
    try:
        response = requests.get(url, headers=headers, timeout=30)
    except requests.RequestException as exc:
        raise ProviderPipelineError(f"Asset download failed: {exc}") from exc

    if response.status_code >= 400:
        raise ProviderPipelineError(f"Asset download failed with status {response.status_code}")
    if not response.content:
        raise ProviderPipelineError("Asset provider returned empty content")
    return response.content


def _pick_orientation(aspect_ratio: str) -> str:
    return "portrait" if aspect_ratio.strip() in {"9:16", "4:5", "3:4"} else "landscape"


def _scene_script_text(scenes: list[Scene]) -> str:
    return "\n".join(
        f"[{scene.scene_id}] ({scene.type}) NARRATION: {scene.narration.strip()} | ONSCREEN: {scene.on_screen_text.strip()}"
        for scene in scenes
    )





def _query_terms(scene: Scene, article_understanding: dict[str, Any]) -> list[str]:
    entities = article_understanding.get("entities") if isinstance(article_understanding, dict) else []
    hooks = article_understanding.get("visual_hooks") if isinstance(article_understanding, dict) else []
    key_points = article_understanding.get("key_points") if isinstance(article_understanding, dict) else []

    terms: list[str] = []
    for raw in entities if isinstance(entities, list) else []:
        if isinstance(raw, str) and raw.strip():
            terms.append(raw.strip())
    for raw in hooks if isinstance(hooks, list) else []:
        if isinstance(raw, str) and raw.strip():
            terms.append(raw.strip())
    for row in key_points if isinstance(key_points, list) else []:
        if isinstance(row, dict) and isinstance(row.get("text"), str):
            terms.append(row["text"].strip())

    terms.extend([
        scene.visual_strategy,
        scene.visual_prompt,
        scene.on_screen_text,
        scene.narration,
    ])

    compact: list[str] = []
    seen: set[str] = set()
    for term in terms:
        cleaned = " ".join(token for token in TOKEN_SPLIT.split(term) if token)
        if not cleaned:
            continue
        lowered = cleaned.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        compact.append(cleaned)
        if len(compact) >= 8:
            break
    return compact


def _score_pexels_photo(photo: dict[str, Any], terms: list[str], orientation: str) -> float:
    width = int(photo.get("width") or 0)
    height = int(photo.get("height") or 0)
    if width <= 0 or height <= 0:
        return 0.0

    ratio = width / height
    target_ratio = 9 / 16 if orientation == "portrait" else 16 / 9
    ratio_score = max(0.0, 1.0 - abs(ratio - target_ratio))

    area = width * height
    clarity_score = min(1.0, area / (1920 * 1080))

    text_blob = " ".join(
        str(photo.get(key, "")) for key in ("alt", "photographer", "url")
    ).lower()
    match_count = sum(1 for term in terms if term.lower() in text_blob)
    relevance = min(1.0, match_count / max(1, len(terms)))

    return (0.5 * relevance) + (0.3 * clarity_score) + (0.2 * ratio_score)


def _search_pexels(query: str, cfg: PipelineConfig) -> list[dict[str, Any]]:
    if not cfg.pexels_api_key:
        return []

    orientation = _pick_orientation(cfg.aspect_ratio)
    headers = {"Authorization": cfg.pexels_api_key}

    def _request() -> list[dict[str, Any]]:
        try:
            response = requests.get(
                PEXELS_SEARCH_URL,
                headers=headers,
                params={"query": query, "per_page": 5, "orientation": orientation},
                timeout=20,
            )
        except requests.RequestException as exc:
            raise ProviderPipelineError(f"Pexels search failed: {exc}") from exc

        if response.status_code in {408, 425, 429, 500, 502, 503, 504}:
            raise ProviderPipelineError(f"Pexels transient failure: {response.status_code}")
        if response.status_code >= 400:
            return []

        data = response.json()
        photos = data.get("photos", [])
        return photos if isinstance(photos, list) else []

    return run_with_retry(_request, retries=2)


def _collect_pexels_pool(queries: list[str], cfg: PipelineConfig) -> list[dict[str, Any]]:
    pool: list[dict[str, Any]] = []
    seen_ids: set[int] = set()
    for query in queries:
        photos = _search_pexels(query, cfg)
        for photo in photos:
            photo_id = photo.get("id")
            if not isinstance(photo_id, int):
                continue
            if photo_id in seen_ids:
                continue
            seen_ids.add(photo_id)
            annotated = dict(photo)
            annotated["_query"] = query
            pool.append(annotated)
    return pool


def _save_placeholder(scene: Scene, job_dir: Path) -> tuple[str, dict[str, Any], str]:
    label = scene.on_screen_text.strip() or scene.narration.strip() or scene.scene_id
    escaped = (
        label.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )
    rel_path = f"assets/placeholders/{scene.scene_id}.svg"
    abs_path = job_dir / rel_path
    abs_path.parent.mkdir(parents=True, exist_ok=True)
    svg = (
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"1080\" height=\"1920\">"
        "<defs><linearGradient id=\"g\" x1=\"0\" y1=\"0\" x2=\"1\" y2=\"1\">"
        "<stop offset=\"0%\" stop-color=\"#0f172a\"/><stop offset=\"100%\" stop-color=\"#1e293b\"/>"
        "</linearGradient></defs>"
        "<rect x=\"0\" y=\"0\" width=\"1080\" height=\"1920\" fill=\"url(#g)\"/>"
        f"<text x=\"540\" y=\"940\" text-anchor=\"middle\" fill=\"#e2e8f0\" font-size=\"44\" font-family=\"Helvetica\">{escaped}</text>"
        "</svg>"
    )
    abs_path.write_text(svg, encoding="utf-8")
    file_hash = sha256(svg.encode("utf-8")).hexdigest()
    return rel_path, {"provider": "placeholder", "reason": "all_providers_failed"}, file_hash


def _call_replicate(prompt: str, cfg: PipelineConfig) -> dict[str, Any] | None:
    if not cfg.replicate_api_token:
        return None

    headers = {
        "Authorization": f"Token {cfg.replicate_api_token}",
        "Content-Type": "application/json",
    }
    # Append news-agency style direction to every prompt
    news_style_suffix = (
        " Use bold, vibrant red and yellow accent colors throughout the image. "
        "Style it as a professional news broadcast graphic with high contrast and clean design."
    )
    styled_prompt = prompt.rstrip(". ") + "." + news_style_suffix

    payload = {
        "input": {
            "prompt": styled_prompt,
            "aspect_ratio": "1:1",
            "width": 1080,
            "height": 1080,
            "output_format": "png",
            "safety_tolerance": 2,
        }
    }

    def _request() -> dict[str, Any] | None:
        try:
            response = requests.post(REPLICATE_PREDICT_URL, headers=headers, json=payload, timeout=30)
        except requests.RequestException as exc:
            raise ProviderPipelineError(f"Replicate request failed: {exc}") from exc

        if response.status_code in {408, 425, 429, 500, 502, 503, 504}:
            raise ProviderPipelineError(f"Replicate transient failure: {response.status_code}")
        if response.status_code >= 400:
            return None
        data = response.json()
        return data if isinstance(data, dict) else None

    prediction = run_with_retry(_request, retries=2)
    if not prediction:
        return None

    status_url = prediction.get("urls", {}).get("get")
    if not isinstance(status_url, str) or not status_url:
        return None

    for _ in range(30):
        try:
            poll = requests.get(status_url, headers=headers, timeout=30)
        except requests.RequestException as exc:
            raise ProviderPipelineError(f"Replicate polling failed: {exc}") from exc
        if poll.status_code >= 400:
            return None
        poll_data = poll.json()
        status = poll_data.get("status")
        if status == "succeeded":
            return poll_data if isinstance(poll_data, dict) else None
        if status in {"failed", "canceled"}:
            return None
        time.sleep(1.0)
    return None


def _extract_replicate_output_url(payload: dict[str, Any]) -> str | None:
    output = payload.get("output")
    if isinstance(output, str) and output:
        return output
    if isinstance(output, list):
        for item in output:
            if isinstance(item, str) and item:
                return item
            if isinstance(item, dict):
                candidate = item.get("url")
                if isinstance(candidate, str) and candidate:
                    return candidate
    if isinstance(output, dict):
        for key in ("url", "image", "output"):
            candidate = output.get(key)
            if isinstance(candidate, str) and candidate:
                return candidate
    return None


def _best_pexels_candidate(
    pool: list[dict[str, Any]],
    terms: list[str],
    orientation: str,
    used_photo_ids: set[int],
) -> dict[str, Any] | None:
    ranked = sorted(
        pool,
        key=lambda row: _score_pexels_photo(row, terms, orientation),
        reverse=True,
    )
    for photo in ranked:
        photo_id = photo.get("id")
        if isinstance(photo_id, int) and photo_id in used_photo_ids:
            continue
        return photo
    return None


def _resolve_scene_asset(
    scene: Scene,
    cfg: PipelineConfig,
    job_dir: Path,
    article_understanding: dict[str, Any],
    pexels_pool: list[dict[str, Any]],
    used_photo_ids: set[int],
    prev_hash: str | None,
    *,
    dry_run: bool,
    logger=None,
) -> tuple[Asset, dict[str, Any], str | None]:
    terms = _query_terms(scene, article_understanding)
    visual_suggestions = scene.visual_suggestions or {}

    # Use the first Pexels query, or fall back to generating one
    pexels_queries = visual_suggestions.get("pexels_search_queries", [])
    query = pexels_queries[0] if pexels_queries else (" ".join(terms[:5]) if terms else scene.narration)
    orientation = _pick_orientation(cfg.aspect_ratio)

    # Determine preferred image source from Gemini's suggestion
    preferred_source = visual_suggestions.get("image_source", "pexels")
    visual_type = visual_suggestions.get("type", "photo")
    replicate_prompt = visual_suggestions.get("replicate_prompt", "").strip()

    # --- Try preferred source first, then fallback to the other ---

    if preferred_source == "replicate" and replicate_prompt and not dry_run and cfg.replicate_api_token:
        # Replicate-preferred: try AI generation first
        prediction = _call_replicate(replicate_prompt, cfg)
        if prediction:
            output_url = _extract_replicate_output_url(prediction)
            if output_url:
                image_bytes = _download_bytes(output_url)
                content_hash = sha256(image_bytes).hexdigest()
                if not (prev_hash and content_hash == prev_hash):
                    rel_path = f"assets/generated/{scene.scene_id}.png"
                    abs_path = job_dir / rel_path
                    abs_path.parent.mkdir(parents=True, exist_ok=True)
                    abs_path.write_bytes(image_bytes)
                    metadata = {
                        "provider": "replicate",
                        "model": "black-forest-labs/flux-1.1-pro",
                        "prompt": replicate_prompt,
                        "visual_type": visual_type,
                        "prediction_id": prediction.get("id"),
                    }
                    asset = Asset(
                        asset_id=f"asset-{scene.scene_id}",
                        scene_id=scene.scene_id,
                        kind="image",
                        source="replicate",
                        path=rel_path,
                        metadata=metadata,
                    )
                    return asset, metadata, content_hash

    # Pexels: try stock photos (either as primary or as fallback for failed replicate)
    if not dry_run and cfg.pexels_api_key:
        candidate = _best_pexels_candidate(pexels_pool, terms, orientation, used_photo_ids)
        if candidate is None and query.strip():
            dynamic_pool = _search_pexels(query, cfg)
            candidate = _best_pexels_candidate(dynamic_pool, terms, orientation, used_photo_ids)

        if candidate is not None:
            src = candidate.get("src") if isinstance(candidate.get("src"), dict) else {}
            image_url = src.get("large2x") or src.get("large") or src.get("original")
            if isinstance(image_url, str) and image_url:
                image_bytes = _download_bytes(image_url)
                content_hash = sha256(image_bytes).hexdigest()
                if not (prev_hash and content_hash == prev_hash):
                    ext = ".jpg"
                    rel_path = f"assets/images/{scene.scene_id}{ext}"
                    abs_path = job_dir / rel_path
                    abs_path.parent.mkdir(parents=True, exist_ok=True)
                    abs_path.write_bytes(image_bytes)

                    photo_id = candidate.get("id")
                    if isinstance(photo_id, int):
                        used_photo_ids.add(photo_id)

                    metadata = {
                        "provider": "pexels",
                        "provider_asset_id": candidate.get("id"),
                        "query": candidate.get("_query") or query,
                        "photographer": candidate.get("photographer"),
                        "orientation": orientation,
                    }
                    asset = Asset(
                        asset_id=f"asset-{scene.scene_id}",
                        scene_id=scene.scene_id,
                        kind="image",
                        source="pexels",
                        path=rel_path,
                        metadata=metadata,
                    )
                    return asset, metadata, content_hash

    # Replicate fallback: if pexels was preferred but failed, or for non-photo types
    if preferred_source != "replicate" and replicate_prompt and not dry_run and cfg.replicate_api_token:
        prediction = _call_replicate(replicate_prompt, cfg)
        if prediction:
            output_url = _extract_replicate_output_url(prediction)
            if output_url:
                image_bytes = _download_bytes(output_url)
                content_hash = sha256(image_bytes).hexdigest()
                if not (prev_hash and content_hash == prev_hash):
                    rel_path = f"assets/generated/{scene.scene_id}.png"
                    abs_path = job_dir / rel_path
                    abs_path.parent.mkdir(parents=True, exist_ok=True)
                    abs_path.write_bytes(image_bytes)
                    metadata = {
                        "provider": "replicate",
                        "model": "black-forest-labs/flux-1.1-pro",
                        "prompt": replicate_prompt,
                        "visual_type": visual_type,
                        "prediction_id": prediction.get("id"),
                    }
                    asset = Asset(
                        asset_id=f"asset-{scene.scene_id}",
                        scene_id=scene.scene_id,
                        kind="image",
                        source="replicate",
                        path=rel_path,
                        metadata=metadata,
                    )
                    return asset, metadata, content_hash

    rel_path, metadata, content_hash = _save_placeholder(scene, job_dir)
    asset = Asset(
        asset_id=f"asset-{scene.scene_id}",
        scene_id=scene.scene_id,
        kind="image",
        source="placeholder",
        path=rel_path,
        metadata=metadata,
    )
    if logger:
        logger.emit("warning", "asset_placeholder_used", scene_id=scene.scene_id)
    return asset, metadata, content_hash


def build_assets_step(
    scenes: list[Scene],
    cfg: PipelineConfig,
    job_dir: Path,
    *,
    dry_run: bool,
    logger=None,
) -> Step3Result:
    article_understanding = _load_json(job_dir / "storyboard" / "article_understanding.json")

    # Extract Pexels queries from all scenes
    all_pexels_queries = []
    for scene in scenes:
        if scene.visual_suggestions and scene.visual_suggestions.get("pexels_search_queries"):
            all_pexels_queries.extend(scene.visual_suggestions["pexels_search_queries"])
    
    # Deduplicate queries
    seen_queries = set()
    unique_queries = []
    for q in all_pexels_queries:
        if q.lower() not in seen_queries:
            unique_queries.append(q)
            seen_queries.add(q.lower())

    pexels_pool: list[dict[str, Any]] = []
    if not dry_run and cfg.pexels_api_key:
        pexels_pool = _collect_pexels_pool(unique_queries, cfg)

    # Ensure at least one scene uses replicate — if none are marked,
    # pick the first scene with a non-empty replicate_prompt and force it
    has_replicate = any(
        (s.visual_suggestions or {}).get("image_source") == "replicate"
        for s in scenes
    )
    if not has_replicate and cfg.replicate_api_token:
        for s in scenes:
            vs = s.visual_suggestions or {}
            if vs.get("replicate_prompt", "").strip():
                vs["image_source"] = "replicate"
                s.visual_suggestions = vs
                break

    assets: list[Asset] = []
    registry_items: list[dict[str, Any]] = []
    prev_hash: str | None = None
    used_photo_ids: set[int] = set()

    for scene in scenes:
        asset, provider_meta, current_hash = _resolve_scene_asset(
            scene,
            cfg,
            job_dir,
            article_understanding,
            pexels_pool,
            used_photo_ids,
            prev_hash,
            dry_run=dry_run,
            logger=logger,
        )
        assets.append(asset)
        
        visual_suggestions = scene.visual_suggestions or {}
        pexels_queries = visual_suggestions.get("pexels_search_queries", [])
        
        registry_items.append(
            {
                "scene_id": scene.scene_id,
                "asset_id": asset.asset_id,
                "source": asset.source,
                "path": asset.path,
                "provenance": provider_meta,
                "llm_plan": {
                    "visual_type": visual_suggestions.get("type", "photo"),
                    "pexels_query": pexels_queries[0] if pexels_queries else "",
                },
            }
        )
        prev_hash = current_hash or prev_hash

    audio_result = build_voiceover_and_subtitles(
        scenes,
        cfg,
        job_dir,
        dry_run=dry_run,
        logger=logger,
    )

    assets_registry_rel_path = "assets/assets_registry.json"
    _write_json(
        job_dir / assets_registry_rel_path,
        {
            "scene_count": len(scenes),
            "visual_plan": {
                "notes": "Generated by Gemini",
                "pexels_query_count": len(unique_queries),
                "pexels_queries": unique_queries,
                "pool_size": len(pexels_pool),
            },
            "assets": registry_items,
            "voiceover": {
                "path": audio_result.voiceover_rel_path,
                "duration_seconds": audio_result.duration_seconds,
            },
            "subtitles": {
                "srt": audio_result.subtitles_srt_rel_path,
                "vtt": audio_result.subtitles_vtt_rel_path,
            },
        },
    )

    # Maintain previous artifact contract while providing richer registry output.
    _write_json(job_dir / "assets" / "assets.json", [asset.model_dump() for asset in assets])

    return Step3Result(
        assets=assets,
        assets_registry_rel_path=assets_registry_rel_path,
        audio=audio_result,
    )


def build_asset_stub(scenes: list[Scene]) -> list[Asset]:
    assets: list[Asset] = []
    for scene in scenes:
        assets.append(
            Asset(
                asset_id=f"asset-{scene.scene_id}",
                scene_id=scene.scene_id,
                kind="image",
                source="stub",
                path=f"assets/{scene.scene_id}.png",
                metadata={"status": "placeholder"},
            )
        )
    return assets
