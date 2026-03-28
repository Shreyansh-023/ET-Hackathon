from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import imghdr
import json
from pathlib import Path
import re
import time
from typing import Any

import requests

from src.audio.pipeline import AudioBuildResult, build_voiceover_and_subtitles
from src.common.config import PipelineConfig
from src.common.errors import ProviderPipelineError
from src.common.models import Asset, Scene
from src.common.retry import run_with_retry


CLIPDROP_TEXT_TO_IMAGE_URL = "https://clipdrop-api.co/text-to-image/v1"
SERPAPI_SEARCH_URL = "https://serpapi.com/search.json"
SERPAPI_ENGINE = "google_images_light"
PEXELS_SEARCH_URL = "https://api.pexels.com/v1/search"
TOKEN_SPLIT = re.compile(r"[^a-zA-Z0-9]+")
MAX_STOCK_QUERY_COUNT = 12
SERPAPI_RESULTS_PER_QUERY = 12
PEXELS_RESULTS_PER_QUERY = 10
SERPAPI_CONFIDENT_RESULT_COUNT = 5
TRANSIENT_HTTP_STATUS_CODES = {408, 425, 429, 500, 502, 503, 504}
LOW_CONFIDENCE_URL_MARKERS = (
    "lookaside.instagram.com",
    "lookaside.fbsbx.com",
    "x.com/",
    "twitter.com/",
)
IMAGE_KIND_TO_EXTENSION = {
    "jpeg": ".jpg",
    "png": ".png",
    "gif": ".gif",
    "webp": ".webp",
    "bmp": ".bmp",
    "tiff": ".tiff",
}
MIN_AI_IMAGES = 1
MAX_AI_IMAGES = 3


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


def _extract_stock_queries(visual_suggestions: dict[str, Any]) -> list[str]:
    raw_queries = visual_suggestions.get("stock_search_queries")
    if not isinstance(raw_queries, list):
        raw_queries = visual_suggestions.get("pexels_search_queries", [])

    cleaned_queries: list[str] = []
    seen: set[str] = set()
    for raw in raw_queries if isinstance(raw_queries, list) else []:
        if not isinstance(raw, str):
            continue
        query = raw.strip()
        if not query:
            continue
        lowered = query.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        cleaned_queries.append(query)
        if len(cleaned_queries) >= MAX_STOCK_QUERY_COUNT:
            break
    return cleaned_queries


def _normalize_image_source(raw_source: Any) -> str:
    source = str(raw_source or "").strip().lower()
    if source in {"serpapi", "cygnusx1"}:
        return "serpapi"
    if source == "replicate":
        return "replicate"
    return "serpapi"


def _safe_query_slug(query: str) -> str:
    slug = "_".join(part.lower() for part in TOKEN_SPLIT.split(query) if part)
    if not slug:
        slug = f"query_{int(time.time() * 1000)}"
    return slug[:80]


def _extract_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _looks_like_html_payload(data: bytes) -> bool:
    sample = data[:4096].lower()
    stripped = sample.lstrip()
    return stripped.startswith((b"<!doctype", b"<html", b"<?xml")) or b"<html" in sample


def _detect_image_kind(data: bytes) -> str | None:
    kind = imghdr.what(None, data)
    if not isinstance(kind, str):
        return None
    return kind if kind in IMAGE_KIND_TO_EXTENSION else None


def _is_valid_image_payload(data: bytes) -> bool:
    if not data or len(data) < 256:
        return False
    if _looks_like_html_payload(data):
        return False
    return _detect_image_kind(data) is not None


def _image_extension_from_payload(data: bytes) -> str:
    kind = _detect_image_kind(data)
    if kind is None:
        return ".jpg"
    return IMAGE_KIND_TO_EXTENSION.get(kind, ".jpg")


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


def _score_stock_photo(photo: dict[str, Any], terms: list[str], orientation: str) -> float:
    width = int(photo.get("width") or 0)
    height = int(photo.get("height") or 0)

    ratio_score = 0.5
    if width > 0 and height > 0:
        ratio = width / height
        target_ratio = 9 / 16 if orientation == "portrait" else 16 / 9
        ratio_score = max(0.0, 1.0 - abs(ratio - target_ratio))

    area = width * height
    if area > 0:
        clarity_score = min(1.0, area / (1920 * 1080))
    else:
        clarity_score = 0.5

    text_blob = " ".join(
        str(photo.get(key, "")) for key in ("title", "source", "link", "raw_link", "url", "_query")
    ).lower()
    match_count = sum(1 for term in terms if term.lower() in text_blob)
    relevance = min(1.0, match_count / max(1, len(terms)))

    source_penalty = 1.0
    candidate_url = str(photo.get("url") or "").strip().lower()
    if any(marker in candidate_url for marker in LOW_CONFIDENCE_URL_MARKERS):
        source_penalty = 0.15

    return ((0.6 * relevance) + (0.25 * clarity_score) + (0.15 * ratio_score)) * source_penalty


def _normalize_serpapi_rows(payload: dict[str, Any], clean_query: str, query_slug: str) -> list[dict[str, Any]]:
    raw_rows = payload.get("images_results")
    if not isinstance(raw_rows, list):
        return []

    rows: list[dict[str, Any]] = []
    for idx, row in enumerate(raw_rows, start=1):
        if not isinstance(row, dict):
            continue

        original_url = str(row.get("original") or "").strip()
        thumbnail_url = str(row.get("thumbnail") or "").strip()
        url = original_url or thumbnail_url
        if not url.startswith("http"):
            continue

        related_id = str(row.get("related_content_id") or "").strip()
        stable_hash = sha256(f"{clean_query}|{url}".encode("utf-8")).hexdigest()[:12]
        row_id = related_id or f"{query_slug}-{idx}-{stable_hash}"

        rows.append(
            {
                "id": row_id,
                "_provider": "serpapi",
                "title": str(row.get("title") or ""),
                "source": str(row.get("source") or ""),
                "link": str(row.get("link") or ""),
                "raw_link": str(row.get("raw_link") or ""),
                "url": url,
                "thumbnail": thumbnail_url,
                "width": _extract_int(row.get("original_width")),
                "height": _extract_int(row.get("original_height")),
                "_query": clean_query,
                "position": _extract_int(row.get("position")) or idx,
            }
        )
        if len(rows) >= SERPAPI_RESULTS_PER_QUERY:
            break

    return rows


def _search_serpapi(query: str, cfg: PipelineConfig, job_dir: Path) -> list[dict[str, Any]]:
    if not cfg.serpapi_api_key:
        return []

    clean_query = query.strip()
    if not clean_query:
        return []

    query_slug = _safe_query_slug(clean_query)
    cache_dir = Path(cfg.cache_root) / "serpapi" / job_dir.name
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{query_slug}.json"

    def _request() -> dict[str, Any]:
        params = {
            "engine": SERPAPI_ENGINE,
            "q": clean_query,
            "api_key": cfg.serpapi_api_key,
            "output": "json",
        }

        try:
            response = requests.get(SERPAPI_SEARCH_URL, params=params, timeout=30)
        except requests.RequestException as exc:
            raise ProviderPipelineError(f"SerpApi search failed: {exc}") from exc

        if response.status_code in TRANSIENT_HTTP_STATUS_CODES:
            raise ProviderPipelineError(f"SerpApi transient failure: {response.status_code}")
        if response.status_code >= 400:
            raise ProviderPipelineError(f"SerpApi search failed with status {response.status_code}")

        try:
            payload = response.json()
        except ValueError as exc:
            raise ProviderPipelineError("SerpApi returned invalid JSON") from exc

        status = str((payload.get("search_metadata") or {}).get("status") or "")
        if status and status.lower() == "error":
            error_message = str(payload.get("error") or "SerpApi reported an error")
            raise ProviderPipelineError(error_message)

        return payload

    payload: dict[str, Any] | None = None
    try:
        payload = run_with_retry(_request, retries=2)
    except ProviderPipelineError:
        payload = None

    if payload is None and cache_path.exists():
        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            payload = None

    if payload is None:
        return []

    rows = _normalize_serpapi_rows(payload, clean_query, query_slug)
    if rows:
        try:
            cache_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        except OSError:
            pass

    return rows


def _normalize_pexels_rows(payload: dict[str, Any], clean_query: str, query_slug: str) -> list[dict[str, Any]]:
    photos = payload.get("photos")
    if not isinstance(photos, list):
        return []

    rows: list[dict[str, Any]] = []
    for idx, photo in enumerate(photos, start=1):
        if not isinstance(photo, dict):
            continue

        src = photo.get("src") if isinstance(photo.get("src"), dict) else {}
        url = ""
        for key in ("large2x", "large", "original", "medium", "small"):
            candidate = str(src.get(key) or "").strip()
            if candidate.startswith("http"):
                url = candidate
                break
        if not url:
            continue

        raw_id = photo.get("id")
        if isinstance(raw_id, int):
            row_id = f"pexels-{raw_id}"
        else:
            stable_hash = sha256(f"{clean_query}|{url}".encode("utf-8")).hexdigest()[:12]
            row_id = f"{query_slug}-{idx}-{stable_hash}"

        rows.append(
            {
                "id": row_id,
                "_provider": "pexels",
                "title": str(photo.get("alt") or ""),
                "source": "Pexels",
                "link": str(photo.get("url") or ""),
                "raw_link": str(photo.get("url") or ""),
                "url": url,
                "thumbnail": str(src.get("tiny") or src.get("small") or ""),
                "width": _extract_int(photo.get("width")),
                "height": _extract_int(photo.get("height")),
                "_query": clean_query,
                "position": idx,
            }
        )
        if len(rows) >= PEXELS_RESULTS_PER_QUERY:
            break

    return rows


def _search_pexels(query: str, cfg: PipelineConfig, orientation: str) -> list[dict[str, Any]]:
    if not cfg.pexels_api_key:
        return []

    clean_query = query.strip()
    if not clean_query:
        return []

    query_slug = _safe_query_slug(clean_query)

    def _request() -> list[dict[str, Any]]:
        try:
            response = requests.get(
                PEXELS_SEARCH_URL,
                headers={"Authorization": cfg.pexels_api_key},
                params={
                    "query": clean_query,
                    "per_page": PEXELS_RESULTS_PER_QUERY,
                    "orientation": orientation,
                },
                timeout=30,
            )
        except requests.RequestException as exc:
            raise ProviderPipelineError(f"Pexels search failed: {exc}") from exc

        if response.status_code in TRANSIENT_HTTP_STATUS_CODES:
            raise ProviderPipelineError(f"Pexels transient failure: {response.status_code}")
        if response.status_code >= 400:
            raise ProviderPipelineError(f"Pexels search failed with status {response.status_code}")

        try:
            payload = response.json()
        except ValueError as exc:
            raise ProviderPipelineError("Pexels returned invalid JSON") from exc

        return _normalize_pexels_rows(payload, clean_query, query_slug)

    return run_with_retry(_request, retries=2)


def _serpapi_rows_are_confident(rows: list[dict[str, Any]]) -> bool:
    usable_rows = 0
    for row in rows:
        url = str(row.get("url") or "").strip().lower()
        if not url.startswith("http"):
            continue
        if any(marker in url for marker in LOW_CONFIDENCE_URL_MARKERS):
            continue
        usable_rows += 1
        if _extract_int(row.get("width")) >= 640 and _extract_int(row.get("height")) >= 360:
            return True
    return usable_rows >= SERPAPI_CONFIDENT_RESULT_COUNT


def _search_stock_with_fallback(
    query: str,
    cfg: PipelineConfig,
    job_dir: Path,
    orientation: str,
) -> list[dict[str, Any]]:
    serpapi_rows = _search_serpapi(query, cfg, job_dir) if cfg.serpapi_api_key else []
    if not cfg.pexels_api_key:
        return serpapi_rows

    if _serpapi_rows_are_confident(serpapi_rows):
        return serpapi_rows

    pexels_rows = _search_pexels(query, cfg, orientation)
    if not serpapi_rows:
        return pexels_rows

    seen_ids = {str(row.get("id") or "").strip() for row in serpapi_rows}
    merged = list(serpapi_rows)
    for row in pexels_rows:
        row_id = str(row.get("id") or "").strip()
        if not row_id or row_id in seen_ids:
            continue
        seen_ids.add(row_id)
        merged.append(row)
    return merged


def _collect_stock_pool(
    queries: list[str],
    cfg: PipelineConfig,
    job_dir: Path,
    orientation: str,
) -> list[dict[str, Any]]:
    pool: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for query in queries:
        photos = _search_stock_with_fallback(query, cfg, job_dir, orientation)
        for photo in photos:
            photo_id = str(photo.get("id") or "").strip()
            if not photo_id:
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


def _call_clipdrop_text_to_image(prompt: str, cfg: PipelineConfig) -> bytes | None:
    if not cfg.clip_drop_api_key:
        return None

    news_style_suffix = (
        " Use bold, vibrant red and yellow accent colors throughout the image. "
        "Style it as a professional news broadcast graphic with high contrast and clean design."
    )
    styled_prompt = prompt.rstrip(". ") + "." + news_style_suffix

    headers = {"x-api-key": cfg.clip_drop_api_key}

    def _request() -> bytes | None:
        try:
            response = requests.post(
                CLIPDROP_TEXT_TO_IMAGE_URL,
                headers=headers,
                files={"prompt": (None, styled_prompt)},
                timeout=60,
            )
        except requests.RequestException as exc:
            raise ProviderPipelineError(f"Clipdrop request failed: {exc}") from exc

        if response.status_code in TRANSIENT_HTTP_STATUS_CODES:
            raise ProviderPipelineError(f"Clipdrop transient failure: {response.status_code}")
        if response.status_code >= 400:
            return None
        if not response.content:
            return None
        return response.content

    return run_with_retry(_request, retries=2)


def _best_stock_candidate(
    pool: list[dict[str, Any]],
    terms: list[str],
    orientation: str,
    used_asset_ids: set[str],
) -> dict[str, Any] | None:
    ranked = sorted(
        pool,
        key=lambda row: _score_stock_photo(row, terms, orientation),
        reverse=True,
    )
    for photo in ranked:
        photo_id = str(photo.get("id") or "").strip()
        if photo_id and photo_id in used_asset_ids:
            continue
        return photo
    return None


def _resolve_scene_asset(
    scene: Scene,
    cfg: PipelineConfig,
    job_dir: Path,
    article_understanding: dict[str, Any],
    stock_pool: list[dict[str, Any]],
    used_asset_ids: set[str],
    prev_hash: str | None,
    ai_images_used: int,
    enforce_ai_generation: bool,
    *,
    dry_run: bool,
    logger=None,
) -> tuple[Asset, dict[str, Any], str | None]:
    terms = _query_terms(scene, article_understanding)
    visual_suggestions = scene.visual_suggestions or {}

    stock_queries = _extract_stock_queries(visual_suggestions)
    query = stock_queries[0] if stock_queries else (" ".join(terms[:5]) if terms else scene.narration)
    orientation = _pick_orientation(cfg.aspect_ratio)

    preferred_source = _normalize_image_source(visual_suggestions.get("image_source", "serpapi"))
    visual_type = visual_suggestions.get("type", "photo")
    ai_prompt = str(visual_suggestions.get("replicate_prompt") or "").strip()

    prefer_ai = preferred_source == "replicate" or enforce_ai_generation
    ai_allowed = (ai_images_used < MAX_AI_IMAGES) or not prefer_ai

    def _try_ai_generation() -> tuple[Asset, dict[str, Any], str] | None:
        if not ai_allowed or not ai_prompt or dry_run or not cfg.clip_drop_api_key:
            return None

        image_bytes = _call_clipdrop_text_to_image(ai_prompt, cfg)
        if not image_bytes:
            return None

        content_hash = sha256(image_bytes).hexdigest()
        if prev_hash and content_hash == prev_hash:
            return None

        rel_path = f"assets/generated/{scene.scene_id}.png"
        abs_path = job_dir / rel_path
        abs_path.parent.mkdir(parents=True, exist_ok=True)
        abs_path.write_bytes(image_bytes)
        metadata = {
            "provider": "clipdrop_text_to_image",
            "model": "clipdrop-text-to-image-v1",
            "prompt": ai_prompt,
            "visual_type": visual_type,
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

    if prefer_ai:
        ai_result = _try_ai_generation()
        if ai_result is not None:
            return ai_result

    if not dry_run and (cfg.serpapi_api_key or cfg.pexels_api_key):
        candidate = _best_stock_candidate(stock_pool, terms, orientation, used_asset_ids)
        if candidate is None and query.strip():
            dynamic_pool = _search_stock_with_fallback(query, cfg, job_dir, orientation)
            candidate = _best_stock_candidate(dynamic_pool, terms, orientation, used_asset_ids)

        if candidate is not None:
            for _attempt in range(4):
                image_url = str(candidate.get("url") or "").strip()
                if not image_url:
                    break

                try:
                    image_bytes = _download_bytes(image_url)
                except ProviderPipelineError:
                    image_bytes = b""

                if image_bytes and not _is_valid_image_payload(image_bytes):
                    if logger:
                        logger.emit(
                            "warning",
                            "asset_invalid_image_payload",
                            scene_id=scene.scene_id,
                            provider=str(candidate.get("_provider") or "unknown"),
                            source_url=image_url,
                        )
                    image_bytes = b""

                if image_bytes:
                    content_hash = sha256(image_bytes).hexdigest()
                    if not (prev_hash and content_hash == prev_hash):
                        ext = _image_extension_from_payload(image_bytes)
                        rel_path = f"assets/images/{scene.scene_id}{ext}"
                        abs_path = job_dir / rel_path
                        abs_path.parent.mkdir(parents=True, exist_ok=True)
                        abs_path.write_bytes(image_bytes)

                        stock_asset_id = str(candidate.get("id") or "").strip()
                        if stock_asset_id:
                            used_asset_ids.add(stock_asset_id)

                        provider_name = str(candidate.get("_provider") or "serpapi").strip().lower()
                        source_value = "pexels" if provider_name == "pexels" else "serpapi"

                        metadata = {
                            "provider": source_value,
                            "provider_asset_id": stock_asset_id,
                            "query": candidate.get("_query") or query,
                            "orientation": orientation,
                            "source_url": image_url,
                            "source_page": str(candidate.get("link") or ""),
                        }
                        asset = Asset(
                            asset_id=f"asset-{scene.scene_id}",
                            scene_id=scene.scene_id,
                            kind="image",
                            source=source_value,
                            path=rel_path,
                            metadata=metadata,
                        )
                        return asset, metadata, content_hash

                provider_name = str(candidate.get("_provider") or "serpapi").strip().lower()
                if provider_name == "pexels" or not cfg.pexels_api_key or not query.strip():
                    break

                pexels_pool = _search_pexels(query, cfg, orientation)
                replacement = _best_stock_candidate(pexels_pool, terms, orientation, used_asset_ids)
                if replacement is None:
                    break
                candidate = replacement

    if not prefer_ai:
        ai_result = _try_ai_generation()
        if ai_result is not None:
            return ai_result

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
    article_language: str = "",
    dry_run: bool,
    logger=None,
) -> Step3Result:
    article_understanding = _load_json(job_dir / "storyboard" / "article_understanding.json")

    all_stock_queries: list[str] = []
    for scene in scenes:
        visual_suggestions = scene.visual_suggestions or {}
        all_stock_queries.extend(_extract_stock_queries(visual_suggestions))

    seen_queries: set[str] = set()
    unique_queries: list[str] = []
    for raw_query in all_stock_queries:
        query = raw_query.strip()
        if not query:
            continue
        lowered = query.lower()
        if lowered in seen_queries:
            continue
        seen_queries.add(lowered)
        unique_queries.append(query)
        if len(unique_queries) >= MAX_STOCK_QUERY_COUNT:
            break

    stock_pool: list[dict[str, Any]] = []
    if not dry_run and (cfg.serpapi_api_key or cfg.pexels_api_key) and unique_queries:
        stock_pool = _collect_stock_pool(unique_queries, cfg, job_dir, _pick_orientation(cfg.aspect_ratio))

    ai_candidate_indices: list[int] = []
    for idx, scene in enumerate(scenes):
        vs = scene.visual_suggestions or {}
        if str(vs.get("replicate_prompt") or "").strip():
            ai_candidate_indices.append(idx)

    preferred_ai_indices = [
        idx
        for idx in ai_candidate_indices
        if _normalize_image_source((scenes[idx].visual_suggestions or {}).get("image_source")) == "replicate"
    ]

    selected_ai_indices = set(preferred_ai_indices[:MAX_AI_IMAGES])
    if len(selected_ai_indices) < MIN_AI_IMAGES and ai_candidate_indices:
        for idx in ai_candidate_indices:
            if idx not in selected_ai_indices:
                selected_ai_indices.add(idx)
            if len(selected_ai_indices) >= MIN_AI_IMAGES:
                break

    assets: list[Asset] = []
    registry_items: list[dict[str, Any]] = []
    prev_hash: str | None = None
    used_asset_ids: set[str] = set()
    ai_images_used = 0

    for idx, scene in enumerate(scenes):
        asset, provider_meta, current_hash = _resolve_scene_asset(
            scene,
            cfg,
            job_dir,
            article_understanding,
            stock_pool,
            used_asset_ids,
            prev_hash,
            ai_images_used,
            enforce_ai_generation=(idx in selected_ai_indices),
            dry_run=dry_run,
            logger=logger,
        )
        assets.append(asset)
        if asset.source == "replicate":
            ai_images_used += 1

        visual_suggestions = scene.visual_suggestions or {}
        stock_queries = _extract_stock_queries(visual_suggestions)

        registry_items.append(
            {
                "scene_id": scene.scene_id,
                "asset_id": asset.asset_id,
                "source": asset.source,
                "path": asset.path,
                "provenance": provider_meta,
                "llm_plan": {
                    "visual_type": visual_suggestions.get("type", "photo"),
                    "stock_query": stock_queries[0] if stock_queries else "",
                    "pexels_query": stock_queries[0] if stock_queries else "",
                },
            }
        )
        prev_hash = current_hash or prev_hash

    audio_result = build_voiceover_and_subtitles(
        scenes,
        cfg,
        job_dir,
        article_language=article_language,
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
                "stock_provider": "serpapi",
                "stock_fallback_provider": "pexels",
                "stock_query_count": len(unique_queries),
                "stock_queries": unique_queries,
                "pool_size": len(stock_pool),
                "pexels_query_count": len(unique_queries),
                "pexels_queries": unique_queries,
                "ai_image_policy": {
                    "min": MIN_AI_IMAGES,
                    "max": MAX_AI_IMAGES,
                    "requested": len(selected_ai_indices),
                    "generated": ai_images_used,
                    "provider": "clipdrop_text_to_image",
                },
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
