from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen
import uuid

from src.common.errors import IOPipelineError
from src.common.models import Article
from src.common.text_utils import detect_language_placeholder, normalize_text


def _read_source(source: str) -> tuple[str, str]:
    maybe_path = Path(source)
    if maybe_path.exists() and maybe_path.is_file():
        return maybe_path.read_text(encoding="utf-8"), f"file:{maybe_path.as_posix()}"

    parsed = urlparse(source)
    if parsed.scheme in {"http", "https"} and parsed.netloc:
        try:
            with urlopen(source, timeout=15) as response:
                data = response.read().decode("utf-8", errors="replace")
            return data, source
        except OSError as exc:
            raise IOPipelineError(f"Failed to fetch URL source: {source}") from exc

    return source, "raw_text"


def ingest_article(source: str, title: str = "") -> Article:
    raw_text, source_label = _read_source(source)
    cleaned = normalize_text(raw_text)
    language = detect_language_placeholder(cleaned)
    return Article(
        article_id=uuid.uuid4().hex,
        source=source_label,
        language=language,
        title=title,
        raw_text=raw_text,
        clean_article_text=cleaned,
    )
