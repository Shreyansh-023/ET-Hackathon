from __future__ import annotations

import re
import langid


MULTISPACE = re.compile(r"\s+")
SPACE_BEFORE_PUNCT = re.compile(r"\s+([,.;:!?])")


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = MULTISPACE.sub(" ", text).strip()
    text = SPACE_BEFORE_PUNCT.sub(r"\1", text)
    return text


def detect_language(text: str) -> str:
    """Return pipeline language label: 'english' or 'hindi'."""
    if not text:
        return "english"

    normalized = normalize_text(text)
    if not normalized:
        return "english"

    language_code, _confidence = langid.classify(normalized)
    if language_code == "hi":
        return "hindi"
    if language_code == "en":
        return "english"

    return "english"


def detect_language_placeholder(text: str) -> str:
    """Backwards-compatible shim for older imports."""
    return detect_language(text)
