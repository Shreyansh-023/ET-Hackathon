from __future__ import annotations

import re


MULTISPACE = re.compile(r"\s+")
SPACE_BEFORE_PUNCT = re.compile(r"\s+([,.;:!?])")


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = MULTISPACE.sub(" ", text).strip()
    text = SPACE_BEFORE_PUNCT.sub(r"\1", text)
    return text


def detect_language_placeholder(text: str) -> str:
    if not text:
        return "unknown"
    lowered = text.lower()
    common_english = {"the", "and", "is", "to", "of", "in"}
    score = sum(1 for token in lowered.split() if token in common_english)
    return "en" if score >= 2 else "unknown"
