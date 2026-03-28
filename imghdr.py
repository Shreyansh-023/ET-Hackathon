from __future__ import annotations

from pathlib import Path
from typing import BinaryIO


def _read_header(file: str | bytes | Path | BinaryIO, size: int = 32) -> bytes:
    if hasattr(file, "read"):
        position = file.tell()
        header = file.read(size)
        file.seek(position)
        return header

    with open(file, "rb") as handle:
        return handle.read(size)


def what(file: str | bytes | Path | BinaryIO, h: bytes | None = None) -> str | None:
    header = h if h is not None else _read_header(file)
    if len(header) < 12:
        return None

    if header.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    if header.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if header[:6] in {b"GIF87a", b"GIF89a"}:
        return "gif"
    if header.startswith(b"RIFF") and header[8:12] == b"WEBP":
        return "webp"
    if header.startswith(b"BM"):
        return "bmp"
    if header[:4] in {b"II*\x00", b"MM\x00*"}:
        return "tiff"

    return None
