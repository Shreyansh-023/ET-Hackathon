from __future__ import annotations

SCHEMA_VERSION = "1.0.0"

STAGE_ORDER = ["ingest", "plan", "assets", "audio", "render", "export"]

JOB_SUBDIRECTORIES = [
    "input",
    "parsed",
    "storyboard",
    "assets",
    "audio",
    "renders",
    "logs",
]
