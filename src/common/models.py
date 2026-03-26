from __future__ import annotations

from hashlib import sha256
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.common.constants import SCHEMA_VERSION, STAGE_ORDER


def utc_now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Article(StrictModel):
    schema_version: str = Field(default=SCHEMA_VERSION)
    article_id: str
    source: str
    language: str = "unknown"
    title: str = ""
    raw_text: str
    clean_article_text: str
    created_at: str = Field(default_factory=utc_now_iso)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ArticleKeyPoint(StrictModel):
    text: str
    importance: int = Field(ge=1, le=5)


class ArticleUnderstanding(StrictModel):
    schema_version: str = Field(default=SCHEMA_VERSION)
    headline: str
    summary: str
    key_points: list[ArticleKeyPoint] = Field(min_length=1)
    entities: list[str] = Field(default_factory=list)
    visual_hooks: list[str] = Field(default_factory=list)
    tone: str
    topic: str


class Scene(StrictModel):
    schema_version: str = Field(default=SCHEMA_VERSION)
    scene_id: str
    index: int
    id: str = ""
    start: float = 0.0
    end: float = 5.0
    type: Literal["hook", "body", "closing"] = "body"
    narration: str
    on_screen_text: str = ""
    visual_strategy: str = "editorial_still"
    visual_prompt: str = ""
    duration_seconds: float = 5.0
    transition_hint: str = ""
    motion_hint: str = ""
    factual_risk: Literal["low", "medium", "high"] = "medium"
    visual_seed: int = 0
    source_key_points: list[str] = Field(default_factory=list)
    visual_suggestions: dict[str, Any] | None = None

    @model_validator(mode="after")
    def _sync_derived_fields(self) -> "Scene":
        if not self.id:
            self.id = self.scene_id
        else:
            self.scene_id = self.id

        if self.end <= self.start:
            self.end = self.start + max(self.duration_seconds, 0.1)

        self.duration_seconds = round(self.end - self.start, 3)

        if self.visual_seed == 0:
            digest = sha256(self.scene_id.encode("utf-8")).hexdigest()
            self.visual_seed = int(digest[:8], 16)

        return self


class Asset(StrictModel):
    schema_version: str = Field(default=SCHEMA_VERSION)
    asset_id: str
    scene_id: str
    kind: Literal["image", "video", "audio", "subtitle", "other"] = "other"
    source: str = "local"
    path: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class RenderJob(StrictModel):
    schema_version: str = Field(default=SCHEMA_VERSION)
    job_id: str
    created_at: str = Field(default_factory=utc_now_iso)
    updated_at: str = Field(default_factory=utc_now_iso)
    current_stage: str = "created"
    completed_stages: list[str] = Field(default_factory=list)
    dry_run: bool = False
    article_id: str | None = None
    paths: dict[str, str] = Field(default_factory=dict)

    def can_run(self, stage: str) -> bool:
        if stage not in STAGE_ORDER:
            return False
        stage_index = STAGE_ORDER.index(stage)
        if stage_index == 0:
            return True
        required_prev = STAGE_ORDER[stage_index - 1]
        return required_prev in self.completed_stages


class Manifest(StrictModel):
    schema_version: str = Field(default=SCHEMA_VERSION)
    job_id: str
    created_at: str = Field(default_factory=utc_now_iso)
    updated_at: str = Field(default_factory=utc_now_iso)
    stages: dict[str, dict[str, Any]] = Field(default_factory=dict)
    artifacts: dict[str, str] = Field(default_factory=dict)
