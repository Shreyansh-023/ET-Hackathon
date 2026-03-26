from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import uuid
from typing import Any

from src.common.constants import JOB_SUBDIRECTORIES, STAGE_ORDER
from src.common.errors import IOPipelineError, ValidationPipelineError
from src.common.models import Manifest, RenderJob
from src.common.validation import validate_payload
from src.observability.json_logger import JsonStageLogger


class JobRepository:
    def __init__(self, jobs_root: str) -> None:
        self.jobs_root = Path(jobs_root)
        self.jobs_root.mkdir(parents=True, exist_ok=True)

    def create_job(self, dry_run: bool) -> RenderJob:
        job_id = datetime.now(tz=timezone.utc).strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
        job_dir = self.job_path(job_id)
        for name in JOB_SUBDIRECTORIES:
            (job_dir / name).mkdir(parents=True, exist_ok=True)

        job = RenderJob(
            job_id=job_id,
            dry_run=dry_run,
            paths={name: str((job_dir / name).as_posix()) for name in JOB_SUBDIRECTORIES},
        )
        manifest = Manifest(job_id=job_id)

        self.save_render_job(job)
        self.save_manifest(manifest)
        return job

    def job_path(self, job_id: str) -> Path:
        return self.jobs_root / job_id

    def _read_json(self, path: Path) -> dict[str, Any]:
        try:
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except OSError as exc:
            raise IOPipelineError(f"Failed reading {path}: {exc}") from exc

    def _write_json(self, path: Path, payload: dict[str, Any]) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, sort_keys=True)
                f.write("\n")
        except OSError as exc:
            raise IOPipelineError(f"Failed writing {path}: {exc}") from exc

    def load_render_job(self, job_id: str) -> RenderJob:
        state_path = self.job_path(job_id) / "job_state.json"
        if not state_path.exists():
            raise ValidationPipelineError(f"Job does not exist: {job_id}")
        payload = self._read_json(state_path)
        return validate_payload(RenderJob, payload)

    def save_render_job(self, job: RenderJob) -> None:
        updated = job.model_copy(update={"updated_at": datetime.now(tz=timezone.utc).isoformat()})
        self._write_json(self.job_path(updated.job_id) / "job_state.json", updated.model_dump())

    def load_manifest(self, job_id: str) -> Manifest:
        payload = self._read_json(self.job_path(job_id) / "manifest.json")
        return validate_payload(Manifest, payload)

    def save_manifest(self, manifest: Manifest) -> None:
        updated = manifest.model_copy(update={"updated_at": datetime.now(tz=timezone.utc).isoformat()})
        self._write_json(self.job_path(updated.job_id) / "manifest.json", updated.model_dump())

    def update_stage_status(self, job_id: str, stage: str, status: str, **extra: Any) -> None:
        manifest = self.load_manifest(job_id)
        stages = dict(manifest.stages)
        stages[stage] = {
            "status": status,
            "updated_at": datetime.now(tz=timezone.utc).isoformat(),
            **extra,
        }
        self.save_manifest(manifest.model_copy(update={"stages": stages}))

    def add_artifact(self, job_id: str, key: str, relative_path: str) -> None:
        manifest = self.load_manifest(job_id)
        artifacts = dict(manifest.artifacts)
        artifacts[key] = relative_path
        self.save_manifest(manifest.model_copy(update={"artifacts": artifacts}))

    def get_stage_logger(self, job_id: str, stage: str) -> JsonStageLogger:
        return JsonStageLogger(self.job_path(job_id) / "logs" / f"{stage}.jsonl")

    def ensure_can_run_stage(self, job: RenderJob, stage: str) -> None:
        if stage not in STAGE_ORDER:
            raise ValidationPipelineError(f"Unknown stage: {stage}")
        if stage in job.completed_stages:
            return
        if not job.can_run(stage):
            raise ValidationPipelineError(
                f"Cannot run stage '{stage}' before required prior stage is completed"
            )

    def mark_stage_started(self, job: RenderJob, stage: str) -> RenderJob:
        self.ensure_can_run_stage(job, stage)
        updated = job.model_copy(update={"current_stage": stage})
        self.save_render_job(updated)
        self.update_stage_status(updated.job_id, stage, "started")
        return updated

    def mark_stage_completed(self, job: RenderJob, stage: str) -> RenderJob:
        completed = list(job.completed_stages)
        if stage not in completed:
            completed.append(stage)
        updated = job.model_copy(update={"current_stage": stage, "completed_stages": completed})
        self.save_render_job(updated)
        self.update_stage_status(updated.job_id, stage, "completed")
        return updated
