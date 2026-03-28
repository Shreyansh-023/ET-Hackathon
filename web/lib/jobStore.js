import fs from "node:fs/promises";
import path from "node:path";

export const STAGE_ORDER = ["ingest", "plan", "assets", "audio", "render", "export"];

export function resolveRepoRoot() {
  return process.env.ET_REPO_ROOT || path.resolve(process.cwd(), "..");
}

export function resolveJobsRoot() {
  return path.join(resolveRepoRoot(), "jobs");
}

export function resolveJobDir(jobId) {
  return path.join(resolveJobsRoot(), jobId);
}

export function normalizeArtifactPath(pathValue) {
  return String(pathValue || "").replace(/\\/g, "/");
}

export async function readJson(filePath) {
  const raw = await fs.readFile(filePath, "utf-8");
  return JSON.parse(raw);
}

export async function getJobStatus(jobId) {
  const jobDir = resolveJobDir(jobId);
  const jobState = await readJson(path.join(jobDir, "job_state.json"));
  const manifest = await readJson(path.join(jobDir, "manifest.json"));

  let language = "unknown";
  try {
    const article = await readJson(path.join(jobDir, "parsed", "article.json"));
    if (article?.language) {
      language = article.language;
    }
  } catch (err) {
    language = "unknown";
  }

  const stages = {};
  for (const stage of STAGE_ORDER) {
    const entry = manifest?.stages?.[stage];
    if (entry) {
      stages[stage] = entry;
      continue;
    }
    if (jobState?.completed_stages?.includes(stage)) {
      stages[stage] = { status: "completed" };
      continue;
    }
    if (jobState?.current_stage === stage) {
      stages[stage] = { status: "started" };
      continue;
    }
    stages[stage] = { status: "pending" };
  }

  const artifacts = manifest?.artifacts || {};
  const videoKey = artifacts.final
    ? "final"
    : artifacts.preview
      ? "preview"
      : artifacts.intermediate_with_audio
        ? "intermediate_with_audio"
        : artifacts.intermediate_raw
          ? "intermediate_raw"
          : null;

  const videoUrl = videoKey ? `/api/jobs/${jobId}/artifact?key=${videoKey}` : null;
  const thumbnailUrl = artifacts.thumbnail
    ? `/api/jobs/${jobId}/artifact?key=thumbnail`
    : null;

  return {
    jobId,
    currentStage: jobState?.current_stage || "created",
    completedStages: jobState?.completed_stages || [],
    stages,
    artifacts,
    language,
    videoUrl,
    thumbnailUrl,
    updatedAt: jobState?.updated_at || manifest?.updated_at || ""
  };
}
