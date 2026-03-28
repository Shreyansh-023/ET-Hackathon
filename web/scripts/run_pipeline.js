import { spawnSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";

function readArg(name) {
  const index = process.argv.indexOf(name);
  if (index === -1 || index + 1 >= process.argv.length) {
    return "";
  }
  return process.argv[index + 1];
}

const jobId = readArg("--job-id");
const pythonBin = readArg("--python") || "python";
const repoRoot = readArg("--repo-root") || path.resolve(process.cwd(), "..");

if (!jobId) {
  process.exit(1);
}

const logPath = path.join(repoRoot, "jobs", jobId, "logs", "web_runner.log");

function logLine(message) {
  try {
    fs.appendFileSync(logPath, `[${new Date().toISOString()}] ${message}\n`);
  } catch (err) {
    process.stdout.write(`${message}\n`);
  }
}

const stages = ["plan", "assets", "audio", "render", "export"];
for (const stage of stages) {
  logLine(`stage_start ${stage}`);
  const result = spawnSync(
    pythonBin,
    ["-m", "src.cli", stage, jobId],
    { cwd: repoRoot, env: process.env, encoding: "utf-8" }
  );

  if (result.stdout) {
    logLine(result.stdout.trim());
  }
  if (result.stderr) {
    logLine(result.stderr.trim());
  }

  if (result.status !== 0) {
    logLine(`stage_failed ${stage} exit=${result.status}`);
    process.exit(1);
  }

  logLine(`stage_completed ${stage}`);
}
